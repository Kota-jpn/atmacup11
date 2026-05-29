"""オーケストレーション: SSL → マルチタスクFT(cls+reg, 5fold) → スタック → 提出。
Colab: python -m atmac.run --data_dir /content/data --out_dir /content/drive/MyDrive/atmaCup11/outputs

SSL重み・fold結果はキャッシュ。再実行で済んだ工程はスキップ（GPU節約）。
"""
import os, argparse, glob
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import CFG
from . import data as D
from .ssl_dino import pretrain_ssl
from .engine import train_one_fold, rmse
from .stack import svd_embed, stack_lgb

try:
    import wandb
except Exception:
    wandb = None


def image_meta(ids, photo_dir):
    rows = []
    for o in ids:
        with Image.open(os.path.join(photo_dir, f"{o}.jpg")) as im:
            w, h = im.size
        rows.append((w, h, w / h, max(w, h)))
    return np.array(rows, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ssl_method", default="dino")
    ap.add_argument("--ssl_epochs", type=int, default=100)
    ap.add_argument("--ssl_img_size", type=int, default=128)
    ap.add_argument("--backbone", default="resnet18d")
    ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--tasks", default="cls,reg")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--ssl_batch", type=int, default=256)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="デバッグ: train/testを各N行に縮小")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    photo_dir = os.path.join(a.data_dir, "photos")
    mdir = os.path.join(a.out_dir, "models"); os.makedirs(mdir, exist_ok=True)
    cdir = os.path.join(a.out_dir, "cache"); os.makedirs(cdir, exist_ok=True)
    sdir = os.path.join(a.out_dir, "submissions"); os.makedirs(sdir, exist_ok=True)

    cfg = CFG(data_dir=a.data_dir, out_dir=a.out_dir, ssl_method=a.ssl_method,
              ssl_epochs=a.ssl_epochs, ssl_img_size=a.ssl_img_size, ssl_batch=a.ssl_batch,
              ssl_backbone=a.backbone, backbone=a.backbone, num_workers=a.num_workers,
              epochs=a.ft_epochs, img_size=a.img_size, batch_size=a.batch_size)

    def wlog(d):
        print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
        if a.wandb and wandb and wandb.run:
            wandb.log(d)

    # ---- データ ----
    train, test, materials, _ = D.load_data(a.data_dir)
    if a.limit:
        train = train.groupby("target").head(max(2, a.limit // 4)).reset_index(drop=True)
        test = test.head(a.limit).reset_index(drop=True)
    train = D.make_folds(train, cfg.n_folds, cfg.seed)
    train, mat_cols = D.build_targets(train, materials, cfg.min_mat_count)
    print(f"train {train.shape} | test {len(test)} | mat_cols {len(mat_cols)}")

    # ---- Phase A: SSL（キャッシュ）。none ならスキップ＝スクラッチFT ----
    if a.ssl_method == "none":
        print("[SSL] skip (scratch baseline)")
        ssl_state = None
    elif os.path.exists(ssl_path := os.path.join(
            mdir, f"ssl_{a.ssl_method}_{a.backbone}_{a.ssl_img_size}_{a.ssl_epochs}ep.pth")):
        print(f"[SSL] cache hit: {ssl_path}")
        ssl_state = torch.load(ssl_path, map_location="cpu")
    else:
        if a.wandb and wandb:
            wandb.init(project="atmacup11", name=f"ssl_{a.ssl_method}_{a.backbone}", reinit=True)
        print(f"[SSL] {a.ssl_method} {a.ssl_epochs}ep on {len(train)+len(test)} imgs ...")
        paths = [os.path.join(photo_dir, f"{o}.jpg")
                 for o in pd.concat([train.object_id, test.object_id])]
        ssl_state = pretrain_ssl(cfg, paths, device=device, log_fn=wlog,
                                 ckpt_path=ssl_path + ".ckpt")
        torch.save(ssl_state, ssl_path)
        if a.wandb and wandb and wandb.run:
            wandb.finish()
        print(f"[SSL] saved {ssl_path}")

    # ---- Phase B: マルチタスクFT（task×fold, キャッシュ）----
    store = {}  # task -> dict(oof, test, val_feat per fold...)
    for task in a.tasks.split(","):
        cfg.task = task
        cfg.use_mixup = (task == "cls")   # 回帰はmixup無効（世紀の中間が無意味）
        oof = np.zeros(len(train)); oof_y = train["target"].to_numpy(float)
        test_pred = np.zeros(len(test))
        val_feat = test_feat = None   # 特徴次元はbackboneから動的に確保
        for fold in range(cfg.n_folds):
            ck = os.path.join(cdir, f"ft_{a.ssl_method}_{task}_{a.backbone}_f{fold}.npz")
            if os.path.exists(ck):
                z = np.load(ck)
                print(f"[FT {task} f{fold}] cache hit (rmse {float(z['best_rmse']):.4f})")
            else:
                if a.wandb and wandb:
                    wandb.init(project="atmacup11", name=f"ft_{task}_{a.backbone}_f{fold}",
                               config=vars(cfg) if hasattr(cfg, "__dict__") else None, reinit=True)
                r = train_one_fold(cfg, train, photo_dir, fold, mat_cols,
                                   ssl_state=ssl_state, test=test, device=device, log_fn=wlog)
                if a.wandb and wandb and wandb.run:
                    wandb.summary["best_val_rmse"] = r["best_rmse"]; wandb.finish()
                np.savez(ck, oof_idx=r["oof_idx"], oof_pred=r["oof_pred"], oof_y=r["oof_y"],
                         val_feat=r["val_feat"], test_pred=r["test_pred"], test_feat=r["test_feat"],
                         best_rmse=r["best_rmse"])
                z = np.load(ck)
            if val_feat is None:   # 初fold で特徴次元を確定して確保
                fdim = z["val_feat"].shape[1]
                val_feat = np.zeros((len(train), fdim), dtype="float32")
                test_feat = np.zeros((len(test), fdim), dtype="float32")
            oof[z["oof_idx"]] = z["oof_pred"]
            val_feat[z["oof_idx"]] = z["val_feat"]
            test_pred += z["test_pred"] / cfg.n_folds
            test_feat += z["test_feat"] / cfg.n_folds
        cv = rmse(np.clip(oof, 0, 3), oof_y)
        print(f"[FT {task}] OOF RMSE = {cv:.4f}")
        store[task] = dict(oof=oof, test=test_pred, val_feat=val_feat, test_feat=test_feat, cv=cv)

    # ---- Phase C: 後処理ブレンド + スタック ----
    sub_ss = pd.read_csv(os.path.join(a.data_dir, "atmaCup#11_sample_submission.csv"))
    tasks = list(store)
    # (1) 単純平均ブレンド
    blend_oof = np.clip(np.mean([store[t]["oof"] for t in tasks], 0), 0, 3)
    blend_test = np.clip(np.mean([store[t]["test"] for t in tasks], 0), 0, 3)
    print(f"[BLEND] OOF RMSE = {rmse(blend_oof, train['target']):.4f}")
    pd.DataFrame({"target": blend_test}).to_csv(
        os.path.join(sdir, f"sub_blend_{a.backbone}.csv"), index=False)

    # (2) LGBスタック（各task予測 + embedding SVD + 画像メタ）
    try:
        meta_tr = image_meta(train.object_id, photo_dir)
        meta_te = image_meta(test.object_id, photo_dir)
        emb_tr, emb_te = svd_embed(store[tasks[0]]["val_feat"], store[tasks[0]]["test_feat"], k=16)
        Xtr = np.column_stack([store[t]["oof"] for t in tasks] + [emb_tr, meta_tr])
        Xte = np.column_stack([store[t]["test"] for t in tasks] + [emb_te, meta_te])
        oof_s, pred_s = stack_lgb(Xtr, train["target"].to_numpy(float), Xte,
                                  train["art_series_id"].to_numpy(), cfg.n_folds, cfg.seed)
        print(f"[STACK] OOF RMSE = {rmse(oof_s, train['target']):.4f}")
        pd.DataFrame({"target": pred_s}).to_csv(
            os.path.join(sdir, f"sub_stack_{a.backbone}.csv"), index=False)
    except Exception as e:
        print("[STACK] skipped:", e)

    assert len(blend_test) == len(test), "予測行数がtestと不一致"
    if not a.limit:
        assert len(blend_test) == len(sub_ss), "提出行数がsample_submissionと不一致"
    print("DONE. submissions ->", sdir)


if __name__ == "__main__":
    main()
