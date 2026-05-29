"""CPUで全工程を最小データ・最小epochで疎通確認する。
.venv/bin/python src/smoke_test.py
GPUは使わず、コードが例外なく一周することのみ検証（スコアは無意味）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from atmac.config import CFG
from atmac import data as D
from atmac.ssl_dino import pretrain_ssl
from atmac.engine import train_one_fold, rmse
from atmac.stack import svd_embed, stack_lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PH = os.path.join(RAW, "photos")
torch.manual_seed(0); np.random.seed(0)

cfg = CFG(data_dir=RAW, img_size=64, ssl_epochs=1, ssl_batch=8,
          epochs=2, batch_size=8, num_workers=0, n_folds=5)

print("1) load + folds + targets")
train, test, materials, techniques = D.load_data(RAW)
train = D.make_folds(train, cfg.n_folds, cfg.seed)
train, mat_cols = D.build_targets(train, materials, cfg.min_mat_count)
print("   train", train.shape, "| mat_cols", len(mat_cols), "| fold dist", train.fold.value_counts().to_dict())
# fold内リーク確認
for f in range(cfg.n_folds):
    leak = set(train[train.fold == f].art_series_id) & set(train[train.fold != f].art_series_id)
    assert len(leak) == 0, f"fold{f} leak"
print("   GroupKFold leak-free ✅")

# 小サブセット（処理済trainから各fold少数を抽出。再処理しない）
sub = train.groupby("fold").head(12).reset_index(drop=True)
sub_test = test.head(10).copy()

print("2) SSL pretrain (dino, 1ep, 40枚)")
paths = [os.path.join(PH, f"{o}.jpg") for o in train.object_id.head(40)]
ssl_state = pretrain_ssl(cfg, paths, device="cpu",
                         log_fn=lambda d: print("   ", d))
print("   ssl encoder keys:", len(ssl_state))

print("3) finetune fold0 (cls, 2ep)")
cfg.task = "cls"; cfg.use_mixup = True
res = train_one_fold(cfg, sub, PH, fold=0, mat_cols=mat_cols,
                     ssl_state=ssl_state, test=sub_test, device="cpu",
                     log_fn=lambda d: print("   ", d))
print("   oof_pred", res["oof_pred"].shape, "test_pred", res["test_pred"].shape,
      "val_feat", res["val_feat"].shape, "best_rmse", round(res["best_rmse"], 4))
assert np.isfinite(res["oof_pred"]).all() and (0 <= res["test_pred"]).all() and (res["test_pred"] <= 3).all()

print("4) reg task fold0 (1ep, no mixup)")
cfg.task = "reg"; cfg.use_mixup = False; cfg.epochs = 1
res2 = train_one_fold(cfg, sub, PH, fold=0, mat_cols=mat_cols, ssl_state=None,
                      test=sub_test, device="cpu", log_fn=None)
print("   reg oof", res2["oof_pred"][:3].round(3), "test", res2["test_pred"][:3].round(3))

print("5) stack (svd embed + lgb)")
# 2モデルのOOF予測 + embedding を特徴に
emb_tr, emb_te = svd_embed(res["val_feat"], res["test_feat"], k=8)
X_tr = np.column_stack([res["oof_pred"], res2["oof_pred"], emb_tr])
X_te = np.column_stack([res["test_pred"], res2["test_pred"], emb_te])
groups = sub.loc[res["oof_idx"], "art_series_id"].to_numpy()
y = res["oof_y"]
oof_s, pred_s = stack_lgb(X_tr, y, X_te, groups, n_folds=3)
print("   stack oof_rmse", round(rmse(oof_s, y), 4), "| pred range",
      round(pred_s.min(), 3), round(pred_s.max(), 3))

print("\nSMOKE TEST PASSED ✅ — 全工程が例外なく一周")
