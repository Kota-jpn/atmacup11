"""ファインチューニング学習ループ + 後処理 + 推論。マルチタスク(main/date/mat)。"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from timm.utils import ModelEmaV2

from .data import ArtDataset, build_transforms
from .model import MultiTaskNet


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def softmax_to_continuous(logits, n_classes=4):
    """分類logit → softmax×クラスindex の加重平均（argmax禁止＝5位の劇的改善）。"""
    p = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    idx = np.arange(n_classes)
    return (p * idx).sum(axis=1)


def _rand_bbox(h, w, lam):
    r = math.sqrt(1.0 - lam)
    cw, ch = int(w * r), int(h * r)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = np.clip(cx - cw // 2, 0, w), np.clip(cy - ch // 2, 0, h)
    x2, y2 = np.clip(cx + cw // 2, 0, w), np.clip(cy + ch // 2, 0, h)
    return x1, y1, x2, y2


def _mix(x, alpha):
    """mixup と cutmix を50%で切替。x を混合し (x, lam, perm) を返す。"""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:
        x = lam * x + (1 - lam) * x[perm]
    else:
        x1, y1, x2, y2 = _rand_bbox(x.size(2), x.size(3), lam)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
    return x, float(lam), perm


def _multitask_loss(cfg, out, y, lam=1.0, perm=None):
    """main + date + mat。mixup時は perm 側を lam で混合。"""
    def blend(loss_fn, pred, t):
        if perm is None:
            return loss_fn(pred, t)
        return lam * loss_fn(pred, t) + (1 - lam) * loss_fn(pred, t[perm])

    if cfg.task == "cls":
        main = blend(lambda p, t: F.cross_entropy(p, t), out["main"], y["target"])
    else:
        main = blend(lambda p, t: F.mse_loss(p.squeeze(1), t.float()),
                     out["main"], y["target"])
    loss = cfg.w_main * main
    loss = loss + cfg.w_date * blend(F.mse_loss, out["date"], y["date"])
    if "mat" in out and y["mat"].numel() > 0:
        loss = loss + cfg.w_mat * blend(F.binary_cross_entropy_with_logits, out["mat"], y["mat"])
    return loss


@torch.no_grad()
def _predict(cfg, model, loader, device, with_feat=False):
    model.eval()
    mains, feats = [], []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        out = model(x, return_feat=with_feat)
        o = out["main"]
        if cfg.use_tta:
            out2 = model(torch.flip(x, dims=[3]), return_feat=False)
            o = (o + out2["main"]) / 2
        mains.append(o.cpu().numpy())
        if with_feat:
            feats.append(out["feat"].cpu().numpy())
    mains = np.concatenate(mains)
    feats = np.concatenate(feats) if with_feat else None
    return mains, feats


def _to_continuous(cfg, main):
    if cfg.task == "cls":
        return np.clip(softmax_to_continuous(main, cfg.n_classes), 0, 3)
    return np.clip(main.squeeze(1), 0, 3)


def train_one_fold(cfg, train, photo_dir, fold, mat_cols,
                   ssl_state=None, test=None, device="cuda", log_fn=None):
    tr_df = train[train.fold != fold]
    va_df = train[train.fold == fold]
    tr_tf, va_tf = build_transforms(cfg, True), build_transforms(cfg, False)
    extra = dict(persistent_workers=True, prefetch_factor=4) if cfg.num_workers > 0 else {}
    mk = lambda df, tf, mode: DataLoader(
        ArtDataset(df, photo_dir, tf, mat_cols, mode),
        batch_size=cfg.batch_size, shuffle=(mode == "train"),
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=(mode == "train"), **extra)
    tr_dl = mk(tr_df, tr_tf, "train")
    va_dl = mk(va_df, va_tf, "val")

    model = MultiTaskNet(cfg.backbone, cfg.task, cfg.n_classes, len(mat_cols)).to(device)
    if ssl_state is not None:
        miss, unexp = model.load_ssl_encoder(ssl_state)
        if log_fn:
            log_fn({"ssl_load_missing": len(miss), "ssl_load_unexpected": len(unexp)})
    ema = ModelEmaV2(model, decay=cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_cuda = device == "cuda"
    scaler = GradScaler(enabled=use_cuda)

    y_val = va_df["target"].to_numpy()
    best = {"rmse": 1e9, "ema_state": None}
    for ep in range(cfg.epochs):
        model.train()
        tl = 0.0
        pbar = tqdm(tr_dl, desc=f"FT {cfg.task} f{fold} ep{ep+1}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
            lam, perm = 1.0, None
            if cfg.use_mixup:
                x, lam, perm = _mix(x, cfg.mixup_alpha)
            with autocast("cuda", enabled=use_cuda):
                out = model(x)
                loss = _multitask_loss(cfg, out, y, lam, perm)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ema.update(model)
            tl += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        main, _ = _predict(cfg, ema.module, va_dl, device)
        v = rmse(_to_continuous(cfg, main), y_val)
        if log_fn:
            log_fn({"fold": fold, "epoch": ep, "train_loss": tl / len(tr_df),
                    "val_rmse": v, "lr": sched.get_last_lr()[0]})
        if v < best["rmse"]:
            best = {"rmse": v, "ema_state": {k: t.clone() for k, t in ema.module.state_dict().items()}}

    # best(ema) で OOF / test 予測 + embedding
    model.load_state_dict(best["ema_state"])
    val_main, val_feat = _predict(cfg, model, va_dl, device, with_feat=True)
    oof = _to_continuous(cfg, val_main)
    res = {"fold": fold, "best_rmse": best["rmse"], "oof_idx": va_df.index.to_numpy(),
           "oof_pred": oof, "oof_y": y_val, "val_feat": val_feat,
           "ema_state": best["ema_state"]}
    if test is not None:
        te_dl = mk(test, va_tf, "test")
        te_main, te_feat = _predict(cfg, model, te_dl, device, with_feat=True)
        res["test_pred"] = _to_continuous(cfg, te_main)
        res["test_feat"] = te_feat
    return res
