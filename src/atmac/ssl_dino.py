"""SSL 事前学習（lightly）。DINO(本命) / SimSiam(実験枠)。
全画像(train+test 9856枚)で学習し、encoder の state_dict を返す。
"""
import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import timm


class _ImgPaths(Dataset):
    """画像パス → transform(img)。DINOはview list、SimSiamは(v0,v1)を返す。"""
    def __init__(self, paths, transform):
        self.paths = paths
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


# ---------------- DINO ----------------
class _DINO(nn.Module):
    def __init__(self, backbone_name, out_dim=2048):
        super().__init__()
        self.student_backbone = timm.create_model(backbone_name, pretrained=False,
                                                   num_classes=0, global_pool="avg")
        feat = self.student_backbone.num_features
        from lightly.models.modules import DINOProjectionHead
        self.student_head = DINOProjectionHead(feat, 512, 64, out_dim, freeze_last_layer=1)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(feat, 512, 64, out_dim)
        from lightly.models.utils import deactivate_requires_grad
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        return self.student_head(self.student_backbone(x))

    def forward_teacher(self, x):
        return self.teacher_head(self.teacher_backbone(x))


def _dino_collate(batch):
    # batch: list(len B) of list(len V) of tensors → list(len V) of (B,C,H,W)
    n_views = len(batch[0])
    return [torch.stack([b[v] for b in batch]) for v in range(n_views)]


def _train_dino(cfg, paths, device, log_fn):
    from lightly.transforms.dino_transform import DINOTransform
    from lightly.loss import DINOLoss
    from lightly.models.utils import update_momentum
    from lightly.utils.scheduler import cosine_schedule

    sz = cfg.ssl_img_size
    tf = DINOTransform(global_crop_size=sz, local_crop_size=sz // 2,
                       n_local_views=cfg.ssl_local_crops,
                       normalize={"mean": cfg.mean, "std": cfg.std})
    dl = DataLoader(_ImgPaths(paths, tf), batch_size=cfg.ssl_batch, shuffle=True,
                    num_workers=cfg.num_workers, collate_fn=_dino_collate,
                    drop_last=True, pin_memory=True)
    model = _DINO(cfg.ssl_backbone).to(device)
    criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ssl_lr, weight_decay=1e-4)
    use_cuda = device == "cuda"
    scaler = GradScaler(enabled=use_cuda)

    for ep in range(cfg.ssl_epochs):
        m = cosine_schedule(ep, cfg.ssl_epochs, 0.996, 1.0)
        update_momentum(model.student_backbone, model.teacher_backbone, m)
        update_momentum(model.student_head, model.teacher_head, m)
        tot = 0.0
        pbar = tqdm(dl, desc=f"DINO ep{ep+1}/{cfg.ssl_epochs}", leave=False)
        for views in pbar:
            views = [v.to(device, non_blocking=True) for v in views]
            with autocast("cuda", enabled=use_cuda):
                teacher_out = [model.forward_teacher(v) for v in views[:2]]
                student_out = [model.forward(v) for v in views]
                loss = criterion(teacher_out, student_out, epoch=ep)
            opt.zero_grad()
            scaler.scale(loss).backward()
            model.student_head.cancel_last_layer_gradients(current_epoch=ep)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        if log_fn:
            log_fn({"ssl_epoch": ep, "ssl_loss": tot / max(1, len(dl)), "momentum": float(m)})
    return model.student_backbone.state_dict()


# ---------------- SimSiam ----------------
class _SimSiam(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        feat = self.backbone.num_features
        from lightly.models.modules import SimSiamProjectionHead, SimSiamPredictionHead
        self.projection = SimSiamProjectionHead(feat, 512, 128)
        self.prediction = SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        z = self.projection(self.backbone(x))
        p = self.prediction(z)
        return z.detach(), p


def _train_simsiam(cfg, paths, device, log_fn):
    from lightly.transforms import SimSiamTransform
    from lightly.loss import NegativeCosineSimilarity

    tf = SimSiamTransform(input_size=cfg.ssl_img_size,
                          normalize={"mean": cfg.mean, "std": cfg.std})
    dl = DataLoader(_ImgPaths(paths, tf), batch_size=cfg.ssl_batch, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    model = _SimSiam(cfg.ssl_backbone).to(device)
    criterion = NegativeCosineSimilarity()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.ssl_lr * cfg.ssl_batch / 256,
                          momentum=0.9, weight_decay=1e-4)
    use_cuda = device == "cuda"
    scaler = GradScaler(enabled=use_cuda)
    for ep in range(cfg.ssl_epochs):
        tot = 0.0
        pbar = tqdm(dl, desc=f"SimSiam ep{ep+1}/{cfg.ssl_epochs}", leave=False)
        for views in pbar:
            x0, x1 = views[0].to(device, non_blocking=True), views[1].to(device, non_blocking=True)
            with autocast("cuda", enabled=use_cuda):
                z0, p0 = model(x0)
                z1, p1 = model(x1)
                loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        if log_fn:
            log_fn({"ssl_epoch": ep, "ssl_loss": tot / max(1, len(dl))})
    return model.backbone.state_dict()


def pretrain_ssl(cfg, image_paths, device="cuda", log_fn=None):
    """全画像で SSL → encoder(state_dict) を返す。"""
    if cfg.ssl_method == "dino":
        return _train_dino(cfg, image_paths, device, log_fn)
    elif cfg.ssl_method == "simsiam":
        return _train_simsiam(cfg, image_paths, device, log_fn)
    raise ValueError(cfg.ssl_method)
