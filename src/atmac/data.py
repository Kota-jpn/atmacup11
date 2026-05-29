"""データ読み込み / CV分割 / 補助ターゲット / Dataset・transform。"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold

# sorting_date 正規化（range 1440-1991）。補助回帰ヘッド用。
DATE_C, DATE_S = 1715.0, 100.0


def load_data(data_dir):
    d = lambda f: os.path.join(data_dir, f)
    train = pd.read_csv(d("train.csv"))
    test = pd.read_csv(d("test.csv"))
    materials = pd.read_csv(d("materials.csv"))
    techniques = pd.read_csv(d("techniques.csv"))
    return train, test, materials, techniques


def make_folds(train, n_folds=5, seed=42):
    """StratifiedGroupKFold: group=art_series_id, stratify=target(=sorting_dateの世紀bin)。"""
    train = train.copy()
    train["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(sgkf.split(train, train["target"], train["art_series_id"])):
        train.loc[va, "fold"] = f
    return train


def build_targets(train, materials, min_mat_count=30):
    """train に date_norm と materials マルチホット列を付与。mat_cols を返す。"""
    train = train.copy()
    train["date_norm"] = (train["sorting_date"] - DATE_C) / DATE_S

    vc = materials["name"].value_counts()
    mat_names = vc[vc >= min_mat_count].index.tolist()
    mat_cols = [f"mat__{n}" for n in mat_names]
    # object_id × material のマルチホット
    sub = materials[materials["name"].isin(mat_names)]
    oh = pd.crosstab(sub["object_id"], sub["name"]).clip(upper=1)
    oh = oh.reindex(columns=mat_names, fill_value=0)
    oh.columns = mat_cols
    train = train.merge(oh, left_on="object_id", right_index=True, how="left")
    train[mat_cols] = train[mat_cols].fillna(0).astype("float32")
    return train, mat_cols


def _pad_to_square(img):
    w, h = img.size
    s = max(w, h)
    bg = Image.new("RGB", (s, s), (255, 255, 255))
    bg.paste(img, ((s - w) // 2, (s - h) // 2))
    return bg


def build_transforms(cfg, train=True):
    base = [T.Lambda(_pad_to_square), T.Resize((cfg.img_size, cfg.img_size))]
    if train:
        aug = [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(0.1, 0.1, 0.1),   # 色は重要なので控えめ（破壊系augは避ける）
        ]
    else:
        aug = []
    return T.Compose(base + aug + [T.ToTensor(), T.Normalize(cfg.mean, cfg.std)])


class ArtDataset(Dataset):
    """mode='train': (img, dict(target,date,mat)) / 'test': img のみ。"""
    def __init__(self, df, photo_dir, transform, mat_cols=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.photo_dir = photo_dir
        self.tf = transform
        self.mat_cols = mat_cols or []
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.photo_dir, f"{r.object_id}.jpg")).convert("RGB")
        img = self.tf(img)
        if self.mode == "test":
            return img
        y = {
            "target": torch.tensor(r["target"], dtype=torch.long),
            "date": torch.tensor(r["date_norm"], dtype=torch.float32),
            "mat": torch.tensor(r[self.mat_cols].to_numpy(dtype="float32")) if self.mat_cols
                   else torch.zeros(0),
        }
        return img, y
