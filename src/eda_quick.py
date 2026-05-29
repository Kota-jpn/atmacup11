"""atmaCup #11 クイックEDA。ローカルvenvで実行し環境検証も兼ねる。
使い方: .venv/bin/python src/eda_quick.py
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from PIL import Image

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

train = pd.read_csv(RAW / "train.csv")
test = pd.read_csv(RAW / "test.csv")
materials = pd.read_csv(RAW / "materials.csv")
techniques = pd.read_csv(RAW / "techniques.csv")

print("=" * 50)
print(f"train {train.shape} / test {test.shape}")
print(f"train columns: {list(train.columns)}")
print(f"test  columns: {list(test.columns)}")

print("\n--- target 分布 ---")
print(train["target"].value_counts().sort_index())
print(f"target mean={train['target'].mean():.3f} std={train['target'].std():.3f}")

print("\n--- sorting_date vs target（境界の確認）---")
print(train.groupby("target")["sorting_date"].agg(["min", "max", "mean"]))

print("\n--- art_series_id ---")
n_series = train["art_series_id"].nunique()
print(f"train rows={len(train)} / unique art_series_id={n_series}")
vc = train["art_series_id"].value_counts()
print(f"1シリーズ複数枚: {(vc > 1).sum()} series（最大 {vc.max()} 枚）")

print("\n--- GroupKFold(art_series_id) 健全性 ---")
gkf = GroupKFold(n_splits=5)
for i, (tr, va) in enumerate(gkf.split(train, train["target"], train["art_series_id"])):
    leak = set(train.iloc[tr]["art_series_id"]) & set(train.iloc[va]["art_series_id"])
    print(f"  fold{i}: train={len(tr)} val={len(va)} "
          f"val_target_mean={train.iloc[va]['target'].mean():.3f} leak_series={len(leak)}")

print("\n--- materials / techniques（train専用補助）---")
print(f"materials: {materials['name'].nunique()} 種, top5:")
print(materials["name"].value_counts().head())
print(f"techniques: {techniques['name'].nunique()} 種, top5:")
print(techniques["name"].value_counts().head())

print("\n--- 画像サイズ サンプル ---")
photos = RAW / "photos"
for oid in train["object_id"].head(5):
    with Image.open(photos / f"{oid}.jpg") as im:
        print(f"  {oid}.jpg: {im.size} {im.mode}")

print("\nEDA OK ✅")
