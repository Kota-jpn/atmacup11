"""atmaCup #11 深掘りEDA。図を outputs/eda/ に保存し、数値も標準出力へ。
.venv/bin/python src/eda_deep.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PH = RAW / "photos"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW / "train.csv")
test = pd.read_csv(RAW / "test.csv")
materials = pd.read_csv(RAW / "materials.csv")
techniques = pd.read_csv(RAW / "techniques.csv")

# ---- 1. 画像メタ(サイズ/アスペクト比/グレースケール/輝度) を train+test 全体で収集 ----
def img_meta(oid):
    with Image.open(PH / f"{oid}.jpg") as im:
        w, h = im.size
        small = im.convert("RGB").resize((32, 32))
    a = np.asarray(small, dtype=np.float32)
    chan_diff = np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()
    return w, h, a.mean(), chan_diff

all_ids = pd.concat([train["object_id"], test["object_id"]]).tolist()
rows = [(oid, *img_meta(oid)) for oid in tqdm(all_ids, desc="img meta")]
meta = pd.DataFrame(rows, columns=["object_id", "w", "h", "brightness", "chan_diff"])
meta["aspect"] = meta["w"] / meta["h"]
meta["longside"] = meta[["w", "h"]].max(axis=1)
meta["is_gray"] = meta["chan_diff"] < 3.0  # ほぼR=G=B
tr = train.merge(meta, on="object_id")

print("=" * 60)
print("[画像サイズ] longside 分布:")
print(meta["longside"].value_counts().head(10).sort_index())
print(f"\nアスペクト比 w/h: mean={meta.aspect.mean():.3f} median={meta.aspect.median():.3f} "
      f"min={meta.aspect.min():.2f} max={meta.aspect.max():.2f}")
print(f"縦長(<0.9): {(meta.aspect<0.9).mean()*100:.1f}%  横長(>1.1): {(meta.aspect>1.1).mean()*100:.1f}%  "
      f"ほぼ正方(0.9-1.1): {((meta.aspect>=0.9)&(meta.aspect<=1.1)).mean()*100:.1f}%")
print(f"\nグレースケール画像: {meta.is_gray.mean()*100:.1f}% (train+test {len(meta)}枚中)")

print("\n[target別] アスペクト比・輝度・グレースケール率:")
print(tr.groupby("target").agg(
    aspect_mean=("aspect", "mean"), aspect_med=("aspect", "median"),
    brightness=("brightness", "mean"), gray_pct=("is_gray", "mean"), n=("object_id", "size")))

# 図: target別アスペクト比 箱ひげ + グレースケール率
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
tr.boxplot(column="aspect", by="target", ax=ax[0]); ax[0].set_title("aspect ratio by target"); ax[0].set_ylim(0, 3)
tr.groupby("target")["is_gray"].mean().plot.bar(ax=ax[1], title="grayscale ratio by target")
tr.groupby("target")["brightness"].mean().plot.bar(ax=ax[2], title="mean brightness by target")
plt.suptitle(""); plt.tight_layout(); plt.savefig(OUT / "01_image_stats_by_target.png", dpi=90); plt.close()

# ---- 2. sorting_date ヒストグラム + target境界 ----
fig, ax = plt.subplots(figsize=(10, 4))
for t in sorted(train.target.unique()):
    ax.hist(train[train.target == t]["sorting_date"], bins=40, alpha=0.6, label=f"target={t}")
ax.set_title("sorting_date distribution by target"); ax.legend(); ax.set_xlabel("year")
plt.tight_layout(); plt.savefig(OUT / "02_sorting_date_hist.png", dpi=90); plt.close()
print("\n[sorting_date] 全体 range:", train.sorting_date.min(), "-", train.sorting_date.max())

# ---- 3. materials / techniques × target （マルチタスク・特徴の有望性確認）----
def name_target_stats(df_long, label):
    m = df_long.merge(train[["object_id", "target"]], on="object_id")
    g = m.groupby("name")["target"].agg(["mean", "std", "count"]).sort_values("count", ascending=False)
    g = g[g["count"] >= 30]
    print(f"\n[{label}] 出現>=30 の name の target平均(年代の傾向、全体mean={train.target.mean():.2f}):")
    print(g.sort_values("mean").to_string())
    return g

mat_g = name_target_stats(materials, "materials")
tec_g = name_target_stats(techniques, "techniques")

# 1作品あたりの素材数 と target
mat_cnt = materials.groupby("object_id").size().rename("n_mat")
tr2 = train.merge(mat_cnt, on="object_id", how="left").fillna({"n_mat": 0})
print("\n[素材数 n_mat × target]:")
print(tr2.groupby("target")["n_mat"].mean())

# ---- 4. art_series 内で target は一致するか ----
series = train.groupby("art_series_id")["target"].agg(["nunique", "count"])
multi = series[series["count"] > 1]
print(f"\n[art_series 内 target一貫性] 複数枚シリーズ {len(multi)}件中、"
      f"target が全て同一: {(multi['nunique']==1).mean()*100:.1f}%")

# ---- 5. クラス別サンプル画像グリッド ----
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for t in range(4):
    ids = train[train.target == t]["object_id"].head(8).tolist()
    for j, oid in enumerate(ids):
        with Image.open(PH / f"{oid}.jpg") as im:
            axes[t, j].imshow(im)
        axes[t, j].axis("off")
        if j == 0:
            axes[t, j].set_ylabel(f"target={t}", rotation=0, labelpad=40, fontsize=12)
    axes[t, 0].set_title(f"target={t}", loc="left")
plt.suptitle("sample images per target (古い→新しい)"); plt.tight_layout()
plt.savefig(OUT / "03_sample_images_by_target.png", dpi=90); plt.close()

print(f"\n図を保存: {OUT}")
print("EDA deep 完了 ✅")
