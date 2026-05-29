"""Colab用 atmaCup#11 ベースラインノートブックを生成する。
.venv/bin/python src/build_notebook.py  ->  notebooks/atmacup11_baseline.ipynb
セル内容(=学習コード)をこのファイルで編集→再生成→Driveへ再アップする運用。
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "atmacup11_baseline.ipynb"

# (cell_type, source) のリスト。md=markdown, code=code
CELLS = [
("md", """# atmaCup #11 ベースライン（スクラッチ学習・回帰）

**競技ルール最重要: 学習済みモデル(ImageNet等)・外部データ 禁止。** → `pretrained=False` でスクラッチ学習。

- タスク: 美術作品画像から `target`(制作年代 0-3) を回帰予測 / 指標: **RMSE**
- CV: `art_series_id` で GroupKFold(5) / 提出: test 順に target 1列
- 連携: データ=Google Drive, 重み/曲線=W&B, 提出csv=Drive outputs/

実行: ランタイム=GPU を選択 → 上から順に実行。"""),

("code", """# 1. 依存。Colabにtorch/torchvisionは既存。wandbのみ追加。
!pip -q install wandb
import torch, torchvision
print("torch", torch.__version__, "cuda", torch.cuda.is_available())"""),

("code", """# 2. Google Drive をマウントしてデータを /content に展開
# Drive側レイアウト: atmaCup11/data/ 配下の任意の場所に photos.zip + 各csv がある前提（再帰探索）
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, glob, shutil
PROJECT = '/content/drive/MyDrive/atmaCup11'      # ← 共有ショートカットをMyDrive直下に置いた前提
DATA_DIR = '/content/data'                          # 学習中はローカルSSDから読む(高速)
photo_dir = f'{DATA_DIR}/photos'
os.makedirs(photo_dir, exist_ok=True)

# photos.zip を data 配下から探して展開
zips = glob.glob(f'{PROJECT}/data/**/photos.zip', recursive=True)
assert zips, f'photos.zip が {PROJECT}/data 配下に無い。データのアップロードを確認'
if len(os.listdir(photo_dir)) < 9000:           # 未展開なら展開（再実行時はスキップ）
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(photo_dir)
    for p in glob.glob(f'{photo_dir}/**/*.jpg', recursive=True):  # サブフォルダ展開を平坦化
        if os.path.dirname(p) != photo_dir:
            shutil.move(p, photo_dir)

# csv を data 配下から DATA_DIR 直下へコピー
for c in glob.glob(f'{PROJECT}/data/**/*.csv', recursive=True):
    shutil.copy(c, DATA_DIR)
print('photos:', len(glob.glob(f'{photo_dir}/*.jpg')),
      'csvs:', [os.path.basename(c) for c in glob.glob(f'{DATA_DIR}/*.csv')])"""),

("code", """# 3. W&B ログイン（初回はAPIキーを貼り付け）
import wandb
wandb.login()
WANDB_PROJECT = 'atmacup11'"""),

("code", """# 4. 設定
from dataclasses import dataclass, asdict
@dataclass
class CFG:
    img_size: int = 128
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_folds: int = 5
    train_fold: int = 0      # まず1foldだけ回す
    num_workers: int = 2
    seed: int = 42
cfg = CFG()

import numpy as np, random
def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything(cfg.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'"""),

("code", """# 5. データ読み込み + GroupKFold(art_series_id)
import pandas as pd
from sklearn.model_selection import GroupKFold
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')

train['fold'] = -1
gkf = GroupKFold(n_splits=cfg.n_folds)
for f, (_, va) in enumerate(gkf.split(train, train['target'], train['art_series_id'])):
    train.loc[va, 'fold'] = f
print(train['fold'].value_counts().sort_index().to_dict())"""),

("code", """# 6. Dataset / 変換（正方形にパッド→リサイズ、軽いaug）
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def pad_to_square(img):
    w, h = img.size; s = max(w, h)
    bg = Image.new('RGB', (s, s), (255, 255, 255))
    bg.paste(img, ((s-w)//2, (s-h)//2)); return bg

train_tf = T.Compose([T.Lambda(pad_to_square), T.Resize((cfg.img_size, cfg.img_size)),
                      T.RandomHorizontalFlip(), T.RandomRotation(10),
                      T.ColorJitter(0.1,0.1,0.1), T.ToTensor()])
val_tf   = T.Compose([T.Lambda(pad_to_square), T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor()])

class ArtDataset(Dataset):
    def __init__(self, df, tf, has_target=True):
        self.df = df.reset_index(drop=True); self.tf = tf; self.has_target = has_target
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = self.tf(Image.open(f'{DATA_DIR}/photos/{r.object_id}.jpg').convert('RGB'))
        if self.has_target:
            return img, torch.tensor(r.target, dtype=torch.float32)
        return img"""),

("code", """# 7. モデル: ResNet18 スクラッチ(pretrained=False) + 回帰ヘッド(1出力)
import torch.nn as nn
def build_model():
    m = torchvision.models.resnet18(weights=None)   # ★ 学習済み禁止 → weights=None
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m"""),

("code", """# 8. 学習(1 fold) + W&B ログ + 重みArtifact保存
import math, time
def rmse(p, y): return math.sqrt(((p - y) ** 2).mean())

def run_fold(fold):
    tr_df, va_df = train[train.fold != fold], train[train.fold == fold]
    tr_dl = DataLoader(ArtDataset(tr_df, train_tf), batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(ArtDataset(va_df, val_tf), batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    crit = nn.MSELoss()

    run = wandb.init(project=WANDB_PROJECT, name=f'resnet18scratch_f{fold}',
                     config=asdict(cfg), reinit=True)
    best = 1e9; best_path = f'/content/best_f{fold}.pth'
    for ep in range(cfg.epochs):
        model.train(); tl = 0
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x).squeeze(1), y); loss.backward(); opt.step()
            tl += loss.item() * len(x)
        sched.step()
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for x, y in va_dl:
                preds.append(model(x.to(device)).squeeze(1).cpu().numpy()); gts.append(y.numpy())
        preds, gts = np.concatenate(preds), np.concatenate(gts)
        v = rmse(np.clip(preds, 0, 3), gts)
        wandb.log({'epoch': ep, 'train_loss': tl/len(tr_df), 'val_rmse': v, 'lr': sched.get_last_lr()[0]})
        print(f'ep{ep:02d} train_loss={tl/len(tr_df):.4f} val_rmse={v:.4f}')
        if v < best:
            best = v; torch.save(model.state_dict(), best_path)
    art = wandb.Artifact(f'resnet18scratch_f{fold}', type='model', metadata={'val_rmse': best})
    art.add_file(best_path); run.log_artifact(art)
    wandb.summary['best_val_rmse'] = best; run.finish()
    print(f'fold{fold} best_val_rmse = {best:.4f}')
    return best_path, best

best_path, best_rmse = run_fold(cfg.train_fold)"""),

("code", """# 9. test 推論 → 提出csv を Drive outputs/ に保存
model = build_model().to(device); model.load_state_dict(torch.load(best_path)); model.eval()
te_dl = DataLoader(ArtDataset(test, val_tf, has_target=False),
                   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
preds = []
with torch.no_grad():
    for x in te_dl:
        preds.append(model(x.to(device)).squeeze(1).cpu().numpy())
sub = pd.DataFrame({'target': np.clip(np.concatenate(preds), 0, 3)})
assert len(sub) == len(test) == 5919
out = f'{PROJECT}/outputs/submission_resnet18scratch_f{cfg.train_fold}.csv'
sub.to_csv(out, index=False)
print('saved:', out, '/ pred mean', sub.target.mean().round(3))
sub.head()"""),

("md", """## 次の一手（改善ループ）
1. **全5fold** 学習して OOF RMSE を確定（`cfg.train_fold` を回す or ループ化）→ test も fold平均
2. **SSL事前学習(SimSiam等)** を photos 全9856枚で行い、その重みからfinetune（atmaCup#11の本命・1st解法路線）
3. 画像サイズ↑(128→224)、aug強化、TTA、EMA、ラベル平滑化(sorting_date活用)
4. backbone多様化(ResNet34 / scratch ConvNeXt-tiny) → アンサンブル

結果(W&B run / 提出csv)は私(Claude)がDrive・W&B経由で確認し、コードを更新します。"""),
]

def cell(t, src):
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    if t == "md":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}

nb = {
    "cells": [cell(t, s) for t, s in CELLS],
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", OUT)
