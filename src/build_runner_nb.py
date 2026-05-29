"""Colab薄ランナーNBを生成。GitHubからコード取得→run.pyを実行。
.venv/bin/python src/build_runner_nb.py  ->  notebooks/run_colab.ipynb
反復はGitHub側コード更新→Colabでセル再実行(git pull)だけ。NB自体は基本不変。
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "run_colab.ipynb"
REPO = "https://github.com/Kota-jpn/atmacup11.git"

CELLS = [
("md", """# atmaCup #11 フルパイプライン ランナー（DINO SSL → マルチタスクFT → スタック）

**仕組み**: コードは [GitHub](https://github.com/Kota-jpn/atmacup11) にあり、このNBは毎回 `git pull` で最新を取得して実行するだけ。
→ Claude がコードを更新したら、下の「実行」セルを再実行すれば最新版で回る（NB自体は触らない）。

手順: ランタイム=**GPU** → 上から実行。データは Drive の atmaCup11/data/（共有ショートカット）から読む。"""),

("code", """# 1. コード取得 + 依存
![ -d atmacup11 ] && (cd atmacup11 && git pull -q) || git clone -q %s
!pip -q install timm lightly lightgbm wandb
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())""" % REPO),

("code", """# 2. Drive マウント + データ展開（/content/data に photos と csv）
from google.colab import drive; drive.mount('/content/drive')
import os, glob, zipfile, shutil
PROJECT = '/content/drive/MyDrive/atmaCup11'   # 共有ショートカットをMyDrive直下に置いた前提
DATA = '/content/data'; os.makedirs(f'{DATA}/photos', exist_ok=True)
zips = glob.glob(f'{PROJECT}/data/**/photos.zip', recursive=True)
assert zips, f'photos.zip が {PROJECT}/data 配下に無い'
if len(os.listdir(f'{DATA}/photos')) < 9000:
    with zipfile.ZipFile(zips[0]) as z: z.extractall(f'{DATA}/photos')
    for p in glob.glob(f'{DATA}/photos/**/*.jpg', recursive=True):
        if os.path.dirname(p) != f'{DATA}/photos': shutil.move(p, f'{DATA}/photos')
for c in glob.glob(f'{PROJECT}/data/**/*.csv', recursive=True): shutil.copy(c, DATA)
print('photos', len(glob.glob(f'{DATA}/photos/*.jpg')), '| csv', [os.path.basename(c) for c in glob.glob(f'{DATA}/*.csv')])"""),

("code", """# 3. W&B ログイン（APIキー貼付）
import wandb; wandb.login()"""),

("code", """# 4.【まず疎通確認】GPUで超小規模に1周（数分）。エラーが無いか確認用。
!cd atmacup11 && git pull -q && OMP_NUM_THREADS=4 PYTHONPATH=src python -m atmac.run \\
  --data_dir /content/data --out_dir /content/atmac_smoke \\
  --ssl_epochs 3 --ssl_img_size 96 --ft_epochs 2 --img_size 128 \\
  --ssl_batch 128 --batch_size 64 --num_workers 8 --limit 400 --tasks cls,reg"""),

("code", """# 5a.【初回ベースライン推奨】SSL無し(scratch)で cls+reg 5fold → 初回LBを取る（~1.5h）
# まずパイプライン全体と提出を検証＆基準スコア確保。SSLは後で「改善」として足す。
!cd atmacup11 && git pull -q && OMP_NUM_THREADS=4 PYTHONPATH=src python -m atmac.run \\
  --data_dir /content/data \\
  --out_dir /content/drive/MyDrive/atmaCup11/outputs \\
  --ssl_method none --ft_epochs 30 \\
  --backbone resnet18d --img_size 224 \\
  --batch_size 128 --num_workers 8 --tasks cls,reg --wandb"""),

("code", """# 5b.【後で】SSLあり本番。SSL(DINO) → cls+reg 5fold FT → blend+stack → 提出csv
# ※ A100想定。num_workers/ssl_batch でデータローダ律速を解消、AMPで高速化。
# SSL重みは ckpt_every 毎に Drive 保存＝切断しても再実行で続きから（ssl_epochs 300 も安全）。
# まず100epで1本→LB確認→300epへ、が安全。
!cd atmacup11 && git pull -q && OMP_NUM_THREADS=4 PYTHONPATH=src python -m atmac.run \\
  --data_dir /content/data \\
  --out_dir /content/drive/MyDrive/atmaCup11/outputs \\
  --ssl_method dino --ssl_epochs 100 --ssl_img_size 128 --ft_epochs 30 \\
  --backbone resnet18d --img_size 224 \\
  --ssl_batch 256 --batch_size 128 --num_workers 8 --tasks cls,reg --wandb"""),

("md", """## 提出
`Drive/atmaCup11/outputs/submissions/` に
- `sub_blend_resnet18d.csv`（cls+reg 平均）
- `sub_stack_resnet18d.csv`（LGBスタック）
が出る。guruguru に提出 → スコアを Claude に共有 → 次の改善へ。

## 反復の流れ
Claude が GitHub のコードを更新 → セル5を再実行（`git pull`で最新化）。済んだ工程はキャッシュでskip。"""),
]


def cell(t, src):
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    if t == "md":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}


nb = {"cells": [cell(t, s) for t, s in CELLS],
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", OUT)
