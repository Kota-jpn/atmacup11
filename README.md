# atmaCup #11 [初心者歓迎! / 画像編]

URL: https://www.guruguru.science/competitions/17/

## 基本情報
- コンペ名: atmaCup #11（初心者歓迎・画像編）
- 主催: atma（@nyk510）
- 公式開催: 2021-07-09 〜 2021-07-22（guruguru 上で過去問として取り組み中）
- 形態: 完全オンライン、NDA 不要

## タスク（確認済み）
- **美術作品の画像から `target`（制作年代を 4 区分: 0,1,2,3）を予測する回帰問題**
- target は古い→新しいの順序を持つ序数。整数クラスだが **回帰**として解く
- 評価指標: **RMSE**
- 提出形式: `target` 列のみの csv（test の順序どおり 5919 行）

## データ（data/raw/、git管理外）
| ファイル | 内容 | 行数 |
|---|---|---|
| `train.csv` | object_id, sorting_date, art_series_id, **target** | 3937 |
| `test.csv` | **object_id のみ** | 5919 |
| `materials.csv` | 画材（name, object_id、多対多） | 9081 |
| `techniques.csv` | 技法（name, object_id、多対多） | 3777 |
| `atmaCup#11_sample_submission.csv` | 提出サンプル | 5919 |
| `photos/*.jpg` | 作品画像 9856 枚（train+test 共通プール） | 9856 |

### target 分布（train）
| target | 件数 |
|---|---|
| 0 | 475 |
| 1 | 896 |
| 2 | 1511 |
| 3 | 1055 |

### 重要な注意点
- **test.csv は object_id のみ**。sorting_date / art_series_id / materials / techniques は **train 専用補助情報**で、test には無い → 予測は基本「画像のみ」から行う
- `art_series_id`: train 3937 行 / ユニーク 3784 → 同一シリーズ複数枚あり。**CV は art_series_id で GroupKFold**（リーク防止）
- `sorting_date`: 制作年。target はこれを 4 区分したものなので**実質ターゲットそのもの**。test に無いのでモデル特徴量には使えない（target 理解・回帰ターゲット平滑化のヒント用途のみ）
- photos 9856 = train 3937 + test 5919（過不足なく一致）

## ⚠️ コンペ最重要ルール
**学習済みモデル（ImageNet pretrained 等）の利用禁止。** 外部データも禁止。
→ スクラッチ学習 or **自己教師あり学習(SSL)** で事前学習する必要がある。
これが atmaCup #11 の教育的肝（1st 解法は SimSiam で SSL pretrain → finetune）。

## 環境 / 実行方針
- **GPU 学習: Google Colab Pro**（ローカル MacBook Air M3 はEDA・軽量検証のみ）
- ローカル: `pip install -r requirements.txt`
- Colab 連携ワークフローは下記参照

## ディレクトリ構成
```
atmacup11/
├── data/raw/          # データ一式（git管理外）
├── notebooks/         # EDA / 実験ノートブック
├── src/               # 再利用コード（dataset, model, train, ssl 等）
├── outputs/
│   ├── submissions/   # 提出csv
│   ├── models/        # 学習済み重み
│   └── logs/          # 学習ログ
├── requirements.txt
└── README.md
```

## 進め方（PDCA）
- [x] データ DL・展開
- [x] フォルダ・環境整備
- [ ] EDA（target/sorting_date/art_series 分布、画像確認）
- [ ] CV 設計（art_series_id GroupKFold）
- [ ] ベースライン①: sorting_date のみ / 画像なしの簡易回帰
- [ ] ベースライン②: スクラッチ CNN（ResNet18 from scratch）
- [ ] SSL pretrain（SimSiam）→ finetune
- [ ] アンサンブル・後処理（clip 0-3, round）

## 解法学習リソース（事後公開あり）
- 1st〜上位解法・discussion が公開済み（SSL/SimSiam, アンサンブル等）。改善ループで参照する。
