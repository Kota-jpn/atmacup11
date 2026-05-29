# atmaCup #11 改善アイデア集（EDA + 上位解法 + 最新論文/Kaggle 統合）

更新: 2026-05-29。出典は SOLUTIONS_NOTES.md（解法精読）/ EDA_FINDINGS.md（自前EDA）/ 下記リサーチ。
※リサーチ①(他解法収集)は実行中、完了後に追記予定。

## 確定した「再現性が最も高い勝ち筋」（1〜10位の共通項で実証）
1. **競技画像のみで SSL 事前学習、特に DINO 300ep以上**（pretrained禁止の本質。1/2/3/7/8位の最良が DINO）
2. **CV = StratifiedGroupKFold(group=art_series_id, stratify=sorting_date) 5fold**（全員のデファクト）
3. **長辺パディング→正方形resize**（EDAで縦長48%/横長47%と実証）
4. **target を 回帰 + 分類の両建て**。**分類は softmax→クラスindex加重平均で連続化（argmax禁止＝5位で劇的改善）**
5. **後処理 np.clip(pred, 0, 3) は必須**（全員）
6. **Mixup + Cutmix（分類のみ。回帰では使わない＝世紀の中間は無意味, 5位）**
7. **LightGBM/CatBoost スタッキング**（分類確率 + SVD圧縮embedding + 画像メタ特徴h/w/aspect を投入）

## ★ EDA 由来の本コンペ固有アイデア
- **materials 補助ヘッド（multi-task）**：materialが年代に強相関を実証（paint0.28 → pencil2.30）。test に無いので特徴量化は不可だが**学習信号として補助タスク化**すると効くはず（3位解法も mat/tech/sorting_date をマルチタスク）。
- **色情報を保持**：グレースケール0.1%・カラーが効く → 過度な grayscale/color-jitter aug は避ける。
- **解像度は224上限**：全画像が長辺224配布。224超リサイズは情報増えない（拡大は計算無駄、SSLは128→224で十分）。
- **sorting_date 回帰補助ヘッド**：target は sorting_date の世紀バケツ。連続値の sorting_date を補助回帰すると粒度の細かい学習信号に（1位も sorting_date 回帰併用）。

---

## 優先度つき実践ロードマップ

### Phase 0: baseline 完走（今ここ）
- ResNet18 scratch 回帰、GroupKFold、224化、Mixup/Cutmix なし。**目標 val RMSE ≈ 0.68**。
- まず1fold→全5fold、OOF と LB の gap 確認。

### Phase 1: すぐ効く低コスト改善（baseline直後・コスト極小から）
| # | 施策 | 根拠 | 期待 |
|---|---|---|---|
| 0 | **CV を StratifiedGroupKFold(group=art_series_id, stratify=sorting_date) に切替** | 全上位デファクト、実装ほぼ0コスト | CV-LB相関担保 |
| 1 | **後処理 clip[0,3] + 分類は softmax→index加重平均（argmax禁止）** | 5位「劇的改善」、clipは全員 | 確実 |
| 2 | **Mixup+Cutmix（分類のみ、回帰では使わない）** | 1/3/4/5/7位 | 安定改善 |
| 3 | **回帰+分類の両建て**（同backbone 2ヘッド or 別学習）→ 平均/スタック | 1/5/7位の共通勝ち筋。7位は分類化で回帰超え | 実証済 |
| 4 | **DLDL（Gaussianソフトラベル）/ 累積順序bin multi-loss** | 隣接相関・順序構造活用 [DLDL](https://arxiv.org/pdf/1611.01731)、4位bin採用 | 序数ノイズ頑健 |
| 5 | **TTA(hflip) + model EMA** | 小幅だが確実 | +小 |
| 6 | **materials/sorting_date 補助ヘッド**（multi-task） | EDAで相関実証＋1/2/3位 | 正則化・汎化 |

### Phase 2: 中核 — SSL 事前学習（最大の伸びしろ）
| # | 施策 | 根拠 | 注意 |
|---|---|---|---|
| 7 | **DINO で train+test 全9856枚を SSL 事前学習（300ep以上）** → FT | **1/2/3/7/8位の最良**。300ep必須(100/200劣る)、2位は800-2400で飽和せず | collapse監視、epoch最重要 |
| 8 | **凍結backbone + 浅いMLP(Dropout0.7, label smoothing)** で学習 | **2位最強単体・8位"remarkably powerful"**。FTはSSL表現を壊す | 低コストで高効果 |
| 9 | **SimSiam を使うなら丁寧版**: 独自mean/std[0.776,0.742,0.669] + RandomGridShuffle(4×4) + collapse監視で150-200ep ckpt | 5位がこれでSimSiam唯一の上位(Priv0.6249) | 雑な100epはscratch並み |
| 10 | **ViT-small + MAE/DINO** を別系統でSSL | アンサンブル多様性、1/2位もViT併用 | Colab Proで現実的 |
| 11 | **FTレシピ**: 450ep / AdamW(wd0.01) / cosine / lr1e-3 scratch・1e-4 SSL後 / 解像度224→360 | 1位実測、2位/6位は大解像度良 | 少データは過学習注意 |

### Phase 3: アンサンブル/仕上げ
| # | 施策 | 根拠 |
|---|---|---|
| 12 | **複数backbone(ResNet18d/34d/50d)+ViT を SSL→FT** → **LightGBM/CatBoost スタック（分類確率 + embeddingをSVD16次元 + 画像メタh/w/aspect を特徴に）** | 1/2/4/6/8位。画像メタはLGB単体でPriv0.7585出る素性 |
| 13 | **seed bagging + Optuna 重み最適化** | CV安定化、4/5位 |
| 14 | **Progressive resizing(128→224/360)** | 高速化＋小幅精度 |
| 15 | **Token Labeling（ViT局所予測）** | 3位 0.7035→0.6486 |

### 実験枠（不確実・要v-baseline比較）
- **SAM / Friendly-SAM**：汎化・ノイズ頑健だが極小データで AdamW を悪化させうる（両刃）。[Friendly-SAM](https://arxiv.org/pdf/2403.12350)
- **年代=スタイル分類への帰着**（補助タスク化）。効果未検証。

## 負の知見（避ける）
- ❌ **ConvNeXt をスクラッチ＋重aug**：失敗報告。使うならSSL必須・aug控えめ。
- ❌ **SimCLR系（大バッチ必須）**：Colab級では非現実的。
- ❌ 学習済み重み・外部データ（WikiArt等）：**規約違反**。手法参考のみ。
- ⚠️ 論文の「+X%改善」数値は別タスク・別データ。**必ず自前CVで検証**。

## 参考出典
- 1位(ishikei, DINO): https://www.docswell.com/s/ishikei/K9N79K-atmacup11-1st-place-solution
- 4-5位(tawatawara, SimSiam, **コード有**): https://github.com/tawatawara/atmaCup-11
- DLDL: https://arxiv.org/pdf/1611.01731 / CORN: https://arxiv.org/pdf/2111.08851
- SimSiam崩壊: https://arxiv.org/pdf/2209.15007 / SSL Cookbook: https://arxiv.org/pdf/2304.12210
- ResNet strikes back: https://arxiv.org/pdf/2110.00476 / MAE小データ: https://arxiv.org/pdf/2504.10021
