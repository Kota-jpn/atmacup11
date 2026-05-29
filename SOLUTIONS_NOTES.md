# atmaCup #11 上位解法インプット（精読メモ）

出典: guruguru discussions（Public2nd/Private1st, 3rd place）を精読。

## 共通する勝ち筋（最重要）
1. **SSL 事前学習が必須**。pretrained 禁止なので photos 全9856枚で自己教師あり学習 → finetune。
   - CNN系: **SimSiam / BYOL**（BYOL 0.685 > SimSiam 0.687）
   - ViT系: **DINO** が最強（MoCov3/EsViT/SimSiam より上）。ただし **epoch を大量に**（300〜800）。
2. **長辺パディングで正方形化 → リサイズ**（年代でアスペクト比が偏るため）。※我々の baseline で既に実装済。
3. **マルチタスク学習**: target だけでなく **sorting_date / materials / techniques** も同時に予測ヘッドとして学習（train専用情報を「正解特徴」でなく「学習信号」として活用＝正則化）。
4. **多様なモデルのアンサンブル/スタッキング**が効く（単体 0.66 → stack 0.60）。
5. CV: **StratifiedGroupKFold(sorting_date, art_series_id)** n=5。※我々は GroupKFold → 昇格する。
6. target を**回帰と分類の両方**で解き、分類のクラス別スコアを stacking 入力に。
7. aug: **Mixup(α0.4)+Cutmix(α0.4) p=0.8 半々**、HFlip, ShiftScaleRotate, Blur, BrightnessContrast。SAM optimizer も有効。

---

## ① Public 2nd / Private 1st
- モデル: ResNet18d, ViT_small, ViT_base
- SSL: **DINO 300ep → finetune 100ep**。初手 SimSiam は失敗、終了2日前に DINO へ。
- 戦略: 単体精度より**モデル数**。stacking は CV overfit 大きい → SSL pretrain モデルを多めに。
- 前処理: 長辺パディング→リサイズ。padding 360×360。
- DINO: 公式設定踏襲、ep 100→300。SSL loss 7.587→3.814（まだ伸びる余地）。
- aug(Albumentations): HFlip, ShiftScaleRotate(0.1/0.1/15°), Blur, RandomBrightnessContrast, Mixup0.4+Cutmix0.4 p0.8。
- CV: StratifiedGroupKFold(sorting_date, art_series_id) n=5。
- 学習: scratch 450ep(LR1e-3, bs128) / SSL後finetune 100ep(LR1e-4, bs32), AdamW wd0.01, CosineAnnealingLR。
- **スコア**:
  | モデル | CV | Public | Private |
  |---|---|---|---|
  | ResNet18d (baseline) | 0.681 | 0.6835 | 0.6688 |
  | ViT_small+DINO | 0.6651 | 0.6576 | 0.6361 |
  | ViT_base+DINO | 0.6723 | 0.6413 | 0.6343 |
  | **LGBM stacking** | **0.5995** | **0.6017** | **0.5878** ←最終 |

## ② 3rd place
- ViT: Swin Tiny / Swin Tiny+token labeling / XCiT Small / XCiT Small+token labeling
- CNN: ResNet34 / ResNet34d（マルチスケール）
- SSL(ViT): EsViT, MoCov3, **DINO(ep800最良, RMSE0.7035)**, SimSiam
- SSL(CNN): SimSiam, **BYOL**（ep800）。BYOL 0.685 / SimSiam 0.6873
- finetune: img_size=224, 400〜500ep, CosineLR, **SAM optimizer**
- aug: cutmix or resizemix
- 事前検証: 80/20 holdout 早期停止(50ep)
- マルチタスク: **tech + sorting_date + mat の3タスク並行**
- アンサンブル: **Netflix Blending**（stacking はしない）
- 最良 Public: ResNet34d SimSiam マルチスケール **0.6739**

---

## 我々のロードマップへの反映
- [x] 長辺パディング（baseline 済）
- [ ] CV を **StratifiedGroupKFold(sorting_date bin, art_series_id)** に昇格
- [ ] baseline 目標 RMSE ≈ **0.68**（ResNet18d scratch 相当）を確認
- [ ] **SimSiam/BYOL を photos 全9856枚で SSL pretrain**（CNN, ep多め）→ finetune
- [ ] **マルチタスクヘッド**（target回帰 + sorting_date回帰 + materials/techniques分類）
- [ ] img_size 128→224、Mixup+Cutmix、SAM
- [ ] DINO（ViT）路線（GPU潤沢なら）
- [ ] 複数 SSL モデルを **Netflix Blending / LGBM stacking**

---

# 全順位リサーチ（1〜10位、エージェント精読 2026-05-29）

## 横断トレンド（最重要）
- **SSL は DINO ≫ SimSiam/BYOL**。1/2/3(ViT)/7/8位の最良ストリームは全部 DINO。SimSiam は多くで scratch並み。**例外=5位**が丁寧に詰めて Priv0.6249（SimSiam唯一の上位）。
- **SSL epoch が支配的**：DINO 300ep必須(100/200は劣る)、2位は800→1600→2400で飽和せず。CNN系SSLも800ep。
- **CV = StratifiedGroupKFold(group=art_series_id, stratify=sorting_date) 5fold が全員のデファクト**。
- **回帰＋分類の両建て**。分類は **softmax→クラスindex加重平均で連続化（argmax禁止＝5位で劇的改善）**。
- **後処理 np.clip(pred,0,3) は全員必須**。
- **最終は LightGBM/CatBoost スタッキング**（1/2/4/6/8位）。単純平均superior。

## 順位別ハイライト
| 順位 | Priv | 核心 |
|---|---|---|
| 1 ishikei | 0.5878 | DINO300ep + ResNet18d/ViT + sorting_date補助 + Mixup/Cutmix + **LGBスタック**(+0.005) |
| 2 | 0.5909 | DINO 800-2400ep + **凍結ViT+3層MLP(Dropout0.7)が最強単体** + 4タスク + patch平均pooling回帰 |
| 3 | (Pub0.6486) | DINO/MoCov3/SimSiam比較(DINO最良) + **Token Labeling**(0.70→0.65) + SAM + Netflix Blend |
| 4/5 tawatawara | 0.616 | DINO + **累積順序bin[0,0,0]/[1,0,0]/..multi-loss** + 3段stack(SVD16次元+画像サイズ特徴)。[code](https://github.com/tawatawara/atmaCup-11) |
| 5 "SimSiam Is My Friend" | 0.6249 | **SimSiam唯一の成功**: 独自mean/std + RandomGridShuffle(4×4) + collapse監視150-200ep + 分類のみMixup |
| 6 | 0.6447 | EffNet-b1 512px + MSE+技法BCE + LGBスタック |
| 7 | 0.6468 | ResNet18d(SimSiam)+ResNet34d(scratch)+ViT(DINO)、分類化で回帰超え |
| 8 | 非開示 | DINO+ViT凍結MLP "remarkably powerful"、CatBoost>LGB、Pseudo無効 |
| 10 | (Pub12) | **SSLなし**xception+resnet34d scratch400ep。SSL/stack/Mixup全部効かずと報告(反例) |

## 細かい有用技
- **凍結backbone+浅いMLP**（fine-tuneはSSL表現を壊す、2位/8位）
- **独自正規化** mean[0.776,0.742,0.669] std[0.115,0.113,0.114]（5位）
- **画像メタ特徴(h,w,aspect)はstackに効く**（LGB単体でPriv0.7585、4位も採用）
- **回帰でMixup禁止**（世紀の中間は無意味、5位）/ color/blur破壊系aug・Cutout・Pseudoは効きにくい報告多数
- nyk510公式SimSiamチュートリアル＝5位の独自正規化baselineが実質その内容
