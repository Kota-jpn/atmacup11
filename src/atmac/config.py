from dataclasses import dataclass, field


@dataclass
class CFG:
    # ---- paths (Colab側で上書き) ----
    data_dir: str = "data/raw"          # photos/ と *.csv がある場所
    out_dir: str = "outputs"            # 重み・OOF・提出の保存先

    # ---- 共通 ----
    seed: int = 42
    n_folds: int = 5
    img_size: int = 224                 # 全画像 長辺224配布。224で十分(>224は無意味)
    num_workers: int = 8                # A100は~12コア。DINOのCPU aug律速を解消
    n_classes: int = 4
    # 5位解法のドメイン特化 正規化統計
    mean: tuple = (0.776, 0.742, 0.669)
    std: tuple = (0.115, 0.113, 0.114)
    min_mat_count: int = 30             # materials補助ヘッドに使う最小出現数

    # ---- SSL 事前学習 ----
    ssl_method: str = "dino"            # dino | simsiam
    ssl_backbone: str = "resnet18d"
    ssl_epochs: int = 100               # 本番は300+推奨
    ssl_img_size: int = 128             # SSLは低解像度で十分高速（FTは img_size=224）
    ssl_batch: int = 256                # A100 40GBで活用（ステップ数半減）
    ssl_lr: float = 5e-4                # 1e-3はDINO不安定(loss上昇)→5e-4に戻す
    ssl_warmup_epochs: int = 10         # SSL LR linear warmup
    ssl_local_crops: int = 6            # DINOローカルクロップ数（減らすと高速）
    ckpt_every: int = 10                # SSLチェックポイント保存間隔（切断対策）

    # ---- ファインチューニング ----
    backbone: str = "resnet18d"
    task: str = "cls"                   # cls(softmax加重平均) | reg(直接回帰)
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    ema_decay: float = 0.999
    # マルチタスク損失の重み（main=target, date=sorting_date, mat=materials）
    w_main: float = 1.0
    w_date: float = 0.2
    w_mat: float = 0.2
    use_mixup: bool = True              # cls のみ true 推奨（regは世紀の中間が無意味）
    mixup_alpha: float = 0.4
    use_tta: bool = True                # 推論時 hflip TTA
