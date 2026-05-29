"""マルチタスクモデル: timm backbone(scratch) + main/date/mat ヘッド。"""
import torch
import torch.nn as nn
import timm


class MultiTaskNet(nn.Module):
    def __init__(self, backbone="resnet18d", task="cls", n_classes=4, n_mat=0):
        super().__init__()
        # ★ pretrained=False 厳守（コンペ規約）
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat = self.encoder.num_features
        self.task = task
        self.head_main = nn.Linear(feat, n_classes if task == "cls" else 1)
        self.head_date = nn.Linear(feat, 1)
        self.head_mat = nn.Linear(feat, n_mat) if n_mat > 0 else None
        self.num_features = feat

    def forward(self, x, return_feat=False):
        f = self.encoder(x)
        out = {"main": self.head_main(f), "date": self.head_date(f).squeeze(1)}
        if self.head_mat is not None:
            out["mat"] = self.head_mat(f)
        if return_feat:
            out["feat"] = f
        return out

    def load_ssl_encoder(self, state_dict):
        """SSL事前学習済 encoder 重みをロード（ヘッドは無視）。"""
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        return missing, unexpected
