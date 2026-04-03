import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class ImageEncoder(nn.Module):
    """
    Image feature extractor used for both DeiT and EfficientNet backbones.
    Uses Generalized Mean (GeM) pooling for CNN backbones and token averaging
    for ViT/DeiT backbones. Produces L2-normalized embeddings of dimension fc_dim.
    """

    def __init__(self, backbone, num_classes, fc_dim=512, s=30, margin=0.5, p=3):
        super().__init__()
        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)
        self.fc = nn.Linear(self.backbone.num_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)

        if isinstance(x, tuple):
            # DeiT distilled: average class + distillation tokens
            x = (x[0] + x[1]) / 2
            x = self.bn(x)
        elif len(x.shape) == 3:
            # ViT/DeiT 3D output (B, N, C) — average first two tokens
            x = (x[:, 0] + x[:, 1]) / 2
            x = self.fc(x)
            x = self.bn(x)
        else:
            # CNN backbone — apply GeM pooling
            x = gem(x, p=self.p).view(batch_size, -1)
            x = self.fc(x)
            x = self.bn(x)
        return x

    def forward(self, x, label=None):
        return self.extract_feat(x)
