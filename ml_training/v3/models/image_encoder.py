"""
Image feature encoder for metric learning.

Architecture matches the inference engine's ShopeeNet exactly so that
checkpoints produced here can be loaded at serving time without any
architecture mismatch.

Supports two backbone families:
  - DeiT (Vision Transformer): token averaging (class + distillation tokens)
  - EfficientNet (CNN): Generalized Mean (GeM) pooling

Output: 512-d embeddings after FC + BatchNorm, ready for L2-normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    """Generalized Mean Pooling.

    Controls the pooling behavior via parameter p:
      p = 1  -> standard average pooling
      p -> inf -> max pooling
      p = 3 (train) / p = 4 (eval for EfficientNet) -> emphasizes
        high activations, focusing on discriminative image regions.

    Critical for retrieval: GeM captures salient features rather than
    averaging over the entire spatial feature map.
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class ImageEncoder(nn.Module):
    """
    Image feature extractor used for both DeiT and EfficientNet backbones.

    The backbone's classifier head is removed. Features are projected to
    fc_dim via a linear layer + batch normalization.

    For DeiT:  avg(class_token, distill_token) -> FC -> BN -> 512-d
    For CNN:   GeM(feature_map) -> FC -> BN -> 512-d
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
            # DeiT distilled: returns (class_token, distill_token) tuple
            x = (x[0] + x[1]) / 2
            x = self.bn(x)
        elif len(x.shape) == 3:
            # DeiT 3D output (B, N, C) in newer timm versions
            # Average first two tokens (class + distillation)
            x = (x[:, 0] + x[:, 1]) / 2
            x = self.fc(x)
            x = self.bn(x)
        else:
            # CNN backbone (EfficientNet) - apply GeM pooling
            x = gem(x, p=self.p).view(batch_size, -1)
            x = self.fc(x)
            x = self.bn(x)
        return x

    def forward(self, x, label=None):
        return self.extract_feat(x)
