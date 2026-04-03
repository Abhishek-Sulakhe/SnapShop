import torch
import torch.nn as nn
from shopee_matching.utils.common import gem

class ShopeeNet(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes,
                 fc_dim=512,
                 s=30, margin=0.5, p=3):
        super(ShopeeNet, self).__init__()

        self.backbone = backbone
        self.backbone.reset_classifier(num_classes=0)  # remove classifier

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
            x = (x[0] + x[1]) / 2
            x = self.bn(x)
        else:
            x = gem(x, p=self.p).view(batch_size, -1)
            x = self.fc(x)
            x = self.bn(x)
        return x

    def forward(self, x, label):
        feat = self.extract_feat(x)
        # Note: In inference context, loss module might not be needed or used
        # x = self.loss_module(feat, label) 
        return feat
