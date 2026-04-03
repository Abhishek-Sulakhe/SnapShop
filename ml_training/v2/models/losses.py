import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace margin-based softmax loss for metric learning."""

    def __init__(self, in_features, out_features, s=30.0, m=0.50,
                 easy_margin=False, ls_eps=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def _l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    return torch.div(x, norm)


class CurricularFace(nn.Module):
    """CurricularFace adaptive margin loss for metric learning."""

    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label):
        embeddings = _l2_norm(embeddings, axis=1)
        kernel_norm = _l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm).clamp(-1, 1)

        target_logit = cosine[torch.arange(0, embeddings.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m

        mask = cosine > cos_theta_m
        final_target_logit = torch.where(
            target_logit > self.threshold, cos_theta_m, target_logit - self.mm
        )

        hard_example = cosine[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t

        cosine[mask] = hard_example * (self.t + hard_example)
        cosine.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cosine * self.s
        return output
