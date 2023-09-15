import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss
from .dice import GeneralizedSoftDiceLoss


class CombinedLoss(nn.Module):
    def __init__(self, gamma, alpha=0.75, **kwargs):
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha
        self._fl = FocalLoss(gamma, **kwargs)
        self._dl = GeneralizedSoftDiceLoss(**kwargs)

    def forward(self, logits, label):
        fl = self._fl(logits, label)
        dl = self._dl(logits, label)
        return self._alpha * fl + (1.0 - self._alpha) * dl
