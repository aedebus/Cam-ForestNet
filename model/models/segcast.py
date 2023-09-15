import torch

from .segnet import SegNet
from util import constants as C


class SegCastNet(SegNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        logits = super().forward(batch)
        mask = batch['mask']
        valid = batch['valid']
        y_logit_seg = (logits * mask)
        y_pred_cls, y_logit_cls = self.aggregate(y_logit_seg, mask)
        
        # Mask the logits to avoid unnecessary gradients flow
        # Removing this line will introduce NaN gradients for 
        # invalid samples (i.e. those with label == IGNORE_VALUE)
        y_logit_cls[~valid,:] = 0
        return y_logit_seg, y_logit_cls, y_pred_cls

    def aggregate(self, y_logit_seg, mask, merge_mapping=None):
        """ Casting segmentation predictions into classification
            ones, using the average to aggregate over segmentation map.
        Args:
            merge_mapping(dict): A mapping from source class to target class.
        """

        # Adding a constant accounting for invalid example
        y_logit_cls = y_logit_seg.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1)
        if merge_mapping is not None:
            merge_mapping = merge_mapping.to(y_logit_cls.device)
            y_prob_flat = torch.nn.functional.softmax(y_logit_cls, dim=1)
            y_prob_flat = torch.mm(y_prob_flat, merge_mapping)
            y_pred_cls = y_prob_flat.argmax(dim=1)
        else:
            y_pred_cls = y_logit_cls.argmax(dim=1)
        return y_pred_cls, y_logit_cls

    def get_seg_logits(self, batch):
        return self.model(batch['image'])
