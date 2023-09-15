import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    def __init__(self, sharpen=0.4, confidence_threshold=0.2):
        super(ConsistencyLoss, self).__init__()
        self.sharpen = sharpen
        self.confidence_threshold = confidence_threshold

    def forward(self, logits_unaugmented, logits_augmented):
        '''
        Compute cross entropy loss between predicted logits on unaugmented
        and augmented images.

        args: logits_unaugmented: logits on unaugmented image of shape (N, C)
        args: logits_augmented: logits on augmented image of shape (N, C)

        Always returns unreduced loss in order to filter out examples
        where the forest loss region is not contained in the augmented image.
        '''
        # Do not backprop through logits on unaugmented images (VAT and UDA)
        logits_unaugmented_detached = logits_unaugmented.detach()

        # Mask out examples that the model is unconfident about (UDA)
        probs = F.softmax(logits_unaugmented_detached, dim=1)
        confidence_mask = (probs.max(1)[0] > self.confidence_threshold)

        # Sharpen predicted logits on unaugmented images (UDA)
        logits_unaugmented_sharpened = logits_unaugmented_detached / self.sharpen
        loss = -(
            F.softmax(logits_unaugmented_sharpened, dim=1) *
            F.log_softmax(logits_augmented, dim=1)
        ).mean(1)

        return loss, confidence_mask
