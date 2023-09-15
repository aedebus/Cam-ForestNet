import torch

from util.constants import *
from .metrics import *
from .combined import *
from .dice import *
from .focal import *


def get_loss_fn(args, class_weights=None, data_split='train'):
    loss_fn = args.get("loss_fn")
    segmentation = args.get("segmentation")

    if args.get("class_weights"):
        if args.get("fixed_class_weights"):
            class_weights = torch.tensor(
                INDONESIA_FIXED_CLASS_WEIGHTS, dtype=torch.float)
        elif class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        if class_weights is not None and args.get("gpus") is not None:
            class_weights = class_weights.cuda()
    else:
        class_weights = None

    reduction = 'mean' if data_split == 'train' else 'none'

    if segmentation:
        if loss_fn == "focal":
            return FocalLoss(gamma=args.get("gamma"),
                             ignore_index=LOSS_IGNORE_VALUE,
                             reduction=reduction)
        elif loss_fn == "multi-focal":
            return MultiFocalLoss(args.get("gamma"),
                                  ignore_index=LOSS_IGNORE_VALUE,
                                  reduction=reduction)
        elif loss_fn == "dice":
            return GeneralizedSoftDiceLoss(reduction=reduction)
        elif loss_fn == "combined":
            return CombinedLoss(args.get("gamma"), reduction=reduction)
        elif loss_fn == "CE":
            return torch.nn.CrossEntropyLoss(class_weights,
                                             ignore_index=LOSS_IGNORE_VALUE,
                                             reduction=reduction)
        else:
            raise ValueError(
                f"loss_fn {args.loss_fn} not supported segmentation.")
    else:
        if loss_fn == "focal":
            return ClassificationFocalLoss(
                gamma=args.get("gamma"),
                reduction=reduction)
        elif loss_fn == "BCE":
            return torch.nn.BCEWithLogitsLoss(
                class_weights, reduction=reduction)
        elif loss_fn == "CE":
            return torch.nn.CrossEntropyLoss(
                class_weights,
                reduction=reduction,
                ignore_index=LOSS_IGNORE_VALUE)
        else:
            raise ValueError(
                f"loss_fn {args.loss_fn} not supported for classification.")
