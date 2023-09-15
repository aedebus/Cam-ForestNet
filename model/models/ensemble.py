import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from .get_model import get_single_model


class EnsembleModel(nn.Module):
    def __init__(self, args):
        super(EnsembleModel, self).__init__()

        self.segmentation = args.segmentation
        self.checkpoint_dir = Path(args.ckpt_path)
        self.use_classification_head = args.use_classification_head
        if self.use_classification_head:
            raise ValueError("Classification head with ensembling " +
                             "not implemented")

        # Note: this code has only been tested with a ResNet18 model.
        print("Loading models in ensemble...")
        self.models = nn.ModuleList([])
        for ckpt_path in tqdm(self.checkpoint_dir.iterdir()):
            args['ckpt_path'] = ckpt_path
            model = get_single_model(args)
            assert (
                model.use_classification_head == self.use_classification_head
            )
            self.models.append(model)

    def forward(self, batch, *args):
        logits_list = []
        mask_logits_list = []
        for model in self.models:
            if self.use_classification_head:
                mask_logits, y_logit_cls = model(batch, args)
            else:
                y_logit_seg, y_logit_cls, _ = model(batch)
            logits_list.append(y_logit_cls)
            mask_logits_list.append(y_logit_seg)

        mask_logits_ensemble = torch.stack(mask_logits_list).mean(dim=0)
        logits_ensemble = torch.stack(logits_list).mean(dim=0)
        logits_cls_ensemble = (
            mask_logits_ensemble.sum(dim=(2, 3)) /
            (batch['mask'].sum(dim=(2, 3)) + 1)
        )
        pred_cls_ensemble = logits_cls_ensemble.argmax(dim=1)

        return mask_logits_ensemble, logits_ensemble, pred_cls_ensemble

    def get_seg_logits(self, batch, *args):
        seg_logits_list = []
        for model in self.models:
            seg_logits_list.append(model.get_seg_logits(batch))
        return torch.stack(seg_logits_list).mean(dim=0)
