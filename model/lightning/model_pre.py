import nni
import torch
import logging
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score
from ignite.metrics import ConfusionMatrix, IoU, mIoU

from eval import get_loss_fn
from .logger import TFLogger
from .model_base import BaseModel 


class PretrainingModel(BaseModel, TFLogger):
    """Standard interface for segmentation pretraining experiments."""
    def __init__(self, params):
        super().__init__(params)
        self.labels = self.hparams.get("labels")
        self.cm = ConfusionMatrix(len(self.labels))
        self.iou = IoU(self.cm)
        self.miou = mIoU(self.cm)

    def forward(self, batch, is_training=True):
        y_true_seg = batch['label_seg']
        y_logit_seg, _, _ = self.model(batch)
        loss = self.loss(y_logit_seg, y_true_seg)
        y_pred_seg = torch.argmax(y_logit_seg, dim=1)
        acc = (y_pred_seg == y_true_seg).sum().float() / y_true_seg.numel()

        if is_training:
            return loss, acc
        else:
            return loss, y_logit_seg

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary or loss and metrics, with:
                loss: loss used to calculate the gradient
                log: metrics to be logged in the TensorBoard and metrics.csv
                progress_bar: metrics to be shown in the progress bar
        """
        loss, acc = self.forward(batch, is_training=True)

        logs = {
            'loss': loss,
            'log': {'train_loss': loss,
                    'train_acc': acc,
                    'epoch': self.current_epoch}}
        return logs

    def validation_step(self, batch, batch_nb):
        y_true_seg = batch['label_seg']
        loss, y_logit_seg = self.forward(batch, is_training=False)
        y_pred_seg = torch.argmax(y_logit_seg, dim=1)
        self.cm.update((y_logit_seg, y_true_seg))

        y_true_seg_np = y_true_seg.cpu().detach().numpy().flatten()
        y_pred_seg_np = y_pred_seg.cpu().detach().numpy().flatten()

        f1_micro = f1_score(y_true_seg_np, y_pred_seg_np, average="micro")
        f1_macro = f1_score(y_true_seg_np, y_pred_seg_np, average="macro")
        acc = accuracy_score(y_true_seg_np, y_pred_seg_np)

        logs = {'val_loss': loss.cpu().detach().numpy(),
                'val_f1_micro': f1_micro,
                'val_f1_macro': f1_macro,
                'val_acc': acc,
                'log': {self.current_epoch}}
        return logs

    def validation_epoch_end(self, outputs):
        metrics = {}
        avg_loss = np.stack([out['val_loss'] for out in outputs]).mean()
        metrics['avg_val_loss'] = avg_loss
        metrics['val_mIoU'] = mIoU = self.miou.compute()
        ious = self.iou.compute()
        for j, l in enumerate(self.labels):
            metrics[f'val_IoU_{l}'] = ious[j]
        self.cm.reset()

        metrics['val_f1_micro'] = np.stack([out['val_f1_micro']
                                            for out in outputs]).mean()
        metrics['val_f1_macro'] = np.stack([out['val_f1_macro']
                                            for out in outputs]).mean()
        avg_acc = np.stack([out['val_acc'] for out in outputs]).mean()
        metrics['val_avg_acc'] = avg_acc
        logs = {'val_loss': avg_loss,
                'log': metrics,
                'progress_bar': {"val_mIoU": mIoU,
                                 "val_loss": avg_loss,
                                 "val_avg_acc": avg_acc,
                                 }}
        nni.report_intermediate_result(metrics['val_f1_macro'])         
        return logs

    def test_step(self, batch, batch_nb):
        y_true_seg = batch['label_seg']
        test_data_split = self.hparams['test_data_split']
        loss, y_logit_seg = self.forward(batch, is_training=False)
        y_pred_seg = torch.argmax(y_logit_seg, dim=1)

        loss = loss.cpu().detach().numpy()

        y_true_seg_np = y_true_seg.cpu().detach().numpy().flatten()
        y_pred_seg_np = y_pred_seg.cpu().detach().numpy().flatten()

        f1_micro = f1_score(y_true_seg_np, y_pred_seg_np, average="micro")
        f1_macro = f1_score(y_true_seg_np, y_pred_seg_np, average="macro")
        acc = accuracy_score(y_true_seg_np, y_pred_seg_np)

        logs = {f'{test_data_split}_loss': loss,
                f'{test_data_split}_f1_micro': f1_micro,
                f'{test_data_split}_f1_macro': f1_macro,
                f'{test_data_split}_acc': acc,
                'log': {f'{test_data_split}_loss': loss,
                        f'{test_data_split}_acc': acc},
                'progress_bar': {f'{test_data_split}_loss': loss}}
        return logs

    def test_epoch_end(self, outputs):
        test_data_split = self.hparams['test_data_split']
        metrics = {}
        avg_loss = np.stack([out[f'{test_data_split}_loss'] for out in outputs]).mean()
        metrics[f'avg_{test_data_split}_loss'] = avg_loss

        metrics[f'{test_data_split}_f1_micro'] = np.stack([
            out[f'{test_data_split}_f1_micro'] for out in outputs
        ]).mean()
        metrics[f'{test_data_split}_f1_macro'] = np.stack([
            out[f'{test_data_split}_f1_macro'] for out in outputs
        ]).mean()
        metrics[f'{test_data_split}_avg_acc'] = np.stack([
            out[f'{test_data_split}_acc'] for out in outputs
        ]).mean()
        logs = {f'avg_{test_data_split}_loss': avg_loss,
                'log': metrics}
        nni.report_final_result(metrics[f'{test_data_split}_f1_macro'])       
        return logs
