import torch
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from sklearn.metrics import f1_score
from scipy.special import entr
import pytorch_lightning as pl

from eval import *
from util.constants import *
from util.merge_scheme import *
from .logger import TFLogger
from .model_base import BaseModel 
from .cam import CAMModule

class ClassificationModel(BaseModel, TFLogger, CAMModule):
    """Standard interface for classification experiments."""

    def __init__(self, params):
        super().__init__(params)
        if self.hparams['cam']:
            self.set_up_cam(params)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary or loss and metrics, with:
                loss: loss used to calculate the gradient
                log: metrics to be logged in the TensorBoard and metrics.csv
                progress_bar: metrics to be shown in the progress bar
        """
        x_dict, y, _ = batch
        y_hat = self.forward(x_dict)
        loss = self.loss(y_hat, y)
        acc = get_accuracy(y_hat, y)
        logs = {
            'loss': loss,
            'train_acc': acc,
            'log': {'train_loss': loss,
                    'train_acc': acc,
                    'epoch': self.epoch},
            'progress_bar': {'train_acc': acc}}
        return logs

    def validation_step(self, batch, batch_nb):
        x_dict, y, _ = batch
        y_hat = self.forward(x_dict)
        _, y_pred = y_hat.max(dim=1)
        loss = self.loss(y_hat, y)
        acc = get_accuracy(y_hat, y)
        logs = {'val_loss': loss,
                'val_acc': acc,
                'y_true': y,
                'y_pred': y_pred,
                'log': {'val_loss': loss,
                        'val_acc': acc,
                        'epoch': self.epoch},
                'progress_bar': {'val_loss': loss}}
        return logs

    def validation_epoch_end(self, outputs):
        metrics = {}
        avg_loss = torch.stack([out['val_loss'] for out in outputs]).mean()
        avg_acc = torch.stack([out['val_acc']
                               for out in outputs if out['val_acc'] >= 0]).mean()
        y_true_cls = torch.cat([out['y_true'] for out in outputs])
        y_pred_cls = torch.cat([out['y_pred'] for out in outputs])
        y_true_numpy = y_true_cls.cpu().detach().numpy()
        y_pred_numpy = y_pred_cls.cpu().detach().numpy()

        metrics['avg_val_loss'] = avg_loss
        metrics['avg_val_acc'] = avg_acc
        metrics['val_f1_macro'] = f1_score(
            y_true_numpy, y_pred_numpy, average="macro")
        logs = {'val_loss': avg_loss,
                'log': metrics}
        self.epoch += 1
        return logs

    def test_step(self, batch, batch_nb):
        x_dict, y, idx = batch
        y_hat = self.inference(batch)

        preds = y_hat.argmax(axis=1)
        # In test step, loss has reduction=None
        loss_by_example = self.test_loss(y_hat, y)
        loss = loss_by_example.sum()
        acc = get_accuracy(y_hat, y)
        logs = {f'{self.test_data_split}_loss': loss_by_example,
                f'{self.test_data_split}_acc': acc,
                'y_true': y,
                'idx': idx,
                'y_pred': preds,
                'y_logits': y_hat,
                'log': {f'{self.test_data_split}_loss': loss,
                        f'{self.test_data_split}_acc': acc},
                'progress_bar': {f'{self.test_data_split}_loss': loss}}
        return logs

    def test_epoch_end(self, outputs):
        metrics = {}
        losses = torch.cat(
            [out[f'{self.test_data_split}_loss'] for out in outputs])
        avg_loss = losses.mean()
        try:
            avg_acc = torch.cat([out[f'{self.test_data_split}_acc']
                                for out in outputs if out[f'{self.test_data_split}_acc'] >= 0]).mean()
        except:
            avg_acc = torch.stack([out[f'{self.test_data_split}_acc']
                    for out in outputs if out[f'{self.test_data_split}_acc'] >= 0]).mean()

        metrics[f'avg_{self.test_data_split}_loss'] = avg_loss
        metrics[f'avg_{self.test_data_split}_acc'] = avg_acc

        y_true = torch.cat([out['y_true'] for out in outputs])
        y_pred = torch.cat([out['y_pred'] for out in outputs])
        y_pred_px = None
        if self.hparams['segmentation']:
            y_pred_px = torch.cat([out['y_pred_px'] for out in outputs])

        if self.hparams['write_logits']:
            y_logits = torch.cat([out['y_logits'] for out in outputs])
            path = f"{self.hparams['default_save_path']}/logits.csv"
            test_dataset = self.test_dataloader().dataset
            lat_lon_year_label_df = test_dataset._image_info[
                ['latitude', 'longitude', 'year', 'label']
            ].reset_index(drop=True)
            y_logits_df = pd.DataFrame(y_logits.cpu().numpy())
            data = pd.concat([lat_lon_year_label_df, y_logits_df], axis=1)
            data.to_csv(path, index=False)

        indices = torch.cat([out['idx'] for out in outputs])

        self.log_results(self.hparams['merge_scheme'],
                         indices,
                         y_true,
                         y_pred,
                         y_pred_px,
                         losses,
                         self.hparams['labels'],
                         self.hparams['label_names'])

        post_merge_scheme = self.hparams['post_merge_scheme_probs'] or self.hparams['post_merge_scheme_preds']
        if post_merge_scheme is not None:
            y_true_post_merge = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                                     post_merge_scheme,
                                                     y_true)
                                             
            y_pred_post_merge = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                                     post_merge_scheme,
                                                     y_pred)

            y_pred_px_post_merge = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                                        post_merge_scheme,
                                                        y_pred_px)

            post_merge_labels = DATASET_MERGE_SCHEMES[post_merge_scheme]['labels']
            post_merge_label_names = DATASET_MERGE_SCHEMES[post_merge_scheme]['label_names']
            
            # TODO: Add new cmap legends for merge_schemes other than 'small-scale-plantation-merge'
            # in merge_scheme.py
            self.log_results(post_merge_scheme,
                             indices,
                             y_true_post_merge,
                             y_pred_post_merge,
                             y_pred_px_post_merge,
                             losses,
                             post_merge_labels,
                             post_merge_label_names)
        
        logs = {f'avg_{self.test_data_split}_loss': avg_loss,
                'log': metrics}
        return logs
    
    def inference(self, batch):
        """Run a forward pass on a set of inputs."""
        x_dict, y, idx = batch
        logits = self.forward(x_dict)
        if self.cam:
            self.save_cams(x_dict, y, logits, idx)
        return logits
