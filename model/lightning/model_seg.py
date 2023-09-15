import torch
import nni
import logging 
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from sklearn.metrics import f1_score, accuracy_score
from scipy.special import entr
import pytorch_lightning as pl

from eval import get_loss_fn, ConsistencyLoss
from util.constants import *
from util.merge_scheme import *
from .logger import TFLogger
from .model_base import BaseModel
import pandas as pd #AD
import os #AD


class SegmentationModel(BaseModel, TFLogger):
    """Standard interface for segmentation experiments."""

    def __init__(self, params):
        super().__init__(params)
        self.labels = self.hparams.get("labels")
        self.cm = ConfusionMatrix(len(self.labels))
        self.iou = IoU(self.cm)
        self.miou = mIoU(self.cm)
        self.merge_mapping = None
        self.list_paths=[] #AD
        self.list_predicted=[] #AD
        self.list_true=[] #AD

        post_merge_scheme_probs = self.hparams['post_merge_scheme_probs']

        if post_merge_scheme_probs is not None:
            post_merge_scheme_mapping = DATASET_MERGE_SCHEMES[post_merge_scheme_probs]['label_mapping']
            output_dim = len(set(post_merge_scheme_mapping))
            self.merge_mapping = torch.zeros(
                (len(self.labels), output_dim))
            for k, v in self.val_dataloader().dataset._label_mapping.items():
                self.merge_mapping[k, post_merge_scheme_mapping[v]] = 1

        args = self.hparams.copy()
        args['loss_fn'] = "CE"
        args['segmentation'] = False
        self.cls_loss = get_loss_fn(args, self.train_class_weights,
                                    data_split="train")
        self.test_cls_loss = get_loss_fn(args, self.train_class_weights,
                                         data_split="test")
        self.con_loss = ConsistencyLoss()

    def forward(self, batch, is_training=True):
        y_true_cls = batch['label_cls']
        y_true_seg = batch['label_seg']
        valid_sample = batch['valid']
        mask = batch['mask']
        y_logit_seg, y_logit_cls, y_pred_cls = self.model(batch)

        loss_seg = self.loss(y_logit_seg, y_true_seg)
        loss_cls = self.cls_loss(y_logit_cls, y_true_cls)
        loss_test_cls = self.test_cls_loss(y_logit_cls, y_true_cls)
        loss_con = self.consistency_loss(batch)
        loss = loss_seg + self.hparams['alpha'] * loss_cls + loss_con
       
        acc = (torch.eq(y_pred_cls, y_true_cls) *
               valid_sample).float().sum() / valid_sample.sum()
        
        if valid_sample.sum() == 0:
            acc = -1
            loss = torch.tensor(1.0, requires_grad=True)
        
        if is_training:
            return loss, acc
        else:
            return loss, y_logit_seg, y_logit_cls, y_pred_cls
    
    def consistency_loss(self, batch):
        if (self.hparams['is_training'] and
            self.hparams['consistency_regularize'] and
            self.current_epoch >= self.hparams['consistency_warmup']):
            
            valid_sample = batch['valid']
        
            _, y_logit_cls_1, _ = self.model(batch)
            batch['image'], batch['image_no_augment'] = batch['image_no_augment'], batch['image']
            _, y_logit_cls_2, _ = self.model(batch)
            batch['image'], batch['image_no_augment'] = batch['image_no_augment'], batch['image']
            con_loss, con_mask = self.con_loss(y_logit_cls_1, y_logit_cls_2)
            # Mask loss based on confident predictions and examples which contain
            loss_mask = con_mask * valid_sample
            cons_loss = (con_loss * loss_mask).sum() / loss_mask.sum()
            
            return cons_loss
        else:
            return 0

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
        loss, y_logit_seg, y_logit_cls, y_pred_cls = self.forward(batch, 
                                                                  is_training=False)
        self.cm.update((y_logit_seg, batch['label_seg']))

        logs = {'val_loss': loss,
                'y_true': batch['label_cls'],
                'y_pred': y_pred_cls,
                'valid': batch['valid'],
                'area': batch['forest_loss'],
                'log': {self.current_epoch}}
        return logs

    def validation_epoch_end(self, outputs):
        metrics = {}
        avg_loss = torch.stack([out['val_loss'] for out in outputs]).mean()
        metrics['avg_val_loss'] = avg_loss
        metrics['val_mIoU'] = mIoU = self.miou.compute()
        ious = self.iou.compute()
        for j, l in enumerate(self.labels):
            metrics[f'val_IoU_{l}'] = ious[j]
        self.cm.reset()

        y_true_cls = torch.cat([out['y_true'] for out in outputs])
        y_pred_cls = torch.cat([out['y_pred'] for out in outputs])
        if self.hparams['eval_by_pixel']:
            area = torch.cat([out['area'] for out in outputs])
        else:
            area = None
        
        valid_sample = torch.cat([out['valid'] for out in outputs])
        acc = (torch.eq(y_pred_cls, y_true_cls) *
               valid_sample).float().sum() / valid_sample.sum()
        y_true_numpy = y_true_cls.cpu().detach().numpy()
        y_pred_numpy = y_pred_cls.cpu().detach().numpy()
        if self.hparams['eval_by_pixel']:
            area = area.cpu().detach().numpy() 
        metrics['val_f1_micro'] = f1_score(
            y_true_numpy, y_pred_numpy, average="micro", sample_weight=area)
        metrics['val_f1_macro'] = f1_score(
            y_true_numpy, y_pred_numpy, average="macro", sample_weight=area)
        metrics['avg_val_acc'] = acc
        logs = {'val_loss': avg_loss,
                'log': metrics,
                'progress_bar': {"val_mIoU": mIoU,
                                 "val_loss": avg_loss,
                                 "val_acc": acc,
                                 }}
        nni.report_intermediate_result(metrics['val_f1_macro'])         
        return logs

    def test_step(self, batch, batch_nb):
        test_data_split = self.hparams['test_data_split']
        loss, _, y_logit_cls, y_pred_cls = self.forward(batch, is_training=False)
        y_logit_seg = self.model.get_seg_logits(batch)

        if self.merge_mapping is not None:
            merge_mapping = merge_mapping.to(y_logit_cls.device)
            y_prob_cls = torch.nn.functional.softmax(y_logit_cls, dim=1)
            y_prob_cls = torch.mm(y_prob_cls, merge_mapping)
            y_pred_cls = y_prob_cls.argmax(dim=1)
        
        valid_sample = batch['valid']
        #if valid_sample.sum() == 0:
            #valid_sample = torch.where(valid_sample ==True, valid_sample, True)
        y_true_cls = batch['label_cls']
        acc = (torch.eq(y_pred_cls, y_true_cls) *
               valid_sample).float().sum() / valid_sample.sum()
        logs = {f'{test_data_split}_loss': loss,
                f'{test_data_split}_acc': acc,
                'y_true': batch['label_cls'],
                'idx': batch['index'],
                'valid': batch['valid'], 
                'y_pred': y_pred_cls,
                'y_pred_px': y_logit_seg.argmax(1),
                'area': batch['forest_loss'],
                'y_logits': y_logit_cls,
                'log': {f'{test_data_split}_loss': loss,
                        f'{test_data_split}_acc': acc},
                'progress_bar': {f'{test_data_split}_loss': loss}}
        
        self.list_paths.append(batch.get('im_path')) #AD
        self.list_predicted.append(y_pred_cls) #AD
        self.list_true.append(y_true_cls) #AD
        
        return logs
    
    #FIXME(@Hao): Fix the testing; be careful with the merge_scheme
    def test_epoch_end(self, outputs):
        test_data_split = self.hparams['test_data_split']
        metrics = {}
        valid = torch.stack([out['valid'].sum() for out in outputs])
        losses = torch.stack([out[f'{test_data_split}_loss'] for out in outputs])
        losses = losses.to('cuda:0') #AD
        avg_loss = (losses * valid).sum() / valid.sum()

        try:
            avg_acc = torch.cat([out[f'{test_data_split}_acc']
                                for out in outputs if out[f'{test_data_split}_acc'] >= 0]).mean()
        except:
            avg_acc = torch.stack([out[f'{test_data_split}_acc']
                    for out in outputs if out[f'{test_data_split}_acc'] >= 0]).mean()

        metrics[f'avg_{test_data_split}_loss'] = avg_loss
        metrics[f'avg_{test_data_split}_acc'] = avg_acc
        y_true = torch.cat([out['y_true'] for out in outputs])
        y_pred = torch.cat([out['y_pred'] for out in outputs])
        y_logits = torch.cat([out['y_logits'] for out in outputs])
        if self.hparams['eval_by_pixel']:
            area = torch.cat([out['area'] for out in outputs])
            area = area.cpu().detach().numpy()
        else:
            area = None
        y_pred_px = None
        if self.hparams['segmentation']:
            y_pred_px = torch.cat([out['y_pred_px'] for out in outputs])

        if self.hparams['write_logits']:
            path = f"{self.hparams['default_save_path']}/logits.csv"
            test_dataset = self.test_dataloader().dataset
            lat_lon_year_label_df = test_dataset._image_info[
                ['latitude', 'longitude', 'year', 'label'] 
            ].reset_index(drop=True)
            y_logits_df = pd.DataFrame(y_logits.cpu().numpy())
            data = pd.concat([lat_lon_year_label_df, y_logits_df], axis=1)
            data.to_csv(path, index=False)

        #AD: add results to csv
        if os.path.exists(os.path.dirname(self.hparams['default_save_path'])) == False:
            os.mkdir(os.path.dirname(self.hparams['default_save_path']))
        if os.path.exists(self.hparams['default_save_path']) == False:
            os.mkdir(self.hparams['default_save_path'])
        path = f"{self.hparams['default_save_path']}/mytest_results.csv"
        test_dataset = self.test_dataloader().dataset
        lat_lon_year_label_df = test_dataset._image_info[
            ['latitude', 'longitude', 'year', 'label', 'merged_label', 'example_path']
        ].reset_index(drop=True)
        y_pred_df = pd.DataFrame(y_pred.cpu().numpy(), columns = ['y_pred_df'])
        list_pred = []
        for k in (y_pred.cpu().numpy()):
            list_pred.append(merge_scheme['label_names'][k])
        y_label_df = pd.DataFrame(list_pred, columns = ['predicted'])
        data = pd.concat([lat_lon_year_label_df, y_pred_df, y_label_df], axis=1)
        data.to_csv(path, index=False)
        #
        indices = torch.cat([out['idx'] for out in outputs])
    
        self.log_results(self.hparams['merge_scheme'],
                         indices,
                         y_true,
                         y_pred,
                         y_logits,
                         y_pred_px,
                         losses,
                         self.hparams['labels'],
                         self.hparams['label_names'],
                         area=area)

        post_merge_scheme = self.hparams['post_merge_scheme_probs'] or self.hparams['post_merge_scheme_preds']
        if post_merge_scheme is not None:
            logging.info(f"Using merge scheme [{post_merge_scheme}]. ")
            y_true_post_merge = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                                     post_merge_scheme,
                                                     y_true)
                                             
            y_pred_post_merge = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                                     post_merge_scheme,
                                                     y_pred)

            y_pred_px_post_merge = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                                        post_merge_scheme,
                                                        y_pred_px)
            
            y_logits_post_merge = None
            post_merge_labels = DATASET_MERGE_SCHEMES[post_merge_scheme]['labels']
            post_merge_label_names = DATASET_MERGE_SCHEMES[post_merge_scheme]['label_names']
            if self.hparams['test_data_split'] == VAL_SPLIT:
                y_true_numpy = y_true_post_merge.cpu().numpy()
                y_pred_numpy = y_pred_post_merge.cpu().numpy()
                f1 = f1_score(y_true_numpy, y_pred_numpy, average="macro", sample_weight=area)
                acc = accuracy_score(y_true_numpy, y_pred_numpy)
                logging.info(f"Post merge f1: {f1}")
                logging.info(f"Post merge acc: {acc}")
                nni.report_final_result({"default": f1, "acc": acc})
     
            # TODO: Add new cmap legends for merge_schemes other than 'small-scale-plantation-merge'
            # in merge_scheme.py
            self.log_results(post_merge_scheme,
                             indices,
                             y_true_post_merge,
                             y_pred_post_merge,
                             y_logits_post_merge,
                             y_pred_px_post_merge,
                             losses,
                             post_merge_labels,
                             post_merge_label_names,
                             area=area)
        
        logs = {f"avg_{test_data_split}_loss": avg_loss,
                'log': metrics}
        
        return logs
