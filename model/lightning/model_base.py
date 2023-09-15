import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from ignite.metrics import ConfusionMatrix, IoU, mIoU
from sklearn.metrics import f1_score
from scipy.special import entr
import pytorch_lightning as pl
import pandas as pd #AD
from models import get_model
from eval import get_loss_fn
from util.constants import *
from util.merge_scheme import *
from util.aux_features import LATE_FUSION_MODE
from data.transforms import get_transforms
from data import (IndonesiaClassificationDataset,
                  IndonesiaSegmentationDataset) 


class BaseModel(pl.LightningModule):
    """Base class for experiments."""

    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = get_model(self.hparams)

        self.train_class_weights = self.get_dataset("train").class_weights()
        self.loss = get_loss_fn(self.hparams,
                                self.train_class_weights,
                                data_split='train')
        self.test_loss = get_loss_fn(self.hparams,
                                     self.train_class_weights,
                                     data_split='test')

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])]

    def get_transforms(self, split):
        is_training = self.hparams['is_training'] and split == TRAIN_SPLIT
        use_landsat = 's2' not in self.hparams['data_version']

        spatial_aug = self.hparams['spatial_augmentation']
        pixel_aug = self.hparams['pixel_augmentation']

        return get_transforms(
            resize=self.hparams['resize'],
            spatial_augmentation=spatial_aug,
            pixel_augmentation=pixel_aug,
            is_training=is_training,
            use_landsat=use_landsat
        )

    def get_dataset(self, split):
        transforms = self.get_transforms(split)
        if split == TRAIN_SPLIT:
            img_option = self.hparams['train_img_option']
        else:
            img_option = self.hparams['eval_img_option']

        no_augment_transforms = self.get_transforms(VAL_SPLIT)

        kwargs = {
            'data_version': self.hparams['data_version'],
            'data_split': split,
            'img_option': img_option,
            'merge_scheme': self.hparams['merge_scheme'],
            'sample_filter': self.hparams['sample_filter'],
            'bands': self.hparams['bands'],
            'masked': self.hparams['masked'],
            'transforms': transforms,
            'ncep': self.hparams['late_fusion_ncep'],
            'no_augment_transforms': no_augment_transforms,
            'aux_features': self.hparams['late_fusion_aux_feats'],
        }

        
        if self.hparams['segmentation']:
            dataset_fn = IndonesiaSegmentationDataset
        else:
            dataset_fn = IndonesiaClassificationDataset

        return dataset_fn(**kwargs) 

    def train_dataloader(self):
        dataset = self.get_dataset(TRAIN_SPLIT)
        if self.hparams['oversample']:
            shuffle = False
            labels=[]  #AD
            for i in range(0, len(dataset._image_info[LABEL_HEADER])): #AD
                labels.append(INDONESIA_ALL_LABELS.index(dataset._image_info.iloc[i][LABEL_HEADER])) #AD
            labels=pd.Series(labels, name='label').map(dataset._label_mapping) #AD
            label2weight = dataset.class_weights()
            weights = labels.apply(lambda label: label2weight[label])
            sampler = WeightedRandomSampler(weights, len(dataset))
            return DataLoader(dataset, #AD
                              batch_size=self.hparams['batch_size'],
                              num_workers=self.hparams['num_dl_workers'],
                              shuffle=shuffle,
                              sampler=sampler)

        else:
            shuffle = True
            return DataLoader(dataset, #AD
                              batch_size=self.hparams['batch_size'],
                              num_workers=self.hparams['num_dl_workers'],
                              shuffle=shuffle) 

    def val_dataloader(self): 
        dataset = self.get_dataset(VAL_SPLIT)
        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_dl_workers'],
                          shuffle=False)

    def test_dataloader(self): 
        dataset = self.get_dataset(self.hparams['test_data_split'])
        return DataLoader(dataset,
                          batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_dl_workers'],
                          shuffle=False)
    
