from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, Callback
)
from pytorch_lightning.loggers.test_tube import TestTubeLogger
import os
import re
import numpy as np

from util.constants import *
from util.merge_scheme import *
from util.util import get_version_num

CKPT_REGEX = r'epoch_[0-9]+_val_loss_([0-9]+\.[0-9]+)\.pt'
CKPT_NAME = '{epoch:02d}-{val_loss:.2f}.pt'


def add_labels_and_label_names(args):
    merge_scheme = args['merge_scheme']
    if merge_scheme is None or args['dataset'] == LANDCOVER_DATASET_NAME:
        labels = DATASET_LABELS_NAMES[args['dataset']]['labels']
        label_names = DATASET_LABELS_NAMES[args['dataset']]['label_names']
    else:
        dataset = args['dataset']
        assert(merge_scheme in DATASET_MERGE_SCHEMES)
        assert(dataset in DATASET_MERGE_SCHEMES[merge_scheme]['datasets'])
        labels = DATASET_MERGE_SCHEMES[merge_scheme]['labels']
        label_names = DATASET_MERGE_SCHEMES[merge_scheme]['label_names']

    args['labels'] = labels
    args['label_names'] = label_names


def get_ckpt_dir(save_path, exp_name):
    ckpt_dir = os.path.join(save_path, exp_name, "ckpts")
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    return ckpt_dir


def get_ckpt_callback(save_path, exp_name, ckpt_metric):
    assert(
        ckpt_metric in [
            'val_loss',
            'val_mIoU',
            'val_f1_micro',
            'val_f1_macro'])
    if 'loss' in ckpt_metric:
        _mode = 'min'
    else:
        _mode = 'max'
    ckpt_dir = get_ckpt_dir(save_path, exp_name)

    ckpt_name = '{epoch}-{' + ckpt_metric + ':.4f}'
    return ModelCheckpoint(filepath=os.path.join(ckpt_dir, ckpt_name),
                           save_top_k=10,
                           verbose=True,
                           monitor=ckpt_metric,
                           mode=_mode,
                           prefix='')


def get_early_stop_callback(patience, ckpt_metric):
    if 'loss' in ckpt_metric:
        _mode = 'min'
    else:
        _mode = 'max'
    return EarlyStopping(monitor=ckpt_metric,
                         patience=patience,
                         verbose=True,
                         mode=_mode)


def get_metric_val_ckpt_name(ckpt_path, ckpt_metric):
    ckpt_reg = rf'epoch=[0-9]+-{ckpt_metric}=([0-9\.]+)\.ckpt'
    r = re.search(ckpt_reg, ckpt_path).groups()
    assert(len(r) == 1)
    metric_val = float(r[0])
    return metric_val


def get_logger(save_path, exp_name):
    exp_dir = os.path.join(save_path, exp_name)
    return TestTubeLogger(save_dir=exp_dir,
                          name='lightning_logs',
                          version=str(get_version_num()))


def get_best_ckpt(save_path, exp_name, ckpt_metric):
    ckpt_dir = get_ckpt_dir(save_path, exp_name)
    ckpts = os.listdir(ckpt_dir)
    ckpt_reg = rf'epoch=[0-9]+-{ckpt_metric}=([0-9\.]+)\.ckpt'
    ckpt_metric_vals = np.array([get_metric_val_ckpt_name(ckpt, ckpt_metric)
                                 for ckpt in ckpts])
    if 'loss' in ckpt_metric:
        best_ckpt = ckpts[np.argmin(ckpt_metric_vals)]
    else:
        best_ckpt = ckpts[np.argmax(ckpt_metric_vals)]

    return os.path.join(ckpt_dir, best_ckpt)


class ResetTrainDataloaderCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        if (pl_module.hparams['consistency_regularize'] and
                pl_module.current_epoch >= pl_module.hparams['consistency_warmup']):
            trainer.reset_train_dataloader(pl_module)
