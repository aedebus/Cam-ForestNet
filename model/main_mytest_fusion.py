'''To test with data not provided by ForestNet but added in a mytest.csv file in another folder 'MyTest' in data'''

import os
import fire
import random
import torch
import logging
import inspect
import csv #AD
import pandas as pd #AD
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from util import *
from util.constants import *
from PIL import Image, ImageDraw, ImageEnhance #AD
import numpy as np #AD
import pickle #AD
import matplotlib.pyplot as plt


from lightning import SegmentationModel, ClassificationModel, PretrainingModel
from lightning.util import *
from data.util import add_labels_and_label_names


from sklearn.metrics import confusion_matrix #AD
import seaborn as sns


def train(exp_name,
          seed=0,
          dataset=CAMEROON_DATASET_NAME,
          data_version="new_splits",
          segmentation=True,
          post_merge_scheme_preds=None,
          post_merge_scheme_probs=None,
          train_img_option=IMG_OPTION_RANDOM,
          eval_img_option=IMG_OPTION_RANDOM,#AD: modified
          merge_scheme='small-scale-plantation-merge',
          resize="none",
          spatial_augmentation="none",
          pixel_augmentation="none",
          test_data_split=VAL_SPLIT,
          save_path=str(SANDBOX_DIR),
          tb_path=str(TB_DIR),
          model="ResNet101",
          architecture="UNet",
          ckpt_path=None,
          log_save_interval=1,
          distributed_backend="dp",
          gradient_clip_val=0.5,
          max_epochs=300,
          train_percent_check=1.0,
          val_percent_check=1.0,
          test_percent_check=1.0,
          ckpt_metric='val_f1_macro',
          patience=100,
          loss_fn="combined",
          sample_filter="historical_merge",
          late_fusion=False,
          late_fusion_detach=False,
          late_fusion_latlon=False,
          late_fusion_stats=False,
          late_fusion_aux_feats=False,
          late_fusion_ncep=True,
          late_fusion_embedding_dim=128,
          late_fusion_dropout=0.2,
          consistency_regularize=False,
          consistency_warmup=75,
          batch_size=16,#AD
          pretrained="none",
          gpus=None,
          num_dl_workers=16,
          lr=1e-4,
          class_weights=False,
          fixed_class_weights=False,
          oversample=False,
          log_images=True,
          masked=False,
          cam=False,
          bands=["red", "green", "blue"],
          use_classification_head=False,
          input_forest_loss_seg=False,
          input_forest_loss_cls=None,
          gamma=3,
          write_logits=False,
          alpha=1,
          eval_by_pixel=False,
          ):
    """
    Run the training experiment.

    Args:
        data_version: Version of the dataset, i.e. suffix of the csvs
                      in the data directory.
        model: Model name
        save_path: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        segmentation: Do segmentation instead of classification.
        log_save_interval: Logging saving frequency (in batch)
        distributed_backend: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        train_percent_check: Proportion of training data to use
        max_epochs: Max number of epochs
        patience: number of epochs with no improvement after
                  which training will be stopped.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        input_forest_loss_seg: Boolean denoting whether to input
                               forest loss area to seg head (after encoder).
        input_forest_loss_cls: How to input forest loss into
                               classification head. One of
                               None: do not input
                               "enc_cls": input to cls after encoder
                               "dec_cls": input to cls after decoder

    Returns: None

    """
    if seed:
        #Reproducibility
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    assert(dataset in DATASET_NAMES and
           test_data_split in DATA_SPLITS and
           train_img_option in IMG_OPTIONS and
           eval_img_option in IMG_OPTIONS)

    # NOTE: Getting this error without this line for late fusion:
    # https://forums.fast.ai/t/runtimeerror-received-0-items-of-ancdata/48935 
    if late_fusion_aux_feats:
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Ensure the specified bands are compatible with the augmentations
    if len(bands) != 3 and (
            'cloud' in spatial_augmentation or pixel_augmentation == 'all'
    ):
        raise ValueError(f"Bands {bands} not compatible with " +
                         f"spatial_augmentation={spatial_augmentation} and " +
                         f"pixel_augmentation={pixel_augmentation}")

    args = Args(locals())
    args['is_training'] = True
    add_labels_and_label_names(args)

    if pretrained == "landcover":
        args['ckpt_path'] = get_best_ckpt(
            save_path, LANDCOVER_EXP_NAMES[model], ckpt_metric
        )

    if segmentation:
        if dataset == LANDCOVER_DATASET_NAME:
            m = PretrainingModel(args)
        else:
            m = SegmentationModel(args)
    else:
        m = ClassificationModel(args)

    init_exp_folder(args)
    trainer = Trainer(
        distributed_backend=distributed_backend,
        logger=get_logger(save_path, exp_name),
        weights_save_path=os.path.join(
            save_path,
            exp_name),
        log_save_interval=log_save_interval,
        checkpoint_callback=get_ckpt_callback(
            save_path,
            exp_name,
            ckpt_metric),
        early_stop_callback=get_early_stop_callback(
            patience,
            ckpt_metric),
        callbacks=[ResetTrainDataloaderCallback()],
        gradient_clip_val=gradient_clip_val,
        train_percent_check=train_percent_check,
        val_percent_check=val_percent_check,
        test_percent_check=test_percent_check,
        max_epochs=max_epochs,
        gpus=gpus,
        weights_summary='top')

    trainer.fit(m)

    best_ckpt_path = get_best_ckpt(save_path, exp_name, ckpt_metric)
    logging.info(f'Running test with best ckpt: [{best_ckpt_path}]')

    for split in [TRAIN_SPLIT, test_data_split]:
        test(log_images=False,
             oversample=False,
             ckpt_path=best_ckpt_path,
             test_data_split=split)


def test(ckpt_path1, ckpt_path2,
         gpus=1,
         **kwargs):
    m1 = _load_from_checkpoint(ckpt_path1, **kwargs) #Landsat
    #m1.hparams['write_logits'] = True
    #m1.hparams['test_data_split'] = 'test_landsat'
    m2 = _load_from_checkpoint(ckpt_path2, **kwargs) #Planet
    #m2.hparams['write_logits'] = True
    #m2.hparams['test_data_split'] = 'test_planet'
    #Trainer(gpus=gpus).test(m1)
    #Trainer(gpus=gpus).test(m2)
    path1 = ckpt_path1.split('/')[0] + '/' + ckpt_path1.split('/')[1] + '/' + ckpt_path1.split('/')[2]
    path2 = ckpt_path2.split('/')[0] + '/' + ckpt_path2.split('/')[1] + '/' + ckpt_path2.split('/')[2]
    log1 = pd.read_csv(os.path.join(path1, 'test_results', ckpt_path1.split('/')[-1][:-5], 'logits.csv')) #Landsat
    log2 = pd.read_csv(os.path.join(path2, 'test_results', ckpt_path2.split('/')[-1][:-5], 'logits.csv')) #Planet

    list_sensor = []
    fus_log = log1.merge(log2, on = ['latitude', 'longitude', 'year'], suffixes = ["_landsat", "_planet"]) #inner join by default
    #for i in range(len(fus_log)):
    #    row = fus_log.loc([i])
    #    for j in range (15):
    #        log_landsat = row[str(j)]
    #        log_planet = row[str(j)]
    #        if log_landsat > log_planet:
    #            list_sensor.append('landsat')
    #        else:
    #            list_sensor.append('planet')

    #fus_log['sensor'] = list_sensor
    info_data_landsat = pd.read_csv(os.path.join(path1, 'test_results', ckpt_path1.split('/')[-1][:-5], 'mytest_results.csv'))
    info_data_planet = pd.read_csv(os.path.join(path2, 'test_results', ckpt_path2.split('/')[-1][:-5], 'mytest_results.csv'))
    info_data = info_data_landsat.merge(info_data_planet, on = ['latitude', 'longitude', 'year'], suffixes = ["_landsat", "_planet"])
    del_labels = len(DATASET_MERGE_SCHEMES[m1.hparams['merge_scheme']]['label_names'])
    
    
    final_label = []
    for i in range (len(fus_log)):
        row = fus_log.iloc[[i]]
        max_logit = -10e9
        max_value = -1
        for j in range(del_labels):
            if float(row[str(j) + '_landsat']) > max_logit:
                max_logit = float(row[str(j) + '_landsat'])
                max_value = int(j)
            if float(row[str(j) + '_planet']) > max_logit:
                max_logit = float(row[str(j) + '_landsat'])
                max_value = int(j)
        final_label.append(DATASET_MERGE_SCHEMES[m1.hparams['merge_scheme']]['label_names'][max_value])
    
    info_data['predicted_label_fusion'] = final_label
    if os.path.exists(os.path.join(SANDBOX_DIR, 'fusion_'+ ckpt_path1.split('/')[-1][:-5] + '_' + ckpt_path2.split('/')[-1][:-5])) == False:
                      os.mkdir(os.path.join(SANDBOX_DIR, 'fusion_'+ ckpt_path1.split('/')[-1][:-5] + '_' + ckpt_path2.split('/')[-1][:-5]))
    
    info_data.to_csv(os.path.join(SANDBOX_DIR, 'fusion_'+ ckpt_path1.split('/')[-1][:-5] + '_' + ckpt_path2.split('/')[-1][:-5], 'fusion.csv'))
    
    
    #Plot result
    #label_names = DATASET_MERGE_SCHEMES[m1.hparams['merge_scheme']]['label_names']
    label_names = ['Fruit plantation', 'Grassland shrubland', 'Hunting', 'Infrastructure', 'Mining', 'Oil palm plantation', 'Other', 'Other large-scale plantations', 'Rubber plantation', 
                   'Selective logging', 'Small-scale maize plantation', 'Small-scale oil palm plantation', 'Small-scale other plantation', 'Timber plantation', 'Wildfire'] 
    #order to match the order in the previous heatmaps
    y_true = info_data['merged_label_landsat']
    y_pred = info_data['predicted_label_fusion']
    conf = confusion_matrix(y_true, y_pred, labels=label_names)
    conf_dataframe = pd.DataFrame(conf, index=label_names, columns=label_names)
    df = pd.DataFrame(data=conf_dataframe, index=label_names, columns=label_names)
    plt.figure(figsize = (20,20))
    heat_map = sns.heatmap(df, cmap = "magma", annot = True)
    heat_map.set(xlabel="Predicted", ylabel="Actual")
    heat_map.set_title('Actual and Predicted Class Labels')
    
    heat_map.figure.savefig(os.path.join(SANDBOX_DIR, 'fusion_'+ ckpt_path1.split('/')[-1][:-5] + '_' + ckpt_path2.split('/')[-1][:-5], 'heatmap.png'))

    plt.clf()
    conf_normalized = confusion_matrix(y_true, y_pred, labels=label_names, normalize='true')
    conf_normalized_dataframe = pd.DataFrame(conf_normalized, index=label_names, columns=label_names)
    df_normalized = pd.DataFrame(data=conf_normalized_dataframe, index=label_names, columns=label_names)
    plt.figure(figsize = (20,20))
    heat_map_normalized = sns.heatmap(df_normalized, cmap = "rocket", annot = True)
    heat_map_normalized.set(xlabel="Predicted", ylabel="Actual")
    heat_map_normalized.set_title('Actual and Predicted Class Labels')

    heat_map_normalized.figure.savefig(os.path.join(SANDBOX_DIR, 'fusion_'+ ckpt_path1.split('/')[-1][:-5] + '_' + ckpt_path2.split('/')[-1][:-5], 'heatmap_normalized.png'))
    

    


def _load_from_checkpoint(ckpt_path, **kwargs):
    hparams = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']
    hparams['test_data_split']='test' #AD
    ensemble = os.path.isdir(ckpt_path)

    if hparams['segmentation']:
        if hparams['dataset'] == LANDCOVER_DATASET_NAME:
            Model = PretrainingModel
        else:
            Model = SegmentationModel
    else:
        Model = ClassificationModel

    if ensemble:
        m = Model(hparams)
    else:
        m = Model.load_from_checkpoint(ckpt_path, 
                                       is_training=False,
                                       ensemble=ensemble,
                                       ckpt_path=None,
                                       test_data_split='test', #AD
                                       data_split='test', #AD
                                       **kwargs)

    m.hparams['default_save_path'] = os.path.join(
        m.hparams['save_path'],
        m.hparams['exp_name'],
        f"{m.hparams['test_data_split']}_results",
        Path(ckpt_path).stem)
    return m


if __name__ == "__main__":
    fire.Fire()
