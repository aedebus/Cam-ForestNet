import os
import torch
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import metrics
from scipy.stats import mode
from matplotlib import pyplot as plt
from collections import defaultdict
import logging

from util.constants import *


def get_accuracy(y_hat, y):
    preds = y_hat.argmax(dim=1)
    return torch.eq(preds, y).float().sum() / y.size(0)


def get_y_true_cls(y):
    y_true_cls = y.max(axis=1)[0].max(axis=1)[0]
    for y_item in y:
        driver = torch.unique(y_item[y_item != LOSS_IGNORE_VALUE])
        if driver.size(0) > 1:
            logging.error(
                f"Find two driver in the loss region: {driver}. " +
                "This is supposed to be a data issue."
            )
    return y_true_cls


def segmentation_to_classification(y_hat, y, merge_mapping=None):
    ignore = (y != LOSS_IGNORE_VALUE)
    y_logit = (y_hat * ignore.unsqueeze(1).float())
    y_logit_flat = y_logit.mean(dim=(2, 3))
    if merge_mapping is not None:
        merge_mapping = merge_mapping.to(y_logit_flat.device)
        y_prob_flat = torch.nn.functional.softmax(y_logit_flat, dim=1)
        y_prob_flat = torch.mm(y_prob_flat, merge_mapping)
        y_pred_cls = y_prob_flat.argmax(dim=1)
    else:
        y_pred_cls = y_logit_flat.argmax(dim=1)
    return y_pred_cls, y_logit_flat


def get_formatted_df(a, label_names):
    class_labels = np.arange(a.shape[0])
    actual_class = np.repeat(label_names, a.shape[1])
    pred_class = np.tile(label_names, a.shape[1])
    data = a.flatten()
    df = pd.DataFrame(data={'Actual': actual_class,
                            'Predicted': pred_class,
                            'Value': data})
    df = df.pivot('Actual', 'Predicted', 'Value')
    return df


def plot_accuracy_by_sc_count(y_true, y_pred, dataset, indices, save_path):
    plt.clf()
    bin_width = 5
    bins = np.linspace(
        0,
        SCENE_LIMIT,
        SCENE_LIMIT //
        bin_width +
        1)  # [0, 5, ..., SCENE_LIMIT]
    correct_counts = np.zeros_like(bins)
    total_counts = np.zeros_like(bins)
    for idx, meta_idx in enumerate(indices):
        image_info = dataset.get_image_info(meta_idx.item())
        num_scenes = image_info[NUM_SC_HEADER]
        scene_bin = np.digitize(num_scenes, bins) - 1
        total_counts[scene_bin] += 1
        correct_counts[scene_bin] += int(y_true[idx] == y_pred[idx])
    acc = correct_counts / total_counts
    plt.bar(bins, acc, width=bin_width, align='edge')
    plt.title('Accuracy by scene count')
    plt.xlabel('Scene count')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(save_path, 'accuracy_by_sc_count.png'))


def plot_heatmaps(confusion_matrix, label_names, save_path):
    """
    Plots two heatmaps of actual and predicted class labels
    for a given experiment: normalized and counts.
    Params:
        matrix (np.ndarray): 2D array where actual labels are rows,
                             predicted labels are cols
        labels (List[str]): labels for the classes
    """
    confusion_matrix_df = get_formatted_df(confusion_matrix, label_names)
    plt.figure(figsize=(20, 20))
    #plt.ylabel('Actual Class Label', fontsize = 35) #AD
    #plt.xlabel('Predicted Class Label', fontsize = 35) #AD
    #plt.title('Actual and Predicted Class Labels', fontsize = 40) #AD
    #ax = sn.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 20}) #AD
    #ax.set_yticklabels(ax.get_yticklabels(), size=25) #AD
    #ax.set_xticklabels(ax.get_xticklabels(), size=25) #AD
    #cbar = ax.collections[0].colorbar #AD
    #cbar.ax.tick_params(labelsize=20) #AD

    plt.ylabel('Actual Class Label')
    plt.xlabel('Predicted Class Label')
    plt.title('Actual and Predicted Class Labels')
    sn.heatmap(confusion_matrix_df, annot=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrix_heatmap.png'))

    confusion_matrix_n = confusion_matrix / \
        confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_n_df = get_formatted_df(confusion_matrix_n, label_names)
    plt.figure(figsize=(20, 20))
    #plt.ylabel('Actual Class Label', fontsize = 35) #AD
    #plt.xlabel('Predicted Class Label', fontsize = 35) #AD
    #plt.title('Normalized Actual and Predicted Class Labels', fontsize = 40) #AD
    #ax = sn.heatmap(confusion_matrix_n_df, annot=True, annot_kws={'size': 20}) #AD
    #ax.set_yticklabels(ax.get_yticklabels(), size=25) #AD
    #ax.set_xticklabels(ax.get_xticklabels(), size=25) #AD
    #cbar = ax.collections[0].colorbar #AD
    #cbar.ax.tick_params(labelsize=20) #AD

    plt.ylabel('Actual Class Label')
    plt.xlabel('Predicted Class Label')
    plt.title('Normalized Actual and Predicted Class Labels')
    sn.heatmap(confusion_matrix_n_df, annot=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrix_n_heatmap.png'))

def save_metrics(
        y_true,
        y_pred,
        labels,
        label_names,
        save_path,
        area=None,
        dataset=None,
        indices=None,
        do_plot_sc_count_acc=False):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, 
                                                labels=labels, 
                                                sample_weight=area)
    plot_heatmaps(confusion_matrix, label_names, save_path)
    if do_plot_sc_count_acc:
        plot_accuracy_by_sc_count(y_true, y_pred, dataset, indices, save_path)
    report = metrics.classification_report(y_true, y_pred, labels=labels, 
                                           target_names=label_names, 
                                           digits=3, sample_weight=area)
    logging.info(report)
    accuracy = metrics.accuracy_score(y_true, y_pred, sample_weight=area)
    np.savetxt(os.path.join(save_path, 'confusion_matrix.txt'),
               confusion_matrix.astype(int),
               fmt='%i',
               )

    with open(os.path.join(save_path, 'report.txt'), 'w') as f:
        f.write(report)
        f.write('\n')
        f.write(f'Overall accuracy: {accuracy}\n')
