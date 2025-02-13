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


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score #AD
import seaborn as sns


def merge(nb_years):
    for i in range(1,6):
        path = os.path.join(os.getcwd(), 'models', 'sandbox', 'test_exp_fold' + str(i)+'_ts_'+ str(nb_years))
        print(path)
        epoch = os.listdir(os.path.join(path, 'val_results'))[0]
        logits = pd.read_csv(os.path.join(path, 'val_results', epoch, 'logits.csv'))
        logits.sort_values(by=['latitude', 'longitude'])
        del_labels = len(DATASET_MERGE_SCHEMES['detailed-temp3']['label_names'])
        final_label = []
        
        ind = np.linspace(0, len(logits)-nb_years, int(len(logits)/nb_years))
        
        for i in ind:
            row1 = logits.iloc[[i]]
            row2 = logits.iloc[[i+1]]
            max_logit1 = -10e9
            max_value1 = -1
            for j in range(del_labels):
                if float(row1[str(j)]) > max_logit1:
                    max_logit1 = float(row1[str(j)])
                    max_value1 = int(j)

            max_logit2 = -10e9
            max_value2 = -1
            for j in range(del_labels):
                if float(row2[str(j)]) > max_logit2:
                    max_logit2 = float(row2[str(j)])
                    max_value2 = int(j)
            if nb_years ==2:
                max_logit = max([max_logit1, max_logit2])
                max_ind = [max_logit1, max_logit2].index(max_logit)
                max_value = [max_value1, max_value2][max_ind]
            if nb_years >=3:
                row3 = logits.iloc[[i+2]]
                max_logit3 = -10e9
                max_value3 = -1
                for j in range(del_labels):
                    if float(row3[str(j)]) > max_logit3:
                        max_logit3 = float(row3[str(j)])
                        max_value3 = int(j)
                if nb_years ==3:
                    max_logit = max([max_logit1, max_logit2, max_logit3])
                    max_ind = [max_logit1, max_logit2, max_logit3].index(max_logit)
                    max_value = [max_value1, max_value2, max_value3][max_ind]
            if nb_years ==4:
                row4 = logits.iloc[[i+3]]
                max_logit4 = -10e9
                max_value4 = -1
                for j in range(del_labels):
                    if float(row4[str(j)]) > max_logit4:
                        max_logit4 = float(row4[str(j)])
                        max_value4 = int(j)
                max_logit = max([max_logit1, max_logit2, max_logit3, max_logit4])
                max_ind = [max_logit1, max_logit2, max_logit3, max_logit4].index(max_logit)
                max_value = [max_value1, max_value2, max_value3, max_value4][max_ind]
            final_label.append(DATASET_MERGE_SCHEMES['detailed-temp3']['label_names'][max_value])

        info_data = pd.read_csv(os.path.join(path, 'val_results', epoch, 'mytest_results.csv'))
        info_data = info_data.loc[ind]    
        info_data['predicted_label_fusion'] = final_label
        if os.path.exists(os.path.join(path, 'val_results', epoch, 'timeseries')) == False:
                        os.mkdir(os.path.join(path, 'val_results', epoch, 'timeseries'))
        
        info_data.to_csv(os.path.join(path, 'val_results', epoch,'timeseries', 'timeseries.csv'))
        
        
        #Plot result
        label_names = ['Fruit plantation', 'Grassland shrubland', 'Hunting', 'Infrastructure', 'Mining', 'Oil palm plantation', 'Other', 'Other large-scale plantations', 'Rubber plantation', 
                    'Selective logging', 'Small-scale maize plantation', 'Small-scale oil palm plantation', 'Small-scale other plantation', 'Timber plantation', 'Wildfire'] 

        y_true = info_data['merged_label']
        y_pred = info_data['predicted']
        conf = confusion_matrix(y_true, y_pred, labels=label_names)
        conf_dataframe = pd.DataFrame(conf, index=label_names, columns=label_names)
        df = pd.DataFrame(data=conf_dataframe, index=label_names, columns=label_names)
        plt.figure(figsize = (20,20))
        heat_map = sns.heatmap(df, cmap = "rocket", annot = True)
        heat_map.set(xlabel="Predicted", ylabel="Actual")
        heat_map.set_title('Actual and Predicted Class Labels')
        
        heat_map.figure.savefig(os.path.join(path, 'val_results', epoch,'timeseries','heatmap.png'))

        plt.clf()
        conf_normalized = confusion_matrix(y_true, y_pred, labels=label_names, normalize='true')
        conf_normalized_dataframe = pd.DataFrame(conf_normalized, index=label_names, columns=label_names)
        df_normalized = pd.DataFrame(data=conf_normalized_dataframe, index=label_names, columns=label_names)
        plt.figure(figsize = (20,20))
        heat_map_normalized = sns.heatmap(df_normalized, cmap = "rocket", annot = True)
        heat_map_normalized.set(xlabel="Predicted", ylabel="Actual")
        heat_map_normalized.set_title('Actual and Predicted Class Labels')

        heat_map_normalized.figure.savefig(os.path.join(path, 'val_results', epoch,'timeseries', 'heatmap_normalized.png'))
        
        #Classification report 
        
        report = classification_report(y_true, y_pred, labels=label_names, 
                                           target_names=label_names, 
                                           digits=3)
        accuracy = accuracy_score(y_true, y_pred)
        with open(os.path.join(path, 'val_results', epoch,'timeseries', 'report.txt'), 'w') as f:
            f.write(report)
            f.write('\n')
            f.write(f'Overall accuracy: {accuracy}\n')




if __name__ == "__main__":
    fire.Fire(merge)
