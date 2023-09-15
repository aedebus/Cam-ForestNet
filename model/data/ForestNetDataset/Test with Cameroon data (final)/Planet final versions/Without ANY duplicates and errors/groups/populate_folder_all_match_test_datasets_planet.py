import os
import shutil
from csv import writer
import fire
import pandas as pd

import random
from matplotlib.colors import ListedColormap
import numpy as np

#constants
MERGED_LABEL_HEADER = 'merged_label'
LABEL_HEADER = 'label'
LATITUDE_HEADER = 'latitude'
LONGITUDE_HEADER = 'longitude'
IMG_PATH_HEADER = 'example_path'
YEAR_HEADER = 'year'


CAMEROON_TRAIN_SPLIT, CAMEROON_VAL_SPLIT, CAMEROON_TEST_SPLIT = .6, .15, .25


CAMEROON_META_COLNAMES = [LABEL_HEADER, #AD
                           MERGED_LABEL_HEADER,
                           LATITUDE_HEADER,
                           LONGITUDE_HEADER,
                           YEAR_HEADER,
                           IMG_PATH_HEADER]

CAMEROON_DATASET_NAME = 'cameroon'
CAMEROON_ALL_LABELS = [
    'Oil palm plantation',
    'Timber plantation',
    'Other large-scale plantations',
    'Grassland shrubland',
    'Small-scale agriculture',
    'Small-scale mixed plantation',
    'Small-scale oil palm plantation',
    'Mining',
    'Fish pond',
    'Logging',
    'Secondary forest',
    'Other',
    '?', #AD
    'Rubber plantation', #AD
    'Timber sales', #AD
    'Fruit plantation', #AD
    'Wildfire', #AD
    'Small-scale cassava plantation', #AD
    'Infrastructure',
    'Selective logging'] #AD

DATASET_MERGE_SCHEMES = {
    "four-class": {
        'datasets': [CAMEROON_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': CAMEROON_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder agriculture',
            'Small-scale mixed plantation': 'Smallholder agriculture',
            'Small-scale oil palm plantation': 'Smallholder agriculture',
            'Small-scale cassava plantation': 'Smallholder agriculture', #AD
            'Mining': 'Other',
            'Fish pond': 'Other',
            'Logging': 'Other',
            'Secondary forest': 'Other',
            'Other': 'Other',
            'Other large-scale plantation': 'Plantation', # AD
            'Rubber plantation': 'Plantation', # AD
            'Timber sales': 'Other', # AD
            'Fruit plantation' : 'Plantation', # AD
            'Infrastructure' : 'Other', #AD
            'Wildfire' : 'Other' }}, #AD
    "four-class2": {
        'datasets': [CAMEROON_DATASET_NAME],
        'label_names': [
            'Plantation',
            'Grassland shrubland',
            'Smallholder agriculture',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': CAMEROON_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Plantation',
            'Timber plantation': 'Plantation',
            'Other large-scale plantations': 'Plantation',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale agriculture': 'Smallholder agriculture',
            'Small-scale oil palm plantation': 'Smallholder agriculture',
            'Small-scale maize plantation': 'Smallholder agriculture', #AD
            'Small-scale other plantation': 'Smallholder agriculture', #AD
            'Selective logging': 'Other', #AD
            'Mining': 'Other',
            'Fish pond': 'Other',
            'Logging': 'Other',
            'Secondary forest': 'Other',
            'Other': 'Other',
            'Other large-scale plantations': 'Plantation', # AD
            'Rubber plantation': 'Plantation', # AD
            'Fruit plantation' : 'Plantation', # AD
            'Infrastructure' : 'Other', #AD
            'Hunting': 'Other', #AD
            'Wildfire' : 'Other'}}, #AD
    "detailed-temp": {
        'datasets': [CAMEROON_DATASET_NAME], 
        'label_names': [
            'Oil palm plantation',
            'Timber plantation',
            'Fruit plantation',
            'Rubber plantation',
            'Other large-scale plantations',
            'Grassland shrubland',
            'Small-scale mixed plantation',
            'Small-scale oil palm plantation',
            'Small-scale cassava plantation',
            'Mining',
            'Logging',
            'Timber sales',
            'Infrastructure',
            'Wildfire'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': CAMEROON_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Fruit plantation': 'Fruit plantation',
            'Rubber plantation': 'Rubber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale mixed plantation': 'Small-scale mixed plantation',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Small-scale cassava plantation': 'Small-scale cassava plantation',
            'Mining': 'Mining',
            'Logging': 'Logging',
            'Timber sales': 'Timber sales',
            'Infrastructure' : 'Infrastructure',
            'Wildfire': 'Wildfire' }},
    "detailed-temp2": {
        'datasets': [CAMEROON_DATASET_NAME], 
        'label_names': [
            'Oil palm plantation',
            'Timber plantation',
            'Fruit plantation',
            'Rubber plantation',
            'Other large-scale plantations',
            'Grassland shrubland',
            'Small-scale mixed plantation',
            'Small-scale oil palm plantation',
            'Small-scale cassava plantation',
            'Mining',
            'Selective logging',
            'Infrastructure',
            'Wildfire'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': CAMEROON_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Fruit plantation': 'Fruit plantation',
            'Rubber plantation': 'Rubber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            'Small-scale mixed plantation': 'Small-scale mixed plantation',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Small-scale cassava plantation': 'Small-scale cassava plantation',
            'Mining': 'Mining',
            'Selective logging': 'Selective logging',
            'Infrastructure' : 'Infrastructure',
            'Wildfire': 'Wildfire'}},

    "detailed-temp3": {
        'datasets': [CAMEROON_DATASET_NAME], 
        'label_names': [
            'Oil palm plantation',
            'Timber plantation',
            'Fruit plantation',
            'Rubber plantation',
            'Other large-scale plantations',
            'Grassland shrubland',
            #'Small-scale mixed plantation',
            'Small-scale oil palm plantation',
            'Small-scale other plantation',
            'Small-scale maize plantation',
            'Mining',
            'Selective logging',
            'Infrastructure',
            'Wildfire',
            'Hunting',
            'Other'],
        'color_scheme': ListedColormap(['y', 'b', 'r', 'c']),
        'original_label_names': CAMEROON_ALL_LABELS,
        'scheme': {
            'Oil palm plantation': 'Oil palm plantation',
            'Timber plantation': 'Timber plantation',
            'Fruit plantation': 'Fruit plantation',
            'Rubber plantation': 'Rubber plantation',
            'Other large-scale plantations': 'Other large-scale plantations',
            'Grassland shrubland': 'Grassland shrubland',
            #'Small-scale mixed plantation': 'Small-scale mixed plantation',
            'Small-scale oil palm plantation': 'Small-scale oil palm plantation',
            'Small-scale maize plantation': 'Small-scale maize plantation',
            'Small-scale other plantation': 'Small-scale other plantation',
            'Mining': 'Mining',
            'Selective logging': 'Selective logging',
            'Infrastructure' : 'Infrastructure',
            'Wildfire': 'Wildfire',
            'Hunting' : 'Hunting', 
            'Other':'Other'}}

    
}

#Adapted from Irvin et al. (2020): Added stratified sampling

def get_split_indices(all_meta, train_split, val_split, merge_scheme):
    direct = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    landsat = pd.read_csv(os.path.join(direct, 'Landsat final versions', 'Without ANY duplicates and errors','groups', 'test.csv'), names=CAMEROON_META_COLNAMES)
    landsat = landsat.iloc[1:]
    planet_all = pd.read_csv(os.path.join(os.getcwd(), 'all.csv'), names=CAMEROON_META_COLNAMES)
    planet_all = planet_all.iloc[1:]
    
    #landsat['year']=landsat['year'].astype(int)
    #landsat['longitude']=landsat['longitude'].astype(float)
    #landsat['latitude']=landsat['latitude'].astype(float)
    planet_all.loc[:, 'index'] = np.linspace(1, len(planet_all), num = len(planet_all))
    
    list_indices =[]
    merged = landsat.merge(planet_all, on = ['latitude', 'longitude', 'year', 'merged_label', 'label'], suffixes = ["_landsat", "_planet"])
    #for j in range(len(merged)):
        #if (merged['merged_label_landsat'][j] != merged['merged_label_planet'][j]) or (merged['label_landsat'][j] != merged['label_planet'][j]):
            #merged.drop(index = j, inplace = True)
    merged.reset_index()
    #for i in range (len(landsat)):
        #list_indices.append((landsat.iloc[i]['longitude'], landsat.iloc[i]['latitude']))

    #Stratified sampling = split by category
    train_indices = []
    val_indices = []
    test_indices = []
    grouped = all_meta.groupby(MERGED_LABEL_HEADER)
    grouped_merged = merged.groupby(MERGED_LABEL_HEADER)


    for label in DATASET_MERGE_SCHEMES[merge_scheme]['label_names']:
        print(label)
        group_label = grouped.get_group(label)
        group_merged_label = grouped_merged.get_group(label)
        indices = list(group_label.index)
        num_examples = len(indices)
        num_train = int(num_examples * CAMEROON_TRAIN_SPLIT)
        num_val = int(num_examples * CAMEROON_VAL_SPLIT)
        #for index in indices:
            #if (all_meta.loc[index]['longitude'], all_meta.loc[index]['latitude']) in list_indices:
                #test_indices.append(index)
                #indices.remove(index)
        test_ind = group_merged_label['index']

        for i in test_ind.index:
            test_indices.append(int(test_ind[i]))
            print(int(test_ind[i]))
            indices.remove(int(test_ind[i]))
        random.shuffle(indices)
        train_indices += indices[:num_train]
        val_indices += indices[num_train:num_train+num_val]
        test_indices += indices[num_train+num_val:]

    return train_indices, val_indices, test_indices


def split_data(meta_fn, merge_scheme):
    random.seed(1)
    all_meta = pd.read_csv(os.path.join(os.getcwd(), meta_fn), names=CAMEROON_META_COLNAMES)
    all_meta = all_meta.iloc[1:]
    train_indices, val_indices, test_indices = get_split_indices(all_meta,
                                                                 CAMEROON_TRAIN_SPLIT,
                                                                 CAMEROON_VAL_SPLIT,
                                                                 merge_scheme)
    #keep same: .6, .15, .25
    train_meta = pd.DataFrame(all_meta.loc[train_indices])
    val_meta = pd.DataFrame(all_meta.loc[val_indices])
    test_meta = pd.DataFrame(all_meta.loc[test_indices])

    train_meta.to_csv('train.csv', index=False, header=True, columns=CAMEROON_META_COLNAMES)
    val_meta.to_csv('val.csv', index=False, header=True, columns=CAMEROON_META_COLNAMES)
    test_meta.to_csv('test.csv', index=False, header=True, columns=CAMEROON_META_COLNAMES)

    train_all = pd.read_csv(os.path.join(os.getcwd(), 'train.csv'))
    val_all = pd.read_csv(os.path.join(os.getcwd(), 'val.csv'))
    test_all = pd.read_csv(os.path.join(os.getcwd(), 'test.csv'))
    print('train:', '\n')
    print(train_all['label'].value_counts(), '\n')

    print('val:', '\n')
    print(val_all['label'].value_counts(), '\n')
    
    print('test:', '\n')
    print(test_all['label'].value_counts(), '\n')




split_data('all.csv', "four-class2")



