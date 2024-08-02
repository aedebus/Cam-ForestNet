import os
import shutil
from csv import writer
import fire
import pandas as pd

import random
from matplotlib.colors import ListedColormap

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



def populate_folder(folder, source, sensor, csv_file, merge_scheme):
    """
    :param folder: folder where to put images
    :param source: folder where parameters are
    :param sensor: landsat or planet
    :param csv_file: csv_files to fill out (containing all data)
    :return: folder populated with images and auxiliary data, csv file filled out
    """
    path_in = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), source)
    years = [2020]
    for year in years:
        path_image = os.path.join(path_in, 'download_images_quick', 'output', str(year))
        path_auxiliary = os.path.join(path_in, 'download_auxiliary_quick', 'output', str(year))
        path_gfc = os.path.join(path_in, 'download_gfc', 'output', str(year))

        list_dir = os.listdir(path_image)
        for d in list_dir:
            if ('production_forest' not in d) and ('small_scale_agriculture' not in d) and ('timber_sales' not in d) and ('Subsistence' not in d) and ('permit' not in d) and ('_sar' not in d): #to reduce files copied # added 
                indices = os.listdir(os.path.join(path_image, d))
                for i in indices:
                    #1. Obtain images
                    images_source = os.path.join(path_image, d, i, sensor + '_nir_ps')
                    if os.path.exists(images_source) == True:
                        
                        images = os.listdir(images_source)
                        lon = images[0].split('_')[1]
                        if 'png' in (images[0].split('_')[2]):
                            lat = (images[0].split('_')[2])[:-4]
                        else:
                            lat = images[0].split('_')[2]
                        coord = lon + '_' + lat
                    
                        destination_images = os.path.join(os.getcwd(), folder, coord, 'images', 'visible')
                        if os.path.exists(os.path.join(os.getcwd(), folder, coord)) == False:
                            os.mkdir(os.path.join(os.getcwd(), folder, coord))
                        if os.path.exists(os.path.join(os.getcwd(), folder, coord, 'images')) == False:
                            os.mkdir(os.path.join(os.getcwd(), folder, coord, 'images'))
                        if os.path.exists(destination_images) == False:
                            os.mkdir(destination_images)
                            source_images = os.path.join(images_source, images[0]) #keep lowest cloud cover (landsat) or first image (planet)
                            shutil.copy2(source_images, destination_images)


                                    #2. Obtain infrared
                                
                            ir_source = os.path.join(path_auxiliary, d, i, sensor)
                            #print(ir_source)
                            destination_ir = os.path.join(os.getcwd(), folder, coord, 'images', 'infrared')
                            if os.path.exists(destination_ir) == False:
                                os.mkdir(destination_ir)
                                source_ir = os.path.join(ir_source, str(year)+'_ir_0.npy')
                                shutil.copy2(source_ir, destination_ir)

                            #3.Obtain auxiliary
                            
                            aux_source1 = os.path.join(path_auxiliary, d, i, sensor)
                            aux_source2 = os.path.join(path_auxiliary, d, i, sensor + '_30')
                            #print(aux_source)
                            destination_aux = os.path.join(os.getcwd(), folder, coord, 'auxiliary')
                            if os.path.exists(destination_aux) == False:
                                os.mkdir(destination_aux)
                                source1 = os.listdir(aux_source1)
                                for s1 in source1:
                                    if 'ncep' not in s1 and 'ir' not in s1:
                                        shutil.copy2(os.path.join(aux_source1, s1), destination_aux)
                                os.mkdir(os.path.join(destination_aux, 'ncep'))
                                source2 = os.listdir(aux_source2)
                                for s2 in source2:
                                    if 'ncep' in s2:
                                        list_ncep = os.listdir(os.path.join(aux_source2, s2))
                                        for n in list_ncep:
                                             shutil.copy2(os.path.join(aux_source2, s2, n), os.path.join(destination_aux, 'ncep'))

                            

                            #4. Obtain forest loss region
                            gfc_source = os.path.join(path_gfc, d, i)
                            destination_gfc = os.path.join(os.path.join(os.getcwd(), folder, coord))
                            source_gfc = os.path.join(gfc_source, 'forest_loss_region.pkl')
                            shutil.copy2(source_gfc, destination_gfc)

                            print('Ok:', os.path.join(os.getcwd(), folder, coord))


if __name__ == "__main__":
    fire.Fire(populate_folder)