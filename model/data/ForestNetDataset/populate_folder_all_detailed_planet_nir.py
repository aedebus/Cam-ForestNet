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

def get_split_indices(all_meta, train_split, val_split, merge_scheme):
    #Stratified sampling = split by category
    train_indices = []
    val_indices = []
    test_indices = []
    #print(all_meta)
    grouped = all_meta.groupby(MERGED_LABEL_HEADER)
    for label in DATASET_MERGE_SCHEMES[merge_scheme]['label_names']:
        group_label = grouped.get_group(label)
        indices = list(group_label.index)
        random.shuffle(indices)
        num_examples = len(indices)
        num_train = int(num_examples * train_split)
        num_val = int(num_examples * val_split)
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



def populate_folder(folder, source, sensor, csv_file, merge_scheme):
    """
    :param folder: folder where to put images
    :param source: folder where parameters are
    :param sensor: landsat or planet
    :param csv_file: csv_files to fill out (containing all data)
    :return: folder populated with images and auxiliary data, csv file filled out
    """
    all_data =[] #to check for duplicates

    path_in = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), source)
    years = [2015, 2016, 2017, 2018, 2019, 2020]
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
                    images_source = os.path.join(path_image, d, i, sensor + '_nir')
                    if os.path.exists(images_source) == True:
                        csv_to_fill = False
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
                            csv_to_fill = True
                            source_images = os.path.join(images_source, images[0]) #keep lowest cloud cover (landsat) or first image (planet)
                            shutil.copy2(source_images, destination_images)


                                    #2. Obtain infrared
                                
                            ir_source = os.path.join(path_auxiliary, d, i, sensor)
                            #print(ir_source)
                            destination_ir = os.path.join(os.getcwd(), folder, coord, 'images', 'infrared')
                            if os.path.exists(destination_ir) == False:
                                os.mkdir(destination_ir)
                                csv_to_fill = True
                                source_ir = os.path.join(ir_source, str(year)+'_ir_0.npy')
                                shutil.copy2(source_ir, destination_ir)

                            #3.Obtain auxiliary
                            
                            aux_source = os.path.join(path_auxiliary, d, i, sensor)
                            #print(aux_source)
                            destination_aux = os.path.join(os.getcwd(), folder, coord, 'auxiliary')
                            if os.path.exists(destination_aux) == False:
                                csv_to_fill = True
                                shutil.copytree(aux_source, destination_aux)
                                list_aux = os.listdir(destination_aux)
                                for a in list_aux:
                                    if 'ir' in a:
                                        os.remove(os.path.join(destination_aux, a))

                            #4. Obtain forest loss region
                            gfc_source = os.path.join(path_gfc, d, i)
                            destination_gfc = os.path.join(os.path.join(os.getcwd(), folder, coord))
                            source_gfc = os.path.join(gfc_source, 'forest_loss_region.pkl')
                            shutil.copy2(source_gfc, destination_gfc)

                            # #5. Fill out csv files: file containing all data
                            
                            list_data=[]
                                #Need to redo because varies with year
                            if merge_scheme == 'four-class':
                                if 'agro_industrial' in d:
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '5') or (i == '23') or (i=='25') or (i=='26') or (i=='27') or (i=='28') \
                                            or (i=='32') or (i=='33') or (i == '39') or (i == '43') or (i == '50') or (i == '51') or (i == '52') \
                                            or (i == '53') or (i == '76'):
                                        list_data.append('Rubber plantation') #https://cdc-cameroon.net/new2014/products/rubber/
                                        #Uncertainty 26, 27, 28, 30, 32, 33, 39
                                    elif (i == '3') or (i == '4') or (i == '6') or (i == '7') or (i == '8') or (i == '9') or \
                                            (i == '10') or (i == '17') or (i == '18') or (i == '19') or (i == '21') or (i == '22') or (i == '24') \
                                            or (i == '29') or (i=='30') or (i=='31') or (i=='34') or (i == '35') or (i=='36') or (i=='38') or (i=='40') \
                                            or (i=='41') or (i == '42') or (i=='47') or (i=='48') or (i=='49') or (i=='54') or (i == '55') or (i == '57') or (i == '58') or (i=='59') or (i=='60') or (i=='61') \
                                            or (i == '62') or (i=='63') or (i=='64') or (i == '65') or (i=='66') or (i=='71')  or (i=='74') or (i == '81') or (i == '82') or (i == '84'):
                                        list_data.append('Oil palm plantation')
                                        #82: Camvert, map from Ruksan
                                    elif (i == '11') or (i == '12') or (i == '13') or (i == '14') or (i == '15') or (i == '16') \
                                            or (i == '20') or (i == '75') or (i == '83'):
                                        list_data.append('Fruit plantation')
                                        #nanana
                                        #https://cdc-cameroon.net/new2014/products/banana/ for 14, 20
                                        #https://www.businessincameroon.com/agriculture/1807-9349-cameroon-to-value-the-plantain-sector-with-construction-of-an-industrial-plantain-processing-plant-in-pouma: for 75, plantain
                                        # (uncertainty but deduced from proximity other banana plantations)
                                    elif (i == '45') or (i == '46'):
                                        list_data.append('Other large-scale plantations') #sugarcane plantation
                                    elif (i == '37') or (i == '67') or (i == '68') or (i == '69') or (i == '70')  or (i == '72') or (i == '73'):
                                        list_data.append('Other large-scale plantations') #tea
                                    # if (i == '78') or (i == '79') or (i == '80'):
                                    #     list_data.append('?')
                                    else:
                                        list_data.append('?')
                                    list_data.append('Plantation')

                                elif ('BioPalm' in d) or ('SocaPalm' in d):
                                    list_data.append('Oil palm plantation')
                                    list_data.append('Plantation')
                                elif ('Forest_Management' in d) and (i != 12): #nb: FMU match with forest production, 12 is Camvert = plantation, 50 encompasses plantation but I checked each image is outside so ok
                                    list_data.append('Logging')
                                    list_data.append('Other')
                                elif ('Hevecam' in d) or ('SudCam' in d):
                                    list_data.append('Rubber plantation')
                                    list_data.append('Plantation')
                                elif ('mining' in d) and ('permit' not in d):
                                    list_data.append('Mining')
                                    list_data.append('Other')

                                elif 'planted_areas' in d:
                                    #4,13,14,15,25,27,28,29,30,31,32,33,34,37,38,39,40,41,43,44,46,47,48, 49, 50, 51,52,63,64: Cannot deduced by proximity
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '3') or (i == '6') or (i == '7') or (i == '8') or (i == '9')\
                                            or (i == '12') or (i == '16') or (i == '19') or (i == '22') or (i == '24') or (i == '36') or (i == '54') \
                                            or (i == '55') or (i == '62') or (i == '65'):
                                        list_data.append('Rubber plantation')
                                        #3,6,7, 16, 19, 24, 36, 54, 55, 62, 65: uncertainty but assuming by proximity
                                    elif (i == '9') or (i == '10') or (i == '11') or (i == '17') or (i == '18') or (i == '20') or (i == '21') or (i == '23')\
                                            or (i == '35') or (i == '56') or (i == '60'):
                                        list_data.append('Oil palm plantation')
                                        #10, 17, 18, 20, 21, 23, 60: uncertainty but assuming by proximity
                                    elif (i == '57') or (i == '61') or (i== '66'):
                                        list_data.append('Fruit plantation')
                                        #57, 61, 66 : uncertainty but assuming by proximity
                                    else:
                                        list_data.append('?')
                                    list_data.append('Plantation')

                                elif 'production_forest' in d:
                                    list_data.append('?')
                                    list_data.append('?')
                                elif ('small_scale_agriculture' in d) or ('Subsistence agriculture.shp' in d):
                                    list_data.append('Small-scale agriculture') 
                                    list_data.append('Smallholder agriculture')
                                    
                                elif 'timber_plantations' in d:
                                    list_data.append('Timber plantation')
                                    list_data.append('Plantation')
                                elif 'timber_sales' in d:
                                    list_data.append('Timber sales')
                                    list_data.append('Other')
                                elif 'food_crops' in d:
                                    list_data.append('Fruit plantation')
                                    list_data.append('Plantation')
                                elif 'WorldCover' in d:
                                    list_data.append('Grassland shrubland') #For future: grassland or shrubland
                                    list_data.append('Grassland shrubland')
                                elif 'Wildfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Other')
                                elif 'cassava' in d:
                                    list_data.append('Small-scale cassava plantation')
                                    list_data.append('Smallholder agriculture')
                                elif 'biopama' in d:
                                    list_data.append('Small-scale oil palm plantation')
                                    list_data.append('Smallholder agriculture')
                                elif 'buildings' in d:
                                    list_data.append('Infrastructure')
                                    list_data.append('Other')
                                else:
                                    list_data.append('?')
                                    list_data.append('?')
                            
                            elif merge_scheme == 'four-class2':
                                if 'agro_industrial' in d:
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '5') or (i == '23') or (i=='25')  \
                                            or (i == '43') or (i == '50') or (i == '51') or (i == '52') \
                                            or (i == '53') or (i == '75') or (i == '76'):
                                        list_data.append('Rubber plantation') #https://cdc-cameroon.net/new2014/products/rubber/
                                        list_data.append('Plantation')
                                        #Uncertainty 26, 27, 28, 30, 32, 33, 39
                                        #or (i=='26') or (i=='27') or (i=='28') or (i=='32') or (i=='33') or (i == '39') 
                                    elif (i == '3') or (i == '4') or (i == '6') or (i == '7') or (i == '8') or (i == '9') or \
                                            (i == '10') or (i == '17') or (i == '18') or (i == '19') or (i == '21') or (i == '22') or (i == '24') \
                                            or (i == '29') or (i=='31') or (i=='34') or (i == '35') or (i=='36') or (i=='38') or (i=='40') \
                                            or (i=='41') or (i == '42') or (i=='47') or (i=='48') or (i=='49') or (i=='54') or (i == '55') or (i == '57') or (i == '58') or (i=='59') or (i=='60') or (i=='61') \
                                            or (i == '62') or (i=='63') or (i=='64') or (i == '65') or (i=='66') or (i=='71')  or (i=='74') or (i == '81') or (i == '82') or (i == '84'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Plantation')
                                    elif ((i == '20') and '2020' in d) or ((i == '83') and '2019' in d):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Plantation')
                                        #82: Camvert, map from Ruksan
                                        #or (i=='30')
                                    elif ((i == '11') or (i == '12') or (i == '13') or (i == '15')) and '2020' not in d:
                                        list_data.append('Fruit plantation')
                                        list_data.append('Plantation')
                        
                                    elif ((i == '11') or (i == '12') or (i == '14') \
                                            or (i == '83')) and '2020' in d:
                                        list_data.append('Fruit plantation')
                                        list_data.append('Plantation')
                                        #banana
                                        #https://cdc-cameroon.net/new2014/products/banana/ for 14, 20
                                        #https://www.businessincameroon.com/agriculture/1807-9349-cameroon-to-value-the-plantain-sector-with-construction-of-an-industrial-plantain-processing-plant-in-pouma: for 75, plantain
                                        # (uncertainty but deduced from proximity other banana plantations) : 14, 16, 20  if not 2020; 15, 13 if 2020
                                    elif (i == '45') or (i == '46'):
                                        list_data.append('Other large-scale plantations') #sugarcane plantation
                                        list_data.append('Plantation') 
                                    elif (i == '37') or (i == '67') or (i == '68') or (i == '69') or (i == '70')  or (i == '72') or (i == '73'):
                                        list_data.append('Other large-scale plantations') #tea
                                        list_data.append('Plantation') 
                                    # if (i == '78') or (i == '79') or (i == '80'):
                                    #     list_data.append('?')
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif ('BioPalm' in d) or ('SocaPalm' in d):
                                    list_data.append('Oil palm plantation')
                                    list_data.append('Plantation')
                                elif ('Forest_Management' in d) and (i != 12): #nb: FMU match with forest production, 12 is Camvert = plantation, 50 encompasses plantation but I checked each image is outside so ok
                                    list_data.append('Selective logging')
                                    list_data.append('Other')
                                elif ('Hevecam' in d) or ('SudCam' in d):
                                    list_data.append('Rubber plantation')
                                    list_data.append('Plantation')
                                elif ('mining' in d) and ('permit' not in d):
                                    list_data.append('Mining')
                                    list_data.append('Other')

                                elif 'planted_areas' in d:
                                    #4,13,14,15,25,27,28,29,30,31,32,33,34,37,38,39,40,41,43,44,46,47,48, 49, 50, 51,52,63,64: Cannot deduced by proximity
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '8'):
                                        list_data.append('Rubber plantation')
                                        list_data.append('Plantation')
                                        #3,6,7, 16, 19, 24, 36, 54, 55, 62, 65: uncertainty but assuming by proximity
                                        #or (i == '3') or (i == '6') or (i == '7') or (i == '16') or (i == '19') (i == '24') or (i == '36') or (i == '54') or (i == '55') or (i == '62') or (i == '65')
                                    elif (i == '9') or (i == '11') or (i == '17') or (i == '35') or (i == '56'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Plantation')
                                        #10, 17, 18, 20, 21, 23, 60: uncertainty but assuming by proximity
                                        #or (i == '10') or (i == '17') or (i == '18') or (i == '20') or (i == '21') or (i == '23') or (i == '60')
                                    #elif (i == '57') or (i == '61') or (i== '66'):
                                        #list_data.append('Fruit plantation')
                                        #list_data.append('Fruit plantation')
                                        #57, 61, 66 : uncertainty but assuming by proximity
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif 'production_forest' in d:
                                    list_data.append('?')
                                    list_data.append('?')
                                elif 'small_scale_agriculture' in d:
                                    list_data.append('?') 
                                    list_data.append('Smallholder agriculture')
                                elif 'Subsistence agriculture' in d:
                                    list_data.append('?')
                                    list_data.append('Smallholder agriculture')
                                    
                                elif 'new_timber_plantations' in d:
                                    list_data.append('Timber plantation')
                                    list_data.append('Plantation')
                                    
                                elif 'timber_sales' in d:
                                    list_data.append('Timber sales')
                                    list_data.append('?')
                                elif 'food_crops' in d:
                                    list_data.append('Fruit plantation')
                                    list_data.append('Plantation')
                                elif 'WorldCover' in d:
                                    list_data.append('Grassland shrubland') 
                                    list_data.append('Grassland shrubland')
                                elif 'Wildfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Other')
                                elif 'globfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Other')
                                elif 'cassava' in d or 'cereal' in d:
                                    list_data.append('Small-scale other plantation')
                                    list_data.append('Smallholder agriculture')
                                elif 'maize' in d:
                                    if (i != '208' and i != '279'):
                                        list_data.append('Small-scale maize plantation')
                                        list_data.append('Smallholder agriculture')
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')
                                elif 'biopama' in d:
                                    list_data.append('Small-scale oil palm plantation')
                                    list_data.append('Smallholder agriculture')
                                elif 'buildings' in d:
                                    list_data.append('Infrastructure')
                                    list_data.append('Other')
                                elif 'hunting' in d:
                                    list_data.append('Hunting')
                                    list_data.append('Other')
                                elif 'water' in d:
                                    list_data.append('Other')
                                    list_data.append('Other')
                                else:
                                    list_data.append('?')
                                    list_data.append('?')


                            
                            
                            elif merge_scheme == 'detailed-temp':
                                if 'agro_industrial' in d:
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '5') or (i == '23') or (i=='25') or (i=='26') or (i=='27') or (i=='28') \
                                            or (i=='32') or (i=='33') or (i == '39') or (i == '43') or (i == '50') or (i == '51') or (i == '52') \
                                            or (i == '53') or (i == '76'):
                                        list_data.append('Rubber plantation') #https://cdc-cameroon.net/new2014/products/rubber/
                                        list_data.append('Rubber plantation')
                                        #Uncertainty 26, 27, 28, 30, 32, 33, 39
                                    elif (i == '3') or (i == '4') or (i == '6') or (i == '7') or (i == '8') or (i == '9') or \
                                            (i == '10') or (i == '17') or (i == '18') or (i == '19') or (i == '21') or (i == '22') or (i == '24') \
                                            or (i == '29') or (i=='30') or (i=='31') or (i=='34') or (i == '35') or (i=='36') or (i=='38') or (i=='40') \
                                            or (i=='41') or (i == '42') or (i=='47') or (i=='48') or (i=='49') or (i=='54') or (i == '55') or (i == '57') or (i == '58') or (i=='59') or (i=='60') or (i=='61') \
                                            or (i == '62') or (i=='63') or (i=='64') or (i == '65') or (i=='66') or (i=='71')  or (i=='74') or (i == '81') or (i == '82') or (i == '84'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                        #82: Camvert, map from Ruksan
                                    elif (i == '11') or (i == '12') or (i == '13') or (i == '14') or (i == '15') or (i == '16') \
                                            or (i == '20') or (i == '75') or (i == '83'):
                                        list_data.append('Fruit plantation')
                                        list_data.append('Fruit plantation')
                                        #banana
                                        #https://cdc-cameroon.net/new2014/products/banana/ for 14, 20
                                        #https://www.businessincameroon.com/agriculture/1807-9349-cameroon-to-value-the-plantain-sector-with-construction-of-an-industrial-plantain-processing-plant-in-pouma: for 75, plantain
                                        # (uncertainty but deduced from proximity other banana plantations)
                                    elif (i == '45') or (i == '46'):
                                        list_data.append('Other large-scale plantations') #sugarcane plantation
                                        list_data.append('Other large-scale plantations') 
                                    elif (i == '37') or (i == '67') or (i == '68') or (i == '69') or (i == '70')  or (i == '72') or (i == '73'):
                                        list_data.append('Other large-scale plantations') #tea
                                        list_data.append('Other large-scale plantations') 
                                    # if (i == '78') or (i == '79') or (i == '80'):
                                    #     list_data.append('?')
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif ('BioPalm' in d) or ('SocaPalm' in d):
                                    list_data.append('Oil palm plantation')
                                    list_data.append('Oil palm plantation')
                                elif ('Forest_Management' in d) and (i != 12): #nb: FMU match with forest production, 12 is Camvert = plantation, 50 encompasses plantation but I checked each image is outside so ok
                                    list_data.append('Logging')
                                    list_data.append('Logging')
                                elif ('Hevecam' in d) or ('SudCam' in d):
                                    list_data.append('Rubber plantation')
                                    list_data.append('Rubber plantation')
                                elif ('mining' in d) and ('permit' not in d):
                                    list_data.append('Mining')
                                    list_data.append('Mining')

                                elif 'planted_areas' in d:
                                    #4,13,14,15,25,27,28,29,30,31,32,33,34,37,38,39,40,41,43,44,46,47,48, 49, 50, 51,52,63,64: Cannot deduced by proximity
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '3') or (i == '6') or (i == '7') or (i == '8') or (i == '9')\
                                            or (i == '12') or (i == '16') or (i == '19') or (i == '22') or (i == '24') or (i == '36') or (i == '54') \
                                            or (i == '55') or (i == '62') or (i == '65'):
                                        list_data.append('Rubber plantation')
                                        list_data.append('Rubber plantation')
                                        #3,6,7, 16, 19, 24, 36, 54, 55, 62, 65: uncertainty but assuming by proximity
                                    elif (i == '9') or (i == '10') or (i == '11') or (i == '17') or (i == '18') or (i == '20') or (i == '21') or (i == '23')\
                                            or (i == '35') or (i == '56') or (i == '60'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                        #10, 17, 18, 20, 21, 23, 60: uncertainty but assuming by proximity
                                    elif (i == '57') or (i == '61') or (i== '66'):
                                        list_data.append('Fruit plantation')
                                        list_data.append('Fruit plantation')
                                        #57, 61, 66 : uncertainty but assuming by proximity
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif 'production_forest' in d:
                                    list_data.append('?')
                                    list_data.append('?')
                                elif 'small_scale_agriculture' in d:
                                    list_data.append('?') 
                                    list_data.append('Smallholder agriculture')
                                elif 'Subsistence agriculture.shp' in d:
                                    list_data.append('Small-scale mixed plantation')
                                    list_data.append('Small-scale mixed plantation')
                                    
                                elif 'timber_plantations' in d:
                                    list_data.append('Timber plantation')
                                    list_data.append('Timber plantation')
                                elif 'timber_sales' in d:
                                    list_data.append('Timber sales')
                                    list_data.append('Timber sales')
                                elif 'food_crops' in d:
                                    list_data.append('Fruit plantation')
                                    list_data.append('Fruit plantation')
                                elif 'WorldCover' in d:
                                    list_data.append('Grassland shrubland') 
                                    list_data.append('Grassland shrubland')
                                elif 'Wildfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Wildfire')
                                elif 'cassava' in d:
                                    list_data.append('Small-scale cassava plantation')
                                    list_data.append('Small-scale cassava plantation')
                                elif 'biopama' in d:
                                    list_data.append('Small-scale oil palm plantation')
                                    list_data.append('Small-scale oil palm plantation')
                                elif 'buildings' in d:
                                    list_data.append('Infrastructure')
                                    list_data.append('Infrastructure')
                                else:
                                    list_data.append('?')
                                    list_data.append('?')

                            elif merge_scheme == 'detailed-temp2':
                                if 'agro_industrial' in d:
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '5') or (i == '23') or (i=='25') or (i=='26') or (i=='27') or (i=='28') \
                                            or (i=='32') or (i=='33') or (i == '39') or (i == '43') or (i == '50') or (i == '51') or (i == '52') \
                                            or (i == '53') or (i == '76'):
                                        list_data.append('Rubber plantation') #https://cdc-cameroon.net/new2014/products/rubber/
                                        list_data.append('Rubber plantation')
                                        #Uncertainty 26, 27, 28, 30, 32, 33, 39
                                    elif (i == '3') or (i == '4') or (i == '6') or (i == '7') or (i == '8') or (i == '9') or \
                                            (i == '10') or (i == '17') or (i == '18') or (i == '19') or (i == '21') or (i == '22') or (i == '24') \
                                            or (i == '29') or (i=='30') or (i=='31') or (i=='34') or (i == '35') or (i=='36') or (i=='38') or (i=='40') \
                                            or (i=='41') or (i == '42') or (i=='47') or (i=='48') or (i=='49') or (i=='54') or (i == '55') or (i == '57') or (i == '58') or (i=='59') or (i=='60') or (i=='61') \
                                            or (i == '62') or (i=='63') or (i=='64') or (i == '65') or (i=='66') or (i=='71')  or (i=='74') or (i == '81') or (i == '82') or (i == '84'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                        #82: Camvert, map from Ruksan
                                    elif (i == '11') or (i == '12') or (i == '13') or (i == '14') or (i == '15') or (i == '16') \
                                            or (i == '20') or (i == '75') or (i == '83'):
                                        list_data.append('Fruit plantation')
                                        list_data.append('Fruit plantation')
                                        #nanana
                                        #https://cdc-cameroon.net/new2014/products/banana/ for 14, 20
                                        #https://www.businessincameroon.com/agriculture/1807-9349-cameroon-to-value-the-plantain-sector-with-construction-of-an-industrial-plantain-processing-plant-in-pouma: for 75, plantain
                                        # (uncertainty but deduced from proximity other banana plantations)
                                    elif (i == '45') or (i == '46'):
                                        list_data.append('Other large-scale plantations') #sugarcane plantation
                                        list_data.append('Other large-scale plantations') 
                                    elif (i == '37') or (i == '67') or (i == '68') or (i == '69') or (i == '70')  or (i == '72') or (i == '73'):
                                        list_data.append('Other large-scale plantations') #tea
                                        list_data.append('Other large-scale plantations') 
                                    # if (i == '78') or (i == '79') or (i == '80'):
                                    #     list_data.append('?')
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif ('BioPalm' in d) or ('SocaPalm' in d):
                                    list_data.append('Oil palm plantation')
                                    list_data.append('Oil palm plantation')
                                elif ('Forest_Management' in d) and (i != 12): #nb: FMU match with forest production, 12 is Camvert = plantation, 50 encompasses plantation but I checked each image is outside so ok
                                    list_data.append('Selective logging')
                                    list_data.append('Selective logging')
                                elif ('Hevecam' in d) or ('SudCam' in d):
                                    list_data.append('Rubber plantation')
                                    list_data.append('Rubber plantation')
                                elif ('mining' in d) and ('permit' not in d):
                                    list_data.append('Mining')
                                    list_data.append('Mining')

                                elif 'planted_areas' in d:
                                    #4,13,14,15,25,27,28,29,30,31,32,33,34,37,38,39,40,41,43,44,46,47,48, 49, 50, 51,52,63,64: Cannot deduced by proximity
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '3') or (i == '6') or (i == '7') or (i == '8') or (i == '9')\
                                            or (i == '12') or (i == '16') or (i == '19') or (i == '22') or (i == '24') or (i == '36') or (i == '54') \
                                            or (i == '55') or (i == '62') or (i == '65'):
                                        list_data.append('Rubber plantation')
                                        list_data.append('Rubber plantation')
                                        #3,6,7, 16, 19, 24, 36, 54, 55, 62, 65: uncertainty but assuming by proximity
                                    elif (i == '9') or (i == '10') or (i == '11') or (i == '17') or (i == '18') or (i == '20') or (i == '21') or (i == '23')\
                                            or (i == '35') or (i == '56') or (i == '60'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                        #10, 17, 18, 20, 21, 23, 60: uncertainty but assuming by proximity
                                    elif (i == '57') or (i == '61') or (i== '66'):
                                        list_data.append('Fruit plantation')
                                        list_data.append('Fruit plantation')
                                        #57, 61, 66 : uncertainty but assuming by proximity
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif 'production_forest' in d:
                                    list_data.append('?')
                                    list_data.append('?')
                                elif 'small_scale_agriculture' in d:
                                    list_data.append('?') 
                                    list_data.append('Smallholder agriculture')
                                elif 'Subsistence agriculture.shp' in d:
                                    list_data.append('Small-scale mixed plantation')
                                    list_data.append('Small-scale mixed plantation')
                                    
                                elif 'new_timber_plantations' in d:
                                    list_data.append('Timber plantation')
                                    list_data.append('Timber plantation')
                                elif 'timber_sales' in d:
                                    list_data.append('Timber sales')
                                    list_data.append('?')
                                elif 'food_crops' in d:
                                    list_data.append('Fruit plantation')
                                    list_data.append('Fruit plantation')
                                elif 'WorldCover' in d:
                                    list_data.append('Grassland shrubland') 
                                    list_data.append('Grassland shrubland')
                                elif 'Wildfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Wildfire')
                                elif 'globfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Wildfire')
                                elif 'cassava' in d:
                                    list_data.append('Small-scale cassava plantation')
                                    list_data.append('Small-scale cassava plantation')
                                elif 'biopama' in d:
                                    list_data.append('Small-scale oil palm plantation')
                                    list_data.append('Small-scale oil palm plantation')
                                elif 'buildings' in d:
                                    list_data.append('Infrastructure')
                                    list_data.append('Infrastructure')
                                else:
                                    list_data.append('?')
                                    list_data.append('?')

                            elif merge_scheme == 'detailed-temp3':
                                if 'agro_industrial' in d:
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '5') or (i == '23') or (i=='25')  \
                                            or (i == '43') or (i == '50') or (i == '51') or (i == '52') \
                                            or (i == '53') or (i == '75') or (i == '76'):
                                        list_data.append('Rubber plantation') #https://cdc-cameroon.net/new2014/products/rubber/
                                        list_data.append('Rubber plantation')
                                        #Uncertainty 26, 27, 28, 30, 32, 33, 39
                                        #or (i=='26') or (i=='27') or (i=='28') or (i=='32') or (i=='33') or (i == '39') 
                                    elif (i == '3') or (i == '4') or (i == '6') or (i == '7') or (i == '8') or (i == '9') or \
                                            (i == '10') or (i == '17') or (i == '18') or (i == '19') or (i == '21') or (i == '22') or (i == '24') \
                                            or (i == '29') or (i=='31') or (i=='34') or (i == '35') or (i=='36') or (i=='38') or (i=='40') \
                                            or (i=='41') or (i == '42') or (i=='47') or (i=='48') or (i=='49') or (i=='54') or (i == '55') or (i == '57') or (i == '58') or (i=='59') or (i=='60') or (i=='61') \
                                            or (i == '62') or (i=='63') or (i=='64') or (i == '65') or (i=='66') or (i=='71')  or (i=='74') or (i == '81') or (i == '82') or (i == '84'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                    elif ((i == '20') and '2020' in d) or ((i == '83') and '2019' in d):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                        #82: Camvert, map from Ruksan
                                        #or (i=='30')
                                    elif ((i == '11') or (i == '12') or (i == '13') or (i == '15')) and '2020' not in d:
                                        list_data.append('Fruit plantation')
                                        list_data.append('Fruit plantation')
                        
                                    elif ((i == '11') or (i == '12') or (i == '14') \
                                            or (i == '83')) and '2020' in d:
                                        list_data.append('Fruit plantation')
                                        list_data.append('Fruit plantation')
                                        #banana
                                        #https://cdc-cameroon.net/new2014/products/banana/ for 14, 20
                                        #https://www.businessincameroon.com/agriculture/1807-9349-cameroon-to-value-the-plantain-sector-with-construction-of-an-industrial-plantain-processing-plant-in-pouma: for 75, plantain
                                        # (uncertainty but deduced from proximity other banana plantations) : 14, 16, 20  if not 2020; 15, 13 if 2020
                                    elif (i == '45') or (i == '46'):
                                        list_data.append('Other large-scale plantations') #sugarcane plantation
                                        list_data.append('Other large-scale plantations') 
                                    elif (i == '37') or (i == '67') or (i == '68') or (i == '69') or (i == '70')  or (i == '72') or (i == '73'):
                                        list_data.append('Other large-scale plantations') #tea
                                        list_data.append('Other large-scale plantations') 
                                    # if (i == '78') or (i == '79') or (i == '80'):
                                    #     list_data.append('?')
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif ('BioPalm' in d) or ('SocaPalm' in d):
                                    list_data.append('Oil palm plantation')
                                    list_data.append('Oil palm plantation')
                                elif ('Forest_Management' in d) and (i != 12): #nb: FMU match with forest production, 12 is Camvert = plantation, 50 encompasses plantation but I checked each image is outside so ok
                                    list_data.append('Selective logging')
                                    list_data.append('Selective logging')
                                elif ('Hevecam' in d) or ('SudCam' in d):
                                    list_data.append('Rubber plantation')
                                    list_data.append('Rubber plantation')
                                elif ('mining' in d) and ('permit' not in d):
                                    list_data.append('Mining')
                                    list_data.append('Mining')

                                elif 'planted_areas' in d:
                                    #4,13,14,15,25,27,28,29,30,31,32,33,34,37,38,39,40,41,43,44,46,47,48, 49, 50, 51,52,63,64: Cannot deduced by proximity
                                    if (i == '0') or (i == '1') or (i == '2') or (i == '8'):
                                        list_data.append('Rubber plantation')
                                        list_data.append('Rubber plantation')
                                        #3,6,7, 16, 19, 24, 36, 54, 55, 62, 65: uncertainty but assuming by proximity
                                        #or (i == '3') or (i == '6') or (i == '7') or (i == '16') or (i == '19') (i == '24') or (i == '36') or (i == '54') or (i == '55') or (i == '62') or (i == '65')
                                    elif (i == '9') or (i == '11') or (i == '17') or (i == '35') or (i == '56'):
                                        list_data.append('Oil palm plantation')
                                        list_data.append('Oil palm plantation')
                                        #10, 17, 18, 20, 21, 23, 60: uncertainty but assuming by proximity
                                        #or (i == '10') or (i == '17') or (i == '18') or (i == '20') or (i == '21') or (i == '23') or (i == '60')
                                    #elif (i == '57') or (i == '61') or (i== '66'):
                                        #list_data.append('Fruit plantation')
                                        #list_data.append('Fruit plantation')
                                        #57, 61, 66 : uncertainty but assuming by proximity
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')

                                elif 'production_forest' in d:
                                    list_data.append('?')
                                    list_data.append('?')
                                elif 'small_scale_agriculture' in d:
                                    list_data.append('?') 
                                    list_data.append('Smallholder agriculture')
                                elif 'Subsistence agriculture' in d:
                                    list_data.append('?')
                                    list_data.append('Smallholder agriculture')
                                    
                                elif 'new_timber_plantations' in d:
                                    list_data.append('Timber plantation')
                                    list_data.append('Timber plantation')
                                    
                                elif 'timber_sales' in d:
                                    list_data.append('Timber sales')
                                    list_data.append('?')
                                elif 'food_crops' in d:
                                    list_data.append('Fruit plantation')
                                    list_data.append('Fruit plantation')
                                elif 'WorldCover' in d:
                                    list_data.append('Grassland shrubland') 
                                    list_data.append('Grassland shrubland')
                                elif 'Wildfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Wildfire')
                                elif 'globfire' in d:
                                    list_data.append('Wildfire')
                                    list_data.append('Wildfire')
                                elif 'cassava' in d or 'cereal' in d:
                                    list_data.append('Small-scale other plantation')
                                    list_data.append('Small-scale other plantation')
                                elif 'maize' in d:
                                    if (i != '208' and i != '279'):
                                        list_data.append('Small-scale maize plantation')
                                        list_data.append('Small-scale maize plantation')
                                    else:
                                        list_data.append('?')
                                        list_data.append('?')
                                elif 'biopama' in d:
                                    list_data.append('Small-scale oil palm plantation')
                                    list_data.append('Small-scale oil palm plantation')
                                elif 'buildings' in d:
                                    list_data.append('Infrastructure')
                                    list_data.append('Infrastructure')
                                elif 'hunting' in d:
                                    list_data.append('Hunting')
                                    list_data.append('Hunting')
                                elif 'water' in d and 'sar' not in d:
                                    list_data.append('Other')
                                    list_data.append('Other')
                                else:
                                    list_data.append('?')
                                    list_data.append('?')


                            list_data.append(lat)
                            list_data.append(lon)
                            list_data.append(year)
                            list_data.append(os.path.join(folder, coord))
                            print(d)
                            print(list_data)
                            if ('?' not in list_data) and (list_data not in all_data): #do not write duplicates
                                with open(csv_file, 'a', newline='') as f_object:
                                    writer_object = writer(f_object)
                                    writer_object.writerow(list_data)
                                    f_object.close()
                            all_data.append(list_data)

    split_data(csv_file, merge_scheme)



if __name__ == "__main__":
    fire.Fire(populate_folder)