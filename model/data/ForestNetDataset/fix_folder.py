import os
import pandas as pd

MERGED_LABEL_HEADER = 'merged_label'
LABEL_HEADER = 'label'
LATITUDE_HEADER = 'latitude'
LONGITUDE_HEADER = 'longitude'
IMG_PATH_HEADER = 'example_path'
YEAR_HEADER = 'year'

CAMEROON_META_COLNAMES = [LABEL_HEADER, #AD
                           MERGED_LABEL_HEADER,
                           LATITUDE_HEADER,
                           LONGITUDE_HEADER,
                           YEAR_HEADER,
                           IMG_PATH_HEADER]

def fix_folders(sensor, suffix):
    path_folder = os.path.join(os.getcwd(), 'my_examples_' + str(sensor) + '_' + str(suffix))
    list_dir = os.listdir(path_folder)
    if 'detailed' in suffix:
        path_reference = os.path.join(os.getcwd(), 'Test with Cameroon data (final)', str(sensor).capitalize() + ' final versions', 'Without ANY duplicates and errors', 'detailed', 'all.csv')
    else:
            path_reference = os.path.join(os.getcwd(), 'Test with Cameroon data (final)', str(sensor).capitalize() + ' final versions', 'Without ANY duplicates and errors', 'groups', 'all.csv')
    data = pd.read_csv(path_reference, names=CAMEROON_META_COLNAMES)
    paths = data['example_path']
    #print(paths)
    coord =[]
    for i in range(1,len(paths)):
        coord.append((paths.iloc[i]).split('/')[-1])
    #print(coord)
    for d in list_dir:
         if d not in coord:
              os.rename(os.path.join(path_folder,d), os.path.join(path_folder, 'not_ok_'+ d))

fix_folders('landsat', 'final')
fix_folders('landsat', 'final_detailed')
fix_folders('planet', 'final')
fix_folders('planet', 'final_detailed')

