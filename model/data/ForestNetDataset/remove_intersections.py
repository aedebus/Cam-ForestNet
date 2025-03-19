import pandas as pd
import geopandas as gpd
import os

csv_file_1 = 'train.csv'
csv_file_2 = 'val.csv'#do all combinations

df1 = pd.read_csv(csv_file_1)
df2  = pd.read_csv(csv_file_2)

csv_intersection = pd.read_csv('intersection_results_between_csvs_with_path_new.csv')
list_to_delete = []

for s in csv_intersection['shapefile_1_path']:
    coord = (s.split('/')[1])[:-4]
    to_delete = 'my_examples_landsat_nir/' + coord #do the same with planet
    list_to_delete.append(to_delete)

df_filtered1 = df1[~df1['example_path'].isin(list_to_delete)]
df_filtered2 = df2[~df2['example_path'].isin(list_to_delete)]

filtered_csv_file1 = 'filtered_train.csv'
df_filtered1.to_csv(filtered_csv_file1 , index=False)

filtered_csv_file2 = 'filtered_val.csv'
df_filtered2.to_csv(filtered_csv_file2 , index=False)
