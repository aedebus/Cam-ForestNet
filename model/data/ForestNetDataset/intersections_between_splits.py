import pandas as pd
import geopandas as gpd
import os

# Load shapefile paths from the three CSV files
csv_file_1 = 'train.csv'
csv_file_2 = 'val.csv'
csv_file_3 = 'test.csv'

# Assuming each CSV contains a column 'shapefile_path' with paths to the shapefiles
shapefiles_csv1_temp = pd.read_csv(csv_file_1)['example_path'].tolist()
shapefiles_csv2_temp = pd.read_csv(csv_file_2)['example_path'].tolist()
shapefiles_csv3_temp = pd.read_csv(csv_file_3)['example_path'].tolist()
shapefiles_csv1 =[]
shapefiles_csv2 = []
shapefiles_csv3 = []

for s in shapefiles_csv1_temp:
    coord = s.split('/')[1]
    shapefiles_csv1.append('forest_loss_landsat/'+ coord + '.shp')

for s in shapefiles_csv2_temp:
    coord = s.split('/')[1]
    shapefiles_csv2.append('forest_loss_landsat/'+ coord + '.shp')

for s in shapefiles_csv3_temp:
    coord = s.split('/')[1]
    shapefiles_csv3.append('forest_loss_landsat/'+ coord + '.shp')

shapefiles_csv1_label = pd.read_csv(csv_file_1)['label'].tolist()
shapefiles_csv2_label = pd.read_csv(csv_file_2)['label'].tolist()
shapefiles_csv3_label = pd.read_csv(csv_file_3)['label'].tolist()

# Print shapefile lists from all CSV files
#print("Shapefiles from CSV 1:", shapefiles_csv1)
#print("Shapefiles from CSV 2:", shapefiles_csv2)
#print("Shapefiles from CSV 3:", shapefiles_csv3)

# Function to load and fix invalid geometries in shapefiles
def load_and_fix_shapefiles(shapefile_paths):
    gdfs = [gpd.read_file(shapefile) for shapefile in shapefile_paths]
    # Fix invalid geometries using buffer(0)
    for i in range(len(gdfs)):
        gdfs[i]['geometry'] = gdfs[i]['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    return gdfs

# Load and fix invalid geometries for all shapefiles from all CSVs
gdfs_csv1 = load_and_fix_shapefiles(shapefiles_csv1)
gdfs_csv2 = load_and_fix_shapefiles(shapefiles_csv2)
gdfs_csv3 = load_and_fix_shapefiles(shapefiles_csv3)

# List to store intersection results (True/False)
intersection_results = []

# Function to check intersections between shapefiles from two lists (CSV)
def check_intersections(gdfs_1, gdfs_2, shapefiles_1, shapefiles_2, labels_1, labels_2, csv_1_name, csv_2_name):
    for i, gdf_1 in enumerate(gdfs_1):
        for j, gdf_2 in enumerate(gdfs_2):
            for idx_1, feature_1 in gdf_1.iterrows():
                for idx_2, feature_2 in gdf_2.iterrows():
                    if feature_1.geometry.intersects(feature_2.geometry):  # Save only when True
                        intersection_results.append({
                            'shapefile_1_path': shapefiles_1[i],
                            'shapefile_2_path': shapefiles_2[j],
                            'shapefile_1_label': labels_1[i],
                            'shapefile_2_label': labels_2[j],
                            'shapefile_1': f'{csv_1_name}_shapefile_{i+1}',
                            'shapefile_2': f'{csv_2_name}_shapefile_{j+1}',
                        })


# Compare shapefiles between CSV 1 and CSV 2
print("Comparing shapefiles from CSV1 and CSV2...")
check_intersections(gdfs_csv1, gdfs_csv2, shapefiles_csv1, shapefiles_csv2, shapefiles_csv1_label, shapefiles_csv2_label, 'CSV1', 'CSV2')

# Compare shapefiles between CSV 1 and CSV 3
print("Comparing shapefiles from CSV1 and CSV3...")
check_intersections(gdfs_csv1, gdfs_csv3, shapefiles_csv1, shapefiles_csv3, shapefiles_csv1_label, shapefiles_csv3_label, 'CSV1', 'CSV3')

# Compare shapefiles between CSV 2 and CSV 3
print("Comparing shapefiles from CSV2 and CSV3...")
check_intersections(gdfs_csv2, gdfs_csv3, shapefiles_csv2, shapefiles_csv3, shapefiles_csv2_label, shapefiles_csv3_label, 'CSV2', 'CSV3')

# Convert results to a DataFrame
intersection_df = pd.DataFrame(intersection_results)

# Save the results to a CSV file
intersection_df.to_csv('intersection_results_between_csvs_with_path_new.csv', index=False)

print("Intersection results saved to 'intersection_results_between_csvs_with_path_new.csv'")
