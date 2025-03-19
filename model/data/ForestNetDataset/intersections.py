import geopandas as gpd
import os
import pandas as pd

shapefiles = []
all_original = os.path.join(os.getcwd(), 'forest_loss_original')
list_areas_og = []
for s in os.listdir(all_original):
    if '.shp' in s:
        shapefile_path = os.path.join(os.getcwd(), all_original, s)
        shapefiles.append(shapefile_path)



# Load all shapefiles into GeoDataFrames
gdfs = [gpd.read_file(shapefile) for shapefile in shapefiles]

def fix_invalid_geometries(gdf):
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    return gdf

# Fix invalid geometries in all shapefiles
gdfs = [fix_invalid_geometries(gdf) for gdf in gdfs]

# Create an empty list to store intersection results (True/False)
intersection_results = []

# Loop through all pairs of shapefiles
for i in range(len(gdfs)):
    for j in range(i + 1, len(gdfs)):  # Avoid comparing the same shapefile to itself
        shapefile_1 = gdfs[i]
        shapefile_2 = gdfs[j]
        
        # Loop through all features in shapefile_1 and check for intersections with shapefile_2
        for idx, feature in shapefile_1.iterrows():
            for _, intersecting_feature in shapefile_2.iterrows():
                # Check for intersection
                intersects = feature.geometry.intersects(intersecting_feature.geometry)
                
                # Append the result to the list
                intersection_results.append({
                    'shapefile_1': f'shapefile_{i+1}',  # To distinguish between shapefiles
                    'shapefile_2': f'shapefile_{j+1}',  # To distinguish between shapefiles
                    'shapefile_1_feature_id': feature['FID'],  # Assuming 'id' is the identifier
                    'shapefile_2_feature_id': intersecting_feature['FID'],  # Assuming 'id' is the identifier
                    'intersection': intersects  # True or False
                })

# Convert the intersection results to a DataFrame
intersection_df = pd.DataFrame(intersection_results)

# Save the results to a CSV file
intersection_df.to_csv('intersection_results_multiple_shapefiles_original.csv', index=False)

print("Intersection results (True/False) saved to 'intersection_results_multiple_shapefiles_original.csv'")