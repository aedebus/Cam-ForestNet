import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split

# File paths
train_val_file = "train_val.csv"  # CSV containing shapefile paths
shapefile_folder = "forest_loss_landsat"  # Folder containing the shapefiles

# Load train_val CSV
df_train_val = pd.read_csv(train_val_file)

# Ensure train_val contains a column with shapefile paths (e.g., 'shapefile_path')
if "example_path" not in df_train_val.columns:
    raise ValueError("train_val.csv must contain a column named 'example_path' with shapefile locations.")

#Load the actual geometries from the shapefiles
gdf_list = []
for idx, row in df_train_val.iterrows():
    shapefile_path = os.path.join(shapefile_folder, row['example_path'].split('/', 1)[1] + '.shp')  # Column with file names
    if os.path.exists(shapefile_path):  
        gdf = gpd.read_file(shapefile_path)  # Read shapefile
        gdf["dataset_index"] = idx  # Store index for merging later
        gdf_list.append(gdf)

# Merge all geometries into a single GeoDataFrame
gdf_train_val = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

# Convert CRS to meters for accurate distance calculations
gdf_train_val = gdf_train_val.to_crs(epsg=3857)

# Define target column (assuming it exists in train_val.csv)
df_train_val["geometry"] = gdf_train_val["geometry"]
target_column = "label"

# Print initial dataset sizes
print(f"📊 Initial dataset sizes:")
print(f"   📂 Train+Validation: {len(df_train_val)} entries")


# Split Train+Validation into 80% Train / 20% Validation (Initial Split)
df_train, df_val = train_test_split(
    df_train_val, test_size=0.2, stratify=df_train_val[target_column], random_state=42
)

# Define minimum buffer distance
buffer_distance = 100  

# Check if the 'geometry' column contains Polygon objects
if isinstance(df_train["geometry"].iloc[0], Polygon):
    print("✅ Geometry column already contains Polygon objects.")
else:
    print("🔄 Converting WKT strings to Polygons...")
    df_train["geometry"] = gpd.GeoSeries.from_wkt(df_train["geometry"])

if not isinstance(df_train, gpd.GeoDataFrame):
    df_train = gpd.GeoDataFrame(df_train, geometry=df_train["geometry"])
    
# Ensure CRS is set for distance calculations
df_train = df_train.to_crs(epsg=3857)  

# Apply spatial filtering to ensure Validation is at least 100m away from Train
spatial_index = df_train.sindex  # Create spatial index for fast lookup

# Keep track of swapped and moved polygons
swapped_polygons = 0
moved_to_train = 0

# List of validation indices that satisfy the buffer condition
valid_val_indices = []

# Go through each validation polygon
for idx, val_polygon in df_val.iterrows():
    # Check if the validation polygon is too close to any training polygon
    possible_matches_index = list(spatial_index.intersection(val_polygon.geometry.buffer(buffer_distance).bounds))
    possible_matches = df_train.iloc[possible_matches_index]

    if possible_matches.geometry.intersects(val_polygon.geometry.buffer(buffer_distance)).any():
        # Polygon is too close! Look for a replacement in training
        class_label = val_polygon[target_column]
        train_candidates = df_train[df_train[target_column] == class_label]  # Get same-class candidates

        # Try to find a replacement polygon that is NOT within 100m of any current train polygons
        swapped = False
        for train_idx, train_polygon in train_candidates.iterrows():
            if not df_train.geometry.intersects(train_polygon.geometry.buffer(buffer_distance)).any():
                # Swap validation polygon with this training polygon
                df_val.at[idx, "geometry"] = train_polygon.geometry
                df_train.at[train_idx, "geometry"] = val_polygon.geometry
                swapped_polygons += 1
                swapped = True
                valid_val_indices.append(idx)
                break  # Stop searching once a replacement is found

        if not swapped:
            # No swap found, move polygon to training
            df_train = pd.concat([df_train, df_val.loc[[idx]]], ignore_index=True)
            moved_to_train += 1
    else:
        # If the polygon is already valid, keep it
        valid_val_indices.append(idx)

# Filter validation set to only include valid polygons
df_val = df_val.loc[valid_val_indices]

# Print final dataset sizes
print(f"\n📊 Final dataset sizes after spatial filtering:")
print(f"   📂 Train: {len(df_train)} entries")
print(f"   📂 Validation: {len(df_val)} entries")

print(f"\n✅ Final Validation set contains {len(df_val)} samples.")
print(f"🔄 {swapped_polygons} polygons were swapped between train and validation.")
print(f"➡️ {moved_to_train} validation polygons were moved to train because no swap was found.")


# Save the updated Train and Validation datasets
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)

print("✅ Train, Validation, and Test datasets saved successfully!")