import geopandas as gpd
import pandas as pd
import os
import numpy as np
from shapely.ops import unary_union
from sklearn.model_selection import StratifiedKFold
import fiona

def load_shapefile_data(csv_file, stratify_column="label"):
    """ Reads a CSV file and returns the dataframe with original columns and shapefile paths. """
    df = pd.read_csv(csv_file)
    df["shapefile_name"] = df['example_path'].apply(lambda x: x.split('/', 1)[1] + '.shp' if '/' in x else x)
    
    if stratify_column not in df.columns:
        df[stratify_column] = "default"  # Add a default category if none exists
    
    return df

def determine_utm_epsg(geo_dataframe):
    """ Determines the UTM EPSG code based on dataset's centroid (assuming WGS 84 input). """
    centroid = geo_dataframe.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    return epsg_code

def load_shapefiles(df, shapefile_folder, utm_crs):
    """ Loads shapefiles, assigns missing CRS, and reprojects to UTM. Returns dataframe with geometry. """
    boundaries = []
    valid_rows = []

    for _, row in df.iterrows():
        shp_path = row["shapefile_name"]
        full_path = os.path.join(shapefile_folder, shp_path)

        if os.path.exists(full_path):
            try:
                with fiona.open(full_path, "r") as source:
                    gdf = gpd.GeoDataFrame.from_features(source)

                if gdf.crs is None:
                    print(f"Missing CRS in {shp_path}! Assigning EPSG:4326...")
                    gdf.set_crs(epsg=4326, inplace=True)

                if gdf.crs.to_epsg() != 4326:
                    raise ValueError(f"Expected WGS 84 (EPSG:4326) but found {gdf.crs}")

                # Reproject to UTM
                gdf_utm = gdf.to_crs(epsg=utm_crs)
                boundary = unary_union(gdf_utm.geometry.boundary)
                
                boundaries.append(boundary)
                valid_rows.append(row)
            except Exception as e:
                print(f"Error loading {shp_path}: {e}")
        else:
            print(f"File not found: {shp_path}")

    return pd.DataFrame(valid_rows), boundaries

def adjust_folds_for_min_distance(df, boundaries, stratify_column, max_distance_m=1000, min_distance_m=100):
    """
    Adjusts folds dynamically to ensure each test polygon has at least one training polygon 
    at the required minimum distance. Starts with 1 km and relaxes down to 100m.
    Uses StratifiedKFold to maintain category balance in folds.
    """
    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Assign initial folds based on stratified sampling
    df["Fold"] = -1  # Initialize fold column
    for fold, (_, test_idx) in enumerate(skf.split(df, df[stratify_column])):
        df.loc[test_idx, "Fold"] = fold

    # Adjust folds to ensure minimum spatial distance
    for i in range(len(df)):
        test_poly = boundaries[i]
        threshold_meters = max_distance_m
        valid_fold_found = False

        while threshold_meters >= min_distance_m and not valid_fold_found:
            for fold in range(num_folds):
                if fold == df.iloc[i]["Fold"]:
                    continue  # Skip same fold (test set)

                # Check if there's a valid training match at the required distance
                valid_match_found = any(
                    test_poly.distance(boundaries[j]) >= threshold_meters
                    for j in range(len(df)) if df.iloc[j]["Fold"] == fold
                )

                if valid_match_found:
                    valid_fold_found = True
                    break

            # If no valid fold found, relax the distance by 100m
            if not valid_fold_found:
                threshold_meters -= 100
                print(f"Relaxing distance to {threshold_meters / 1000:.1f} km for {df.iloc[i]['shapefile_name']}")

        # If still no valid fold, assign a random fold as fallback
        if not valid_fold_found:
            alternative_folds = [f for f in range(num_folds) if f != df.iloc[i]["Fold"]]
            df.at[i, "Fold"] = np.random.choice(alternative_folds)

    return df

def compute_min_distances(boundaries_train, boundaries_test, min_distance_m=100):
    """ Computes minimum edge distance between test and training shapefiles. """
    results = []

    for test_boundary in boundaries_test:
        min_distance = float("inf")  # Start with a high distance

        for train_boundary in boundaries_train:
            distance = test_boundary.distance(train_boundary)
            if distance >= min_distance_m:
                min_distance = min(min_distance, distance)

        results.append(None if min_distance == float("inf") else min_distance)  # Save distance

    return results

def five_fold_cross_validation(csv1, csv2, csv3, shapefile_folder, output_csv, stratify_column="label", max_distance_km=1.0, min_distance_km=0.1):
    """ Performs 5-fold cross-validation while ensuring stratified sampling and dynamically relaxing distance constraints. """
    
    # Load CSVs with metadata
    df1 = load_shapefile_data(csv1, stratify_column)
    df2 = load_shapefile_data(csv2, stratify_column)
    df3 = load_shapefile_data(csv3, stratify_column)
    df_all = pd.concat([df1, df2, df3], ignore_index=True)

    # Load a sample shapefile to determine UTM projection
    with fiona.open(os.path.join(shapefile_folder, df_all["shapefile_name"].iloc[0]), "r") as source:
        gdf_sample = gpd.GeoDataFrame.from_features(source)
    if gdf_sample.crs is None:
        gdf_sample.set_crs(epsg=4326, inplace=True)
    utm_crs = determine_utm_epsg(gdf_sample)

    # Load shapefiles with geometry
    df_all, boundaries_all = load_shapefiles(df_all, shapefile_folder, utm_crs)

    # Adjust fold assignments dynamically
    df_all = adjust_folds_for_min_distance(df_all, boundaries_all, stratify_column, max_distance_km * 1000, min_distance_km * 1000)

    fold_results = []

    for fold in range(5):
        print(f"\nProcessing Fold {fold+1}...\n")

        df_test = df_all[df_all["Fold"] == fold].copy()
        df_train = df_all[df_all["Fold"] != fold].copy()

        boundaries_test = [boundaries_all[i] for i in df_test.index]
        boundaries_train = [boundaries_all[i] for i in df_train.index]

        # Compute minimum distances
        df_test["Min_Distance_m"] = compute_min_distances(boundaries_train, boundaries_test, min_distance_m=min_distance_km * 1000)

        # Append results
        fold_results.append(df_test)

    # Save results
    final_results = pd.concat(fold_results, ignore_index=True)
    final_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


# Example usage:
csv_file_1 = "train.csv"
csv_file_2 = "val.csv"
csv_file_3 = "test.csv"
shapefile_directory = "forest_loss_landsat"
output_results = "shapefile_5fold_dynamic_1km_100m.csv"

five_fold_cross_validation(csv_file_1, csv_file_2, csv_file_3, shapefile_directory, output_results)
