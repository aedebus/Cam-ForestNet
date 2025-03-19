import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union, nearest_points
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split

### Step 1: Load CSV & Extract Shapefile Edges ###
def load_shapefile_edges(csv_file, path_column='example_path', label_column='merged_label'):
    data = pd.read_csv(csv_file)

    # Normalize column names (lowercase, strip spaces)
    data.columns = data.columns.str.lower().str.strip()
    
    # Keep track of all original columns
    original_columns = data.columns.tolist()

    extracted_edges = []

    for _, row in data.iterrows():
        path = row[path_column]
        pkl_path = os.path.join(path, 'forest_loss_region.pkl')

        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    shapefile_data = pickle.load(f)

                if isinstance(shapefile_data, (Polygon, MultiPolygon)):
                    geometries = [shapefile_data]  
                elif isinstance(shapefile_data, list):
                    geometries = shapefile_data  
                elif isinstance(shapefile_data, dict) and 'features' in shapefile_data:
                    geometries = [shape(feature['geometry']) for feature in shapefile_data['features']]
                else:
                    print(f"Unknown format in {pkl_path}, skipping...")
                    continue

                # Extract edges and keep all original column values
                for geom in geometries:
                    if geom.boundary:
                        row_dict = row.to_dict()
                        row_dict['edge'] = geom.boundary
                        extracted_edges.append(row_dict)

            except Exception as e:
                print(f"Error loading {pkl_path}: {e}")
        else:
            print(f"Warning: File not found {pkl_path}")

    df = pd.DataFrame(extracted_edges)

    for col in original_columns:
        if col not in df.columns:
            df[col] = np.nan  

    return df[original_columns + ['edge']] 

### Step 2: Compute Real-World Edge-to-Edge Distance ###
def real_world_edge_distance(edge1, edge2):
    """Find the closest real-world distance (in km) between two edges."""
    p1, p2 = nearest_points(edge1, edge2)
    coord1 = (p1.y, p1.x)
    coord2 = (p2.y, p2.x)
    return geodesic(coord1, coord2).km  

### Step 3: Dynamic Distance Relaxation ###
def assign_with_dynamic_distance(data, existing_splits, min_distance=1.0, max_steps=10, min_threshold=0.1):
    """
    Assigns points ensuring they are at least `min_distance` away from all existing splits.
    Tracks the threshold for each assigned point.
    """

    assigned = []
    unassigned = data.copy()
    step_size = (min_distance - min_threshold) / max_steps
    distance_counts = {}  # Store assigned points per threshold

    for step in range(max_steps + 1):
        current_distance = min_distance - (step * step_size)
        if current_distance < min_threshold:
            break  # Stop if we reach the lowest threshold

        remaining_unassigned = []
        assigned_at_this_threshold = 0

        for _, point in unassigned.iterrows():
            # Ensure distance from ALL existing splits (Train, Test, Validation)
            too_close = any(
                real_world_edge_distance(point['edge'], other['edge']) < current_distance
                for dataset in existing_splits for other in dataset
            )

            if not too_close:
                point['assigned_distance'] = current_distance  # Store threshold
                assigned.append(point)
                assigned_at_this_threshold += 1
            else:
                remaining_unassigned.append(point)

        unassigned = pd.DataFrame(remaining_unassigned)
        distance_counts[current_distance] = assigned_at_this_threshold

        if unassigned.empty:
            break  # Stop early if everything is assigned

    return pd.DataFrame(assigned), unassigned, distance_counts



### Step 4: Split Data & Handle Unassigned Points ###
def stratified_split_with_distance(data, label_col='merged_label', min_distance=1.0, train_ratio=0.6, test_ratio=0.25):
    train_list, test_list, val_list = [], [], []
    unassigned_points = []
    distance_counts_all = {}

    for label, group in data.groupby(label_col):
        train_size = int(len(group) * train_ratio)
        test_size = int(len(group) * test_ratio)
        val_size = len(group) - train_size - test_size  

        # Stratified split
        train_group, temp_group = train_test_split(group, train_size=train_size, stratify=group[label_col])
        test_group, val_group = train_test_split(temp_group, train_size=test_size, stratify=temp_group[label_col])

        # Assign with distance enforcement across ALL datasets
        train_assigned, train_unassigned, train_dist_counts = assign_with_dynamic_distance(
            train_group, existing_splits=[], min_distance=min_distance
        )
        
        test_assigned, test_unassigned, test_dist_counts = assign_with_dynamic_distance(
            test_group, existing_splits=[train_assigned.to_dict('records')], min_distance=min_distance
        )

        val_assigned, val_unassigned, val_dist_counts = assign_with_dynamic_distance(
            val_group, existing_splits=[train_assigned.to_dict('records'), test_assigned.to_dict('records')], min_distance=min_distance
        )
        print(len(train_assigned))
        print(len(val_assigned))
        print(len(test_assigned))
        # Save threshold info
        train_assigned['split'] = 'train'
        test_assigned['split'] = 'test'
        val_assigned['split'] = 'validation'

        # Store unassigned points
        unassigned_points += train_unassigned.to_dict('records')
        unassigned_points += test_unassigned.to_dict('records')
        unassigned_points += val_unassigned.to_dict('records')

        train_list.append(train_assigned)
        test_list.append(test_assigned)
        val_list.append(val_assigned)

        # Merge all `distance_counts` from Train, Test, and Validation
        for dist, count in {**train_dist_counts, **test_dist_counts, **val_dist_counts}.items():
            distance_counts_all[dist] = distance_counts_all.get(dist, 0) + count

    # Combine all assigned data
    train_data = pd.concat(train_list).reset_index(drop=True)
    test_data = pd.concat(test_list).reset_index(drop=True)
    val_data = pd.concat(val_list).reset_index(drop=True)

    return train_data, test_data, val_data, distance_counts_all




### Step 4: Remove Intersecting Geometries and Reassign ###

def remove_intersections_between_splits(train_data, test_data, val_data):
    """Strictly remove intersecting geometries between Train, Test, and Validation datasets."""

    def detect_and_remove_intersections(source_data, reference_geom):
        """Detect intersections with reference geometries and remove overlapping ones."""
        intersections = source_data['edge'].apply(lambda x: reference_geom.intersects(x))
        removed_data = source_data[intersections]
        cleaned_data = source_data[~intersections]
        return cleaned_data, removed_data

    # Convert dataset geometries into unified geometry collections
    train_geom = unary_union(train_data['edge'].tolist())
    test_geom = unary_union(test_data['edge'].tolist())
    val_geom = unary_union(val_data['edge'].tolist())

    # Step 1: Remove Test-Train Overlaps
    test_data, removed_test = detect_and_remove_intersections(test_data, train_geom)

    # Step 2: Remove Validation-Train and Validation-Test Overlaps
    val_data, removed_val = detect_and_remove_intersections(val_data, unary_union([train_geom, test_geom]))

    print(f"Removed {len(removed_test)} intersecting geometries from Test dataset.")
    print(f"Removed {len(removed_val)} intersecting geometries from Validation dataset.")

    # Step 3: Reassign Removed Points to the Closest Split
    reassigned_train = []
    reassigned_test = []
    reassigned_val = []

    for _, row in removed_test.iterrows():
        if real_world_edge_distance(row['edge'], train_geom) < real_world_edge_distance(row['edge'], val_geom):
            reassigned_train.append(row)
        else:
            reassigned_test.append(row)

    for _, row in removed_val.iterrows():
        if real_world_edge_distance(row['edge'], train_geom) < real_world_edge_distance(row['edge'], test_geom):
            reassigned_train.append(row)
        else:
            reassigned_val.append(row)

    # Append reassignments
    train_data = pd.concat([train_data, pd.DataFrame(reassigned_train)], ignore_index=True)
    test_data = pd.concat([test_data, pd.DataFrame(reassigned_test)], ignore_index=True)
    val_data = pd.concat([val_data, pd.DataFrame(reassigned_val)], ignore_index=True)

    print(f"Reassigned {len(reassigned_train)} points to Train, {len(reassigned_test)} to Test, and {len(reassigned_val)} to Validation.")

    return train_data, test_data, val_data


def check_dataset_intersections(train_data, test_data, val_data):
    """Check if train, test, and validation datasets have overlapping geometries."""

    # Convert all edges into unified geometries
    train_geom = unary_union(train_data['edge'].tolist())
    test_geom = unary_union(test_data['edge'].tolist())
    val_geom = unary_union(val_data['edge'].tolist())

    # Check for spatial overlaps between all sets
    train_test_overlap = train_geom.intersects(test_geom)
    train_val_overlap = train_geom.intersects(val_geom)
    test_val_overlap = test_geom.intersects(val_geom)

    # Print intersection results
    print("\n**Intersection Check Between All Splits**:")
    print(f"Train-Test Overlap: {'Yes' if train_test_overlap else 'No'}")
    print(f"Train-Validation Overlap: {' Yes' if train_val_overlap else 'No'}")
    print(f"Test-Validation Overlap: {'Yes' if test_val_overlap else 'No'}")

    return train_test_overlap, train_val_overlap, test_val_overlap

### Step 5: Save Data ###
def save_dataframes(train_data, test_data, val_data, original_csv_path):
    original_data = pd.read_csv(original_csv_path)
    original_columns = original_data.columns.tolist()

    for col in original_columns:
        for df in [train_data, test_data, val_data]:
            if col not in df.columns:
                df[col] = np.nan  

    train_data[original_columns].to_csv("train.csv", index=False)
    test_data[original_columns].to_csv("test.csv", index=False)
    val_data[original_columns].to_csv("val.csv", index=False)

    print("Train, Test, and Validation sets saved.")

### Step 6: Run Processing ###
csv_path = "all.csv"
df_edges = load_shapefile_edges(csv_path, path_column='example_path', label_column='merged_label')

train_data, test_data, val_data, distance_counts_all = stratified_split_with_distance(
    df_edges, label_col='merged_label', min_distance=1.0, train_ratio=0.6, test_ratio=0.25
)

train_data, test_data, val_data = remove_intersections_between_splits(train_data, test_data, val_data)

save_dataframes(train_data, test_data, val_data, csv_path)
check_dataset_intersections(train_data, test_data, val_data)

### Step 7: Print & Plot Assigned Distance Thresholds ###
print("\nProportion of Data Assigned at Each Distance Threshold:")
total_assigned = sum(distance_counts_all.values())
threshold_proportions = {}

for dist, count in sorted(distance_counts_all.items(), reverse=True):
    proportion = (count / total_assigned) * 100
    threshold_proportions[dist] = proportion
    print(f"Threshold {dist:.2f} km: {count} samples ({proportion:.2f}%)")

# ### Step 8: Plot Distribution of Assigned Distances ###
# plt.figure(figsize=(8, 5))
# plt.bar([f"{d:.2f} km" for d in threshold_proportions.keys()], 
#         list(threshold_proportions.values()), color='blue')
# plt.xlabel("Distance Threshold (km)")
# plt.ylabel("Percentage of Assigned Data")
# plt.title("Proportion of Data Assigned at Each Distance Threshold")
# plt.xticks(rotation=45)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.show()

### Step 9: Print Final Dataset Sizes ###
print("\nFinal Dataset Sizes:")
print(f"Train: {len(train_data)}")
print(f"Test: {len(test_data)}")
print(f"Validation: {len(val_data)}")


# Compute total points
total_points = len(train_data) + len(test_data) + len(val_data)
train_points = len(train_data)
test_points = len(test_data)
val_points = len(val_data)

# Print actual proportions
print("\nFinal Dataset Proportions:")
print(f"Train: {train_points} ({train_points/total_points:.2%})")
print(f"Test: {test_points} ({test_points/total_points:.2%})")
print(f"Validation: {val_points} ({val_points/total_points:.2%})")




