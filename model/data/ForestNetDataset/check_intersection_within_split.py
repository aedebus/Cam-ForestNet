import pandas as pd
import geopandas as gpd
import os
import fiona

splits = ['train', 'val', 'test']

for s in splits:

    # Define CSV file path
    csv_file = s + ".csv"  # Update with your actual CSV file path
    output_csv = "overlapping_shapefiles_" + s + "_filtered.csv"  # Output CSV file


    # Load CSV file (assuming it has columns: 'file_name' and 'file_path')
    df = pd.read_csv(csv_file)

    # Read all shapefiles into a single GeoDataFrame
    gdfs = []
    for index, row in df.iterrows():
        shapefile_path = os.path.join(os.getcwd(), 'forest_loss_landsat', row["example_path"].split('/')[1] + '.shp')  # Construct full path
        if os.path.exists(shapefile_path) and shapefile_path.endswith(".shp"):
            with fiona.open(shapefile_path, "r") as source:
                        gdf = gpd.GeoDataFrame.from_features(source)
            gdf["source_file"] = row["example_path"].split('/')[1] + '.shp' # Track file name for reference
            gdfs.append(gdf)
        else:
            print(f"Warning: Shapefile {shapefile_path} not found.")

    # Combine all geometries into one GeoDataFrame
    if gdfs:
        full_gdf = pd.concat(gdfs, ignore_index=True)
    else:
        print("No valid shapefiles found. Exiting.")
        exit()

    # Ensure geometries are valid polygons
    full_gdf = full_gdf[full_gdf.geometry.notna() & full_gdf.geometry.apply(lambda geom: geom.geom_type in ["Polygon", "MultiPolygon"])]

    # Check for overlaps
    overlaps = []

    for i in range(len(full_gdf)):
        for j in range(i + 1, len(full_gdf)):  # Avoid duplicate comparisons
            if full_gdf.geometry.iloc[i].intersects(full_gdf.geometry.iloc[j]):
                overlaps.append((full_gdf.source_file.iloc[i], full_gdf.source_file.iloc[j]))

    overlap_df = pd.DataFrame(overlaps, columns=["Shapefile 1", "Shapefile 2"])
    overlap_df.to_csv(output_csv, index=False)

    # Print confirmation
    print(f"Overlap results saved to {output_csv}")
