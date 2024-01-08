# Cam-ForestNet
Model to classify direct deforestation drivers in Cameroon.

Folders
------
- 'model': This folder contains the classification model and most of the code was written by [Irvin et al. (2020)](https://arxiv.org/pdf/2011.05479.pdf). Our changes are all identified with *#AD* in the code. These changes were made to adapt the model for our case study/different types of input data/different tests for improvements including data fusion and time series analysis.

- 'prepare_files': This folder contains all of the steps needed to build our reference dataset.
  
- 'environments': This folder contains the environments needed to run the model or steps in 'prepare_files'. See below the use for each task.

Guidelines for use
------
Description of the sub-folders and files
------
- In 'prepare files':

| Sub-folder    | Description sub-folder          | File | Description file | Environment to use | 
| ------------- | ------------- |------------- |------------- |------------- |
| download auxiliary quick  | To download auxiliary data  |  download_gain_quick.py | To download forest gain | ggdrive |
|   |   |  download_ir_quick.py | To download infrared bands | ggdrive |
|   |   |  download_ncep_file.py | To generate NCEP data using the downloaded NCEP files (need to [download](https://www.nco.ncep.noaa.gov/pmb/products/cfs/) those beforehand and put the result in an 'ncep' folder in the 'input' subfolder) | ggdrive |
|   |   |  download_ncep_quick.py | To download NCEP data using Google Earth Engine | ggdrive |
|   |   |  download_osm_quick.py | To download OpenStreetMap data | ggdrive |
|   |   |  download_srtm_quick.py | To download SRTM data | ggdrive |
|   |   |  get_peat_quick.py | To generate data on the present of peat using the downloaded file from [Global Forest Watch](https://data.globalforestwatch.org/datasets/aed14a0e0a8d40c69a73321275caf3e8/explore?location=10.103967%2C-99.210783%2C1.87)| ggdrive |
| download gfc  | To download the Global Forest Change (GFC) forest loss polygons  |  create_gfc.py | To create shapefiles from the GFC TIFF images | pygdal |
|   |   |  create_grassland.py | To create a shapefile from the ESA WorldCover 2020 map for grassland | pygdal |
|   |   |  create_grassland_small.py | To create a shapefile from the ESA WorldCover 2020 map for grassland but limit output to a selected number of shapes| pygdal |
|   |   |  extract_polygon_gfc_additional.py | To generate more GFC forest loss patches where we know the land use for large-scale plantations and mining | pygdal |
|   |   |  extract_polygon_gfc_additional_fruit.py | To generate more GFC forest loss patches where we know the land use for fruit plantations | pygdal |
|   |   |  extract_polygon_gfc_additional_mining.py | To generate more GFC forest loss patches where we know the land use for mining | pygdal |
|   |   |  extract_polygon_gfc_all.py | To generate GFC forest loss patches where we know the land use by selecting the shapefile and year | pygdal |
|   |   |  extract_polygon_gfc_geowiki.py | To generate GFC forest loss patches where we know the land use for Geowiki data | pygdal |
|   |   |  extract_polygon_gfc_small_scale_oil_palm.py | To generate GFC forest loss patches where we know the land use for Biopama data | pygdal |
|   |   |  extract_polygon_gfc_worldcereal.py | To generate  GFC forest loss patches where we know the land use for WorldCereal data | pygdal |
|   |   |  extract_polygon_loss_geowiki.py | To generate shapefile where we know the land use from a Geowiki csv file | pygdal |
|   |   |  extract_polygon_loss_maize.py | To generate shapefile where we know the land use from a WorldCereal csv file | pygdal |
|   |   |  extract_polygon_water.py | To generate GFC forest loss patches where we know the land use for Worldcover water data | pygdal |
|   |   |  extract_polygon_worldcover.py | To generate GFC forest loss patches where we know the land use for Worldcover data | pygdal |


Data licenses
------
The NICFI PlanetScope images fall under the same license as the NICFI data program license agreement. OpenStreetMap® is open data, licensed under the Open Data Commons Open Database License (ODbL) by the OpenStreetMap Foundation (OSMF). The documentation is licensed under the Creative Commons Attribution-ShareAlike 2.0 license (CC BY-SA 2.0). The rest of the data (including data in the input folders shared with the code) is under a Creative Commons Attribution 4.0 International License. The data has been transformed following the code in this repository.


