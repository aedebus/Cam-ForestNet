# Cam-ForestNet
Model to classify direct deforestation drivers in Cameroon.

Folders
------
- 'model': This folder contains the classification model and most of the code was written by [Irvin et al. (2020)](https://arxiv.org/pdf/2011.05479.pdf). Our changes are all identified with *#AD* in the code. These changes were made to adapt the model for our case study/different types of input data/different tests for improvements including data fusion and time series analysis.

- 'prepare_files': This folder contains all of the steps needed to build our reference dataset.
  
- 'environments': This folder contains the environments needed to run the model or steps in 'prepare_files'. See below the use for each task.

Guidelines for use
------
1. Download 'model' locally and the fnet environment
3. Download datasets and unzip in model>data>ForestNetDataset: https://zenodo.org/records/8325259
4. Choose the csv file for the wanted approach and sensor and add in model>data>ForestNetDataset
5. Run:

a) Train from scratch

`conda activate fnet`

`python3 main_mytrain_all.py train --exp_name test_exp --gpus [0] --data_version ls8_dynamic --merge_scheme four-class --resize aggressive --spatial_augmentation affine --pixel_augmentation hazy --model EfficientNet-b2 --architecture FPN --loss_fn CE --gamma 10 --alpha 10 --late_fusion True --late_fusion_stats True --late_fusion_aux_feats True --late_fusion_ncep True --late_fusion_embedding_dim 128 --late_fusion_dropout 0.1`


`ulimit -n 4096`

`python3 main_mytest_all.py test --ckpt_path models/sandbox/test_exp/ckpts/epoch=xx-val_f1_macro=xx.ckpt`

**Note 1: Fill out xx values with the best epoch by examining the sandbox**

**Note 2: Possibility to choose timeseries or fusion approach too  by changing the .py file used**

b) Use the trained model

Best performing option for Landsat-8:

[Download](https://www.dropbox.com/scl/fi/ffujhxhmmo54j9qato43x/epoch-epoch-126-val_f1_macro-val_f1_macro-0.8027.ckpt?rlkey=zkbldwnb2voo06k463wjbqydk&st=0qd62ags&dl=0) .ckpt file in a sandbox>test_exp>ckpts folder

`ulimit -n 4096`

`python3 main_mytest_all.py test --ckpt_path models/sandbox/test_exp/ckpts/epoch=126-val_f1_macro=0.8027.ckpt`

Best performing option for NICFI PlanetScope:

[Download](https://www.dropbox.com/scl/fi/d99evxfjzfgowa6vqqm97/epoch-140-val_f1_macro-0.8208.ckpt?rlkey=cipa5j6txk5lmysbztg94afdx&st=oyv0xtyf&dl=0) .ckpt file in a sandbox>test_exp>ckpts folder

`ulimit -n 4096`

`python3 main_mytest_all.py test --ckpt_path models/sandbox/test_exp/ckpts/epoch=140-val_f1_macro=0.8208.ckpt`


6. See the results in models>sandbox>test_exp>test_results


Description of the sub-folders and files
------
-In 'prepare_files': 

| Sub-folder    | Description sub-folder          | File | Description file | Environment to use | 
| ------------- | ------------- |------------- |------------- |------------- |
| download auxiliary quick  | To download auxiliary data  |  download_gain_quick.py | To download forest gain | ggdrive |
|   |   |  download_ir_quick.py | To download infrared bands | ggdrive |
|   |   |  download_ncep_file.py | To generate NCEP data using the downloaded NCEP files (need to [download](https://www.nco.ncep.noaa.gov/pmb/products/cfs/) those beforehand and put the result in an 'ncep' folder in the 'input' subfolder) | ggdrive |
|   |   |  download_ncep_quick.py | To download NCEP data using Google Earth Engine | ggdrive |
|   |   |  download_ncep_landsat_30.py | To download NCEP data for non-pansharpened Landsat-8 data using Google Earth Engine | ggdrive |
|   |   |  download_osm_quick.py | To download OpenStreetMap data | ggdrive |
|   |   |  download_osm_landsat30.py | To download OpenStreetMap data for non-pansharpened Landsat-8 data| ggdrive |
|   |   |  download_srtm_quick.py | To download SRTM data | ggdrive |
|   |   |  get_peat_quick.py | To generate data on the present of peat using the downloaded file from [Global Forest Watch](https://data.globalforestwatch.org/datasets/aed14a0e0a8d40c69a73321275caf3e8/explore?location=10.103967%2C-99.210783%2C1.87)| ggdrive |
| download gfc  | To download the Global Forest Change (GFC) forest loss polygons  |  create_gfc.py | To create shapefiles from the GFC TIFF images which need to be [downloaded](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/download.html) for coordinates 0-10N, 0-10E; 0-10N, 10-20E; 10-20N, 10-20E and added in the 'input' subfolder| pygdal |
|   |   |  create_grassland.py | To create a shapefile from the ESA WorldCover 2020 map for grassland | pygdal |
|   |   |  create_grassland_small.py | To create a shapefile from the ESA WorldCover 2020 map for grassland but limit output to a selected number of shapes| pygdal |
|   |   |  extract_polygon_gfc_additional.py | To generate more GFC forest loss patches where we know the land use for large-scale plantations and mining | pygdal |
|   |   |  extract_polygon_gfc_additional_fruit.py | To generate more GFC forest loss patches where we know the land use for fruit plantations | pygdal |
|   |   |  extract_polygon_gfc_additional_mining.py | To generate more GFC forest loss patches where we know the land use for mining | pygdal |
|   |   |  extract_polygon_gfc_all.py | To generate GFC forest loss patches where we know the land use by selecting the shapefile and year | pygdal |
|   |   |  extract_polygon_gfc_geowiki.py | To generate GFC forest loss patches where we know the land use for Geowiki data | pygdal |
|   |   |  extract_polygon_gfc_small_scale_oil_palm.py | To generate GFC forest loss patches where we know the land use for Biopama data | pygdal |
|   |   |  extract_polygon_gfc_worldcereal.py | To generate GFC forest loss patches where we know the land use for WorldCereal data | pygdal |
|   |   |  extract_polygon_loss_geowiki.py | To generate shapefile where we know the land use from a Geowiki csv file: need to [download ILUC_DARE_campaign_x_y.csv](https://pure.iiasa.ac.at/id/eprint/17539/) and add it to the 'Geowiki' subfolder in the 'input' subfolder| pygdal |
|   |   |  extract_polygon_loss_maize.py | To generate shapefile where we know the land use from a WorldCereal csv file | pygdal |
|   |   |  extract_polygon_water.py | To generate GFC forest loss patches where we know the land use for Worldcover water data | pygdal |
|   |   |  extract_polygon_worldcover.py | To generate GFC forest loss patches where we know the land use for Worldcover data ([download WorldCover](https://esa-worldcover.org/en/data-access) file and add it in 'ESA_WorldCover_10m_2020_v100_N03E009_Map' subfolder in the 'input' and 'WorldCover' subfolders to reproduce the conversion from TIFF to shapefile)| pygdal |
| download images quick  | To download the reference images from Google Earth Engine |  clean_up.py | To remove blank images (i.e. 'errors') | ggdrive |
|  |  |  download_landsat_quick.py | To download single pan-sharpened RGB Landsat-8 images (TOA) centred on the GFC forest loss patches created | ggdrive |
|  |  |  download_landsat_quick_nir.py | To download single non-pansharpened RGB + NIR Landsat-8 images (TOA) centred on the GFC forest loss patches created | ggdrive |
|  |  |  download_landsat_quick_nir_pansharpened.py | To download single pansharpened RGB + NIR Landsat-8 images (TOA) centred on the GFC forest loss patches created | ggdrive |
|  |  |  download_landsat_quick_sr_nir.py | To download single non-pansharpened RGB + NIR Landsat-8 images (SR) centred on the GFC forest loss patches created | ggdrive |
|  |  |  download_planetscope_quick_fix.py | To download single RGB NICFI PlanetScope images (monthly composites) centred on the GFC forest loss patches created (NB: 'fix' because the filtering in the previous version was not properly done)| ggdrive |
|  |  |  download_planetscope_quick_fix_missing.py | To download RGB NICFI PlanetScope images that were not properly downloaded| ggdrive |
|  |  |  download_planetscope_quick_fix_missing2.py | To download RGB NICFI PlanetScope images that were not properly downloaded| ggdrive |
|  |  |  download_planetscope_quick_nir.py | To download single RGB + NIR NICFI PlanetScope images (monthly composites) centred on the GFC forest loss patches created (NB: 'fix' because the filtering in the previous version was not properly done)| ggdrive |
|  |  |  download_planetscope_quick_nir_biannual.py | To download single RGB + NIR NICFI PlanetScope images (biannual composites) centred on the GFC forest loss patches created (NB: 'fix' because the filtering in the previous version was not properly done)| ggdrive |


-In 'model': 

| Sub-folder    | Description sub-folder          | File | Description file | Environment to use | 
| ------------- | ------------- |------------- |------------- |------------- |
| data > ForestNetDataset | To format and store the Cameroon dataset |  fix_folder.py | To remove images with high uncertainty | fnet |
|  |  |  fix_names_all.py | To fix the names of NCEP data to match the ForestNet formatting for names | fnet |
| / | / |  main_my_test_all.py | To test Cam-ForestNet with a single image approach | fnet |
|  |  |  main_my_test_fusion.py | To test Cam-ForestNet with a decision-based fusion approach | fnet |
|  |  |  main_my_train_all.py | To train Cam-ForestNet | fnet |
|  |  |  populate_folder_all_detailed.py | To create a formatted data folder (RGB only) to train and test Cam-ForestNet using the steps in 'prepare_files'; split the data into training, validation and testing datasets; and generate a reference csv file with labels | fnet |
|  |  |  populate_folder_all_detailed_landsat_nir.py | To create a formatted data folder for NIR + RGB Landsat-8 (non-pansharpened, TOA) data to train and test Cam-ForestNet using the steps in 'prepare_files'; split the data into training, validation and testing datasets; and generate a reference csv file with labels | fnet |
|  |  |  populate_folder_all_detailed_landsat_nir_ps.py | To create a formatted data folder for NIR + RGB Landsat-8 (pansharpened, TOA) data to train and test Cam-ForestNet using the steps in 'prepare_files'; split the data into training, validation and testing datasets; and generate a reference csv file with labels | fnet |
|  |  |  populate_folder_all_detailed_planet_nir.py | To create a formatted data folder for NIR + RGB NICFI PlanetScope (monthly composites) data to train and test Cam-ForestNet using the steps in 'prepare_files'; split the data into training, validation and testing datasets; and generate a reference csv file with labels | fnet |
|  |  |  populate_folder_all_detailed_planet_nir_biannual.py | To create a formatted data folder for NIR + RGB NICFI PlanetScope (biannual composites) data to train and test Cam-ForestNet using the steps in 'prepare_files'; split the data into training, validation and testing datasets; and generate a reference csv file with labels | fnet |
|  |  |  populate_folder_all_match_test_datasets_planet_nir.py | To select the available Landsat-8 and NICFI PlanetScope data with the same centroid coordinates; split the data into training, validation and testing datasets; and generate a reference csv file with labels to test data fusion | fnet |
|  |  |  populate_folder_all_detailed_landsat_sr_nir.py | To create a formatted data folder for NIR + RGB Landsat-8 (non-pansharpened, SR) data to train and test Cam-ForestNet using the steps in 'prepare_files'; split the data into training, validation and testing datasets; and generate a reference csv file with labels | fnet |

Data licenses
------
The NICFI PlanetScope images fall under the same license as the [NICFI data program license agreement](https://assets.planet.com/docs/Planet_ParticipantLicenseAgreement_NICFI.pdf). [OpenStreetMap®](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines) is open data, licensed under the Open Data Commons Open Database License (ODbL) by the OpenStreetMap Foundation (OSMF). The documentation is licensed under the [Creative Commons Attribution-ShareAlike 2.0 license (CC BY-SA 2.0)](https://creativecommons.org/licenses/by-sa/2.0/). The rest of the data (including data in the input folders shared with the code) is under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). The data has been transformed following the code in this repository.


