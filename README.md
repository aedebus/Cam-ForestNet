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
|   |   |  get_peat_quick.py | To generate data on the present of peat | ggdrive |

