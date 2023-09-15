"""Define constants to be used throughout the repository."""
from pathlib import Path


# Main paths
SHARED_DEEP_DIR = Path('') #MODIFIED
DATA_BASE_DIR = SHARED_DEEP_DIR / 'data'
MODEL_BASE_DIR = SHARED_DEEP_DIR / 'models'
SANDBOX_DIR = MODEL_BASE_DIR / 'sandbox'
TB_DIR = SANDBOX_DIR / 'tb'

# Dataset constants
HANSEN_DATASET_NAME = 'hansen'
INDONESIA_DATASET_NAME = 'indonesia'
LANDCOVER_DATASET_NAME = 'landcover'
CAMEROON_DATASET_NAME = 'cameroon' #AD
DATASET_NAMES = [
    HANSEN_DATASET_NAME, INDONESIA_DATASET_NAME, LANDCOVER_DATASET_NAME, CAMEROON_DATASET_NAME #AD
]

TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'
MY_TRAIN_SPLIT = 'mytrain' #AD
MY_VAL_SPLIT = 'myval' #AD
MY_TEST_SPLIT = 'mytest' #AD
DATA_SPLITS = [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, MY_TRAIN_SPLIT, MY_VAL_SPLIT, MY_TEST_SPLIT] #AD

# Hansen constants
# HANSEN_DIR = DATA_BASE_DIR / 'hansen_drivers'
# HANSEN_TRAIN_PATH = HANSEN_DIR / 'train_latlon_v1.csv'

# HANSEN_TRAIN_V2_PATH = HANSEN_DIR / 'train_latlon_v2.csv'
# HANSEN_VAL_V2_PATH = HANSEN_DIR / 'val_latlon_v2.csv'

# HANSEN_TRAIN_V3_PATH = HANSEN_DIR / 'train_latlon_v3.csv'
# HANSEN_VAL_V3_PATH = HANSEN_DIR / 'val_latlon_v3.csv'

# Label constants
# HANSEN_LABELS = ['Commodity Driven Deforestation', 'Shifting Agriculture',
#                   'Forestry', 'Wildfire', 'Urbanization',
#                   'Other Natural Disturbance', 'Uncertain']
HANSEN_LABELS_V3 = ['Commodity Driven Deforestation', 'Shifting Agriculture',
                    'Forestry', 'Wildfire', 'Urbanization']
# HANSEN_IGNORED_LABELS = ['Other Natural Disturbance', 'Uncertain']
# HANSEN_IGNORED_LABEL_IDXS = [HANSEN_LABELS.index(label)
#                               for label in HANSEN_IGNORED_LABELS]

# HANSEN_NUM_CLASSES = len(HANSEN_LABELS_V3)

# NOTE: based on successfully downloaded images from
# train_latlon_v3.csv. Should be adjusted when missing images found.
HANSEN_V3_TRAIN_FREQS = [848, 1166, 1549, 749, 222]
HANSEN_V3_NUM_EXS = sum(HANSEN_V3_TRAIN_FREQS)
HANSEN_V3_CLASS_WEIGHTS = [HANSEN_V3_NUM_EXS / freq
                            for freq in HANSEN_V3_TRAIN_FREQS]
# TODO: add ignored class idxs

# Indonesia dataset constants
INDONESIA_COORD_PROJ = '+proj=cea +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs'
INDONESIA_DIR = DATA_BASE_DIR / 'ForestNetDataset' #MODIFIED
INDONESIA_TRAIN_SPLIT, INDONESIA_VAL_SPLIT, INDONESIA_TEST_SPLIT = .6, .15, .25
# INDONESIA_SHAPEFILE_PATH = DATA_BASE_DIR / 'ind_images_new_download/shapefiles'
# INDONESIA_SHAPEFILE_PATH_SENT2 = DATA_BASE_DIR / 'ind_images_SENT2/shapefiles'
# DRIVER_SHAPEFILE_PATH = INDONESIA_DIR / "shapefiles" / "all_samp_drivers.shp"
POLYGON_SHAPE = 'Polygon'
MULTIPOLYGON_SHAPE = 'MultiPolygon'
# Computed from all polygon areas in the training set
# TODO: Change this before deploying the model if we
# input forest loss area in the final model.
MAX_POLYGON_AREA_IN_PIXELS = 86385.37110881005

INDONESIA_TRAIN_SPLIT, INDONESIA_VAL_SPLIT, INDONESIA_TEST_SPLIT = .6, .15, .25

INDONESIA_ALL_LABELS = [
    'Oil palm plantation',
    'Timber plantation',
    'Other large-scale plantations',
    'Grassland shrubland',
    'Small-scale agriculture',
    'Small-scale mixed plantation',
    'Small-scale oil palm plantation',
    'Mining',
    'Fish pond',
    'Logging',
    'Secondary forest',
    'Other',
    '?', #AD
    'Rubber plantation', #AD
    'Timber sales', #AD
    'Fruit plantation', #AD
    'Wildfire', #AD
    'Small-scale cassava plantation', #AD
    'Small-scale maize plantation', #AD
    'Small-scale other plantation', #AD
    'Infrastructure', #AD
    'Hunting', #AD
    'Selective logging'] #AD

INDONESIA_LABELS = ['Oil palm plantation', 'Timber plantation',
                    'Other large-scale plantations', 'Grassland shrubland',
                    'Small-scale agriculture', 'Small-scale mixed plantation',
                    'Small-scale oil palm plantation', 'Mining', 'Fish pond',
                    'Logging', 'Secondary forest']
INDONESIA_IGNORED_LABELS = ['Other']
INDONESIA_HISTORICAL_IGNORED_LABELS = [
    'Grassland shrubland', 'Logging', 'Secondary forest']

INDONESIA_IGNORED_LABEL_IDXS = [INDONESIA_ALL_LABELS.index(label)
                                for label in INDONESIA_IGNORED_LABELS]
INDONESIA_HISTORICAL_IGNORED_LABEL_IDXS = [INDONESIA_ALL_LABELS.index(
    label) for label in INDONESIA_HISTORICAL_IGNORED_LABELS]

INDONESIA_NUM_CLASSES = len(INDONESIA_LABELS)

LANDCOVER_LABELS = [
    'Unknown', 'Primary Dryland Forest', 'Secondary Dryland Forest',
    'Primary Swamp Forest', 'Secondary Swamp Forest', 'Primary Mangrove Forest',
    'Secondary Mangrove Forest', 'Bush/Shrub', 'Swamp Shrub', 'Grass Land',
    'Plantation Forest', 'Estate Cropplantation', 'Dryland Agriculture',
    'Shrub-Mixed Dryland Farm', 'Transmigration Area', 'Transmigration Area',
    'Rice Field', 'Fish Pond', 'Barren Land', 'Mining Area', 'Settlement Area',
    'Airport', 'Swamp', 'Cloud Covered', 'Bodies of Water'
]

# LANDCOVER_INDICES = [
#     0, 1, 2, 5, 51, 4, 41, 7, 71, 3, 6, 10, 91, 92,
#     122, 60, 93, 94, 14, 141, 12, 121, 11, 25, 200
# ]

# INDONESIA_PLACES_PATH = DATA_BASE_DIR / "deployment" / "indonesia.pkl"

#INDONESIA_AUX_DATA_PATH = DATA_BASE_DIR / 'indonesia_auxiliary'
NCEP_DIR = 'ncep'

# INDONESIA_ALL_PATH = INDONESIA_DIR / 'all.csv'
# INDONESIA_ALL_PATH_SENT2 = INDONESIA_DIR / 'all_latlon_SENT2.csv'
INDONESIA_TRAIN_PATH = INDONESIA_DIR / 'train.csv' #MODIFIED
INDONESIA_VAL_PATH = INDONESIA_DIR / 'val.csv' #MODIFIED
INDONESIA_TEST_PATH = INDONESIA_DIR / 'test.csv' #MODIFIED

# INDONESIA_TRAIN_PATH_NEW = INDONESIA_DIR / \
#     'train_latlon_new_download_multiple.csv'
# INDONESIA_VAL_PATH_NEW = INDONESIA_DIR / 'val_latlon_new_download_multiple.csv'
# INDONESIA_TEST_PATH_NEW = INDONESIA_DIR / \
#     'test_latlon_new_download_multiple.csv'

# INDONESIA_RELABEL_DIR = INDONESIA_DIR
# INDONESIA_VAL_RELABEL = INDONESIA_RELABEL_DIR / 'relabel_val.csv'
# INDONESIA_TEST_RELABEL = INDONESIA_RELABEL_DIR / 'relabel_test.csv'
# INDONESIA_NEW_VAL_RELABEL = INDONESIA_RELABEL_DIR / 'relabel_new_val.csv'
# INDONESIA_NEW_TEST_RELABEL = INDONESIA_RELABEL_DIR / 'relabel_new_test.csv'
# INDONESIA_GS_RELABEL = INDONESIA_RELABEL_DIR / 'relabel_gs.csv'

IS_CORRECT_HEADER = 'Correct'
ORIGINAL_LABEL_HEADER = 'Original Label'
CORRECTED_LABEL_HEADER = 'Corrected Label (if applicable)'

# INDONESIA_PEAT_JSON = 'Indonesia_peat_lands.geojson' 

DATASET_LABELS_NAMES = {
    INDONESIA_DATASET_NAME: {
        'label_names': INDONESIA_LABELS,
        'labels': list(range(INDONESIA_NUM_CLASSES)),
    },
    HANSEN_DATASET_NAME: {
        'label_names': HANSEN_LABELS_V3,
        'labels': list(range(len(HANSEN_LABELS_V3))),
    },
    LANDCOVER_DATASET_NAME: {
        'label_names': LANDCOVER_LABELS,
        'labels': list(range(len(LANDCOVER_LABELS)))
    }
}

# Cameroon dataset constants: AD
CAMEROON_TRAIN_SPLIT, CAMEROON_VAL_SPLIT, CAMEROON_TEST_SPLIT = .6, .15, .25

CAMEROON_ALL_LABELS = [
    'Oil palm plantation',
    'Timber plantation',
    'Other large-scale plantations',
    'Grassland shrubland',
    'Small-scale agriculture',
    'Small-scale mixed plantation',
    'Small-scale oil palm plantation',
    'Mining',
    'Fish pond',
    'Logging',
    'Secondary forest',
    'Other',
    '?', #AD
    'Rubber plantation', #AD
    'Timber sales', #AD
    'Fruit plantation', #AD
    'Wildfire', #AD
    'Small-scale cassava plantation', #AD
    'Small-scale maize plantation', #AD
    'Small-scale other plantation', #AD
    'Infrastructure', #AD
    'Hunting', #AD
    'Selective logging'] #AD

CAMEROON_DIR = DATA_BASE_DIR / 'ForestNetDataset'
CAMEROON_TRAIN_PATH = CAMEROON_DIR / 'mytrain.csv'
CAMEROON_VAL_PATH = CAMEROON_DIR / 'myval.csv'
CAMEROON_TEST_PATH = CAMEROON_DIR / 'mytest.csv'
CAMEROON_META_PATH = CAMEROON_DIR / 'myall.csv'

SINGLE_IMG_REGEX = r'[0-9]{4}_[0-9]{2}_[0-9]{2}_cloud_([0-9]+)\.png'
IMG_OPTION_CLOUD = 'least_cloudy'
IMG_OPTION_CLOSEST_YEAR = 'closest_year'
IMG_OPTION_FURTHEST_YEAR = 'furthest_year'
IMG_OPTION_COMPOSITE = 'composite'
IMG_OPTION_RANDOM = 'random'
IMG_OPTIONS = [IMG_OPTION_CLOUD,
               IMG_OPTION_CLOSEST_YEAR,
               IMG_OPTION_FURTHEST_YEAR,
               IMG_OPTION_COMPOSITE,
               IMG_OPTION_RANDOM]

# NOTE: based on successfully downloaded images from
# train_latlon_year.csv. Should be adjusted when missing images found.
INDONESIA_POST_2009_TRAIN_FREQS = [174, 110, 96, 220, 206,
                                   60, 42, 37, 8, 63, 123]  # 27]

INDONESIA_POST_2009_NUM_EXS = sum(INDONESIA_POST_2009_TRAIN_FREQS)
INDONESIA_POST_2009_CLASS_WEIGHTS = [
    INDONESIA_POST_2009_NUM_EXS /
    freq for freq in INDONESIA_POST_2009_TRAIN_FREQS]

INDONESIA_FULL_TRAIN_FREQS = [352, 246, 112, 395, 356, 113,
                              87, 49, 25, 184, 233]  # 53]
INDONESIA_FULL_NUM_EXS = sum(INDONESIA_FULL_TRAIN_FREQS)
INDONESIA_FULL_CLASS_WEIGHTS = [INDONESIA_FULL_NUM_EXS / freq
                                for freq in INDONESIA_FULL_TRAIN_FREQS]

DATASET_FREQS = {HANSEN_DATASET_NAME: HANSEN_V3_TRAIN_FREQS,
                 INDONESIA_DATASET_NAME: INDONESIA_FULL_TRAIN_FREQS}

INDONESIA_FIXED_CLASS_WEIGHTS = [ 
    3.0310,
    5.4481,
    3.8705,
    43.9184,
    86.0800,
    11.6957,
    9.2361] #NOT USED IN MODEL

NUM_RGB_CHANNELS = 3
NUM_IR_CHANNELS = 3
NUM_MASKED_CHANNELS = 1

# Both PIL and imgaug complain about having negative values,
# and we use -100 as a target sentinel to ignore loss for
# certain outputs. So, we use LABEL_IGNORE_VALUE
# and switch these to -100 (LOSS_IGNORE_VALUE) after the transforms.
LABEL_IGNORE_VALUE = 255
LOSS_IGNORE_VALUE = -100

# CRS strings for projection transformations
# We define a proj4 string, since IGH does not have an EPSG number
IGH_PROJ4 = '+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs'
LAT_LON_EPSG = 'epsg:4326'

# DL Band names
RED_BAND = 'red'
GREEN_BAND = 'green'
BLUE_BAND = 'blue'
NIR_BAND = 'nir'
SWIR1_BAND = 'swir1'
SWIR2_BAND = 'swir2'
CLOUD_MASK_BAND = 'cloud-mask'
CLOUD_MASK_BAND_LS7 = "derived:visual_cloud_mask"
NDVI_BAND = "ndvi"
NDVI_BAND_LS7 = "derived:ndvi"
BRIGHT_MASK_BAND = 'bright-mask'
CIRRUS_BAND = 'cirrus'
TREECOVER_BAND = 'treecover2000'
LOSSYEAR_BAND = 'lossyear'
RGB_BANDS = [RED_BAND, GREEN_BAND, BLUE_BAND]
IR_BANDS = [NIR_BAND, SWIR1_BAND, SWIR2_BAND]
CLOUD_BANDS = [CLOUD_MASK_BAND, BRIGHT_MASK_BAND, CIRRUS_BAND]
BANDS_LS8 = RGB_BANDS + IR_BANDS + CLOUD_BANDS
BANDS_LS8_NO_IR = RGB_BANDS + CLOUD_BANDS
BANDS_LS7 = RGB_BANDS + IR_BANDS + [CLOUD_MASK_BAND_LS7] + [NDVI_BAND_LS7]
BANDS_LS7_NO_IR = RGB_BANDS + [CLOUD_MASK_BAND_LS7] + [NDVI_BAND_LS7]
GFC_GAIN_BAND = "gain"
GFC_COVER_BAND = "treecover2000"
GFC_BANDS = [GFC_GAIN_BAND, GFC_COVER_BAND]

NCEP_TEMP_AVG = 'tavg'
NCEP_TEMP_MAX = 'tmax'
NCEP_TEMP_MIN = 'tmin'
NCEP_TEMP_BANDS = [NCEP_TEMP_AVG, NCEP_TEMP_MAX, NCEP_TEMP_MIN]

# Download all NCEP bands, except for 3 snow-related bands, 
# and transpiration (not available in early version of product.)

ALBEDO_BAND = 'albedo'
DOWN_LONG_FLUX_BAND = 'clear-sky_downward_longwave_flux'
DOWN_SOLAR_FLUX_BAND = 'clear-sky_downward_solar_flux'
UP_LONG_FLUX_BAND = 'clear-sky_upward_longwave_flux'
UP_SOLAR_FLUX_BAND = 'clear-sky_upward_solar_flux'
EVAP_BAND = 'direct_evaporation_bare_soil'
DOWN_LONG_RAD_FLUX_BAND = 'downward_longwave_radiation_flux'
UP_LONG_RAD_FLUX_BAND = 'upward_longwave_radiation_flux'
DOWN_SHORT_RAD_FLUX_BAND = 'downward_shortwave_radiation_flux'
UP_SHORT_RAD_FLUX_BAND = 'upward_shortwave_radiation_flux'
GROUND_HEAT_BAND = 'ground_heat_net_flux'
LATENT_HEAT_BAND = 'latent_heat_net_flux'
MAX_HUMIDITY_BAND = 'max_specific_humidity'
MIN_HUMIDITY_BAND = 'min_specific_humidity'
POTENTIAL_EVAP_BAND = 'potential_evaporation_rate'
PREC_BAND = 'prec'
SENSIBLE_HEAT_BAND = 'sensible_heat_net_flux'
SOIL_MOIST1_BAND = 'soilmoist1'
SOIL_MOIST2_BAND = 'soilmoist2'
SOIL_MOIST3_BAND = 'soilmoist3'
SOIL_MOIST4_BAND = 'soilmoist4'
HUMIDITY_BAND = 'specific_humidity'
SUBLIMATION_BAND = 'sublimation'
SURFACE_PRESSURE_BAND = 'surface_pressure'
U_WIND_BAND = 'u_wind_10m'
V_WIND_BAND = 'v_wind_10m'
WATER_RUNOFF_BAND = 'water_runoff'

NCEP_BANDS = [
    ALBEDO_BAND,
    DOWN_LONG_FLUX_BAND,
    DOWN_SOLAR_FLUX_BAND,
    UP_LONG_FLUX_BAND,
    UP_SOLAR_FLUX_BAND,
    EVAP_BAND,
    DOWN_LONG_RAD_FLUX_BAND,
    DOWN_SHORT_RAD_FLUX_BAND,
    GROUND_HEAT_BAND,
    LATENT_HEAT_BAND,
    MAX_HUMIDITY_BAND,
    MIN_HUMIDITY_BAND,
    POTENTIAL_EVAP_BAND,
    PREC_BAND,
    SENSIBLE_HEAT_BAND,
    SOIL_MOIST1_BAND,
    SOIL_MOIST2_BAND,
    SOIL_MOIST3_BAND,
    SOIL_MOIST4_BAND,
    HUMIDITY_BAND,
    SUBLIMATION_BAND,
    SURFACE_PRESSURE_BAND,
    NCEP_TEMP_AVG,
    NCEP_TEMP_MAX,
    NCEP_TEMP_MIN,
    U_WIND_BAND,
    UP_LONG_RAD_FLUX_BAND,
    UP_SHORT_RAD_FLUX_BAND,
    V_WIND_BAND,
    WATER_RUNOFF_BAND
]

ALTITUDE_BAND = 'altitude'
ASPECT_BAND = 'aspect'
SLOPE_BAND = 'slope'
SRTM_BANDS = [ALTITUDE_BAND, ASPECT_BAND, SLOPE_BAND]
NUM_SRTM_CHANNELS = len(SRTM_BANDS)

ALL_SUPPORTED_BANDS = RGB_BANDS + IR_BANDS + SRTM_BANDS + [NDVI_BAND]

# SRTM constants for feature scaling (computed from train_latlon_ls8_dynamic.csv)
ALTITUDE_MIN = -23.0
ALTITUDE_MAX = 4228.0
SLOPE_MIN = 0.0
SLOPE_MAX = 8421.0
ASPECT_MIN = -17954.0
ASPECT_MAX = 18000.0

SRTM_SCALING = {
    ALTITUDE_BAND: (ALTITUDE_MIN, ALTITUDE_MAX),
    SLOPE_BAND: (SLOPE_MIN, SLOPE_MAX),
    ASPECT_BAND: (ASPECT_MIN, ASPECT_MAX)
}

# GFC constants for feature scaling
TREE_COVER_MIN = 0.0
TREE_COVER_MAX = 100.0
TREE_COVER_SCALING = (TREE_COVER_MIN, TREE_COVER_MAX)

# NCEP constants for feature scaling (computed from train_latlon_ls8_dynamic.csv)
NCEP_SCALING = {
    ALBEDO_BAND: (268., 1103.),
    DOWN_LONG_FLUX_BAND: (274., 434.),
    DOWN_SOLAR_FLUX_BAND: (233., 346.),
    UP_LONG_FLUX_BAND: (366., 503.),
    UP_SOLAR_FLUX_BAND: (13., 48.),
    EVAP_BAND: (0., 262.),
    DOWN_LONG_RAD_FLUX_BAND: (276., 449.),
    DOWN_SHORT_RAD_FLUX_BAND: (5., 345.),
    GROUND_HEAT_BAND: (-37., 21.),
    LATENT_HEAT_BAND: (-2., 789.),
    MAX_HUMIDITY_BAND: (7., 405.),
    MIN_HUMIDITY_BAND: (7., 405.),
    POTENTIAL_EVAP_BAND: (0., 2171.),
    PREC_BAND: (0., 3218.),
    SENSIBLE_HEAT_BAND: (-604., 239.),
    SOIL_MOIST1_BAND: (0., 4550.),
    SOIL_MOIST2_BAND: (0., 4525.),
    SOIL_MOIST3_BAND: (0., 4523.),
    SOIL_MOIST4_BAND: (0., 4604.),
    HUMIDITY_BAND: (49., 222.),
    SUBLIMATION_BAND: (0., 11.),
    SURFACE_PRESSURE_BAND: (7553., 10216.),
    NCEP_TEMP_AVG: (27286., 31693.),
    NCEP_TEMP_MAX: (27286., 31693.),
    NCEP_TEMP_MIN: (27286., 31693.),
    U_WIND_BAND: (-968., 1215.),
    UP_LONG_RAD_FLUX_BAND: (366., 502.),
    UP_SHORT_RAD_FLUX_BAND: (0., 43.),
    V_WIND_BAND: (-1148., 1065.),
    WATER_RUNOFF_BAND: (0., 29115.),
}

# OSM constants for feature scaling in km (computed from train_latlon_ls8_dynamic.csv)
STREET_MIN = 0.00327
CITY_MIN = 0.19590
STREET_MAX = 513.49534
CITY_MAX = 513.49534

OSM_SCALING = {
    'city': (CITY_MIN, CITY_MAX),
    'street': (STREET_MIN, STREET_MAX)
}

SCENE_LIMIT = 200
NDVI_IMG_MEAN_LS7 = 48000
SINGLE_IMG_CLOUD_FRAC = 0.005
SINGLE_IMG_CLOUD_FRAC_LS7 = 0.015
SMALL_COMP_SC_NUM = 5
SMALL_COMP_CLOUD_FRAC = 0.05

SINGLE_IMG_DOWNLOAD_METHOD = 'single image'
SMALL_COMPOSITE_DOWNLOAD_METHOD = 'small composite'
FULL_COMPOSITE_DOWNLOAD_METHOD = 'full composite'

# Tile download constants
KM_TO_DEG = 0.008
HANSEN_TILE_SIZE_KM = 10
INDONESIA_TILE_SIZE_KM = 5
CLOUD_FRACTION = 0.5

# DL product constants
LANDSAT8_TIER1_PRODUCT_NAME = 'landsat:LC08:01:T1:TOAR'
LANDSAT8_PRE_COLLECTION_PRODUCT_NAME = 'landsat:LC08:PRE:TOAR'
# LANDSAT7_PRE_COLLECTION_PRODUCT_NAME = 'landsat:LE07:PRE:TOAR'
# LANDSAT5_PRE_COLLECTION_PRODUCT_NAME = 'landsat:LT05:PRE:TOAR'
# SENTINEL2_PRODUCT_NAME = 'sentinel-2:L1C'
# INDONESIA_LANDCOVER_PRODUCT_NAME = 'descarteslabs:indonesia_land_cover'
NCEP_PRODUCT_NAMES = ['ncep:cfsr-v1:daily:v1', 'ncep:cfsr-v2:daily:v1']
LANDSAT8_TIER1_PRODUCT_RES = 15
# SENTINEL2_PRODUCT_RES = 10
DOWNLOAD_MASKED = 'masked'
GFC_PRODUCT = '42b24cbb9a71ed9beb967dbad04ea61d7331d5af:global_forest_change_v0'
PLACE_INDONESIA = 'asia_indonesia'
SRTM_PRODUCT_NAME = 'srtm:GL1003'

# Data CSV headers
MERGED_LABEL_HEADER = 'merged_label'
LABEL_HEADER = 'label'
X_CENTROID_HEADER = 'x_coord'
Y_CENTROID_HEADER = 'y_coord'
LATITUDE_HEADER = 'latitude'
LONGITUDE_HEADER = 'longitude'
IMG_PATH_HEADER = 'example_path' #MODIFIED
NUM_SC_HEADER = 'num_scenes'
YEAR_HEADER = 'year'
SHAPEFILE_HEADER = 'shape_paths'
AREA_HEADER = 'area_ha'
IR_PATH_HEADER = 'ir_paths'
DOWNLOAD_METHOD_HEADER = 'download_method'
NUM_IMGS_DOWNLOADED = 'num_imgs_downloaded'
IMG_COMPOSITE_IS_LS8 = 'composite_is_landsat8'
TILE_ID_HEADER = 'tile_id'
SNAPSHOT_ID_HEADER = 'snapshot_id'
CLASS_PRED_HEADER = 'class_preds'
SEGMENTATION_MAP_HEADER = 'segmap_paths'
LANDCOVER_MAP_PATH_HEADER = 'landcover_map'
AUX_PATH_HEADER = 'aux_paths'

# Train/Valid data CSV headers
# HANSEN_META_COLNAMES = [LABEL_HEADER,
#                         LATITUDE_HEADER,
#                         LONGITUDE_HEADER,
#                         IMG_PATH_HEADER,
#                         NUM_SC_HEADER]

INDONESIA_META_COLNAMES = [LABEL_HEADER,
                           MERGED_LABEL_HEADER,
                           LATITUDE_HEADER,
                           LONGITUDE_HEADER,
                           YEAR_HEADER,
                           IMG_PATH_HEADER]

CAMEROON_META_COLNAMES = [LABEL_HEADER, #AD
                           MERGED_LABEL_HEADER,
                           LATITUDE_HEADER,
                           LONGITUDE_HEADER,
                           YEAR_HEADER,
                           IMG_PATH_HEADER]
#AD: INDONESIA_META_COLNAMES MODIFIED TO FIT THE CSV FILES RECEIVED WITH TRAINING DATA








# LANDCOVER_META_COLNAMES = [IMG_PATH_HEADER,
#                            IMG_OPTION_CLOSEST_YEAR,
#                            IMG_OPTION_COMPOSITE,
#                            LANDCOVER_MAP_PATH_HEADER]

# Model architecture
NUM_INPLANES = 64
SHAPE_META_COLNAMES = [LABEL_HEADER,
                       LATITUDE_HEADER,
                       LONGITUDE_HEADER,
                       YEAR_HEADER,
                       AREA_HEADER,
                       IMG_PATH_HEADER,
                       NUM_SC_HEADER]


DEPLOY_SHAPE_META_COLNAMES = [LATITUDE_HEADER,
                              LONGITUDE_HEADER,
                              YEAR_HEADER,
                              SHAPEFILE_HEADER,
                              SNAPSHOT_ID_HEADER,
                              CLASS_PRED_HEADER]


DEPLOY_META_COLNAMES = [LATITUDE_HEADER,
                        LONGITUDE_HEADER,
                        YEAR_HEADER,
                        SHAPEFILE_HEADER]

SEG_LEGEND_PATH = 'util/color_legend.png'

# CAM constants
MODEL2CAM_LAYER = {"DenseNet121": "model.features",
                   "ResNet152": "model.layer4.2.conv3",
                   "Inceptionv4": "model.features.21.branch3.1.conv",
                   "ResNet101": "model.layer4.2.conv3"}
CAM_DIR = 'cams'
CAM_PATH = 'CAM_path'
IMAGE_PATH = 'image_path'
TARGET_PATH = 'target_path'
NUMPY_PATH = 'numpy_path'
PROB = 'probability'
PRED = 'prediction'
INDEX = 'index'
TARGET = 'target'

# Shapefile creation constants
# YEAR_TO_INT_CONVERSION = 2000
# FORESTED_THRESHOLD = 30
# SHAPE_META_NAME = 'shape_info.csv'
# TILE_META_NAME = 'tile_info.csv'
# SHAPE_SIZE_THRESHOLD = 2
# INDONESIA_PLACE_NAME = 'indonesia_place.json'
# EXP_SHAPE_META_NAME = 'exp_shape_info.csv'
# EXP_TILE_META_NAME = 'exp_tile_info.csv'
# SNAPSHOT_TO_SHAPE_IDX_NAME = 'snapshot_to_shape_idx.json'
# SHAPE_PIXEL_HEIGHT = 166
# SHAPE_PIXEL_WIDTH = 166

# Deploy constants
OVERLAY_EXTENSION = '_overlay.png'
POLYGON_EXTENSION = '_polygon.png'

# Geo encoding constants
INDONESIA_LAT_MIN = -11.0
INDONESIA_LAT_MAX = 6.1 
INDONESIA_LON_MIN = 95.0
INDONESIA_LON_MAX = 141.1

# approximate radius of earth in km
EARTH_RADIUS = 6373.0

# Landcover pretraining constants
# LANDCOVER_EXP_NAMES = {
#     'ResNet50': "pretrain0910_ResNet50_0.0001_0",
#     'EfficientNet-b2': "pretrain0910_EfficientNet-b2_0.0005_0"
# }

# Baseline model configuration 
Y_COL = ['label_cls']
META_COL = ['index', 'n_pixel']
AREA_COL = ['n_pixel']
