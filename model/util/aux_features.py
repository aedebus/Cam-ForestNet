from .constants import *

# 3 'modes' we use aux data in.
PX_MODE = 'pixel' # pixel baseline
REGION_MODE = 'loss_region' # region baseline
LATE_FUSION_MODE = 'late_fusion' # late fusion for segmentation
ML_BASELINE_MODES = [PX_MODE, REGION_MODE]

IMG_REGION = 'image'
FL_REGION  = REGION_MODE
PX_REGION  = PX_MODE
                                    
'''
There are three cases of features: 
1) pixel-wise features
2) fl-wise features (forest loss)
3) image-wise features 

All "types" of feature names are defined below.
'''

'''
Auxiliary features with constant values, ie. 
the features are the same across all three cases.  
'''
COORD_FEATURES = ['lat', 'lon']
OSM_FEATURES = ['street_dist', 'city_dist']
PEAT_FEATURES = ['peat']
# For temp, avg min and max were bands already defined by NCEP.   
TEMP_FEATURES = NCEP_TEMP_BANDS 
NCEP_FEATURES = []
for band in NCEP_BANDS:
    if band in TEMP_FEATURES:
        NCEP_FEATURES.append(band)
    else:
        NCEP_FEATURES.extend([f'{band}_min', f'{band}_max', f'{band}_avg'])

CONSTANT_FEATURES = COORD_FEATURES + OSM_FEATURES + PEAT_FEATURES #+ NCEP_FEATURES
# No lat, lon for late fusion because we handle with geoencoding
LATE_FUSION_CONSTANT_FEATURES = OSM_FEATURES + PEAT_FEATURES #+ NCEP_FEATURES

'''
Auxiliary features with values that change
across the three cases above. These are the cases where we have per-pixel data.
'''
# Summary features computed for pixel-wise features
# (for late fusion + region baseline)
SUMMARY_FEATURES = (lambda band, region: [f'{band}_{region}_min', 
                                          f'{band}_{region}_max',
                                          f'{band}_{region}_avg',
                                          f'{band}_{region}_std'])
'''
For pixel-wise, feature names are just names of the bands, ie 'red', etc. 
For loss_region, feature names are ie 'red_loss_region_avg', etc.
For full region, feature names are 'red_image_avg', etc. 
'''
def get_feature_types(bands):
    feature_types = {
        PX_REGION: bands, 
        IMG_REGION: [], 
        FL_REGION: [],
    }
    for band in bands:
        feature_types[IMG_REGION].extend(SUMMARY_FEATURES(band, IMG_REGION))
        feature_types[FL_REGION].extend(SUMMARY_FEATURES(band, FL_REGION))
    return feature_types
    

RGB_FEATURES = get_feature_types(RGB_BANDS)
# NOTE: Removing 'gain' from 
# GFC aux data for 10/06 submission (temporal leakage).
GFC_FEATURES = get_feature_types([GFC_GAIN_BAND]) #AD
SRTM_FEATURES = get_feature_types(SRTM_BANDS)
IR_FEATURES = get_feature_types(IR_BANDS)
NDVI_FEATURES = get_feature_types([NDVI_BAND])

PIXEL_FEATURES = (RGB_FEATURES[PX_REGION] +
                  GFC_FEATURES[PX_REGION] + 
                  SRTM_FEATURES[PX_REGION] +
                  IR_FEATURES[PX_REGION] + 
                  NDVI_FEATURES[PX_REGION])

FL_FEATURES = (RGB_FEATURES[FL_REGION] + 
               GFC_FEATURES[FL_REGION] +
               SRTM_FEATURES[FL_REGION] +
               IR_FEATURES[FL_REGION] +
               NDVI_FEATURES[FL_REGION]) 

IMG_FEATURES = (RGB_FEATURES[IMG_REGION] + 
                GFC_FEATURES[IMG_REGION] +
                SRTM_FEATURES[IMG_REGION] +
                IR_FEATURES[IMG_REGION] + 
                NDVI_FEATURES[IMG_REGION]) 
                            
# For late fusion, we'd like to have features computed based on the
# forest loss region as well as the entire image. 
# True for RGB features only. 
AUX_FEATURES = {
    PX_MODE: {
        True: RGB_FEATURES[PX_REGION],
        False: CONSTANT_FEATURES + PIXEL_FEATURES,
    },
    REGION_MODE: {
        True: RGB_FEATURES[FL_REGION] + RGB_FEATURES[IMG_REGION],
        False: CONSTANT_FEATURES + FL_FEATURES + IMG_FEATURES,
    },
    LATE_FUSION_MODE: {
        True: RGB_FEATURES[FL_REGION] + RGB_FEATURES[IMG_REGION],
        False: LATE_FUSION_CONSTANT_FEATURES + FL_FEATURES,
    }
}
