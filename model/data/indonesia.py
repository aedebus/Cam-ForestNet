import pandas
import numpy as np
import os
import json
from geopy import distance

from util import *
from .classification_dataset import ClassificationDataset, ClassificationAddedBandsDataset
from .segmentation_dataset import SegmentationDataset
from .ml_dataset import MachineLearningDataset


class IndonesiaDataset:

    DATA_SPLIT_TO_RELABEL = DATA_SPLITS

    def process_file(self):
        """This function processes image meta-CSV into a list of
        (label, lat, long, image path) tuples.
        """
        self.meta_filename = INDONESIA_DIR / \
                             f"{self._data_split}.csv"
        self._image_info = pandas.read_csv(self.meta_filename,
                                           header=0,
                                           names=INDONESIA_META_COLNAMES)
        print(self._image_info)
        # self.correct_labels()
        #self.filter_examples()

    def filter_examples(self):
        # TODO: right now we drop images that don't exist
        # from the CSV (image_path == None). We need to go
        # retrieve these.
        # NOTE: we also drop labels 'Other' for now.
        valid_img_paths = self._image_info[IMG_PATH_HEADER] != 'None'
        #valid_shape_paths = self._image_info[SHAPEFILE_HEADER] != 'None'
        valid_labels = self._image_info[LABEL_HEADER].isin(
            self._original_labels)
        
        if self._sample_filter == "historical_merge":
            # ignore Logging, Grassland/Shrubland, Secondary Forest before 2012
            valid_years = self._image_info[YEAR_HEADER] >= '2012'
            valid_historical_labels = ~self._image_info[LABEL_HEADER].isin(
                INDONESIA_HISTORICAL_IGNORED_LABEL_IDXS)
            valid_years_labels = valid_years | valid_historical_labels
            filter_condition = valid_img_paths & valid_labels & valid_years_labels
        elif self._sample_filter == "post_2012":
            # ignore all samples before 2012
            valid_years = self._image_info[YEAR_HEADER] >= '2012'
            filter_condition = valid_img_paths & valid_labels & valid_years
        else:
            filter_condition = valid_img_paths & valid_labels

        if self._use_ir:
            valid_ir_paths = self._image_info[IMG_PATH_HEADER] != 'None'
            filter_condition = filter_condition & valid_ir_paths

        self._image_info = self._image_info[filter_condition]

    # def correct_labels(self):
    #     if self._data_split not in self.DATA_SPLIT_TO_RELABEL:
    #         return
    #     relabel_paths = self.DATA_SPLIT_TO_RELABEL[self._data_split]
    #     relabel_examples = pandas.concat([pandas.read_csv(relabel_path)\.drop(columns=['Image URL'], errors='ignore')for relabel_path in relabel_paths])
    #     relabel_examples = relabel_examples[relabel_examples[IS_CORRECT_HEADER] == 'Incorrect']
    #     for _, relabel in relabel_examples.iterrows():
    #         year = relabel[YEAR_HEADER]
    #         lat= relabel[LATITUDE_HEADER]
    #         lon = relabel[LONGITUDE_HEADER]
    #         lat, lon = round(lat, 4), round(lon, 4)
    #         query = ((self._image_info[YEAR_HEADER] == year) &
    #                  (self._image_info[LATITUDE_HEADER].round(4) == lat) &
    #                  (self._image_info[LONGITUDE_HEADER].round(4) == lon))
    #         row_to_correct = self._image_info[query]
    #         assert(len(row_to_correct) <= 1)
    #         if len(row_to_correct) == 1:
    #             new_label = relabel[CORRECTED_LABEL_HEADER]
    #             if new_label == 'Plantation':
    #                 new_label = 'Oil palm plantation'
    #             elif new_label == 'Fallow Land':
    #                 new_label = 'Small-scale agriculture'
    #             new_label = INDONESIA_LABELS.index(new_label)
    #             self._image_info.at[row_to_correct.iloc[0].name, LABEL_HEADER] = new_label

    def _get_ncep_features(self, aux_path, feature_scale=False):
        features = {}
        ncep_path = os.path.join(aux_path, NCEP_DIR)
        for feat_name in NCEP_FEATURES:
            path = os.path.join(ncep_path, f'{feat_name}.npy')
            feat = np.load(path, allow_pickle=True) #AD
            
            # NOTE: Pretty hacky right now for retreiving feature scaling keys 
            # from feature names, should fix this eventually. 
            if feature_scale:
                if feat_name in TEMP_FEATURES:
                    min_scale, max_scale = NCEP_SCALING[feat_name]
                else:
                    min_scale, max_scale = NCEP_SCALING[feat_name[:-4]]
                feat = self._feature_scale(feat, min_scale, max_scale, False)
            
            features[feat_name] = feat
            
        return features
    
    def _get_peat_features(self, aux_path):
        peat_path = os.path.join(aux_path, 'peat.json')
        peat = json.loads(open(peat_path).read())['peat']
        peat_val = 1. if peat else 0.
        return {'peat': peat_val}
        
    # Using geodesic distance provided by geopy
    # See: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
    def _get_osm_features(self, aux_path, lat, lon, feature_scale=False):
        street_path = os.path.join(aux_path, 'closest_street.json')
        street = json.loads(open(street_path).read())
        city_path = os.path.join(aux_path, 'closest_city.json')        
        city = json.loads(open(city_path).read())
        street_dist = distance.distance((lat, lon),
                                        (street.get('lat'), street.get('lon'))).km                                  
        city_dist = distance.distance((lat, lon),
                                      (city.get('lat'), city.get('lon'))).km
                                      
        if feature_scale:
            street_min, street_max = OSM_SCALING['street']
            street_dist = self._feature_scale(street_dist, street_min, street_max, False)
            city_min, city_max = OSM_SCALING['city']
            city_dist = self._feature_scale(city_dist, city_min, city_max, False)
                                      
        features = {'street_dist': street_dist,
                    'city_dist': city_dist}        
        return features      
        
    def _get_img_summary_stats(self,
                               bands,
                               band_names,
                               mask=None):
        features = {}
        if type(bands) == float:  # AD
            max_val = bands
            min_val = bands
            avg_val = bands
            std_val = bands
            if mask is not None:
                feature_type = FL_REGION
            else:
                feature_type = IMG_REGION
        elif type(bands) != float and len(bands.shape) == 1: #AD
            max_val = bands.max(axis=0)
            min_val = bands.min(axis=0)
            avg_val = bands.mean(axis=0)
            std_val = bands.std(axis=0)
            if mask is not None:
                feature_type = FL_REGION
            else:
                feature_type = IMG_REGION
        else:
            if mask is not None:
                feature_type = FL_REGION
                if bands.shape[0] != mask.shape[0]: #AD
                    bands = np.transpose(bands)
                max_val = bands[mask].max(axis=0) #AD
                min_val = bands[mask].min(axis=0) #AD
                avg_val = bands[mask].mean(axis=0) #AD
                std_val = bands[mask].std(axis=0) #AD
            else:
                feature_type = IMG_REGION
                max_val = bands.max(axis=(0, 1))
                min_val = bands.min(axis=(0, 1))
                avg_val = bands.mean(axis=(0, 1))
                std_val = bands.std(axis=(0, 1))
        if not isinstance(max_val, np.ndarray): #AD
            max_val=np.array([max_val]) #AD
        if not isinstance(min_val,  np.ndarray): #AD
            min_val=np.array([min_val]) #AD
        if not isinstance(avg_val,  np.ndarray):#AD
            avg_val=np.array([avg_val]) #AD
        if not isinstance(std_val,  np.ndarray): #AD
            std_val=np.array([std_val]) #AD
        for band_idx, band_name in enumerate(band_names):
            if band_idx<len(max_val): #AD
                features[f'{band_name}_{feature_type}_min'] = min_val[band_idx]                    
                features[f'{band_name}_{feature_type}_max'] = max_val[band_idx]
                features[f'{band_name}_{feature_type}_avg'] = avg_val[band_idx] 
                features[f'{band_name}_{feature_type}_std'] = std_val[band_idx]
        
        return features 
                
    def _get_gfc_img(self, aux_path, band_name):
        gfc_img = np.load(os.path.join(aux_path, f'{band_name}.npy'), allow_pickle=True )[0] #AD
        return gfc_img         
                
    def _get_srtm_img(self, index):
        srtm_img = np.stack([self._get_srtm(index, band) for band in SRTM_BANDS], axis = 2) #AD: remove axis = 2
        return srtm_img    
                                       
    def _get_raw_imgs(self, index, im_path, feature_scale=False):
        aux_path = self._get_aux_path(index)
        rgb_img = self._get_image(im_path)
        ir_img = self._get_ir(im_path)
        srtm_img = self._get_srtm_img(index)
        ndvi_img = self._get_ndvi(rgb_img, ir_img)[..., np.newaxis]
        gfc_gain_img = self._get_gfc_img(aux_path, GFC_GAIN_BAND)
        #gfc_cover_img = self._get_gfc_img(aux_path, GFC_COVER_BAND) #AD: modified
        
        label = self._get_label(index)
        polygon = self._get_polygon(index)
        mask = self._get_mask(rgb_img, label, polygon)
                
        mask[mask == LABEL_IGNORE_VALUE] = 0
        mask[mask != 0] = 1        
        mask = mask.squeeze().astype(np.bool)
              
        if feature_scale:
            # Scale to [0-1] without re-scaling.
            # GFC Gain already [0-1].
            rgb_img = self._feature_scale(rgb_img, 0, 255, False)
            ir_img = self._feature_scale(ir_img, 0, 255, False)
            srtm_img = self._feature_scale(srtm_img, 0, 255, False)
            ndvi_img = self._feature_scale(ndvi_img, 0, 255, False)            
            cover_min, cover_max = TREE_COVER_SCALING
            # gfc_cover_img = self._feature_scale(gfc_cover_img, cover_min, cover_max, False) #AD: modified
            
        # gfc_img = np.stack([gfc_gain_img, gfc_cover_img], axis=2) #AD: modified
                
        imgs = {
            'rgb': {'image': rgb_img, 'band_names': RGB_BANDS},
            'ir': {'image': ir_img, 'band_names': IR_BANDS},
            'ndvi': {'image': ndvi_img, 'band_names': [NDVI_BAND]},
            'srtm': {'image': srtm_img, 'band_names': SRTM_BANDS}, 
            'gfc': {'image': gfc_gain_img, 'band_names': [GFC_GAIN_BAND]} #AD: modified
            }         
        return imgs, mask
        
    def _get_constant_features(self, index, feature_scale=False):
        aux_path = self._get_aux_path(index)
        features = {}
        # Lat/Lon never need to be feature scaled.
        lat, lon = self._get_latlon(index)
        features['lat'] = lat
        features['lon'] = lon
        # Peat already [0-1].
        peat_features = self._get_peat_features(aux_path)
        osm_features = self._get_osm_features(aux_path, lat, lon, feature_scale)
        # TODO: Add flag to include/exclude ncep features
        if self._ncep:
            ncep_features = self._get_ncep_features(aux_path, feature_scale)
            features.update(ncep_features)
        features.update(peat_features)
        features.update(osm_features)
        return features 
        
    
class IndonesiaClassificationDataset(IndonesiaDataset,
                                     ClassificationDataset):
    def __init__(self, *args, **kwargs):
        IndonesiaDataset.__init__(self)
        ClassificationDataset.__init__(self, *args, **kwargs)


class IndonesiaSegmentationDataset(IndonesiaDataset, #AD: The one used in the model
                                   SegmentationDataset):
    def __init__(self, *args, **kwargs):
        IndonesiaDataset.__init__(self)
        SegmentationDataset.__init__(self, *args, **kwargs)
        # Features should be scaled for late fusion
        self.feature_scale = True
        
    def _get_aux_features(self, index, im_path):
        features = self._get_constant_features(index,
                                               self.feature_scale)
        imgs, mask = self._get_raw_imgs(index,
                                        im_path,
                                        self.feature_scale)
        
        for img_name, img_dict in imgs.items():
            # TODO: add flag to include/exclude image-wise stats
            #img_stats = self._get_img_summary_stats(img_dict['image'],
            #                                        img_dict['band_names'])
            region_stats = self._get_img_summary_stats(img_dict['image'],
                                                       img_dict['band_names'],
                                                       mask)
            #features.update(img_stats)
            features.update(region_stats)

        test = np.array([0]) # operation with features not possible with arrays
        for k,v in features.items(): #AD
            if type(v) == type(test):
                features[k] = torch.tensor(np.float64(v), dtype=torch.float32)
            else:
                features[k] = torch.tensor(v, dtype=torch.float32)
        #features = {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}
        return features
        
    def __getitem__(self, index):
        batch = super().__getitem__(index) #Super to give access to the properties of parent
        if self._aux_features: 
            im_path = batch.get('im_path')
            aux_features = self._get_aux_features(index, im_path)
            batch.update(aux_features)
        
        return batch 


class IndonesiaMachineLearningRegionDataset(IndonesiaDataset,
                                            MachineLearningDataset):
    def __init__(self, *args, **kwargs):
        IndonesiaDataset.__init__(self)
        MachineLearningDataset.__init__(self, *args, **kwargs)
        # Features need not be scaled for ML Baselines (RFs)
        self.feature_scale = False

    def _get_aux_features(self, index, im_path):
        features = self._get_constant_features(index,
                                               feature_scale=self.feature_scale)
        imgs, mask = self._get_raw_imgs(index,
                                        im_path,
                                        feature_scale=self.feature_scale)
                                  
        for img_name, img_dict in imgs.items():
            img_stats = self._get_img_summary_stats(img_dict['image'],
                                                    img_dict['band_names'])
            region_stats = self._get_img_summary_stats(img_dict['image'],
                                                       img_dict['band_names'],
                                                       mask)
            features.update(img_stats)
            features.update(region_stats)
            
        return features

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        im_path = batch.get('im_path')
        aux_features = self._get_aux_features(index, im_path)
        batch.update(aux_features)
        return batch 

class IndonesiaMachineLearningPixelDataset(IndonesiaDataset,
                                           MachineLearningDataset):
    def __init__(self, *args, **kwargs):
        IndonesiaDataset.__init__(self)
        MachineLearningDataset.__init__(self, *args, **kwargs)
        # Features need not be scaled for ML Baselines (RFs)
        self.feature_scale = False
        
    def _get_aux_features(self, index, im_path):
        features = self._get_constant_features(index,
                                               self.feature_scale)
        imgs, mask = self._get_raw_imgs(index,
                                        im_path,
                                        self.feature_scale)
        
        for img_name, img_dict in imgs.items():
            # Pixel level stats
            img, band_names = img_dict['image'], img_dict['band_names']
            for band_idx, band_name in enumerate(band_names):
                px = img[:, :, band_idx][mask]
                features[band_name] = px   
    
        return features

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        aux_features = self._get_aux_features(index)
        im_path = batch.get('im_path')
        batch.update(aux_features, im_path)
        return batch 
