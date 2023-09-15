import torch
import pickle
import numpy as np
from PIL import Image, ImageEnhance
import os
import random
import re
import pandas as pd
from util.constants import *
from util.merge_scheme import *


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_version,
                 data_split,
                 img_option,
                 merge_scheme=None,
                 sample_filter=None,
                 masked=False,
                 ncep=False,
                 bands=["red", "green", "blue"],
                 transforms=None,
                 no_augment_transforms=None,
                 aux_features=None):
                 
        """
        Params:
            filename (str): tab-separated Meta-CSV minimally requiring
                            (label, lat, long, image path)
            transforms (object): composed transform
        """
        if data_version is None or len(data_version) == 0:
            self._data_version = ""
        elif data_version.startswith("_"):
            self._data_version = data_version
        else:
            self._data_version = f"_{data_version}"

        # Ensure all specified bands are supported.
        for band in bands:
            if band not in ALL_SUPPORTED_BANDS:
                raise ValueError(f"Band {band} not supported.")

        self._data_split = data_split
        self._img_option = img_option
        self._merge_scheme = merge_scheme
        self._original_labels = DATASET_MERGE_SCHEMES[self._merge_scheme]['original_label_index']
        self._label_mapping = DATASET_MERGE_SCHEMES[self._merge_scheme]['label_mapping']
        self._image_info = None
        self._sample_filter = sample_filter 
        self._masked = masked
        self._bands = bands
        self._use_ir = (any(ir_band in bands for ir_band in IR_BANDS)
                        or NDVI_BAND in bands)
        self._transforms = transforms
        self._no_augment_transforms = no_augment_transforms
        self._aux_features = aux_features
        self._ncep = ncep
        self.process_file()

    def process_file(self):
        raise Exception(NotImplementedError)
    
    def _get_example_meta(self, index):
        im_path = self._get_image_path(index, self._img_option)
        label = self._get_label(index)
        polygon = self._get_polygon(index)
        rgb_image = self._get_image(im_path)
        mask = self._get_mask(rgb_image, label, polygon)
        aux_path = self._get_aux_path(index)
        meta = {
            'index': index,
            'im_path': im_path,
            'label': label,
            'polygon': polygon,
            'mask': mask,
            'aux_path': aux_path,
        }   

        if self._use_ir:
            ir_image = self._get_ir(im_path)

        band_list = []
        for band in self._bands:
            if band in RGB_BANDS:
                band_list.append(rgb_image[:, :, RGB_BANDS.index(band)])
            elif band in IR_BANDS:
                band_list.append(ir_image[:, :, IR_BANDS.index(band)])
            elif band == NDVI_BAND:
                band_list.append(self._get_ndvi(rgb_image, ir_image))
            elif band in SRTM_BANDS:
                band_list.append(self._get_srtm(index, band))

        image = np.stack(band_list, axis=2)
        meta['image'] = image
        return meta

    def class_weights(self):
        labels_series=[] #AD
        for i in range(0, len(self._image_info[LABEL_HEADER])): #AD
            labels_series.append(INDONESIA_ALL_LABELS.index(self._image_info.iloc[i][LABEL_HEADER])) #AD
        labels_series=pd.Series(labels_series, name='label') #AD
        freq=labels_series.map(self._label_mapping).value_counts(normalize=True).sort_index() #AD
        return 1.0 / freq

    def get_image_info(self, index):
        return self._image_info.iloc[index].to_dict()

    def __len__(self):
        return len(self._image_info)

    def _get_image_path(self, index, img_option):
        image_info = self.get_image_info(index)
        if img_option is None:
            return image_info[IMG_PATH_HEADER]
        im_dir= os.path.join(INDONESIA_DIR,image_info[IMG_PATH_HEADER]) #AD
        im_dir = os.path.join(im_dir, "images") #AD
        im_path = None

        if img_option == IMG_OPTION_RANDOM:
            im_paths = os.listdir(os.path.join(im_dir, 'visible'))
            #print(im_dir)
            #print(im_paths)
            im_path = random.choice(im_paths)
            im_path = os.path.join('visible', im_path)
        else:
            im_path = image_info[img_option]
            # NOTE: In case image does not exist for previous options
            # (no single images retrieved), we fall back on composite
            if im_path == 'None':
                im_path = image_info[IMG_OPTION_COMPOSITE]
        return os.path.join(im_dir, im_path)

    def _get_image(self, im_path, feature_scale=False):
        pil_image = Image.open(im_path).convert('RGB')
        pil_image = ImageEnhance.Brightness(pil_image).enhance(1.5)
        image = np.array(pil_image)
        
        if feature_scale:
            image = self.feature_scale()
        
        return image

    def _get_ir(self, im_path, feature_scale=False):
        ir_path = im_path.replace('visible', 'infrared')
        if 'small_composite' in im_path or 'full_composite' in im_path:
            ir_path += '.npy'
        else:
            #ir_path = ir_path.replace('png', 'npy') #AD
            ir_name = str(os.path.split(ir_path)[1][0:4]) + '_ir_0.npy' #AD
            ir_path = os.path.join(os.path.dirname(ir_path), ir_name) #AD
        ir = np.load(ir_path).astype(np.uint8)
        return ir

    def _get_label(self, index):
        label = int(INDONESIA_ALL_LABELS.index(self._image_info.iloc[index][LABEL_HEADER])) # AD: MODIFIED + iloc=return part based on  and _image_info=csv_file
        label = self._label_mapping[label]
        label = torch.tensor(np.uint8(label), dtype=torch.long)
        return label

    def _get_polygon(self, index):
        image_info = self.get_image_info(index) #AD
        poly_dir=os.path.join(INDONESIA_DIR, image_info[IMG_PATH_HEADER]) #AD
        with open(os.path.join(poly_dir,'forest_loss_region.pkl'), 'rb') as f: #AD: MODIFIED
            polygon = pickle.load(f)
        return polygon

    def _get_landcover(self, index):
        landcover = np.load(
            self._image_info.iloc[index][LANDCOVER_MAP_PATH_HEADER]
        ).squeeze(0)
        # 255 is not in https://catalog.descarteslabs.com/?/product/descarteslabs
        landcover[landcover == 255] = 0
        landcover2index = np.vectorize(lambda x: LANDCOVER_INDICES.index(x))
        landcover_index = landcover2index(landcover)
        return landcover_index.astype(np.int32)

    def _get_aux_path(self, index):
        aux_path=os.path.join(INDONESIA_DIR, self._image_info.iloc[index][IMG_PATH_HEADER]) #AD
        return os.path.join(aux_path, "auxiliary") #AD: MODIFIED
    
    def _get_srtm(self, index, srtm_band):
        srtm_path=os.path.join(INDONESIA_DIR, self._image_info.iloc[index][IMG_PATH_HEADER]) #AD
        srtm_path = os.path.join(srtm_path, "auxiliary") #AD: MODIFIED
        srtm_unscaled = np.load(os.path.join(srtm_path, f'{srtm_band}.npy'))
        srtm_min, srtm_max = SRTM_SCALING[srtm_band]
        srtm_scaled = self._feature_scale(srtm_unscaled, srtm_min, srtm_max)
        return srtm_scaled 

    def _get_ndvi(self, rgb_image, ir_image):
        red_band_index = RGB_BANDS.index(RED_BAND)
        red_band = rgb_image[:, :, red_band_index].astype('float')
        ir_band_index = IR_BANDS.index(NIR_BAND)
        nir_band = ir_image[:, :, ir_band_index].astype('float')
        ndvi_unscaled = ((nir_band - red_band + 1e-6) /
                         (nir_band + red_band + 1e-6))
        # NDVI ranges between -1 and 1
        ndvi_scaled = self._feature_scale(
            ndvi_unscaled, -1, 1
        )

        return ndvi_scaled

    def _get_latlon(self, index):
        lat = float(self._image_info.iloc[index][LATITUDE_HEADER])
        lon = float(self._image_info.iloc[index][LONGITUDE_HEADER]) 
        return lat, lon
    
    def __getitem__(self, index):
        batch = {}
        batch['index'] = index
        lat, lon = self._get_latlon(index)
        batch['lat'] = lat
        batch['lon'] = lon
        return batch

    # Scale array to [0, 1] based on (min_val, max_val) values.
    # Then rescale to [0, 255] to match the other bands.
    def _feature_scale(self, arr, min_val, max_val, rescale=True):
        if arr is None: #AD
            arr = 0.0

        elif type(arr) == float and arr is not None: #AD
            arr= (arr- min_val) / (max_val - min_val)

        elif arr.size == 1 and arr != None: #AD
            arr = (arr - min_val) / (max_val - min_val)

        elif type(arr) != float and arr.size > 1: #AD
            if len(arr.shape) == 1:
                for i in range(arr.shape[0]):
                    if arr[i] == None:
                        arr[i] = 0

            elif len(arr.shape) == 2:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        if arr[i][j] == None:
                            arr[i][j] = 0

            elif len(arr.shape) == 3:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        for k in range(arr.shape[2]):
                            if arr[i][j][k] == None:
                                arr[i][j][k] = 0

            arr = (arr - min_val) / (max_val - min_val)
            
        if rescale:
            arr = (arr * 255) #AD
            arr= np.uint8(arr) #AD
            
        return arr
