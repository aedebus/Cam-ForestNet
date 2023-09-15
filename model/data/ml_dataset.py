import torch
import numpy as np
import imgaug.augmenters as iaa
import albumentations as A
from PIL import Image, ImageDraw

from util.constants import *
from .util import draw_img_roi, get_area
from .segmentation_dataset import SegmentationDataset 
from .base_dataset import BaseDataset


class MachineLearningDataset(SegmentationDataset):
    
    def _get_label(self, index):
        label = int(self._image_info.iloc[index][LABEL_HEADER])
        label = self._label_mapping[label]
        label = np.uint8(label)
        return label
    """ 
    def _get_example_meta(self, index):
        im_path = self._get_image_path(index, self._img_option)
        image = self._get_image(im_path)
        label = self._get_label(index)
        polygon = self._get_polygon(index)
        mask = self._get_mask(image, label, polygon)
        return image, label, polygon, mask
    """
    def _get_mask(self, image, label, polygon):
        mask = Image.new('L', image.shape[:2], LABEL_IGNORE_VALUE)
        # Use label + 1 to handle issue with iaa Affine filling
        # segmentation mask with 0s
        draw_img_roi(ImageDraw.Draw(mask), polygon, label + 1) #AD: mask.show() will show the forest loss region on a picture 
        mask = np.array(mask).astype(np.int32)
        return mask

    def _post_process_example(self, 
                              im_path,
                              image, 
                              mask, 
                              polygon):
        mask[mask == LABEL_IGNORE_VALUE] = LOSS_IGNORE_VALUE
        # Affine transform fills with 0s - ignore these values
        mask[mask == 0] = LOSS_IGNORE_VALUE

        # Shift label back by 1
        keep = mask != LOSS_IGNORE_VALUE
        mask[keep] -= 1

        area = get_area(polygon)
        # Make x a dictionary but not label or index
        # which simplifies downstream implementation
        batch = {"im_path": im_path,
                 "image": image,
                 "forest_loss": area,
                 "label_seg": mask,
                 "mask": keep}

        return batch

    def __getitem__(self, index):
        batch = BaseDataset.__getitem__(self, index)
        meta = self._get_example_meta(index)
        im_path = meta.get('im_path')
        image, label = meta.get('image'), meta.get('label')
        polygon, mask = meta.get('polygon'), meta.get('mask')         
        processed = self._post_process_example(im_path, image, mask, polygon)
        batch = {**batch, **processed}

        # Sample-wise valid indicator
        batch['valid'] = valid = batch['mask'].max() > 0
        batch['label_cls'] = label
        batch['pixel'] = batch['image'][batch['mask'], :] 
        batch['n_pixel'] = batch['pixel'].shape[0]             

        return batch