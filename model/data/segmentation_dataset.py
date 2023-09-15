import torch
import numpy as np
import imgaug.augmenters as iaa
import albumentations as A
from PIL import Image, ImageDraw

from util.constants import *
from util.aux_features import LATE_FUSION_MODE
from .util import draw_img_roi, get_area
from .base_dataset import BaseDataset


class SegmentationDataset(BaseDataset):

    def _get_mask(self, image, label, polygon):
        mask = Image.new('L', image.shape[:2], LABEL_IGNORE_VALUE) #AD: image.shape[:2] = number of pixels
        # Use label + 1 to handle issue with iaa Affine filling
        # segmentation mask with 0s
        #AD: need to convert lat/lon polygon in draw_img_roi
        draw_img_roi(ImageDraw.Draw(mask), polygon, label.item() + 1) #AD: mask.show() will show the forest loss region on a picture
        # Add extra dimension to
        # (1) the beginning for number of segmentation masks (1)
        # (2) the end for class dim.
        # Required for SegmentationImgAug.
        mask = np.array(mask)[np.newaxis, :, :, np.newaxis]
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
        mask = mask.long().squeeze(0)

        area = get_area(polygon)
        # Make x a dictionary but not label or index
        # which simplifies downstream implementation
        batch = {"im_path": im_path,
                 "image": image,
                 "forest_loss": area,
                 "label_seg": mask,
                 "mask": keep}

        return batch

    def _apply_transform(self, t, image, mask):
        if issubclass(type(t), iaa.meta.Augmenter):
            image, mask = t(image=image, segmentation_maps=mask)
        # NOTE: this line is pretty hacky. Was not able
        # to figure out how to generalize this for albumentations
        # like the line above.
        elif isinstance(t, A.augmentations.transforms.RGBShift):
            image = t(image=image)['image']
        else:
            # necessary to avoid the negative stride
            image = t(np.ascontiguousarray(image))
            # necessary so ToTensor() doesn't convert to [0., 1.]
            mask = mask.astype(np.float32)
            # have to drop the extra dimension
            mask = t(np.ascontiguousarray(mask[0]))

        return image, mask

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        meta = self._get_example_meta(index)
        im_path = meta.get('im_path')
        orig_image, orig_mask = meta.get('image'), meta.get('mask')
        label, polygon = meta.get('label'), meta.get('polygon')
        image = orig_image
        mask = orig_mask
        for t in self._transforms:
            image, mask = self._apply_transform(t, image, mask)

        processed = self._post_process_example(im_path, image, mask, polygon)
        batch = {**batch, **processed}

        if self._no_augment_transforms is not None:
            # Apply deterministic transforms to get "original" image
            image2 = orig_image
            mask2 = orig_mask
            for t in self._no_augment_transforms:
                image2, mask2 = self._apply_transform(t, image2, mask2)
            batch2 = self._post_process_example(im_path, image2, mask2, polygon)
            batch["image_no_augment"] = batch2["image"]

        # Sample-wise valid indicator
        batch['valid'] = valid = batch['mask'].max() > 0
        label[~valid] = LOSS_IGNORE_VALUE
        batch['label_cls'] = label

        return batch
