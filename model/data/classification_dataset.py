import numpy as np
import albumentations as A
import imgaug.augmenters as iaa
from PIL import Image, ImageDraw

from util.constants import *
from .base_dataset import BaseDataset
from .util import draw_img_roi, get_area


class ClassificationDataset(BaseDataset):

    def _post_process_example(self, image, polygon):
        # Make x a dictionary but not label or index
        # which simplifies downstream implementation
        x_dict = {"image": image}

        area = get_area(polygon)
        x_dict["forest_loss"] = area

        return x_dict

    def _apply_transform(self, t, image):
        if issubclass(type(t), iaa.meta.Augmenter):
            image = t(image=image)
        # NOTE: this line is pretty hacky. Was not able 
        # to figure out how to generalize this for albumentations
        # like the line above.
        elif isinstance(t, A.augmentations.transforms.RGBShift):
            image = t(image=image)['image']
        else:
            # necessary to avoid the negative stride
            image = t(np.ascontiguousarray(image))

        return image 

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        meta = self._get_example_meta(index)
        image, label, polygon = meta.get('image'), meta.get('label'), meta.get('polygon')

        if self._transforms:
            for t in self._transforms:
                image = self._apply_transform(t, image)

        processed = self._post_process_example(image, polygon)
        batch = {**batch, **processed} 
        batch['label_cls'] = label
        batch['index'] = meta.get('index')
        return batch


class ClassificationAddedBandsDataset(BaseDataset):

    def _get_image(self, index):
        label = self._get_label(index)

        im_path = self._image_info.iloc[index][IMG_PATH_HEADER]
        rgb = np.array(Image.open(im_path).convert('RGB'))
        all_bands = [rgb]

        # NOTE: ir not up-to-date with multiple downloaded images and segmentation. 
        # should be updated if we plan to use it in the future 
        if self._ir:
            nir = np.load(im_path.replace("png", "npy")).astype(np.uint8)
            nir = np.transpose(nir, (1, 2, 0))
            all_bands.append(nir)

        if self._masked:
            polygon = self._get_polygon(index)
            mask = Image.new('L', rgb.shape[:2], 0)  # default value is 0
            draw_img_roi(ImageDraw.Draw(mask), polygon, 1)
            all_bands.append(np.array(mask)[...,np.newaxis])

        image = np.concatenate(all_bands, axis=-1)
        return image
