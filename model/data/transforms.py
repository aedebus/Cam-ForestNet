import imgaug.augmenters as iaa
import torchvision.transforms as T
import albumentations as A

from util.constants import *


# Dataset constants
# Hao: input size should be divisible by 32 and more than 64 pixels.
# https://github.com/qubvel/segmentation_models.pytorch/issues/18
RESIZE_SIZE = 300  # TODO: These dims should be cross-validated
RESIZE_SIZE_SENT2 = 450
INPUT_SIZE = 224
INPUT_SIZE_SENT2 = 320
AGGRESSIVE_INPUT_SIZE = 160
AGGRESSIVE_INPUT_SIZE_SENT2 = 256
SMALL_RESIZE_SIZE = 150
SMALL_RESIZE_SIZE_SENT2 = 240

SPATIAL_AUGMENTATIONS = {
    "none": [iaa.Identity()],
    "flip": [iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)])],
    "affine":
    [iaa.SomeOf(2, [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Rot90(1, 3)
    ])],
    "cloud":
    [iaa.Sequential([
        iaa.SomeOf(2, [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(1, 3)]),
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.Clouds(),
                          iaa.Fog(),
                          iaa.Snowflakes()]))])],
    "affine_cloud":
    [iaa.Sequential([
        iaa.SomeOf(2, [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                cval=LABEL_IGNORE_VALUE,
                mode='constant'),
            iaa.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                cval=LABEL_IGNORE_VALUE,
                mode='constant'),
            iaa.Rot90(1, 3)]),
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.Clouds(),
                          iaa.Fog(),
                          iaa.Snowflakes()]))])],
    "sap":
    [iaa.SomeOf(2, [
        iaa.SaltAndPepper([0.0, 0.01]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            rotate=(-5, 5),
            cval=LABEL_IGNORE_VALUE,
            mode='constant')
    ])],
    "aggressive":
    [iaa.SomeOf(2, [
        iaa.SaltAndPepper([0.0, 0.1]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            rotate=(-5, 5),
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            shear=(-5, 5),
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.ElasticTransformation(alpha=(0.0, 40.0), sigma=(4.0, 8.0))
    ])]
}

# These intervals were validated by looking at the differences in channel
# pixel values between hazy and non-hazy images for a given example. 
R_SHIFT_INTERVAL = (6, 16)
G_SHIFT_INTERVAL = (14, 25)
B_SHIFT_INTERVAL = (34, 46)
HAZY_AUGMENTATION = A.RGBShift(R_SHIFT_INTERVAL,
                               G_SHIFT_INTERVAL,
                               B_SHIFT_INTERVAL,
                               p=0.2)


PIXEL_AUGMENTATIONS = {
    "none": [iaa.Identity()],
    "hazy": [HAZY_AUGMENTATION],
    "all": 
    [
        iaa.Sometimes(0.2,
            iaa.SomeOf((0, 2), [
            iaa.MultiplyBrightness((0.9, 1.1)),
            iaa.LogContrast((0.9, 1.1))
        ])),
        HAZY_AUGMENTATION
    ]
}

TOTENSOR_TRANSFORM = [T.ToTensor()]
TOIMAGE_TRANSFORM = T.ToPILImage()

DEPLOY_RESIZE = iaa.Resize((INPUT_SIZE, INPUT_SIZE))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_NET_TRANSFORMS = [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def get_resize_crop_transform(resize, is_training, use_landsat):
    if use_landsat:
        input_size = INPUT_SIZE
        aggressive_input_size = AGGRESSIVE_INPUT_SIZE
        resize_size = RESIZE_SIZE
        small_resize_size = SMALL_RESIZE_SIZE
    else:
        input_size = INPUT_SIZE_SENT2
        aggressive_input_size = AGGRESSIVE_INPUT_SIZE_SENT2
        resize_size = RESIZE_SIZE_SENT2
        small_resize_size = SMALL_RESIZE_SIZE_SENT2

    resize_crop_transform_dict = {
        "none": {
            True: [
                iaa.CropToFixedSize(input_size, input_size)
            ],
            False: [
                iaa.CenterCropToFixedSize(input_size, input_size)
            ]
        },
        "aggressive": {
            True: [
                iaa.CropToFixedSize(
                    aggressive_input_size, aggressive_input_size
                )
            ],
            False: [
                iaa.CenterCropToFixedSize(
                    aggressive_input_size, aggressive_input_size
                )
            ]
        },
        "small": {
            True: [
                iaa.Sequential([
                    iaa.Resize((resize_size, resize_size)),
                    iaa.CropToFixedSize(input_size, input_size)
                ])
            ],
            False: [
                iaa.Sequential([
                    iaa.Resize((resize_size, resize_size)),
                    iaa.CenterCropToFixedSize(input_size, input_size)
                ])
            ]
        }
    }

    return resize_crop_transform_dict[resize][is_training]


def get_transforms(resize,
                   spatial_augmentation,
                   pixel_augmentation,
                   is_training,
                   use_landsat):
    transforms = []

    # Resizing
    transforms += get_resize_crop_transform(
        resize=resize, is_training=is_training, use_landsat=use_landsat
    )

    # Spatial and pixel augmentation
    if is_training:
        transforms += SPATIAL_AUGMENTATIONS[spatial_augmentation]
        transforms += PIXEL_AUGMENTATIONS[pixel_augmentation]

    # ToTensor
    transforms += TOTENSOR_TRANSFORM

    return transforms
