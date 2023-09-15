from util import *
from imageio import imsave
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def unnormalizer(mean, std, tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean: mean to use for unnormalizing
        std: std to use for unnormalizing
    Returns:
        Tensor: Normalized image.
    """

    assert(len(tensor.size()) == 3),\
        ('Image tensor should have 3 dimensions. ' +
            f'Got tensor with {len(tensor.size())} dimensions.')

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor


def unnormalize_image_tensor(img_tensor, mean, std):
    """Unnormalize a PyTorch Tensor seen by the model into a NumPy array of
    pixels fit for visualization. If using raw Hounsfield Units,
    window the input.
    Args:
        img_tensor: Normalized tensor using mean and std.
                    Tensor with pixel values in range (-1, 1).
    Returns:
        unnormalized_img: Numpy ndarray with values between 0 and 1.
    """

    # Make a copy, as we don't want to unnormalize in place.
    # The unnormalizer affects the input tensor.
    img_tensor_copy = img_tensor.clone()
    unnormalized_img = unnormalizer(mean, std, img_tensor_copy)
    unnormalized_img = unnormalized_img.cpu().float().numpy()

    return unnormalized_img


def _normalize_png(img):
    """Normalizes img to be in the range 0-255."""
    img -= np.amin(img)
    img /= (np.amax(img) + 1e-7)
    img *= 255
    return img


def add_heat_map(original_image, intensities_np, alpha_img=0.33,
                 color_map='magma', normalize=True):
    """Add a CAM heat map as an overlay on a PNG image.
    Args:
        original_image: Pixels to add the heat map on top of.
                        Must be in range (0, 1).
        intensities_np: Intensity values for the heat map.
                        Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map.
                   Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range
                   exactly from 0 to 1.
    Returns:
        Original pixels with heat map overlaid.
    """
    assert(np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0)
    assert(np.max(original_image) <= 1 and np.min(original_image) >= 0),\
        (f'np.max: {np.max(original_image)} and ' +
         f'np.min: {np.min(original_image)}')
    color_map_fn = plt.get_cmap(color_map)
    

    if normalize:
        # Returns pixel values between 0 and 255
        intensities_np = _normalize_png(intensities_np)
    else:
        intensities_np *= 255

    # Get heat map (values between 0 and 1
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
   
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = (alpha_img * original_image.astype(np.float32) +
               (1. - alpha_img) * heat_map.astype(np.float32))

    new_img = np.uint8(_normalize_png(new_img))

    return new_img


def normalize_gradients(grads):
    return grads / (torch.norm(grads).item() + 1e-5)


def write_batch_cams(batch_x,
                     batch_y,
                     batch_y_hat,
                     batch_cams,
                     idxs,
                     save_dir,
                     dataset,
                     zoomed_regions,
                     save_images=True,
                     ):
    """Write the cams created for the batch to output files."""
    master_filename = save_dir / "master.csv"

    # Create directories for saving CAMs.
    numpy_dir = save_dir / "numpy"
    numpy_dir.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(batch_y_hat)
    row_list = []

    for (input_, target, y_hat, cam, prob, index) in (
            zip(batch_x, batch_y, batch_y_hat, batch_cams, probs, idxs)):

        pred = torch.argmax(y_hat)
        prob_print = float(prob[pred.item()])
        _, height, width = input_.shape
        # Upscale the CAM to the dimensions of the original image.
        cam = cam[pred]
        resized_cam = cv2.resize(cam, (height, width))

        index = index.item()
        np_filename = numpy_dir / f"{index}_{target}_p{prob_print:.3f}.npy"
        np.save(np_filename, resized_cam)

        cam_filename = orig_filename = np.nan
        if save_images:
            # Draw shape on original image
            image_info = dataset.get_image_info(index)
            orig_img = Image.open(image_info[IMG_PATH_HEADER])
            draw = ImageDraw.Draw(orig_img)
            shape = pickle.load(open(image_info[SHAPEFILE_HEADER], 'rb'))
            draw_img_roi(draw, shape)
            # Use val crops on image so CAM overlay matches
            if zoomed_regions:
                transform_list = [
                        T.Resize((SMALL_RESIZE_HEIGHT, SMALL_RESIZE_WIDTH)),
                        T.CenterCrop((SMALL_INPUT_HEIGHT, SMALL_INPUT_WIDTH)),
                        T.ToTensor()
                    ]
            else:
                transform_list = [
                        T.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
                        T.CenterCrop((INPUT_HEIGHT, INPUT_WIDTH)),
                        T.ToTensor()
                    ]
            transforms = T.Compose(transform_list)
            img_tensor = transforms(orig_img)
            input_img = np.moveaxis(img_tensor.numpy(), 0, 2)
            # Add the CAM to the original image.
            cam_img = add_heat_map(input_img, resized_cam, alpha_img=0.8,
                                   normalize=True)
            if target == pred:  # correct prediction
                error = 'TP'
            else:
                error = 'FN'

            target_class_dir = save_dir / f'class_{target}'
            error_dir = target_class_dir / error
            target_class_dir.mkdir(parents=True, exist_ok=True)
            error_dir.mkdir(parents=True, exist_ok=True)
            cam_filename, orig_filename = save_images_to_dir(error_dir, index, target.item(), prob_print, cam_img, orig_img)

            if error == 'FN':
                pred_class_dir = save_dir / f'class_{pred}'
                pred_error_dir = pred_class_dir / 'FP'
                pred_class_dir.mkdir(parents=True, exist_ok=True)
                pred_error_dir.mkdir(parents=True, exist_ok=True)
                save_images_to_dir(pred_error_dir, index, target.item(), prob_print, cam_img, orig_img)
                
        row = {INDEX: index, TARGET: target.item(), PRED: pred.item(),
               PROB: prob_print, NUMPY_PATH: np_filename,
               CAM_PATH: cam_filename, IMAGE_PATH: orig_filename}
        row_list.append(row)

    # Pandas dataframe storing file paths and outputs.
    file_df = pd.DataFrame(row_list)
    file_df.to_csv(master_filename, mode='a+', header=True)


def save_images_to_dir(error_dir, index, target, prob_print, cam_img, orig_img):
    cam_img = Image.fromarray(cam_img)
    cam_filename = (error_dir /
                        f'{index}_{target}_p{prob_print:.3f}.png')
    orig_filename = (error_dir /
                        f'orig_{index}_{target}_p{prob_print:.3f}.png')
    cam_img.save(cam_filename)
    orig_img.save(orig_filename)
    return cam_filename, orig_filename
