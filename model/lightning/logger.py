import os 
import torch
import logging
import torch.nn.functional as F
import torchvision.transforms as T
import imgaug.augmenters as iaa
from torch.distributions import Categorical

import pickle 
from collections import defaultdict
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
from tqdm import tqdm 
import numpy as np
from matplotlib import pyplot as plt

from util.constants import *
from data.util import draw_img_roi
from data.transforms import *
from util.merge_scheme import DATASET_MERGE_SCHEMES
from eval.metrics import *


class TFLogger:

    def log_results(self,
                    merge_scheme,
                    indices,
                    y_true,
                    y_pred,
                    y_logits,
                    y_pred_px,
                    losses,
                    labels,
                    label_names,
                    area=None):

        save_path = os.path.join(self.hparams['default_save_path'], 
                                 merge_scheme)
        os.makedirs(save_path, exist_ok=True)
        test_dataset = self.test_dataloader().dataset

        if self.hparams['log_images']:
            self.log_images(test_dataset,
                            indices,
                            y_true,
                            y_pred,
                            y_logits,
                            losses,
                            labels,
                            label_names,
                            merge_scheme,
                            y_pred_px)
        
        y_true_numpy = y_true.detach().cpu().numpy()
        y_pred_numpy = y_pred.detach().cpu().numpy()

        save_metrics(y_true_numpy,
                     y_pred_numpy,
                     labels=labels,
                     label_names=label_names,
                     save_path=save_path,
                     dataset=test_dataset,
                     indices=indices,
                     do_plot_sc_count_acc=False,
                     area=area)

    def log_images(self, 
                   dataset, 
                   indices, 
                   y_true, 
                   y_pred,
                   y_logits,
                   losses,
                   labels, 
                   label_names, 
                   merge_scheme,
                   y_pred_px
                   ):
        """
        Log images and optionally detection to tensorboard
        :param logger: [Tensorboard Logger] Tensorboard logger object.
        :param images: [tensor] batch of images indexed
                    [batch, channel, size1, size2]
        """
        use_landsat =  's2' not in self.hparams['data_version']
        images = get_images_from_indices(dataset,
                                         indices, 
                                         self.hparams['resize'],
                                         self.hparams['eval_img_option'],
                                         use_landsat)
        images = prep_images_for_logging(images, 
                                         self.hparams['pretrained'])
        images = images.cpu()
        images_by_type = get_images_by_type(images, 
                                            y_true, 
                                            y_pred,
                                            labels,
                                            label_names)
        
        cmap = DATASET_MERGE_SCHEMES[merge_scheme]['color_scheme']
        img_h, img_w = images.size()[-2:]
        color_legend = get_color_legend(img_h, img_w)

        loss_outlier_imgs, loss_outlier_indices = \
            get_loss_outlier_images(images.detach().clone(), losses, y_true, y_pred, label_names)

        if self.hparams['segmentation']:
            loss_outlier_imgs = process_segmentation_images(
                loss_outlier_imgs, loss_outlier_indices, y_pred_px, cmap, color_legend
            )

        self.logger.experiment.add_images('High loss', loss_outlier_imgs)

        if y_logits is not None:
            low_entropy_imgs, low_entropy_indices, high_entropy_imgs, high_entropy_indices\
                = get_entropy_outlier_images(images, y_logits, y_true, y_pred, label_names)
            
            if self.hparams['segmentation']:
                high_entropy_imgs = process_segmentation_images(
                    high_entropy_imgs, high_entropy_indices, y_pred_px, cmap, color_legend
                )
                low_entropy_imgs = process_segmentation_images(
                    low_entropy_imgs, low_entropy_indices, y_pred_px, cmap, color_legend
                )
            self.logger.experiment.add_images('High entropy', high_entropy_imgs)
            self.logger.experiment.add_images('Low entropy', low_entropy_imgs)

        for tag, imgs_dict in images_by_type.items():
            imgs = imgs_dict.get('imgs')
            idxs = imgs_dict.get('idxs')
            logging.info(f"Found {imgs.size()[0]} images for [{tag}]")

            if self.hparams['segmentation']:
                imgs = process_segmentation_images(
                    imgs, idxs, y_pred_px, cmap, color_legend
                )

            self.logger.experiment.add_images(tag, imgs)


def process_segmentation_images(images, indices, y_pred_px, cmap, color_legend):
    images_seg = torch.stack(
        [seg_pred_overlay(img, y_pred_px[indices[i].item()], cmap) 
         for i, img in enumerate(images)]
    )
    # concatenate images along width dimension rather than adding two separate
    # so they stay on the same row 
    images_seg_concatenated = torch.cat([images, images_seg], dim=3)
    # Add the color legend to beginning and end of tab
    images_seg_concatenated = torch.cat(
        [color_legend, images_seg_concatenated, color_legend], dim=0
    )

    return images_seg_concatenated


def get_color_legend(img_h, img_w):
    color_legend = Image.open(SEG_LEGEND_PATH).convert('RGB')
    color_legend = T.Resize((img_h, img_w))(color_legend)
    color_legend = T.ToTensor()(color_legend).unsqueeze(0)
    # add empty space to match example image dims 
    color_legend = torch.cat([color_legend, torch.zeros(color_legend.size())], dim=3)
    return color_legend

def seg_pred_overlay(image, preds, cmap, alpha=0.2): 
    preds = preds.detach().cpu().numpy()
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    cmap_fn = plt.get_cmap(cmap)
    pred_overlay = cmap_fn(preds.astype(np.uint8))[:, :, :3]
    overlay = (1. - alpha) * image.astype(np.float32) + alpha * pred_overlay.astype(np.float32)
    overlay -= np.amin(overlay)
    overlay /= (np.amax(overlay) + 1e-7)
    overlay *= 255 
    overlay = np.uint8(overlay)
    return T.ToTensor()(overlay)

def get_loss_outlier_images(images, losses, y_true, y_pred, label_names, top_n=50):
    high_loss_indices = torch.argsort(-losses)
    losses = losses[high_loss_indices][:top_n]
    images = images[high_loss_indices][:top_n]
    y_true = y_true[high_loss_indices][:top_n]
    y_pred = y_pred[high_loss_indices][:top_n]

    for idx in range(len(images)):
        gt = label_names[y_true[idx].item()]
        pred = label_names[y_pred[idx].item()]
        loss = losses[idx].item()
        text = f'Pred:{pred}\nGT:{gt}\nLoss:{round(loss,2)}' 
        images[idx] = draw_on_image(images[idx], text)
    
    return images, high_loss_indices


def get_entropy_outlier_images(images, y_logits, y_true, y_pred, label_names, top_n=50):
    probs = torch.softmax(y_logits, dim=1)
    entropies = Categorical(probs).entropy()
    entropy_sorted_inds = torch.argsort(entropies)

    low_entropy_indices = entropy_sorted_inds[:top_n]
    high_entropy_indices = entropy_sorted_inds[-top_n:]
    stat2indices = {
        'low': low_entropy_indices,
        'high': high_entropy_indices
    }

    stat2images = {}

    for stat, indices in stat2indices.items():
        image_list = []
        for index in indices:
            entropy_value = entropies[index].item()
            if y_true[index] != torch.tensor([-100],  device='cuda:0'): #AD
                gt = label_names[y_true[index].item()]
            pred = label_names[y_pred[index].item()]
            image = images[index]
            text = (
                f'Pred:{pred}\n' + 
                f'GT:{gt}\n' + 
                f'Entropy:{round(entropy_value,2)}'
            )
            image = draw_on_image(image, text)
            image_list.append(image)
        stacked_image = torch.stack(image_list)
        stat2images[stat] = stacked_image

    low_entropy_imgs = stat2images['low']
    high_entropy_imgs = stat2images['high']

    return low_entropy_imgs, low_entropy_indices, high_entropy_imgs, high_entropy_indices


def get_images_from_indices(dataset, indices, resize, img_option, use_landsat):
    pil_to_tensor = T.ToTensor()
    images = []
    logging.info('Processing images...')
    for idx in tqdm(indices):
        idx = idx.item()
        image_info = dataset.get_image_info(idx)
        img_path = dataset._get_image_path(idx, img_option)
        img = dataset._get_image(img_path)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        poly_dir=os.path.join(INDONESIA_DIR, image_info[IMG_PATH_HEADER]) #AD
        shape = pickle.load(open(os.path.join(poly_dir,'forest_loss_region.pkl'), 'rb')) #AD
        draw_img_roi(draw, shape, label=None)
        transforms = get_transforms(
            resize=resize,
            spatial_augmentation="none",
            pixel_augmentation="none",
            is_training=False,
            use_landsat=use_landsat
        )
        img = np.array(img)
        for t in transforms:
            if isinstance(t, iaa.Augmenter):
                img = t(image=img)
            else:
                # necessary to avoid the negative stride
                img = t(np.ascontiguousarray(img))
        img = T.ToPILImage()(img)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default() 
        year = image_info[YEAR_HEADER]
        num_sc = len(os.listdir(os.path.join(INDONESIA_DIR, image_info['example_path'], 'images','visible'))) #AD
        text = (f'{idx}/{year}/{num_sc}')
        draw.text((10, img.height-20), text, (255,255,255), font=font)
        lat, lon = image_info[LATITUDE_HEADER], image_info[LONGITUDE_HEADER]
        text = (f'{lat:.4f}/{lon:.4f}')
        draw.text((10, img.height-40), text, (255,255,255), font=font)
        images.append(pil_to_tensor(img))
    return torch.stack(images).cuda() 


def get_images_by_type(images, 
                       y_true, 
                       y_pred, 
                       labels,
                       label_names):
    images_by_type = {}
    for idx, label in enumerate(labels):
        label_name = label_names[idx]
        
        gt_indices = torch.nonzero(y_true == label, as_tuple=False)
        tp_indices = ((y_true == label) & (y_pred == label)).nonzero()
        fp_indices = ((y_true != label) & (y_pred == label)).nonzero()
        fn_indices = ((y_true == label) & (y_pred != label)).nonzero()

        gt_images = images[gt_indices].squeeze(1)
        if len(gt_images) > 0:
            images_by_type[f'{label_name} GT'] = {
                'imgs': gt_images, 
                'idxs': gt_indices
            }
        
        tp_images = images[tp_indices].squeeze(1)
        if len(tp_images) > 0:
            images_by_type[f'{label_name} TP'] = {
                'imgs': tp_images, 
                'idxs': tp_indices
            }

        fp_images = []
        
        for idx in fp_indices:
            true_label = y_true[idx]
            if true_label != torch.tensor([-100],  device='cuda:0'):
                true_label_name = label_names[true_label]
                image = draw_on_image(images[idx].squeeze(0), true_label_name)
                fp_images.append(image)

        if len(fp_images) > 0:
            images_by_type[f'{label_name} FP'] = {
                'imgs': torch.stack(fp_images),
                'idxs': fp_indices
            }

        fn_images = []
        for idx in fn_indices:
            pred_label = y_pred[idx]
            pred_label_name = label_names[pred_label]
            image = draw_on_image(images[idx].squeeze(0), pred_label_name)
            fn_images.append(image)

        if len(fn_images) > 0:
            images_by_type[f'{label_name} FN'] = {
                'imgs': torch.stack(fn_images),
                'idxs': fn_indices
            }
    
    return images_by_type


def draw_on_image(image, text):
    image = image.cpu()
    image = T.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default() 
    draw.text((0, 0), text, (255,255,255), font=font)
    return T.ToTensor()(image)


def prep_images_for_logging(images, pretrained="none"):
    """
    Prepare images to be logged
    :param images: [tensor] batch of images indexed
                   [channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :param size: [int] new size of the image to be rescaled
    :return: images that are reversely normalized
    """
    if pretrained == "imagenet":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    images = normalize_inverse(images, mean, std)
    return images


def normalize_inverse(images, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Reverse Normalization of Pytorch Tensor
    :param images: [tensor] batch of images indexed
                   [batch, channel, size1, size2]
    :param mean: [list] mean values used to normalize images
    :param std: [list] standard deviation values used to normalize images
    :return: images that are reversely normalized
    """
    mean_inv = torch.FloatTensor(
        [-m/s for m, s in zip(mean, std)]).view(1, 3, 1, 1)
    std_inv = torch.FloatTensor([1/s for s in std]).view(1, 3, 1, 1)
    if torch.cuda.is_available():
        mean_inv = mean_inv.cuda()
        std_inv = std_inv.cuda()
    return (images - mean_inv) / std_inv
