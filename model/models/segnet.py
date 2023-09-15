import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F

from data.util import get_num_channels
from util.constants import *


class SegNet(nn.Module):

    ARCHITECTURE = {
        "UNet": smp.Unet,
        # Deeplab was just added to the smp package.
        # Use it if you have the latest version installed
        "DeepLabV3": smp.DeepLabV3,
        "FPN": smp.FPN
    }
    CONV1_KEY = {
        "ResNet18": ("resnet18", "encoder.conv1.weight", "imagenet"),
        "ResNet34": ("resnet34", "encoder.conv1.weight", "imagenet"),
        "ResNet50": ("resnet50", "encoder.conv1.weight", "imagenet"),
        "ResNet101": ("resnet101", "encoder.conv1.weight", "imagenet"),
        "ResNet152": ("resnet152", "encoder.conv1.weight", "imagenet"),
        "DenseNet121": ("densenet121", "encoder.conv1.weight", "imagenet"),
        "DenseNet161": ("densenet161", "encoder.conv1.weight", "imagenet"),
        "ResNeXt50": ("resnext50_32x4d", "encoder.conv1.weight", "instagram"),
        "SEResNeXt50": ("se_resnext50_32x4d", "encoder.conv1.weight", "instagram"),
        "ResNeXt101": ("resnext101_32x8d", "encoder.conv1.weight", "instagram"),
        "EfficientNet-b0": ("efficientnet-b0", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b1": ("efficientnet-b1", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b2": ("efficientnet-b2", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b3": ("efficientnet-b3", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b4": ("efficientnet-b4", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b5": ("efficientnet-b5", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b6": ("efficientnet-b6", "encoder._conv_stem.weight", "imagenet"),
        "EfficientNet-b7": ("efficientnet-b7", "encoder._conv_stem.weight", "imagenet")
    }

    def __init__(self, model_args):
        super().__init__()
        self.hparams = model_args
        self.model = model_args.get("model")
        self.backbone, self.conv1_key, self.encoder_weights = (
            self.CONV1_KEY[self.model]
        )
        self.pretrained = model_args.get("pretrained")
        self.model_fn = self.ARCHITECTURE[model_args.get("architecture")]
        self.input_forest_loss_seg = model_args.get("input_forest_loss_seg")
        self.input_forest_loss_cls = model_args.get("input_forest_loss_cls")
        self.input_forest_loss_size = (
            self.input_forest_loss_seg or
            self.input_forest_loss_cls is not None
        )
        self.use_classification_head = model_args.get("use_classification_head")
        if (self.input_forest_loss_size and
                model_args.get("architecture") != "UNet"):
            raise ValueError("Forest loss input only supported for UNet.")
        if self.use_classification_head and model_args.get(
                "architecture") != "UNet":
            raise ValueError("Classification head only supported for UNet.")

        if (self.input_forest_loss_cls is not None and
                not self.use_classification_head):
            raise ValueError("classification_head must be True " +
                             "when using input_forest_loss_size=" +
                             f"{self.input_forest_loss_cls}.")

        num_channels = get_num_channels(model_args)
        self.num_classes = len(model_args.get("labels"))

        assert num_channels >= 3

        # Set default seg model params if we want to change this down the line
        self.decoder_channels = [256, 128, 64, 32, 16]

        if model_args.get("architecture") == "UNet":
            self.model = self.model_fn(in_channels=num_channels,
                                       classes=self.num_classes,
                                       encoder_name=self.backbone,
                                       decoder_channels=self.decoder_channels)

        else:
            self.model = self.model_fn(in_channels=num_channels,
                                       classes=self.num_classes,
                                       encoder_name=self.backbone)

        if self.pretrained == "imagenet":
            # Copy the pre-trained weights from the red channel
            model_rgb = self.model_fn(in_channels=3,
                                      classes=self.num_classes,
                                      encoder_name=self.backbone,
                                      encoder_weights=self.encoder_weights)
            state_dict = model_rgb.state_dict()
            conv1_weight = state_dict[self.conv1_key]
            conv1_weight = torch.cat(
                [conv1_weight] + [conv1_weight[:, :1, :, :]] * (num_channels - 3),
                dim=1
            )
            state_dict[self.conv1_key] = conv1_weight
            self.model.load_state_dict(state_dict)

        self.set_segmentation_head()
        if self.use_classification_head:
            self.set_classification_head()

    def set_segmentation_head(self):
        """Override segmentation head only if using forest loss as input
        to the segmentation head."""
        if self.input_forest_loss_seg:
            # Remake segmentation head. Adapted from
            # https://github.com/qubvel/segmentation_models.pytorch/blob/master/
            # segmentation_models_pytorch/unet/model.py#L75-L80
            # In channels fixed in our experiments to 16, +1 for forest loss
            self.model.segmentation_head = smp.base.SegmentationHead(
                in_channels=self.decoder_channels[-1] + 1,
                out_channels=self.num_classes,
                kernel_size=3
            )

    def set_classification_head(self):
        """Set the classification head of the model.

        Creates a small sequence of fully connected layers when forest loss
        is input to this head. Otherwise, it uses the default classification
        head in segmentation_models.pytorch, which is a single layer.

        """
        dropout = None
        if self.input_forest_loss_cls == "dec_cls":
            in_channels = self.decoder_channels[-1]
            # Adapted from
            # https://github.com/qubvel/segmentation_models.pytorch/blob/master/
            # segmentation_models_pytorch/base/heads.py#L14-L24
            self.pool = nn.AdaptiveAvgPool2d(1)
            flatten = smp.base.modules.Flatten()
            if dropout:
                dropout = nn.Dropout(p=dropout, inplace=True)
            else:
                dropout = nn.Identity()
            linear = nn.Linear(in_channels + 1, 20)
            activation = nn.ReLU()
            linear2 = nn.Linear(20, 10)
            activation2 = nn.ReLU()
            classifier = nn.Linear(10, self.num_classes)

            self.model.classification_head = nn.Sequential(
                flatten, dropout,
                linear, activation,
                linear2, activation2,
                classifier
            )
        elif self.input_forest_loss_cls == "enc_cls":
            self.pool = nn.AdaptiveAvgPool2d(1)
            flatten = smp.base.modules.Flatten()
            dropout = nn.Dropout(p=0.2, inplace=True)
            in_channels = self.model.encoder.out_channels[-1]
            linear = nn.Linear(in_channels + 1, self.num_classes)
            self.model.classification_head = nn.Sequential(
                flatten, dropout, linear
            )
        else:
            in_channels = self.model.encoder.out_channels[-1]
            self.model.classification_head = smp.base.ClassificationHead(
                in_channels=in_channels, classes=self.num_classes, dropout=dropout)

    def segmentation_forward(self, decoder_output, forest_expanded):
        """Segmentation forward pass for models that input forest loss size."""
        if self.input_forest_loss_seg:
            # Add forest loss size as input to the segmentation head.

            # Concatenate forest loss features
            # (B, 1, 1, 1) -> (B, 1, 224, 224)
            img_size = decoder_output.shape[2]
            forest_repeated = forest_expanded.repeat(
                (1, 1, img_size, img_size)
            )
            seg_head_input = torch.cat([decoder_output, forest_repeated],
                                       dim=1)

            masks = self.model.segmentation_head(seg_head_input)

        else:
            # Forest loss not input to segmentation head.
            # No need to change this forward pass.
            masks = self.model.segmentation_head(decoder_output)

        return masks

    def classification_forward(self, image_features, decoder_output,
                               forest_expanded, forest_loss_region_mask):
        """Classification forward pass for models that input forest loss size."""
        if self.input_forest_loss_cls in ["dec_cls", "enc_cls"]:
            if self.input_forest_loss_cls == "dec_cls":
                # Change boolean mask of shape (B, 224, 244)
                # to float mask of shape (B, 1, 224, 224)
                forest_loss_mask = forest_loss_region_mask.unsqueeze(1).float()
                # Only pool output "pixels" that are within the
                # forest loss region
                features = decoder_output * forest_loss_mask
                pooled = self.pool(features)

            else:
                pooled = self.pool(image_features[-1])

            cls_head_input = torch.cat([pooled, forest_expanded], dim=1)
            logits = self.model.classification_head(cls_head_input)

        else:
            # Forest loss not input to the classification head.
            # No need to change this forward pass.
            logits = self.model.classification_head(image_features[-1])

        return logits

    def forward(self, x, image_key='image', forest_loss_region_mask=None):
        if not self.input_forest_loss_size:
            # Forest loss is not input to either head.
            # Keep the forward pass the same.
            return self.model(x[image_key])

        else:
            # Modify forward pass to support forest loss area as input
            # Adapted from
            # https://github.com/qubvel/segmentation_models.pytorch/blob/
            # master/segmentation_models_pytorch/base/model.py#L13-L24

            # Get image features
            image_features = self.model.encoder(x[image_key])
            decoder_output = self.model.decoder(*image_features)

            # Prepare forest loss features
            # (B,) -> (B, 1, 1, 1)
            forest_expanded = x['forest_loss'].float()[(..., ) + (None, ) * 3]

            masks = self.segmentation_forward(decoder_output, forest_expanded)
            if self.use_classification_head:
                logits = self.classification_forward(
                    image_features, decoder_output,
                    forest_expanded, forest_loss_region_mask
                )
                return masks, logits
            else:
                return masks
