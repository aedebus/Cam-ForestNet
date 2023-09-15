import torch
from torch import nn
import numpy as np 

from .segnet import SegNet
from eval.metrics import segmentation_to_classification
from util import constants as C
from util.aux_features import *


class FusionNet(SegNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detach = self.hparams.get("late_fusion_detach", False)
        self.stats = self.hparams.get("late_fusion_stats", False)
        self.latlon = self.hparams.get("late_fusion_latlon", False)
        self.use_aux = self.hparams.get("late_fusion_aux_feats", False)
        self.embedding_dim = self.hparams.get("late_fusion_embedding_dim", 128)
        self.dropout_rate = self.hparams.get("late_fusion_dropout", 0.2)
        self.image_stats = self.hparams.get("late_fusion_image_stats", False)
         
        # TODO(@hao): The following are not in the CLI args.
        # Add to args after the workshop paper submission.
        self.N = self.hparams.get("geo_embedding_dim", 5) if self.latlon else 0
        self.max_wavelength = self.hparams.get("geo_embedding_wave_length", 10)
        # Don't think we actually need this but just in case. 
        self.late_fusion_aux_rgb = self.hparams.get("late_fusion_aux_rgb", False)
        
        self.aux_features = []
        if self.use_aux:
            self.aux_features = AUX_FEATURES[LATE_FUSION_MODE][self.late_fusion_aux_rgb]
        
        num_latlon_features = self.N * 4
        num_logit_features = self.num_classes * (1 + 3 * self.stats) + 4 * self.num_classes * self.image_stats
        num_aux_features =  self.use_aux * len(self.aux_features)
        
        self.input_dim = num_latlon_features + num_logit_features + num_aux_features
        self.output_dim = self.num_classes 
                
        self.fc = nn.Sequential(
                    nn.Linear(self.input_dim, self.embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.embedding_dim, self.output_dim)
                )

    def latlon_encoding(self, lat, lon):
        lat = self.positional_encoding(
            lat, C.INDONESIA_LAT_MIN, C.INDONESIA_LAT_MAX)
        lon = self.positional_encoding(
            lon, C.INDONESIA_LON_MIN, C.INDONESIA_LON_MAX)
        return torch.cat([lat, lon], dim=-1)

    def positional_encoding(self, value, value_min, value_max):
        """
        Geo/positional encoding following:
        1. Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
        2. Improving Urban-Scene Segmentation via
        Height-driven Attention Networks: https://arxiv.org/abs/2003.05128.pdf
        """

        value = (value - value_min) / (value_max - value_min)
        encoding = []
        for i in range(self.N):
            s = torch.sin(value / (self.max_wavelength ** (i / self.N)))
            c = torch.cos(value / (self.max_wavelength ** (i / self.N)))
            encoding.append(s)
            encoding.append(c)
        return torch.stack(encoding, dim=-1)

    def get_seg_logits(self, batch):
        return super().forward(batch)
        
    def forward(self, batch):
        mask = batch['mask']
        lat, lon = batch['lat'], batch['lon']
        valid = batch['valid']
        logits = self.get_seg_logits(batch)
                
        y_logit_flatten = logits.view(
            logits.size(0), logits.size(1), -1)    
        mask_flatten = mask.view(
            mask.size(0), mask.size(1), -1)
        
        # Logit stats for full image            
        y_logit_mean = y_logit_flatten.mean(dim=-1)
        y_logit_std = y_logit_flatten.std(dim=-1)
        y_logit_min, _ = y_logit_flatten.min(dim=-1)
        y_logit_max, _ = y_logit_flatten.max(dim=-1)
        
        # Logit stats for image region
        y_logit_fl_mean = (y_logit_flatten * mask_flatten).sum(dim=-1) / (mask_flatten.sum(dim=-1) + 1)
        y_logit_fl_std = (torch.pow(y_logit_flatten - y_logit_fl_mean.unsqueeze(-1), 2) * mask_flatten).sum(dim=-1) / (mask_flatten.sum(dim=-1) + 1)
        y_logit_fl_std = torch.sqrt(y_logit_fl_std)
        
        y_logit_min_placeholder = y_logit_flatten.clone()
        y_logit_min_placeholder[~mask_flatten.expand_as(y_logit_flatten)] = torch.tensor(np.inf).type_as(logits)
        y_logit_fl_min, _ = y_logit_min_placeholder.min(dim=-1)
        y_logit_fl_min[y_logit_fl_min == torch.tensor(np.inf)] = torch.tensor(0.).type_as(logits)
        
        y_logit_max_placeholder = y_logit_flatten.clone()
        y_logit_max_placeholder[~mask_flatten.expand_as(y_logit_flatten)] = torch.tensor(-np.inf).type_as(logits)
        y_logit_fl_max, _ = y_logit_max_placeholder.max(dim=-1)
        y_logit_fl_max[y_logit_fl_max == torch.tensor(-np.inf)] = torch.tensor(0.).type_as(logits)
                                
        if self.detach:
            y_logit_mean = y_logit_mean.detach()
            y_logit_fl_mean = y_logit_fl_mean.detach()
           
        fusion_feats = [y_logit_fl_mean.float()]
        if self.stats:
            # Always detach min, max, std to avoid gradients instability
            logit_stats = [
                y_logit_fl_std.detach().float(),
                y_logit_fl_min.detach().float(),
                y_logit_fl_max.detach().float()   
            ]
            fusion_feats.extend(logit_stats)            
        
        if self.image_stats:
            logit_stats = [
                y_logit_mean.float(),
                y_logit_std.detach().float(),
                y_logit_min.detach().float(),
                y_logit_max.detach().float(),
            ]
            fusion_feats.extend(logit_stats)            

        if self.latlon:
            latlon_encoding = self.latlon_encoding(lat, lon)
            fusion_feats.append(latlon_encoding.float())

        for feat_name in self.aux_features:
            feat = batch.get(feat_name)
            feat = feat.unsqueeze(1)
            fusion_feats.append(feat.float())
                
        fusion = torch.cat(fusion_feats, dim=1)
        
        y_logit_cls = self.fc(fusion)
        y_pred_cls = y_logit_cls.argmax(dim=1)

        # Mask the logits to avoid unnecessary gradients flow
        # Removing this line will introduce NaN gradients for
        # invalid samples (i.e. those with label == IGNORE_VALUE)

        y_logit_cls[~valid, :] = 0
        return logits, y_logit_cls, y_pred_cls
