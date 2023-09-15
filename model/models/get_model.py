import torch
import logging

from .pretrained import *
from .segcast import SegCastNet
from .fusion import FusionNet 

def get_single_model(args):
    model_classes = globals().copy()
    if args.get("segmentation") and args.get("late_fusion"):
        model_name = "FusionNet"
    elif args.get("segmentation"):
        model_name = "SegCastNet"
    else:
        model_name = args.get("model")
    model = model_classes[model_name](args)
    if args['ckpt_path'] is not None:
        logging.info(f"Loading checkpoint at {args['ckpt_path']}")
        # If this throws a key mismatch, it may be because the "model"
        # arg does not match the checkpoint model.
        state_dict = torch.load(args['ckpt_path'])['state_dict']
        
        # Truncate layer names from model.model... -> model
        # since loading outside of lightning.
        state_dict = {
            layer_name.replace("model.", "", 1): layer
            for layer_name, layer in state_dict.items()
        }
        
        model_dict = model.state_dict()
        for k in model_dict.keys():
            if (args['pretrained'] == 'landcover' and args['is_training'] and "model.segmentation_head.0" in k):
                continue
            if k in state_dict:
                model_dict[k] = state_dict[k]
 
        model.load_state_dict(model_dict)

    return model
