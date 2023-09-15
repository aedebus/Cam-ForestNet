from .get_model import get_single_model
from .ensemble import EnsembleModel


def get_model(args):
    if args.get("ensemble"):
        model = EnsembleModel(args)
    else: 
        model = get_single_model(args)
    return model


