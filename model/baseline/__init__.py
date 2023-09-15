import os

from util import constants as C
from util import aux_features as A
from util import Args
from data.util import add_labels_and_label_names

from .base import BaselineModel
from .pixel import PixelBaselineModel

def train_and_evaluate(exp_name,
                       dataset=C.INDONESIA_DATASET_NAME,
                       save_path=str(C.SANDBOX_DIR),
                       data_version="new_splits",
                       late_fusion=False,
                       late_fusion_ncep=True,
                       post_merge_scheme_preds=None,
                       post_merge_scheme_probs=None,
                       merge_scheme='small-scale-plantation-merge',
                       sample_filter="historical_merge",
                       train_img_option=C.IMG_OPTION_COMPOSITE,
                       eval_img_option=C.IMG_OPTION_CLOSEST_YEAR,
                       eval_by_pixel=False,
                       debug_samples=False,
                       mode=A.REGION_MODE,
                       rgb=False,
                       model='rf',
                       test_data_split=C.VAL_SPLIT,
                       ):
    
    assert mode in A.ML_BASELINE_MODES

    args = Args(locals())
    args['default_save_path'] = os.path.join(save_path, exp_name)
 
    add_labels_and_label_names(args)
    if mode == A.PX_MODE:
        model = PixelBaselineModel(args)
    else:
        model = BaselineModel(args)
    model.train_and_eval()
