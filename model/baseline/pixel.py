import logging
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import pandas as pd

from data import IndonesiaMachineLearningPixelDataset
from util import constants as C
from util.merge_scheme import *
from util.aux_features import *

from .base import BaselineModel


class PixelBaselineModel(BaselineModel):
    def __init__(self, args):
        super().__init__(args)

    def get_dataset(self, split):
        kwargs = self.get_dataset_args(split)
        dataset = IndonesiaMachineLearningPixelDataset(**kwargs)
        return dataset

    def collect(self, split):
        logging.info(f"Start collecting data for [{split}]...")
        dataset = self.get_dataset(split)

        df = []
        for sample in tqdm(dataset, total=len(dataset)):
            sample_dict = {}
            for i in range(sample['n_pixel']):
                for f in self.col:
                    v = sample[f]
                    if isinstance(v, np.ndarray):
                        # size instead of len because saved numpy arrs have no len
                        if v.size > 1:
                            v = v[i]                                                    
                    sample_dict[f] = v
                df.append(sample_dict)
        return pd.DataFrame(df)

    def predict(self):
        self.val_dataset['y_pred'] = self.m.best_estimator_.predict(
            self.val_dataset[self.features])
        y_pred = self.val_dataset.groupby(
            ["index"])['y_pred'].agg(
            pd.Series.mode).to_numpy()
        y_true = self.val_dataset.groupby(
            ["index"])[
            self.target_col].agg(
            pd.Series.mode).to_numpy()
        return y_pred, y_true

