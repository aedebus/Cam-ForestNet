import pandas as pd
import os
import torch
from tqdm import tqdm
from sklearn import metrics
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from data import IndonesiaMachineLearningRegionDataset
from eval.metrics import save_metrics
from util import constants as C
from util import *


logging.getLogger().setLevel(logging.INFO)


PARAMETERS = {
    'dt': {
        'max_depth': [1, 3, 5],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ["balanced"],
        'random_state': [0]
    },
    'rf': {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 1, 3, 5],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ["balanced"],
        'random_state': [0]
    },
    'knn': {
        'n_neighbors': [3, 5, 20],
        'weights': ['uniform', 'distance'],
    },
    'ridge': {
        'alpha': [0.1, 1, 10],
        'normalize': [True],
    },
    'lr': {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
    },
    'mlp': {
        'hidden_layer_sizes': [(64,), (64, 128), (64, 128, 128)],
        'solver': ['lbfgs', 'adam'],
        'learning_rate_init': [0.01, 0.001, 0.0001],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [100000],
        'tol': [1e-7],
        'activation': ['tanh', 'relu'],
        'alpha': [0.01, 0.001],
    },
}

MODELS = {
    'dt': DecisionTreeClassifier,
    'rf': RandomForestClassifier,
    'knn': KNeighborsClassifier,
    'ridge': RidgeClassifier,
    'lr': LogisticRegression,
    'mlp': MLPClassifier,
}


class BaselineModel:
    def __init__(self, args):
        self.hparams = args
        self.features = AUX_FEATURES[self.hparams['mode']][self.hparams['rgb']]
        self.target_col = C.Y_COL
        self.meta_col = C.META_COL
        self.area_col = C.AREA_COL 
        self.m = self.get_model()

    def get_model(self):
        if self.hparams['model'] in MODELS:
            estimator = MODELS[self.hparams['model']]()
        else:
            raise ValueError(f"{self.hparams['model']} not supported.")

        model = GridSearchCV(
            estimator=estimator,
            param_grid=PARAMETERS[self.hparams['model']],
            cv=3,
            scoring='f1_macro',
            n_jobs=1
        )
        return model

    def get_dataset_args(self, split):
        transforms = []
        if split == TRAIN_SPLIT:
            img_option = self.hparams['train_img_option']
        else:
            img_option = self.hparams['eval_img_option']

        kwargs = {
            'data_version': self.hparams['data_version'],
            'data_split': split,
            'img_option': img_option,
            'merge_scheme': self.hparams['merge_scheme'],
            'sample_filter': self.hparams['sample_filter'],
            'transforms': [],
            'ncep': self.hparams['late_fusion_ncep'],
            'aux_features': True
        }

        return kwargs

    def get_dataset(self, split):
        kwargs = self.get_dataset_args(split)
        dataset = IndonesiaMachineLearningRegionDataset(**kwargs)
        return dataset

    def collect(self, split):
        logging.info(f"Start collecting data for [{split}]...")
        dataset = self.get_dataset(split)
        df = []
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            sample_dict = {}
            for f in self.col:
                v = sample[f]
                sample_dict[f] = v
            df.append(sample_dict)
            if self.hparams['debug_samples'] and i > self.hparams['debug_samples']:
                break
        return pd.DataFrame(df)

    def fit(self):
        X = self.train_dataset[self.features]
        y = self.train_dataset[self.target_col].values[:, 0]
        self.m.fit(X, y)

    def predict(self):
        y_pred = self.m.best_estimator_.predict(self.val_dataset[self.features])
        y_true = self.val_dataset[self.target_col].values[:, 0]
        return y_pred, y_true

    def evaluate(self, y_pred, y_true):
        post_merge_scheme = (
            self.hparams['post_merge_scheme_probs'] or
            self.hparams['post_merge_scheme_preds']
        )
        labels = self.hparams['labels']
        label_names = self.hparams['label_names']
        if post_merge_scheme is not None:
            logging.info(f"Using merge scheme [{post_merge_scheme}]. ")
            labels = DATASET_MERGE_SCHEMES[post_merge_scheme]['labels']
            label_names = DATASET_MERGE_SCHEMES[post_merge_scheme]['label_names']

            y_true = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                          post_merge_scheme,
                                          y_true)

            y_pred = map_scheme_to_scheme(self.hparams['merge_scheme'],
                                          post_merge_scheme,
                                          y_pred)
        if self.hparams['eval_by_pixel']:
            area = self.val_dataset[self.area_col].values[:, 0]
        else: 
            area = None
        save_path = self.hparams['default_save_path']
        os.makedirs(save_path, exist_ok=True)
        save_metrics(y_true,
                     y_pred,
                     labels,
                     label_names,
                     save_path,
                     area=area)

        if self.hparams['model'] in ['dt', 'rf']:
            feature_importances = list(
                zip(self.features,
                    list(self.m.best_estimator_.feature_importances_))
            )
            feature_importances.sort(key=lambda l: -l[1])
            feature_importances_df = pd.DataFrame(
                feature_importances, columns=[
                    "Feature", "Importance"])
            feature_importances_df.to_csv(
                os.path.join(
                    save_path,
                    'feature_importance.csv'),
                index=False)
            for (feature, importance) in feature_importances:
                logging.info(f'{feature}: {importance:.4f}')

    def train_and_eval(self):
        self.col = self.features + self.target_col + self.meta_col
        self.train_dataset = self.collect(C.TRAIN_SPLIT)
        self.val_dataset = self.collect(self.hparams['test_data_split'])
        self.fit()
        y_pred, y_true = self.predict()
        self.evaluate(y_pred, y_true)
