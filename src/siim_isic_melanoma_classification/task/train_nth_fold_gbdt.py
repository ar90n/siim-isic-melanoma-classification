import sys
import json
from dataclasses import asdict

import numpy as np
import xgboost as xgb
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from wandb.xgboost import wandb_callback

from ..net import create_ef_lowlevel_features_model, load_from_checkpoint
from ..logger import get_logger
from ..config import Config
from ..util import is_tpu_available, get_my_isic2020_experiments_root
from ..datasource import DataSource, get_folds_by
from ..dataset import MelanomaDataset
from ..lightning import Classifier
from .common import get_logger_name


def train_nth_fold_gbdt(
    config: Config, all_source: DataSource, transforms,
):
    experiment_root = get_my_isic2020_experiments_root() / config.experiment_name
    experiment_index_path = experiment_root / "index.json"
    experiment = json.load(experiment_index_path.open("r"))

    model_type = experiment["model_type"]
    for ckpt in experiment["checkpoints"]:
        fold_index = ckpt["fold_index"]
        n_fold = ckpt["n_fold"] + 1
        ckpt_path = experiment_root / ckpt["file"]

        logger_base_name = get_logger_name(config)
        logger_name = f"{logger_base_name}-{fold_index}-{n_fold}"
        logger = get_logger(name=logger_name)
        logger.log_hyperparams(asdict(config))

        train_source, val_source = get_folds_by(all_source, fold_index, n_fold)
        train_loader = DataLoader(
            MelanomaDataset(train_source, train=False, transforms=transforms),
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=is_tpu_available(),
        )

        val_loader = DataLoader(
            MelanomaDataset(val_source, train=False, transforms=transforms),
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            drop_last=is_tpu_available(),
        )

        print(
            f"Train using {str(ckpt_path)} - {fold_index} / {n_fold}", file=sys.stderr,
        )
        model = create_ef_lowlevel_features_model(
            load_from_checkpoint(model_type, ckpt_path), 5
        )
        x_train = Classifier(model).predict(train_loader)
        y_train = np.vstack([y for y in train_source.df["target"]]).ravel()
        x_val = Classifier(model).predict(val_loader)
        y_val = np.vstack([y for y in val_source.df["target"]]).ravel()

        xgb_train = xgb.DMatrix(x_train, label=y_train)
        xgb_val = xgb.DMatrix(x_val, label=y_val)
        xgb_params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb_model = xgb.train(
            xgb_params, xgb_train, num_boost_round=100, callbacks=[wandb_callback()]
        )
        y_val_pred = xgb_model.predict(xgb_val)

        auc = roc_auc_score(y_val, y_val_pred)
        print(
            f"Use {config.experiment_name} - {fold_index} / {n_fold}: auc={auc}",
            file=sys.stderr,
        )

        model_name = f"fold_{fold_index}_{n_fold}_val_auc_{auc}.model"
        xgb_model.save_model(model_name)
