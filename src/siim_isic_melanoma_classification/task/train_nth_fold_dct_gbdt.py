from dataclasses import asdict
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from wandb.xgboost import wandb_callback

from ..logger import get_logger
from ..config import Config
from ..datasource import DataSource, get_folds_by
from ..dataset import MelanomaDataset
from ..transforms import ZigZagFlatten
from .common import get_logger_name


def train_nth_fold_dct_gbdt(
    config: Config, all_source: DataSource, fold_index: int, n_fold: int
):
    logger_base_name = get_logger_name(config)
    logger_name = f"{logger_base_name}-{fold_index}-{n_fold}"
    logger = get_logger(name=logger_name)
    logger.log_hyperparams(asdict(config))

    train_source, val_source = get_folds_by(all_source, fold_index, n_fold)

    train_dataset = MelanomaDataset(
        source=train_source,
        train=True,
        meta_features=["sex", "age_approx", "anatom_site_general_challenge"],
        dct=True,
        transforms=ZigZagFlatten(8),
    )

    val_dataset = MelanomaDataset(
        source=val_source,
        train=True,
        meta_features=["sex", "age_approx", "anatom_site_general_challenge"],
        dct=True,
        transforms=ZigZagFlatten(8),
    )

    x_train, y_train = zip(
        *[(np.hstack(dct_coeffs), meta) for dct_coeffs, meta in train_dataset]
    )
    x_train = np.stack(x_train)
    y_train = np.array(y_train)

    x_val, y_val = zip(
        *[(np.hstack(dct_coeffs), meta) for dct_coeffs, meta in val_dataset]
    )
    x_val = np.stack(x_val)
    y_val = np.array(y_val)

    clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=256)

    clf.fit(
        x_train,
        y_train,
        eval_metric="auc",
        eval_set=[(x_train, y_train), (x_val, y_val),],
        early_stopping_rounds=64,
        callbacks=[wandb_callback()],
    )
    y_hat = clf.predict_proba(x_val)[:, 1]

    auc = roc_auc_score(y_val, y_hat)
    print(f"auc: {auc}")

    model_name = f"fold_{fold_index}_{n_fold}_val_auc_{auc}.pickle"
    pickle.dump(clf, Path(model_name).open("wb"))
