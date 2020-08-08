from dataclasses import asdict

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from wandb.xgboost import wandb_callback

from ..logger import get_logger
from ..config import Config
from ..datasource import DataSource, get_folds_by
from ..dataset import MetaDataset
from .common import get_logger_name


def train_nth_fold_meta_gbdt(
    config: Config, all_source: DataSource, fold_index: int, n_fold: int
):
    logger_base_name = get_logger_name(config)
    logger_name = f"{logger_base_name}-{fold_index}-{n_fold}"
    logger = get_logger(name=logger_name)
    logger.log_hyperparams(asdict(config))

    train_source, val_source = get_folds_by(all_source, fold_index, n_fold)

    train_dataset = MetaDataset(train_source, train=True)
    val_dataset = MetaDataset(val_source, train=True)

    x_train, y_train = zip(*[v for v in train_dataset])
    x_train = np.stack(x_train)
    y_train = np.array(y_train)

    x_val, y_val = zip(*[v for v in val_dataset])
    x_val = np.stack(x_val)
    y_val = np.array(y_val)

    clf = xgb.XGBClassifier(objective="binary:logistic", n_estimators=8192)

    clf.fit(
        x_train,
        y_train,
        eval_metric="auc",
        eval_set=[(x_train, y_train), (x_val, y_val),],
        early_stopping_rounds=64,
        callbacks=[wandb_callback()],
    )
    y_hat = clf.predict_proba(x_val)[:,1]

    auc = roc_auc_score(y_val, y_hat)
    print(f"auc: {auc}")

    model_name = f"fold_{fold_index}_{n_fold}_val_auc_{auc}.model"
    clf.save_model(model_name)
