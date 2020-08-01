import os
from typing import Optional
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor

from torch.utils.data import DataLoader
import numpy as np

from ..logger import get_logger
from ..config import Config
from ..util import clean_up, is_tpu_available, search_best_model_path
from ..datasource import kfold_split, DataSource
from ..dataset import MelanomaDataset
from ..lightning import Trainer, Classifier
from .common import get_logger_name


def _task(
    Net,
    config: Config,
    train_source: DataSource,
    val_source: DataSource,
    test_source: DataSource,
    transforms,
    fold_index: int,
    n_fold: int,
    logger_base_name: str,
):
    # TODO: if model isn't initialized hbefore logger initializzation, an exception occur.
    model = Net(config)

    logger_name = f"{logger_base_name}-{fold_index}"
    logger = get_logger(name=logger_name)
    logger.log_hyperparams(asdict(config))

    test_loader = DataLoader(
        MelanomaDataset(test_source, train=False, transforms=transforms,),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    train_loader = DataLoader(
        MelanomaDataset(train_source, train=True, transforms=transforms),
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=is_tpu_available(),
    )

    val_loader = DataLoader(
        MelanomaDataset(val_source, train=True, transforms=transforms),
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=is_tpu_available(),
    )

    trainer = Trainer(config, fold_index=fold_index, n_fold=n_fold, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    clean_up()
    best_model_path = search_best_model_path(trainer.best_model_path)
    if best_model_path is not None:
        model = Net.load_from_checkpoint(best_model_path)

    return Classifier(model, tta_epochs=5).predict(test_loader)


def kfold_cv_tta(
    Net,
    config: Config,
    all_source: DataSource,
    test_source: DataSource,
    transforms,
    n_workers: Optional[int] = 1,
    n_fold: int = 8,
) -> np.ndarray:

    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")
    if n_workers is None:
        n_workers = n_fold

    logger_base_name = get_logger_name(config)
    _loop = _kfold_cv_tta_sp if n_workers == 1 else _kfold_cv_tta_mp
    all_results = _loop(
        Net,
        config,
        all_source,
        test_source,
        transforms,
        logger_base_name,
        n_workers,
        n_fold,
    )
    return np.average(np.hstack(all_results), axis=1)


def _kfold_cv_tta_mp(
    Net,
    config: Config,
    all_source: DataSource,
    test_source: DataSource,
    transforms,
    logger_base_name: str,
    n_workers: int,
    n_fold: int,
) -> np.ndarray:
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for fold_index, (train_source, val_source) in enumerate(
            kfold_split(all_source, n_fold=n_fold)
        ):
            future = executor.submit(
                _task,
                Net,
                config,
                train_source,
                val_source,
                test_source,
                transforms,
                fold_index,
                logger_base_name,
            )
            futures.append(future)
        return [f.result() for f in futures]


def _kfold_cv_tta_sp(
    Net,
    config: Config,
    all_source: DataSource,
    test_source: DataSource,
    transforms,
    logger_base_name,
    n_workers: int,
    n_fold: int,
) -> np.ndarray:
    futures = []
    for fold_index, (train_source, val_source) in enumerate(
        kfold_split(all_source, n_fold=n_fold)
    ):
        future = _task(
            Net,
            config,
            train_source,
            val_source,
            test_source,
            transforms,
            fold_index,
            n_fold,
            logger_base_name,
        )
        futures.append(future)
    return futures
