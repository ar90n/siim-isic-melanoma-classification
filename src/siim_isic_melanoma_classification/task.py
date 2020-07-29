import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import DataLoader
import numpy as np

from .logger import get_logger
from .config import Config
from .util import clean_up, is_tpu_available, search_best_model_path, get_random_name
from .datasource import kfold_split, DataSource
from .dataset import MelanomaDataset
from .lightning import Trainer, Classifier


def _get_logger_name(fold_index: int) -> str:
    pid = os.getpid()
    name = get_random_name(pid)
    return f"{name}-{fold_index}"


def kfold_cv_tta(
    Net,
    config: Config,
    all_source: DataSource,
    test_source: DataSource,
    transforms,
    n_workers: Optional[int] = 1,
    n_split: int = 5,
    stratify: Optional[str] = "target",
) -> np.ndarray:
    def _task(
        train_source: DataSource,
        val_source: DataSource,
        test_source: DataSource,
        fold_index: int,
    ):
        clean_up()
        logger = get_logger(use_fake_logger=(fold_index != 0))
        logger.log_hyperparams(config.__dict__)

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

        model = Net(config)

        trainer = Trainer(config, fold_index=fold_index, logger=logger)
        trainer.fit(model, train_loader, val_loader)

        best_model_path = search_best_model_path(trainer.best_model_path)
        if best_model_path is not None:
            model = Net.load_from_checkpoint(best_model_path)

        return Classifier(model, tta_epochs=5).predict(test_loader)

    if 8 < n_split:
        raise ValueError("n_split must be less than 8")
    if n_workers is None:
        n_workers = n_split

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for fold_index, (train_source, val_source) in enumerate(
            kfold_split(all_source, n_split=n_split, stratify=stratify)
        ):
            future = executor.submit(
                _task, train_source, val_source, test_source, fold_index,
            )
            futures.append(future)
        all_results = [f.result() for f in futures]

    return np.average(np.hstack(all_results), axis=1)

