import sys
from dataclasses import asdict
from typing import Optional
from pathlib import Path
import json

from torch.utils.data import DataLoader

from ..logger import get_logger
from ..config import Config
from ..util import is_tpu_available
from ..datasource import DataSource, get_folds_by
from ..dataset import MelanomaDataset
from ..lightning import Trainer
from .common import get_logger_name
from ..util import get_my_isic2020_experiments_root
from ..net import load_from_checkpoint


def train_nth_fold(
    Net,
    config: Config,
    all_source: DataSource,
    transforms,
    fold_index: int,
    n_fold: int,
    experiment_name: Optional[str] = None,
):
    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")

    train_source, val_source = get_folds_by(all_source, fold_index, n_fold)

    if experiment_name is not None:
        experiment_root = get_my_isic2020_experiments_root() / experiment_name
        experiment_index_path = experiment_root / "index.json"
        experiment = json.load(experiment_index_path.open("r"))

        model_type = experiment["model_type"]
        for ckpt in experiment["checkpoints"]:
            if fold_index == ckpt["fold_index"] and n_fold == (ckpt["n_fold"] + 1):
                ckpt_path = experiment_root / ckpt["file"]
                print(f"Retrain using {ckpt_path}", file=sys.stderr)
                model = load_from_checkpoint(model_type, ckpt_path)
                break
        else:
            raise ValueError("Wrong fold_index and n_fold are given.")
    else:
        model = Net(config)

    logger_base_name = get_logger_name(config)
    logger_name = f"{logger_base_name}-{fold_index}-{n_fold}"
    logger = get_logger(name=logger_name)
    logger.log_hyperparams(asdict(config))

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
