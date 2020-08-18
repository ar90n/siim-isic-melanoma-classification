import sys
from dataclasses import asdict
from typing import Optional
from pathlib import Path
import json

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ..logger import get_logger
from ..config import Config
from ..util import is_tpu_available
from ..datasource import DataSource
from ..dataset import MelanomaDataset
from ..lightning import Trainer
from .common import get_logger_name
from ..util import get_my_isic2020_experiments_root
from ..net import load_from_checkpoint


def train_split_val(
    Net, config: Config, all_source: DataSource, transforms,
):
    train_df, val_df = train_test_split(all_source.df, test_size=0.2, random_state=42)
    train_source = DataSource(train_df, all_source.roots, 1)
    val_source = DataSource(val_df, all_source.roots, 1)

    model = Net(config)

    logger_base_name = get_logger_name(config)
    logger_name = f"{logger_base_name}"
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

    trainer = Trainer(config, logger=logger)
    trainer.fit(model, train_loader, val_loader)
