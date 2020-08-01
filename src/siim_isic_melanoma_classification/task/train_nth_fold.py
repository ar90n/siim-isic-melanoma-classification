from dataclasses import asdict

from torch.utils.data import DataLoader

from ..logger import get_logger
from ..config import Config
from ..util import is_tpu_available
from ..datasource import DataSource, get_folds_by
from ..dataset import MelanomaDataset
from ..lightning import Trainer
from .common import get_logger_name


def train_nth_fold(
    Net,
    config: Config,
    all_source: DataSource,
    transforms,
    fold_index: int,
    n_fold: int = 8,
):

    if n_fold < 0 or 8 < n_fold:
        raise ValueError("n_fold must be best_model_path 1 with 8")

    train_source, val_source = get_folds_by(all_source, fold_index, n_fold)

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
