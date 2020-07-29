import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int
    learning_rate: float
    num_workers: int
    max_epochs: int
    image_size: int
    gpus: Optional[int]
    tpus: Optional[int]
    precision: int
    early_stop_patience: int
    max_data_size: Optional[int]
    label_smoothing: float
    pos_weight: float


def get_config() -> Config:
    batch_size = int(os.environ.get("KAGGLE_BATCH_SIZE", 32))
    learning_rate = float(os.environ.get("KAGGLE_LEARNING_RATE", 0.008))
    num_workers = int(os.environ.get("KAGGLE_NUM_WORKERS", 4))
    max_epochs = int(os.environ.get("KAGGLE_MAX_EPOCHS", 4))
    image_size = int(os.environ.get("KAGGLE_IMAGE_SIZE", 256))
    gpus = os.environ.get("KAGGLE_GPUS")
    if gpus is not None:
        gpus = int(gpus)
    tpus = os.environ.get("KAGGLE_TPUS")
    if tpus is not None:
        tpus = int(tpus)
    precision = int(os.environ.get("KAGGLE_PRECISION", 16))
    early_stop_patience = int(os.environ.get("KAGGLE_EARLY_STOP_PATIENCE", 5))
    max_data_size = os.environ.get("KAGGLE_MAX_DATASIZE")
    if max_data_size is not None:
        max_data_size = int(max_data_size)
    label_smoothing = float(os.environ.get("KAGGLE_LABEL_SMOOTHING", 0.0))
    pos_weight = os.environ.get("KAGGLE_POS_WEIGHT", 1.0)
    if pos_weight is not None:
        pos_weight = float(pos_weight)

    return Config(
        batch_size,
        learning_rate,
        num_workers,
        max_epochs,
        image_size,
        gpus,
        tpus,
        precision,
        early_stop_patience,
        max_data_size,
        label_smoothing,
        pos_weight
    )
