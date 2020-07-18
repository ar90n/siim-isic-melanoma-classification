import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int
    learning_rate: float
    num_workers: int
    max_epochs: int
    gpus: int
    precision: int
    max_data_size: Optional[int]


def get_config() -> Config:
    batch_size = int(os.environ.get("KAGGLE_BATCH_SIZE", 32))
    learning_rate = float(os.environ.get("KAGGLE_LEARNING_RATE", 0.008))
    num_workers = int(os.environ.get("KAGGLE_NUM_WORKERS", 4))
    max_epochs = int(os.environ.get("KAGGLE_MAX_EPOCHS", 4))
    gpus = int(os.environ.get("KAGGLE_GPUS", 1))
    precision = int(os.environ.get("KAGGLE_PRECISION", 16))
    max_data_size = os.environ.get("KAGGLE_MAX_DATASIZE")
    if max_data_size is not None:
        max_data_size = int(max_data_size)

    return Config(
        batch_size,
        learning_rate,
        num_workers,
        max_epochs,
        gpus,
        precision,
        max_data_size,
    )
