from typing import Optional
import os

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger


def get_logger(name: Optional[str] = None, use_fake_logger: bool = False):
    use_fake_logger |= "WANDB_API_KEY" not in os.environ
    if use_fake_logger:
        return get_fake_logger()
    return get_wandb_logger(name)


def get_wandb_logger(name: Optional[str] = None):
    offline = "WANDB_API_KEY" not in os.environ
    project = os.environ.get("WANDB_PROJECT", "siim_isic_melanoma_classification")
    return WandbLogger(name=name, offline=offline, project=project, log_model=True)


def get_fake_logger(*args, **kwargs):
    return DummyLogger()
