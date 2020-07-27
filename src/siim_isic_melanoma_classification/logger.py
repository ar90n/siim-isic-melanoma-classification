from typing import Optional
import os

from pytorch_lightning.loggers.wandb import WandbLogger


def get_logger(name: Optional[str] = None):
    offline = "WANDB_API_KEY" not in os.environ
    project = os.environ.get("WANDB_PROJECT", "siim_isic_melanoma_classification")
    return WandbLogger(name=name, offline=offline, project=project, log_model=True)
