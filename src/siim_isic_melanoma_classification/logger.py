import os

from pytorch_lightning.loggers.wandb import WandbLogger


def get_logger():
    offline = "WANDB_API_KEY" not in os.environ
    project = os.environ.get("WANDB_PROJECT", "siim_isic_melanoma_classification")
    return WandbLogger(offline=offline, project=project, log_model=True)
