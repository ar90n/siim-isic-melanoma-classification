from typing import Optional
import os

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.comet import CometLogger
from pytorch_lightning.loggers.base import DummyLogger


def get_logger(name: Optional[str] = None, use_fake_logger: bool = False):
    if "COMETML_API_KEY" in os.environ and not use_fake_logger:
        return get_cometml_logger(name)
    if "WANDB_API_KEY" in os.environ and not use_fake_logger:
        return get_wandb_logger(name)
    return get_fake_logger(name)


def get_wandb_logger(name: Optional[str] = None):
    offline = "WANDB_API_KEY" not in os.environ
    project = os.environ.get("WANDB_PROJECT", "siim_isic_melanoma_classification")
    return WandbLogger(name=name, offline=offline, project=project, log_model=True)


def get_cometml_logger(name: Optional[str] = None):
    api_key = os.environ.get("COMETML_API_KEY")
    project_name = os.environ.get(
        "COMETML_PROJECT", "sandbox"
    )
    workspace = os.environ.get("COMETML_WORKSPACE", "melanoma_classification")
    return CometLogger(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace,
        experiment_name=name,
        log_git_metadata=False,
        log_git_patch=False,
    )


def get_fake_logger(*args, **kwargs):
    return DummyLogger()
