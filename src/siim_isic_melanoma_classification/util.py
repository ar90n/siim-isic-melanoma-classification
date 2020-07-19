import warnings
from typing import Iterable
from pathlib import Path
import os
import gc
import subprocess

import torch
from pytorch_lightning import seed_everything as pl_seed_evertything

from .config import Config


try:
    import apex

    has_apex = True
except ImportError:
    has_apex = False

try:
    import torch_xla

    has_xla = True
except ImportError:
    has_xla = False


def seed_everything(seed: int = 47) -> None:
    pl_seed_evertything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def initialize(config: Config) -> None:
    warnings.simplefilter("ignore")
    seed_everything()
    install_xla(config)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input() -> Path:
    input_path = os.environ.get("KAGGLE_INPUT_PATH")
    if input_path is not None:
        return Path(input_path)
    return Path.cwd().parent / "input"


def get_isic_melanoma_classification_root() -> Path:
    return get_input() / "siim-isic-melanoma-classification"


def get_jpeg_melanoma_root(size: int) -> Path:
    dataset_name = f"jpeg-melanoma-{size}x{size}"
    return get_input() / dataset_name


def clean_up():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def to_device(tensors, device=None):
    if device is None:
        device = get_device()

    if isinstance(tensors, Iterable):
        ret = [t.to(device) for t in tensors]
        if isinstance(tensors, tuple):
            ret = tuple(ret)
        return ret

    return tensors.to(device)


def install_xla(config: Config, version="nightly"):
    if config.tpus is None:
        return

    global has_xla

    curl_args = [
        "curl",
        "https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py",
        "-o",
        "pytorch-xla-env-setup.py",
    ]
    subprocess.run(curl_args)

    install_args = ["python", "pytorch-xla-env-setup.py", "--version", version]
    subprocess.run(install_args)
    has_xla = True


def is_apex_available() -> bool:
    return torch.cuda.is_available() and has_apex


def is_tpu_available() -> bool:
    if not has_xla:
        return False

    import torch_xla.core.xla_model as xm

    devices = xm.get_xla_supported_devices(devkind="TPU")
    return devices is not None
