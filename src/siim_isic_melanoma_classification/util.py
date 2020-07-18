import warnings
from pathlib import Path
import os

import torch
from pytorch_lightning import seed_everything as pl_seed_evertything


def seed_everything(seed: int = 47) -> None:
    pl_seed_evertything(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def initialize() -> None:
    warnings.simplefilter("ignore")
    seed_everything()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_input() -> Path:
    input_path = os.environ.get("KAGGLE_INPUT_PATH")
    if input_path is not None:
        return Path(input_path)
    return Path.cwd().parent / "input"


def get_jpeg_melanoma_root(size: int) -> Path:
    dataset_name = f"jpeg-melanoma-{size}x{size}"
    return get_input() / dataset_name
