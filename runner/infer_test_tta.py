# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#%load_ext autoreload
#%autoreload 2

# %%
import torch
from torch.utils.data import DataLoader
from torch import nn
import torchtoolbox.transform as transforms
import os
import json

# %%
from siim_isic_melanoma_classification import (
    net,
    io,
    util,
    task,
    transforms as my_transforms,
)
from siim_isic_melanoma_classification.config import get_config
from siim_isic_melanoma_classification.lightning import Classifier
from siim_isic_melanoma_classification.dataset import MelanomaDataset

# %%
config = get_config()

# %%
util.initialize(config)
if util.is_kaggle():
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()

# %%
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        my_transforms.Microscope(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
_, test_source = io.load_my_isic2020_csv(
    size=config.image_size, is_sanity_check=config.sanity_check
)

# %%
test_loader = DataLoader(
    MelanomaDataset(test_source, train=False, transforms=train_transform),
    batch_size=config.batch_size,
    num_workers=config.num_workers,
)

# %%
experiment_name = os.environ["KAGGLE_EXPERIMENT_NAME"]
task.infer_tta(config, test_source, train_transform, experiment_name)
