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
import torchtoolbox.transform as transforms
import os

# %%
from siim_isic_melanoma_classification import (
    net,
    io,
    util,
    task,
    transforms as my_transforms,
)
from siim_isic_melanoma_classification.config import get_config

# %%
config = get_config()

# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
all_source, _ = io.load_my_isic2020_csv(
    size=config.image_size, is_sanity_check=config.sanity_check
)

# %%
task.train_nth_fold_gbdt(config, all_source, transform)
