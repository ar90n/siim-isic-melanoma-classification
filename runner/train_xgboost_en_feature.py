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
import os
import torchtoolbox.transform as transforms

# %%
from siim_isic_melanoma_classification import (
    io,
    task,
    transforms as my_transforms,
)
from siim_isic_melanoma_classification.config import get_config

# %%
config = get_config()

# %%
util.initialize(config)
if util.is_kaggle():
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()

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
experiment_name = os.environ["KAGGLE_EXPERIMENT_NAME"]
fold_index = int(os.environ["KAGGLE_TRAIN_FOLD_INDEX"])
n_fold = int(os.environ["KAGGLE_N_FOLD"])
feature_layer = int(os.environ["KAGGLE_EN_FAETURE_LAYER"])
task.train_nth_fold_gbdt(
    config,
    all_source,
    transform,
    experiment_name,
    fold_index=fold_index,
    n_fold=n_fold,
    feature_layer=feature_layer,
)
