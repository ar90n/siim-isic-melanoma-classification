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

# %%
from siim_isic_melanoma_classification import io, task
from siim_isic_melanoma_classification.config import get_config

# %%
config = get_config()

# %%
all_source, _ = io.load_my_isic2020_csv(
    size=config.image_size, is_sanity_check=config.sanity_check
)

# %%
fold_index = int(os.environ["KAGGLE_TRAIN_FOLD_INDEX"])
n_fold = int(os.environ["KAGGLE_N_FOLD"])
task.train_nth_fold_dct_gbdt(config, all_source, fold_index=fold_index, n_fold=n_fold)
