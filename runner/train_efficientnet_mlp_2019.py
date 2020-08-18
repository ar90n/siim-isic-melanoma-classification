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
    util,
    task,
    net,
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
all_source, _ = io.load_my_isic2020_csv(
    size=config.image_size, is_sanity_check=config.sanity_check
)
print(len(all_source))
all_source.df = all_source.df[all_source.df["dataset"] == "isic2019"]
print(len(all_source))

# %%
model_name = os.environ["KAGGLE_MODEL_NAME"]
model_class = net.get_model_class(model_name)
task.train_split_val(
    model_class,
    config,
    all_source,
    train_transform,
)
