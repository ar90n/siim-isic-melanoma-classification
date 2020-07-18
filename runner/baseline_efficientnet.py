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
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torchtoolbox.transform as transforms
import pandas as pd
import numpy as np
from pathlib import Path
import timm
import os

# %%
from siim_isic_melanoma_classification import (
    io,
    util,
    transforms as my_transforms,
    datasource,
)
from siim_isic_melanoma_classification.config import Config, get_config
from siim_isic_melanoma_classification.logger import get_logger
from siim_isic_melanoma_classification.dataset import MelanomaDataset
from siim_isic_melanoma_classification.lightning import LightningModelBase, Trainer

# %%
logger = get_logger()

# %%
util.initialize()
if "KAGGLE_CONTAINER_NAME" in os.environ:
    import kaggle_timm_pretrained

    kaggle_timm_pretrained.patch()


# %%
class Net(LightningModelBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.backbone = timm.create_model(
            "efficientnet_b0", num_classes=1, pretrained=True
        )

    def forward(self, inputs):
        x, y = inputs
        x = self.backbone(x)
        return torch.squeeze(x)


# %%
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        my_transforms.Microscope(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
config = get_config()
logger.log_hyperparams(config.__dict__)

# %%
all_source, test_source = io.load_jpeg_melanoma(max_data_size=config.max_data_size)

# %%
test_loader = DataLoader(
    MelanomaDataset(test_source, train=False, transforms=test_transform),
    batch_size=config.batch_size,
    num_workers=config.num_workers,
)

# %%
train_source, val_source = datasource.train_validate_split(all_source, val_size=0.2)
train_loader = DataLoader(
    MelanomaDataset(train_source, train=True, transforms=train_transform),
    shuffle=True,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
)
val_loader = DataLoader(
    MelanomaDataset(val_source, train=True, transforms=train_transform),
    shuffle=True,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
)
# %%
model = Net(config)

# %%
checkpoint_callback = ModelCheckpoint(filepath=os.getcwd(), verbose=True)
trainer = Trainer(config, logger=logger, checkpoint_callback=checkpoint_callback)
trainer.fit(model, train_loader, val_loader)

# %%
# model = torch.load(model_path)
# gc.collect()
device = util.get_device()
model.eval().to(device)
result = []
with torch.no_grad():
    for i, x_test in enumerate(test_loader):
        x_test[0] = x_test[0].to(device)
        x_test[1] = x_test[1].to(device)
        z_test = model(x_test)
        z_test = torch.sigmoid(z_test)
        result.append(z_test.cpu().numpy())
result = np.hstack(result)

# %%
sub = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sub.iloc[: len(result)]["target"] = result
sub.to_csv("submission.csv", index=False)
