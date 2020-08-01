# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.5.2
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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchtoolbox.transform as transforms
from torchtoolbox.nn import Swish
import timm
import os

# %%
from siim_isic_melanoma_classification import (
    io,
    util,
    transforms as my_transforms,
    datasource,
)
from siim_isic_melanoma_classification.config import get_config
from siim_isic_melanoma_classification.logger import get_logger
from siim_isic_melanoma_classification.dataset import MelanomaDataset
from siim_isic_melanoma_classification.lightning import (
    LightningModelBase,
    Trainer,
    Classifier,
)

# %%
from category_encoders import TargetEncoder

# %%
config = get_config()

# %%
logger = get_logger()
logger.log_hyperparams(config.__dict__)

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
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# %%
all_source, test_source = io.load_my_isic2020_csv(
    size=config.image_size, is_sanity_check=(not util.is_kaggle())
)
# %%
test_loader = DataLoader(
    MelanomaDataset(test_source, train=False, transforms=test_transform),
    batch_size=config.batch_size,
    num_workers=config.num_workers,
)

# %%
train_source, val_source = datasource.train_validate_split(all_source)

train_loader = DataLoader(
    MelanomaDataset(train_source, train=True, transforms=train_transform),
    shuffle=True,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    drop_last=util.is_tpu_available(),
)
val_loader = DataLoader(
    MelanomaDataset(val_source, train=True, transforms=train_transform),
    shuffle=True,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    drop_last=util.is_tpu_available(),
)
# %%
class Net(LightningModelBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.backbone = timm.create_model(
            "efficientnet_b0", num_classes=500, pretrained=True
        )
        self.meta = nn.Sequential(
            nn.Linear(9, 500),
            nn.BatchNorm1d(500),
            Swish(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 250),  # FC layer output will have 250 features
            nn.BatchNorm1d(250),
            Swish(),
            nn.Dropout(p=0.2),
        )
        self.ouput = nn.Linear(500 + 250, 1)

    def forward(self, inputs):
        x, y = inputs
        x = self.backbone(x)
        y = self.meta(y)
        features = torch.cat((x, y), dim=1)
        output = self.ouput(features)
        return output.view(output.size(0), -1)


# %%
model = Net(config)

# %%
trainer = Trainer(config, logger=logger)
trainer.fit(model, train_loader, val_loader)

# %%
best_model_path = util.search_best_model_path(trainer.best_model_path)
if best_model_path is not None:
    model = Net.load_from_checkpoint(best_model_path)

# %%
result = Classifier(model, tta_epochs=5).predict(test_loader)

# %%
io.save_result(result)
