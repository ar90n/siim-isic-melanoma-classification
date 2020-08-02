from pathlib import Path

import timm
from torch import nn
from torchtoolbox.nn import Swish
import torch

from .lightning import LightningModelBase


def load_from_checkpoint(model_name, path_to_cktp: Path) -> LightningModelBase:
    return {EfficientNetB0MLP.name: EfficientNetB0MLP}[model_name].load_from_checkpoint(
        str(path_to_cktp)
    )


class EfficientNetB0MLP(LightningModelBase):
    name: str = "en_b0_mlp"

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._backbone = timm.create_model(
            "efficientnet_b0", num_classes=512, pretrained=True
        )
        self._meta = nn.Sequential(
            nn.Linear(9, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),  # FC layer output will have 256 features
            nn.BatchNorm1d(256),
            Swish(),
            nn.Dropout(p=0.2),
        )
        self._classifier = nn.Linear(512 + 256, 1)

    def backbone(self, x):
        return self._backbone(x)

    def meta(self, x):
        return self._meta(x)

    def features(self, inputs):
        x, y = inputs
        x = self.backbone(x)
        y = self.meta(y)
        return torch.cat((x, y), dim=1)

    def forward(self, inputs):
        features = self.features(inputs)
        output = self._classifier(features)
        return output.view(output.size(0), -1)
