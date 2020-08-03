from pathlib import Path

import imageio as io
import numpy as np
import torch
from torch.utils.data import Dataset
from jpeg2dct.numpy import load as dct_load


from .datasource import DataSource


class MelanomaDataset(Dataset):
    def __init__(
        self,
        source: DataSource,
        train: bool = True,
        transforms=None,
        meta_features=None,
        dct: bool = False,
    ):
        self.source = source
        self.transforms = transforms
        self.train = train
        self.dct = dct

        if meta_features is None:
            self.meta_features = self.get_one_hot_encoding_columns(source)
        else:
            self.meta_features = meta_features

    def _load_image(self, path: Path):
        if self.dct:
            return dct_load(str(path))
        else:
            return io.imread(path)

    def __getitem__(self, index):
        img_root = self.get_img_root(index)
        img_path = img_root / f"{self.source.df.iloc[index]['image_name']}.jpg"
        x = self._load_image(img_path)
        meta = np.array(
            self.source.df.iloc[index][self.meta_features].values, dtype=np.float32
        )

        if self.transforms:
            x = self.transforms(x)

        if self.train:
            y = torch.tensor(self.source.df.iloc[index]["target"]).view(1)
            return (x, meta), y
        else:
            return (x, meta)

    def __len__(self):
        return len(self.source.df)

    def get_img_root(self, index: int):
        if isinstance(self.source.roots, Path):
            return self.source.roots
        else:
            return self.source.roots[self.source.df.iloc[index]["dataset"]]

    @classmethod
    def get_one_hot_encoding_columns(cls, source: DataSource):
        return ["sex", "age_approx"] + [
            col for col in source.df.columns if col.startswith("site_")
        ]
