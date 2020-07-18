from pathlib import Path
import imageio as io
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .datasource import DataSource


class MelanomaDataset(Dataset):
    def __init__(
        self,
        source: DataSource,
        train: bool = True,
        transforms=None,
        meta_features=None,
    ):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (Path): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.source = source
        self.transforms = transforms
        self.train = train

        if meta_features is None:
            self.meta_features = ["sex", "age_approx"] + [
                col for col in source.df.columns if col.startswith("site_")
            ]
        else:
            self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = self.source.root / f"{self.source.df.iloc[index]['image_name']}.jpg"
        x = io.imread(im_path)
        meta = np.array(
            self.source.df.iloc[index][self.meta_features].values, dtype=np.float32
        )

        if self.transforms:
            x = self.transforms(x)

        if self.train:
            y = self.source.df.iloc[index]["target"]
            return (x, meta), y
        else:
            return (x, meta)

    def __len__(self):
        return len(self.source.df)
