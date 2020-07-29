from typing import Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import pandas as pd


@dataclass
class DataSource:
    df: pd.DataFrame
    root: Path

    def __len__(self) -> int:
        return len(self.df)


def train_validate_split(
    source: DataSource, val_size: float, stratify: Optional[str] = "target"
) -> Tuple[DataSource, DataSource]:
    if stratify is not None:
        stratify = source.df[stratify]
    train_df, val_df = train_test_split(
        source.df, test_size=val_size, stratify=stratify, shuffle=True
    )
    return DataSource(train_df, source.root), DataSource(val_df, source.root)


def _kfold_split(
    source: DataSource, n_split: int, stratify: Optional[str]
) -> Union[KFold, StratifiedKFold]:
    if stratify is not None:
        return StratifiedKFold(n_split, shuffle=True).split(
            source.df, source.df[stratify]
        )
    return KFold(n_split, shuffle=True).split(source.df)


def kfold_split(source: DataSource, n_split=5, stratify: Optional[str] = "target"):
    for train_idx, val_idx in _kfold_split(source, n_split, stratify):
        train_source = DataSource(source.df.iloc[train_idx], source.root)
        val_source = DataSource(source.df.iloc[val_idx], source.root)
        yield train_source, val_source
