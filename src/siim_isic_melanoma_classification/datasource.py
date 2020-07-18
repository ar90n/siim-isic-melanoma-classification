from typing import Tuple
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold

import pandas as pd


@dataclass
class DataSource:
    df: pd.DataFrame
    root: Path


def train_validate_split(
    source: DataSource, val_size: float
) -> Tuple[DataSource, DataSource]:
    train_df, val_df = train_test_split(source.df, test_size=val_size)
    return DataSource(train_df, source.root), DataSource(val_df, source.root)


def kfold_split(source: DataSource, n_split=5):
    kfold = KFold(n_split)
    for train_df, val_df in kfold.split(source.df):
        yield DataSource(train_df, source.root), DataSource(val_df, source.root)
