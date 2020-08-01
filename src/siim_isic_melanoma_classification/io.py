from typing import Tuple

import pandas as pd
import numpy as np

from .util import (
    get_jpeg_melanoma_root,
    get_jpeg_isic2019_root,
    get_my_isic2020_csv_root,
)
from .datasource import DataSource


def load_my_isic2020_csv(size: int = 256, is_sanity_check: bool = False) -> Tuple[DataSource, DataSource]:
    train_df, test_df = load_my_isic2020_csv_dataframe()
    if is_sanity_check:
        train_df = train_df[train_df["sanity_check"] == 1]
        test_df = test_df[test_df["sanity_check"] == 1]


    isic2020_root_path = get_jpeg_melanoma_root(size)
    isic2019_root_path = get_jpeg_isic2019_root(size)
    train_img_roots = {
        "isic2020": isic2020_root_path / "train",
        "isic2019": isic2019_root_path / "train",
    }
    test_img_roots = {
        "isic2020": isic2020_root_path / "test",
        "isic2019": isic2019_root_path / "test",
    }

    train_folds = list(train_df["fold"].unique())
    test_folds = list(test_df["fold"].unique())
    return (
        DataSource(train_df, train_img_roots, train_folds),
        DataSource(test_df, test_img_roots, test_folds),
    )


def load_my_isic2020_csv_dataframe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root_path = get_my_isic2020_csv_root()

    train_csv_path = dataset_root_path / "train.csv"
    train_df = pd.read_csv(train_csv_path)

    test_csv_path = dataset_root_path / "test.csv"
    test_df = pd.read_csv(test_csv_path)

    return train_df, test_df


def save_result(result: np.ndarray) -> None:
    sample_submission_path = get_my_isic2020_csv_root() / "sample_submission.csv"
    sub = pd.read_csv(sample_submission_path)
    sub["target"].iloc[: len(result)] = result.reshape(-1)
    sub.to_csv("submission.csv", index=False)
