from typing import Tuple, cast
from pathlib import Path

import pandas as pd
import numpy as np

from .util import (
    get_jpeg_melanoma_root,
    get_jpeg_isic2019_root,
    get_my_isic2020_csv_root,
    get_malignant_v2_root,
)
from .datasource import DataSource


def _preprop(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    concat = pd.concat(
        [
            train_df["anatom_site_general_challenge"],
            test_df["anatom_site_general_challenge"],
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix="site")
    train_df = pd.concat([train_df, dummies.iloc[: train_df.shape[0]]], axis=1)
    test_df = pd.concat(
        [test_df, dummies.iloc[train_df.shape[0] :].reset_index(drop=True)], axis=1
    )

    # Sex features
    train_df["sex"] = train_df["sex"].map({"male": 1.0, "female": 0.0, "unknown": 0.5})
    test_df["sex"] = test_df["sex"].map({"male": 1.0, "female": 0.0, "unknown": 0.5})
    train_df["sex"] = train_df["sex"].fillna(0.5)
    test_df["sex"] = test_df["sex"].fillna(0.5)

    # Age features
    train_df["age_approx"] /= 100.0
    test_df["age_approx"] /= 100.0
    train_df["age_approx"] = train_df["age_approx"].fillna(0.0)
    test_df["age_approx"] = test_df["age_approx"].fillna(0.0)

    train_df["patient_id"] = train_df["patient_id"].fillna(0)
    return train_df, test_df


def load_my_isic2020_csv(
    size: int = 256, is_sanity_check: bool = False
) -> Tuple[DataSource, DataSource]:
    train_df, test_df = load_my_isic2020_csv_dataframe()
    train_df, test_df = _preprop(train_df, test_df)
    if is_sanity_check:
        train_df = train_df[train_df["sanity_check"] == 1]
        test_df = test_df.iloc[: len(train_df)]

    isic2020_root_path = get_jpeg_melanoma_root(size)
    isic2019_root_path = get_jpeg_isic2019_root(size)
    malignant_v2_root_path = get_malignant_v2_root(size)
    train_img_roots = {
        "isic2020": isic2020_root_path / "train",
        "isic2019": isic2019_root_path / "train",
        "malignant_v2": malignant_v2_root_path / f"jpeg{size}",
    }
    test_img_roots = isic2020_root_path / "test"

    train_folds = list(train_df["fold"].unique())
    test_folds = []

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


def save_result(result: np.ndarray, dst_path: Path = Path("./submission.csv")) -> None:
    sample_submission_path = get_my_isic2020_csv_root() / "sample_submission.csv"
    sub = cast(pd.DataFrame, pd.read_csv(sample_submission_path))
    sub["target"].iloc[: len(result)] = result.reshape(-1)
    sub.to_csv(str(dst_path), index=False)
