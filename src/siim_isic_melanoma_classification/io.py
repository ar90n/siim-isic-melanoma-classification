from typing import Tuple
import pandas as pd
import numpy as np

from .util import get_jpeg_melanoma_root, get_isic_melanoma_classification_root
from .datasource import DataSource


def load_jpeg_melanoma(
    size: int = 256, is_raw: bool = False, max_data_size: int = None
) -> Tuple[DataSource, DataSource]:
    train_df, test_df = load_jpeg_melanoma_dataframe(size, is_raw)
    dataset_root_path = get_jpeg_melanoma_root(size)
    train_img_root = dataset_root_path / "train"
    test_img_root = dataset_root_path / "test"

    if max_data_size is not None:
        train_df = train_df[:max_data_size]
        test_df = test_df[:max_data_size]
    return DataSource(train_df, train_img_root), DataSource(test_df, test_img_root)


def load_jpeg_melanoma_dataframe(
    size: int = 256, is_raw: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root_path = get_jpeg_melanoma_root(size)

    train_csv_path = dataset_root_path / "train.csv"
    train_df = pd.read_csv(train_csv_path)

    test_csv_path = dataset_root_path / "test.csv"
    test_df = pd.read_csv(test_csv_path)

    if is_raw:
        return train_df, test_df

    # One-hot encoding of anatom_site_general_challenge feature
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
    train_df["sex"] = train_df["sex"].map({"male": 1, "female": 0})
    test_df["sex"] = test_df["sex"].map({"male": 1, "female": 0})
    train_df["sex"] = train_df["sex"].fillna(-1)
    test_df["sex"] = test_df["sex"].fillna(-1)

    # Age features
    train_df["age_approx"] /= train_df["age_approx"].max()
    test_df["age_approx"] /= test_df["age_approx"].max()
    train_df["age_approx"] = train_df["age_approx"].fillna(0)
    test_df["age_approx"] = test_df["age_approx"].fillna(0)

    train_df["patient_id"] = train_df["patient_id"].fillna(0)

    return train_df, test_df


def save_result(result: np.ndarray) -> None:
    sample_submission_path = (
        get_isic_melanoma_classification_root() / "sample_submission.csv"
    )
    sub = pd.read_csv(sample_submission_path)
    sub["target"].iloc[: len(result)] = result.reshape(-1)
    sub.to_csv("submission.csv", index=False)
