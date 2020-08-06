import import pdb; pdb.set_trace()
import json
from typing import cast

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from ..lightning import Classifier
from ..dataset import MelanomaDataset
from ..datasource import DataSource, get_folds_by
from ..config import Config
from ..util import get_my_isic2020_experiments_root, get_my_isic2020_csv_root
from ..net import load_from_checkpoint


def infer_test_tta(
    config: Config, test_source: DataSource, transforms, experiment_name
):
    experiment_root = get_my_isic2020_experiments_root() / experiment_name
    experiment_index_path = experiment_root / "index.json"
    experiment = json.load(experiment_index_path.open("r"))

    test_loader = DataLoader(
        MelanomaDataset(test_source, train=False, transforms=transforms),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    sample_submission_path = get_my_isic2020_csv_root() / "sample_submission.csv"
    label = cast(pd.DataFrame, pd.read_csv(sample_submission_path))["image_name"].iloc[
        : len(test_source.df)
    ]

    inferences = []
    model_type = experiment["model_type"]
    for ckpt in experiment["checkpoints"]:
        fold_index = ckpt["fold_index"]
        n_fold = ckpt["n_fold"]
        ckpt_path = experiment_root / ckpt["file"]

        print(
            f"Infer test data using {str(ckpt_path)} - {fold_index} / {n_fold}",
            file=sys.stderr,
        )
        model = load_from_checkpoint(model_type, ckpt_path)
        inference = Classifier(model, tta_epochs=config.tta_epochs).predict(test_loader)
        inferences.append((f"fold-{fold_index}", inference.ravel()))
    result = pd.concat([label, pd.DataFrame(dict(inferences))], axis=1)
    result.to_csv(f"cv_test_{model_type}.csv", index=False)


def infer_val_tta(config: Config, all_source: DataSource, transforms, experiment_name):
    experiment_root = get_my_isic2020_experiments_root() / experiment_name
    experiment_index_path = experiment_root / "index.json"
    experiment = json.load(experiment_index_path.open("r"))

    inferences = []
    model_type = experiment["model_type"]
    for ckpt in experiment["checkpoints"]:
        fold_index = ckpt["fold_index"]
        n_fold = ckpt["n_fold"]
        ckpt_path = experiment_root / ckpt["file"]

        print(
            f"Infer val data using {str(ckpt_path)} - {fold_index} / {n_fold}",
            file=sys.stderr,
        )

        _, val_source = get_folds_by(all_source, fold_index, n_fold + 1)
        label = cast(pd.DataFrame, val_source.df)["image_name"]

        val_loader = DataLoader(
            MelanomaDataset(val_source, train=False, transforms=transforms),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model = load_from_checkpoint(model_type, ckpt_path)
        target = Classifier(model, tta_epochs=config.tta_epochs).predict(val_loader)
        inference = pd.concat(
            [label, pd.DataFrame({"target": target.ravel()}, index=label.index)], axis=1
        )
        inferences.append(inference)
    result = pd.concat(inferences, axis=0)
    result.to_csv(f"cv_val_{model_type}.csv", index=False)
