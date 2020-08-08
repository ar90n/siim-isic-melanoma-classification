import sys
import json
from typing import cast
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from ..dataset import MetaDataset
from ..datasource import DataSource
from ..config import Config
from ..util import get_my_isic2020_experiments_root, get_my_isic2020_csv_root


def infer_test_tta(config: Config, test_source: DataSource, experiment_name):
    experiment_root = get_my_isic2020_experiments_root() / experiment_name
    experiment_index_path = experiment_root / "index.json"
    experiment = json.load(experiment_index_path.open("r"))

    test_dataset = MetaDataset(test_source, train=False)

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

        x_test = np.stack([v for v in test_dataset])
        clf = pickle.load(Path(ckpt_path).open("rb"))
        y_hat = clf.predict_proba(x_test)[:, 1]
        inferences.append((f"fold-{fold_index}", y_hat.ravel()))
    result = pd.concat([label, pd.DataFrame(dict(inferences))], axis=1)
    result.to_csv(f"cv_test_{model_type}.csv", index=False)


# def infer_val_tta(config: Config, all_source: DataSource, transforms, experiment_name):
#    experiment_root = get_my_isic2020_experiments_root() / experiment_name
#    experiment_index_path = experiment_root / "index.json"
#    experiment = json.load(experiment_index_path.open("r"))
#
#    inferences = []
#    features = []
#    model_type = experiment["model_type"]
#    for ckpt in experiment["checkpoints"]:
#        fold_index = ckpt["fold_index"]
#        n_fold = ckpt["n_fold"]
#        ckpt_path = experiment_root / ckpt["file"]
#
#        print(
#            f"Infer val data using {str(ckpt_path)} - {fold_index} / {n_fold}",
#            file=sys.stderr,
#        )
#
#        _, val_source = get_folds_by(all_source, fold_index, n_fold + 1)
#        label = cast(pd.DataFrame, val_source.df)["image_name"]
#
#        val_loader = DataLoader(
#            MelanomaDataset(val_source, train=False, transforms=transforms),
#            batch_size=config.batch_size,
#            num_workers=config.num_workers,
#        )
#
#        model = load_from_checkpoint(model_type, ckpt_path)
#        target = Classifier(
#            # model, tta_epochs=config.tta_epochs, with_features=True
#            model,
#            tta_epochs=config.tta_epochs,
#            with_features=False,
#        ).predict(val_loader)
#
#        inference = pd.concat(
#            [label, pd.DataFrame({"target": target.ravel()}, index=label.index)], axis=1
#        )
#        inferences.append(inference)
#
#        # if feature is not None:
#        #    concat_feature = pd.concat(
#        #        [
#        #            pd.DataFrame([pd.DataFrame(f).values.tolist()])
#        #            for f in np.transpose(feature, [1, 0, 2])
#        #        ]
#        #    )
#        #    concat_feature.index = label.index
#        #    feature = pd.concat([label, concat_feature], axis=1)
#        #    features.append(feature)
#    result = pd.concat(inferences, axis=0)
#    result.to_csv(f"cv_val_{model_type}.csv", index=False)
#
#    # feature_result = pd.concat(features, axis=0)
#    # feature_result.to_pickle(f"embedding_val_{model_type}.pickle")
