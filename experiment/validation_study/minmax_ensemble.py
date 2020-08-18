import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc


def min_max_best_stacking(best, concats, cutoff_lo=0.66, cutoff_hi=0.33):
    max_values = concats.max(axis=1)
    min_values = concats.min(axis=1)

    result = best.copy()
    result = np.where(
        np.all(concats.iloc[:, :] > cutoff_lo, axis=1),
        max_values,
        np.where(
            np.all(concats.iloc[:, :] < cutoff_hi, axis=1), min_values, best,
        ),
    )

    return result


def load_csvs(train_csv_path, best_csv_path, val_output_csv_paths):
    expect = pd.read_csv(train_csv_path, index_col=0)["target"]
    best = pd.read_csv(best_csv_path, index_col=0)
    pred_source = {
        str(p.stem): pd.read_csv(p, index_col=0).loc[expect.index]
        for p in val_output_csv_paths
    }
    preds = pd.concat(pred_source.values(), axis=1)
    preds.columns = list(pred_source.keys())

    return expect, best, preds


def calc_auc(expect, pred):
    fpr, tpr, _ = roc_curve(expect, pred)
    return auc(fpr, tpr)


def main(train_csv_path, best_csv_path, val_output_csv_paths):
    expect, best, preds = load_csvs(train_csv_path, best_csv_path, val_output_csv_paths)

    result = min_max_best_stacking(best, preds)
    auc_score = calc_auc(expect, result)

    print(f"auc_score: {auc_score}")
    filename = f"cv_auc_{auc_score}.csv"
    print(f"save to {filename}")
    result.to_csv(filename)


if __name__ == "__main__":
    try:
        train_csv_path = Path(sys.argv[1])
        best_csv_path = Path(sys.argv[2])
        val_output_csv_paths = [Path(arg) for arg in sys.argv[3:]]
    except IndexError:
        print(
            f"python calc_metric.py <train.csv>  <best.csv> <validation outout.csv>  <validation outout.csv> ....",
            file=stderr,
        )
        sys.exit(1)

    main(train_csv_path, best_csv_path, val_output_csv_paths)
