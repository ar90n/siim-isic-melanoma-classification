import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc
from scipy.stats import rankdata


def load_csvs(train_csv_path, val_output_csv_paths):
    expect = pd.read_csv(train_csv_path, index_col=0)["target"]
    pred_source = {
        str(p.stem): pd.read_csv(p, index_col=0).loc[expect.index]
        for p in val_output_csv_paths
    }
    preds = pd.concat(pred_source.values(), axis=1)
    preds.columns = list(pred_source.keys())

    return expect, preds

def calc_auc(expect, pred):
    fpr, tpr, _ = roc_curve(expect, pred)
    return auc(fpr, tpr)


def main(train_csv_path, val_output_csv_paths):
    expect, preds = load_csvs(train_csv_path, val_output_csv_paths)

    rank = rankdata(rankdata(preds, axis=0).mean(axis=1), method='min')
    auc_score = calc_auc(expect, rank)

    print(f"auc_score: {auc_score}")
    filename=f"cv_auc_{auc_score}.csv"
    print(f"save to {filename}")
    expect["target"] = rank
    expect.to_csv(filename)


if __name__ == "__main__":
    try:
        train_csv_path = Path(sys.argv[1])
        val_output_csv_paths = [Path(arg) for arg in sys.argv[2:]]
    except IndexError:
        print(
            f"python calc_metric.py <train.csv> <validation outout.csv>  <validation outout.csv> ....",
            file=sys.stderr,
        )
        sys.exit(1)

    main(train_csv_path, val_output_csv_paths)
