import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc


def main(train_csv_path, val_output_csv_path):
    train_target = pd.read_csv(train_csv_path)

    val_output = pd.read_csv(val_output_csv_path)
    val_output = val_output.set_index("image_name").loc[train_target["image_name"]]

    fpr, tpr, _ = roc_curve(
        train_target["target"].values, val_output["target"].values
    )
    auc_val = auc(fpr, tpr)
    print(f"auc: {auc_val}")


if __name__ == "__main__":
    try:
        train_csv_path = Path(sys.argv[1])
        val_output_csv_path = Path(sys.argv[2])
    except IndexError:
        print(
            f"python calc_metric.py <train.csv> <validation outout.csv> ",
            file=sys.stderr,
        )
        sys.exit(1)

    main(train_csv_path, val_output_csv_path)
