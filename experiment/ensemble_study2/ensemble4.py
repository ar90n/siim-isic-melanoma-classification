import sys
from pathlib import Path

import pandas as pd
import numpy as np
import os


def min_max_best_stacking(sub_best, concat_sub):
    concat_sub["is_iceberg_max"] = concat_sub.iloc[:, 1:].max(axis=1)
    concat_sub["is_iceberg_min"] = concat_sub.iloc[:, 1:].min(axis=1)

    cutoff_lo = 0.7
    cutoff_hi = 0.01

    concat_sub["is_iceberg_base"] = sub_best["target"]
    concat_sub["target"] = np.where(
        np.all(concat_sub.iloc[:, 1:-3] > cutoff_lo, axis=1),
        concat_sub["is_iceberg_max"],
        np.where(
            np.all(concat_sub.iloc[:, 1:-3] < cutoff_hi, axis=1),
            concat_sub["is_iceberg_min"],
            concat_sub["is_iceberg_base"],
        ),
    )

    return concat_sub[["image_name", "target"]]


def main(paths):
    sub_best = pd.read_csv(paths[0])
    subs = [pd.read_csv(p, index_col=0) for p in paths[1:]]

    concat_sub = pd.concat([s for s in subs], axis=1)
    cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)

    result = min_max_best_stacking(sub_best, concat_sub)
    result.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python ensemble.py <path to submission.csv> ...", file=sys.stderr)
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    main(paths)
