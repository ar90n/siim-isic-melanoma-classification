import sys
from pathlib import Path

import pandas as pd


def main(path0, path1, w):
    sub = pd.read_csv(path0)
    label = sub["image_name"]
    target0 = sub["target"]

    sub = pd.read_csv(path1)
    target1 = sub["target"]

    target = (1.0 - w) * target0 + w * target1

    result = pd.DataFrame()
    result["image_name"] = label
    result["target"] = target
    result.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("python ensemble.py <path to submission.csv> ...", file=sys.stderr)
        sys.exit(1)

    path0 = Path(sys.argv[1])
    path1 = Path(sys.argv[2])
    w = float(sys.argv[3])
    main(path0, path1, w)
