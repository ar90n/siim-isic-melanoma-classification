import sys
from pathlib import Path

import pandas as pd


def main(path):
    sub = pd.read_csv(path)
    label = sub["image_name"]
    target = sub[[str(i) for i in range(4)]].mean(axis=1)

    result = pd.DataFrame()
    result["image_name"] = label
    result["target"] = target
    result.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python ensemble.py <path to submission.csv> ...", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    main(path)
