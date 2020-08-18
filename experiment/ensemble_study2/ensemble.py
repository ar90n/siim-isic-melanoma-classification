import sys
from pathlib import Path

import pandas as pd


def main(paths):
    subs = [pd.read_csv(p) for p in paths]
    target = pd.concat([s["target"] for s in subs], axis=1).mean(axis=1)

    result = pd.DataFrame()
    result["image_name"] = subs[0]["image_name"]
    result["target"] = target
    result.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python ensemble.py <path to submission.csv> ...", file=sys.stderr)
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    main(paths)
