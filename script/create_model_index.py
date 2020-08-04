import sys
from pathlib import Path
import re
import json

import yaml


def main(model_name: str, root_path: Path):
    checkpoints = []
    for model_path in root_path.glob("*.ckpt"):
        match = re.match(r"^fold_(\d+)_(\d+).*", model_path.stem)
        if match is None:
            continue

        fold_index = int(match.group(1))
        n_fold = int(match.group(2))
        checkpoints.append(
            {"fold_index": fold_index, "n_fold": n_fold, "file": model_path.name}
        )

    config_path = root_path / "config.yaml"
    config = yaml.load(config_path.open("r"))

    index = {
        "model_type": model_name,
        "checkpoints": checkpoints,
        "image_size": config["image_size"]["value"],
    }
    index_path = root_path / "index.json"
    index_path.write_text(json.dumps(index, indent=4))


if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        root_path = Path(sys.argv[2])
    except:
        print(
            "usage : python create_model_index.py <model name> <path to model dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    main(model_name, root_path)
