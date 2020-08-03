#/usr/local/bin/bash
unset $(env | grep KKT | awk -F = '{print $1}')

export KKT_KAGGLE_GPUS=1
export KKT_KAGGLE_BATCH_SIZE=64
export KKT_KAGGLE_IMAGE_SIZE=384
export KKT_KAGGLE_N_FOLD=4
export KKT_WANDB_PROJECT=isic2020_infer_en_b0_mlp_0
export KKT_KAGGLE_EXPERIMENT_NAME=en_b0_mlp_0

#export KKT_KAGGLE_SANITY_CHECK=1

PROJ_ROOT=$(python -c "from pathlib import Path; print(str(Path('$0').absolute().parent.parent.parent))")
poetry run jupytext --to notebook  ${PROJ_ROOT}/runner/infer_test_tta.py
poetry run kkt push --target .infer_test_tta
