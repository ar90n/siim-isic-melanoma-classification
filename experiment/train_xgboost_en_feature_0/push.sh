#/usr/local/bin/bash
FOLD_INDEX=${1}

unset $(env | grep KKT | awk -F = '{print $1}')

export KKT_KAGGLE_BATCH_SIZE=64
export KKT_KAGGLE_IMAGE_SIZE=128
export KKT_KAGGLE_N_FOLD=4
export KKT_KAGGLE_TRAIN_FOLD_INDEX=${FOLD_INDEX}
export KKT_WANDB_PROJECT=isic2020_train_xgboost_en_feature
export KKT_KAGGLE_EXPERIMENT_NAME=en_b0_mlp_0
export KKT_KAGGLE_EN_FAETURE_LAYER=5

#export KKT_KAGGLE_SANITY_CHECK=1

PROJ_ROOT=$(python -c "from pathlib import Path; print(str(Path('$0').absolute().parent.parent.parent))")
poetry run jupytext --to notebook  ${PROJ_ROOT}/runner/train_xgboost_en_feature.py
poetry run kkt push --target .train_xgboost_en_feature
