#/usr/local/bin/bash
FOLD_INDEX=${1}

unset $(env | grep KKT | awk -F = '{print $1}')

export KKT_KAGGLE_N_FOLD=4
export KKT_KAGGLE_TRAIN_FOLD_INDEX=${FOLD_INDEX}
export KKT_WANDB_PROJECT=isic2020_train_xgboost_dct

#export KKT_KAGGLE_SANITY_CHECK=1

PROJ_ROOT=$(python -c "from pathlib import Path; print(str(Path('$0').absolute().parent.parent.parent))")
poetry run jupytext --to notebook  ${PROJ_ROOT}/runner/train_xgboost_dct.py
poetry run kkt push --target .runner.train_xgboost_dct
