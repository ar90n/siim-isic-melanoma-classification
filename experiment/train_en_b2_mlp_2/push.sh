#/usr/local/bin/bash
FOLD_INDEX=${1}

unset $(env | grep KKT | awk -F = '{print $1}')

export KKT_KAGGLE_MAX_EPOCHS=6
export KKT_KAGGLE_LEARNING_RATE=0.001
export KKT_KAGGLE_GPUS=1
export KKT_KAGGLE_BATCH_SIZE=64
export KKT_KAGGLE_IMAGE_SIZE=384
export KKT_KAGGLE_LABEL_SMOOTHING=0.15
export KKT_KAGGLE_POS_WEIGHT=4
export KKT_KAGGLE_EARLY_STOP_PATIENCE=12
export KKT_KAGGLE_N_FOLD=4
export KKT_KAGGLE_TRAIN_FOLD_INDEX=${FOLD_INDEX}
export KKT_WANDB_PROJECT=isic2020_train_en_b2_mlp
export KKT_KAGGLE_EXPERIMENT_NAME=en_b2_mlp_0
export KKT_KAGGLE_MODEL_NAME=en_b2_mlp

#export KKT_KAGGLE_SANITY_CHECK=1

PROJ_ROOT=$(python -c "from pathlib import Path; print(str(Path('$0').absolute().parent.parent.parent))")
poetry run jupytext --to notebook  ${PROJ_ROOT}/runner/train_efficientnet_mlp.py
poetry run kkt push --target .runner.train_en_mlp
