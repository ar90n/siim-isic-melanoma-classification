#/usr/local/bin/bash
FOLD_INDEX=${1}

unset $(env | grep KAGGLE_ | awk -F = '{print $1}')

export KAGGLE_MAX_EPOCHS=12
export KAGGLE_LEARNING_RATE=0.001
export KAGGLE_GPUS=1
export KAGGLE_BATCH_SIZE=64
export KAGGLE_IMAGE_SIZE=384
export KAGGLE_LABEL_SMOOTHING=0.15
export KAGGLE_POS_WEIGHT=0.25
export KAGGLE_EARLY_STOP_PATIENCE=12
export KAGGLE_N_FOLD=4
export KAGGLE_TRAIN_FOLD_INDEX=${FOLD_INDEX}
export WANDB_PROJECT=isic2020_train_en_b1_mlp

#export KKT_KAGGLE_SANITY_CHECK=1

PROJ_ROOT=$(python -c "from pathlib import Path; print(str(Path('$0').absolute().parent.parent.parent))")
poetry run python ${PROJ_ROOT}/runner/train_efficientnet_b1_mlp.py
