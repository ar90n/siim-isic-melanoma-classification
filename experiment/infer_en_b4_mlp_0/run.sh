#/usr/local/bin/bash
unset $(env | grep KKT | awk -F = '{print $1}')

export KAGGLE_GPUS=1
#export KAGGLE_TPUS=1
export KAGGLE_BATCH_SIZE=64
export KAGGLE_IMAGE_SIZE=256
export KAGGLE_N_FOLD=4
export WANDB_PROJECT=isic2020_infer_en_b4_mlp_0
export KAGGLE_EXPERIMENT_NAME=en_b4_mlp_0

#export KAGGLE_SANITY_CHECK=1

PROJ_ROOT=$(python -c "from pathlib import Path; print(str(Path('$0').absolute().parent.parent.parent))")
poetry run python ${PROJ_ROOT}/runner/infer_tta.py
