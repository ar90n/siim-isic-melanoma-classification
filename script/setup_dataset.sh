#!/usr/bin/env bash
PROJ_ROOT=$(cd $(dirname $0); pwd)/../
INPUT_DIR=${PROJ_ROOT}/../input

function setup_dataset () {
    SLUG=$(echo ${1} | sed 's/\/.*//')
    DATANAME=$(echo ${1} | sed 's/.*\///')
    mkdir -p ${INPUT_DIR}/${DATANAME}
    pushd ${INPUT_DIR}/${DATANAME}
    kaggle d download ${1}
    unzip ${DATANAME}.zip
    rm ${DATANAME}.zip
    popd
}

mkdir -p $INPUT_DIR

setup_dataset "cdeotte/jpeg-melanoma-128x128"
setup_dataset "cdeotte/jpeg-melanoma-256x256"
setup_dataset "cdeotte/jpeg-melanoma-384x384"
setup_dataset "cdeotte/jpeg-isic2019-128x128"
setup_dataset "cdeotte/jpeg-isic2019-256x256"
setup_dataset "cdeotte/jpeg-isic2019-384x384"
setup_dataset "cdeotte/malignant-v2-128x128"
setup_dataset "cdeotte/malignant-v2-256x256"
setup_dataset "cdeotte/malignant-v2-384x384"
setup_dataset "nroman/melanoma-hairs"
setup_dataset "ar90ngas/my-isic2020-csv"
setup_dataset "ar90ngas/my-isic2020-experiments"
