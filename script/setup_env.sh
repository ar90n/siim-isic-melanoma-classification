#!/usr/bin/env bash

git config --global user.email argon.argon.argon@gmail.com
git config --global user.name "Masahiro Wada"

conda create -y -n py37 python=3.7
conda init bash
