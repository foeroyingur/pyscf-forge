#!/usr/bin/env bash

set -e

sudo apt-get -qq install \
    gcc \
    libblas-dev \
    cmake \
    curl

python -m pip install --upgrade pip
pip install "scipy<1.16"
pip install pytest
pip install .
pip install -U jax

pip install trexio
pip install mcfun
pip install --no-deps pyscf-dispersion==1.5.0

pip install git+https://gitlab.com/pyframe-project/pyframe.git@trajectory_embedding

# TODO: check if pyscf code is changed using dist-info file
#pip uninstall -y pyscf-forge
