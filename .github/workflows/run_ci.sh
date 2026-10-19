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

# PyFraME requires Python 3.11 or newer; on older versions the embedding tests
# skip themselves.
if python -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    pip install pyframe==0.5.0
fi

# TODO: check if pyscf code is changed using dist-info file
#pip uninstall -y pyscf-forge
