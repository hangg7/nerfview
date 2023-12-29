#!/bin/bash
#
# File   : setup.sh
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
# Date   : 12/26/2023
#
# Distributed under terms of the MIT license.

set -e -x

# Setup environment.
conda create --prefix .env python=3.10 -y
conda activate "$(pwd)/.env"

# Clone the repo. We use a specific commit to use our patch.
git clone https://github.com/nerfstudio-project/nerfacc --recursive
pushd nerfacc
git checkout e6647a0
git popd

# Install dependencies (will take a while).
pip install numpy
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e nerfacc --config-settings editable_mode=compat
pip install -r nerfacc/examples/requirements.txt
# Install nerfview.
pip install -e ../../ --config-settings editable_mode=compat

# Apply patch.
pushd nerfacc
git am ../nerfview.patch
popd
