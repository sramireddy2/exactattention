#!/bin/bash
# One-time environment setup on Georgia Tech PACE (Phoenix or ICE). Run on a login node:
#   bash scripts/pace/setup.sh
# Creates a venv in ~/scratch (keeps your home quota free) with CUDA PyTorch, Triton and the
# test dependencies, then builds the C++ kernel.
set -euo pipefail
module load python/3.12.5 2>/dev/null || module load python 2>/dev/null || true
VENV=${EA_VENV:-$HOME/scratch/exactattn-venv}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch            # Linux wheels bundle CUDA and Triton
pip install -q numpy mpmath pytest hypothesis
make -C "$(dirname "$0")/../../csrc"
# Baselines for scripts/bench_vs_flashinfer.py (EA_PLAN=flash). FlashInfer JIT-compiles, so it
# needs an nvcc matching torch's CUDA; PACE's cuda modules are older, so take nvcc from pip.
CU=$(python -c "import torch; print(torch.version.cuda)")          # e.g. 13.2
pip install -q flashinfer-python ninja "nvidia-cuda-nvcc==${CU}.*"
L=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu${CU%%.*}/lib
[ -e "$L/libcudart.so" ] || ln -s "libcudart.so.${CU%%.*}" "$L/libcudart.so"   # for -lcudart
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda build', torch.version.cuda)"
echo "venv ready: $VENV"
