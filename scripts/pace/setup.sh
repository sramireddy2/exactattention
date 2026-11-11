#!/bin/bash
# One-time environment setup on Georgia Tech PACE (Phoenix or ICE). Run on a login node:
#   bash scripts/pace/setup.sh
# Creates a venv in ~/scratch (keeps your home quota free) with CUDA PyTorch, Triton and the
# test dependencies, then builds the C++ kernel.
set -euo pipefail
module load anaconda3 2>/dev/null || module load python 2>/dev/null || true
VENV=${EA_VENV:-$HOME/scratch/exactattn-venv}
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch            # Linux wheels bundle CUDA and Triton
pip install -q numpy mpmath pytest hypothesis
make -C "$(dirname "$0")/../../csrc"
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda build', torch.version.cuda)"
echo "venv ready: $VENV"
