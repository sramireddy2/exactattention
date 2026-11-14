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
# nvidia-cuda-nvcc does not pin its nvvm (cicc) or crt headers: unpinned, pip takes the newest,
# whose PTX the older ptxas rejects ("Unsupported .version 9.5; current version is '9.2'").
pip install -q flashinfer-python ninja "nvidia-cuda-nvcc==${CU}.*" "nvidia-nvvm==${CU}.*" "nvidia-cuda-crt==${CU}.*"
# FlashInfer links with -L$CUDA_HOME/lib64 -L$CUDA_HOME/lib64/stubs -lcudart -lcuda; the pip layout
# has only lib/ and no driver stub, so add both (the stub comes from a PACE cuda module).
C=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu${CU%%.*}
mkdir -p "$C/lib64/stubs"
ln -sfn "../lib/libcudart.so.${CU%%.*}" "$C/lib64/libcudart.so"
STUB=$(module show "cuda/${CU%%.*}" 2>&1 | grep -o '"CUDA_HOME","[^"]*' | head -1 | cut -d'"' -f4)/lib64/stubs/libcuda.so
if [ -e "$STUB" ]; then ln -sfn "$STUB" "$C/lib64/stubs/libcuda.so"; else echo "WARNING: no libcuda stub found; FlashInfer JIT will fail to link"; fi
python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__, 'cuda build', torch.version.cuda)"
echo "venv ready: $VENV"
