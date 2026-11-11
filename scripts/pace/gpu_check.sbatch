#!/bin/bash
# Bit-identity and kernel tests on one NVIDIA GPU, on PACE's free "embers" tier.
#   sbatch scripts/pace/gpu_check.sbatch
# Before the first submit, edit the -A line. Run `pace-quota` to see your charge accounts.
# embers jobs are free but preemptible, and capped at 8 h. This job takes a few minutes.
#SBATCH -J exactattn-gpu
#SBATCH -A gts-REPLACE_ME
#SBATCH -q embers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o exactattn-gpu-%j.out
# To request a specific GPU, replace the --gres line with one of these (check `sinfo` for names):
#   --gres=gpu:A100:1    --gres=gpu:H100:1    --gres=gpu:V100:1    --gres=gpu:RTX_6000:1

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source "${EA_VENV:-$HOME/scratch/exactattn-venv}/bin/activate"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

echo "== fixture: same SHA-256 as the laptop (Intel Arc GPU + x86 CPU) =="
python scripts/fixture.py check --backend cpp
python scripts/fixture.py check --backend torch:cuda
python scripts/fixture.py check --backend triton:cuda

echo "== kernel test suites on CUDA =="
EA_TRITON_DEVICE=cuda python -m pytest -q -p no:cacheprovider tests/test_triton.py tests/test_torch_impl.py tests/test_cpu_kernel.py
