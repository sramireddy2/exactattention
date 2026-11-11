#!/bin/bash
# Bit-identity and kernel tests on one NVIDIA GPU, on PACE Phoenix's FREE "embers" QOS.
# embers has UsageFactor 0 (`sacctmgr show qos embers`), so nothing is charged. It runs on idle
# nodes, can be preempted, and is capped at 8 h. Pass the account on the command line:
#   sbatch -A <your-gts-account> scripts/pace/gpu_check.sbatch
# Listing several partitions lets Slurm start the job wherever a GPU frees up first. All of
# them are sm_80 or newer, which the int8 tensor-core path needs.
#SBATCH -J exactattn-gpu
#SBATCH -q embers
#SBATCH -p gpu-a100,gpu-h100,gpu-h200,gpu-l40s,gpu-rtxpro-blackwell
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o exactattn-gpu-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source "${EA_VENV:-$HOME/scratch/exactattn-venv}/bin/activate"
echo "partition: ${SLURM_JOB_PARTITION:-?}  node: $(hostname)  qos: ${SLURM_JOB_QOS:-?}"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__, torch.cuda.get_device_name(0))"

echo "== fixture: same SHA-256 as the laptop (Intel Arc GPU + x86 CPU) =="
python scripts/fixture.py check --backend cpp
python scripts/fixture.py check --backend torch:cuda
python scripts/fixture.py check --backend triton:cuda
python scripts/fixture.py check --backend tritontc:cuda

echo "== kernel test suites on CUDA =="
EA_TRITON_DEVICE=cuda python -m pytest -q -p no:cacheprovider tests/test_triton.py tests/test_torch_impl.py tests/test_cpu_kernel.py
