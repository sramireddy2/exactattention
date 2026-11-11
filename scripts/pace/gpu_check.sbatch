#!/bin/bash
# GPU checks and benchmarks on PACE Phoenix's FREE embers QOS (UsageFactor 0: nothing is charged).
#
# On embers only the first hour is guaranteed; after that the job can be cancelled (not
# requeued) with about 30 s warning. So this job asks for 55 min and runs scripts/pace/run_stages.sh,
# which checkpoints per stage and chains further jobs until every stage is done.
#
# Start a run (the account is only needed for embers access; embers is never charged):
#   sbatch -A <gts-account> --export=ALL,EA_RUN=$(date +%m%d-%H%M) scripts/pace/gpu_check.sbatch
#SBATCH -J exactattn-gpu
#SBATCH -q embers
#SBATCH -p gpu-a100,gpu-h100,gpu-h200,gpu-l40s,gpu-rtxpro-blackwell
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:55:00
#SBATCH --signal=B:TERM@60
#SBATCH -o exactattn-gpu-%j.out

# --signal=B:TERM@60 gives the stage runner 60 s of notice before the time limit as well.
exec bash "${EA_REPO:-$HOME/scratch/exactattention}/scripts/pace/run_stages.sh"
