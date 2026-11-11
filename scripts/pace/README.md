# Running on Georgia Tech PACE (free tier only)

Everything here uses Phoenix's **free `embers` QOS**. Its UsageFactor is 0, so nothing is
charged; check with `sacctmgr show qos embers`.

## How embers behaves, and how the job copes

- **Only the first hour is guaranteed.** After that a priority job can take the GPU, and the
  embers job is **cancelled, not requeued** (`PreemptMode=CANCEL`), with about 30 s warning.
  Busy partitions such as L40S lose it at about 1 h almost every time, even with an 8 h
  request.
- So each job asks for **55 minutes**, inside the guarantee. Shorter requests also backfill
  sooner.
- The work is split into **stages** (`run_stages.sh`), each with its own time cap, and each
  writes a `.done` marker when it finishes. A new job skips finished stages.
- When stages remain, the job **submits the next job in the chain** itself (at most
  `EA_MAX_CHAIN`, default 6). On SIGTERM, whether preemption or the time limit, the running
  stage stays not-done and the chain resubmits.
- A stage that hits its cap twice is recorded as `TIMEOUT` and skipped, so the chain cannot
  loop forever.

## Setup (once, on a login node)

```bash
git clone https://github.com/sramireddy2/exactattention.git ~/scratch/exactattention
cd ~/scratch/exactattention && bash scripts/pace/setup.sh
```

## Start a run

```bash
cd ~/scratch/exactattention
sbatch -A <gts-account> --export=ALL,EA_RUN=$(date +%m%d-%H%M) -o ~/scratch/ea_jobs/exactattn-%j.out scripts/pace/gpu_check.sbatch
squeue -u $USER
cat results/pace/run-<RUN>/summary.md        # updated by every job in the chain
```

Stages, in order:

1. environment
2. C++ fixture
3. torch-CUDA fixture
4. torch-CUDA tests
5. C++ tests
6. Triton CUDA tests
7. Triton fixture: windowed scores
8. Triton fixture: tensor-core scores
9. decode benchmarks at 1K, 4K and 16K keys

Every fixture stage should print `ALL MATCH`, meaning the same SHA-256 as the laptop's Intel GPU
and CPU.
