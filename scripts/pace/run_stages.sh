#!/bin/bash
# Preemption-safe stage runner for PACE's free embers QOS.
#
# On embers, a job is guaranteed its first hour. After that a priority job can take the GPU,
# and the embers job is CANCELLED, not requeued, with about 30 s warning. So:
#   * each job asks for <= 55 min, inside the guarantee (shorter jobs also backfill sooner);
#   * the work is split into small stages, and each finished stage writes a .done marker;
#   * a job skips stages that are already done, runs the rest within its time budget, and if
#     anything is left it submits the next job in the chain (at most EA_MAX_CHAIN jobs);
#   * on SIGTERM (preemption or time limit) it stops the running stage, leaves it not-done,
#     resubmits, and exits.
#
# Results go to results/pace/run-<RUN>/, one log per stage, plus summary.md.
set -uo pipefail

REPO="${EA_REPO:-$HOME/scratch/exactattention}"
RUN="${EA_RUN:?set EA_RUN to a run name}"
CHAIN="${EA_CHAIN:-1}"
MAX_CHAIN="${EA_MAX_CHAIN:-6}"
BUDGET_S="${EA_BUDGET_S:-3000}"          # leave about 5 min of the 55-min limit for setup and resubmit
MAX_TIMEOUTS="${EA_MAX_TIMEOUTS:-2}"     # after this many timeouts, a stage is recorded as TIMEOUT and skipped
OUT="$REPO/results/pace/run-$RUN"
mkdir -p "$OUT"
cd "$REPO"
source "${EA_VENV:-$HOME/scratch/exactattn-venv}/bin/activate"
START=$(date +%s)

# name | per-stage cap in seconds | command
STAGES=(
  "env|120|nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv && python -c \"import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.get_device_name(0))\""
  "fixture_cpp|600|python scripts/fixture.py check --backend cpp"
  "fixture_torch_cuda|900|python scripts/fixture.py check --backend torch:cuda"
  "tests_torch_cuda|900|python -m pytest -q -p no:cacheprovider tests/test_torch_impl.py"
  "tests_cpu_kernel|900|python -m pytest -q -p no:cacheprovider tests/test_cpu_kernel.py"
  "tests_triton_cuda|1500|EA_TRITON_DEVICE=cuda python -m pytest -q -p no:cacheprovider tests/test_triton.py"
  "fixture_triton_cuda|1500|python scripts/fixture.py check --backend triton:cuda"
  "fixture_tritontc_cuda|1500|python scripts/fixture.py check --backend tritontc:cuda"
  "bench_n1024|900|python scripts/bench_decode.py --rows 16 --keys 1024 --out $OUT/bench_n1024.md"
  "bench_n4096|1200|python scripts/bench_decode.py --rows 16 --keys 4096 --out $OUT/bench_n4096.md"
  "bench_n16384|1800|python scripts/bench_decode.py --rows 16 --keys 16384 --out $OUT/bench_n16384.md"
)

resubmit() {
  if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
    sbatch -A "${SLURM_JOB_ACCOUNT:-gts-agarg35}" -q embers \
      --export=ALL,EA_RUN="$RUN",EA_CHAIN=$((CHAIN + 1)) \
      -o "$HOME/scratch/ea_jobs/exactattn-$RUN-%j.out" "$REPO/scripts/pace/gpu_check.sbatch" \
      && echo "chain: submitted job $((CHAIN + 1))/$MAX_CHAIN"
  else
    echo "chain: reached EA_MAX_CHAIN=$MAX_CHAIN, not resubmitting"
  fi
}

write_summary() {
  {
    echo "# PACE run $RUN"; echo
    echo "| stage | result | last lines |"; echo "|---|---|---|"
    for entry in "${STAGES[@]}"; do
      name="${entry%%|*}"
      if [ -f "$OUT/$name.done" ]; then res=$(cat "$OUT/$name.done"); else res="pending"; fi
      last=$(tail -n 2 "$OUT/$name.log" 2>/dev/null | tr '\n|' ' /' | cut -c1-160)
      echo "| $name | $res | $last |"
    done
    echo; cat "$OUT"/bench_n*.md 2>/dev/null | grep -E '^\| [0-9]' | sed 's/^/bench: /'
  } > "$OUT/summary.md"
}

CHILD=0
on_term() {
  echo "SIGTERM at $(date) - stopping stage, leaving it not-done, resubmitting"
  [ "$CHILD" -ne 0 ] && kill -TERM "$CHILD" 2>/dev/null
  write_summary
  resubmit
  exit 143
}
trap on_term TERM

remaining() { echo $(( BUDGET_S - ($(date +%s) - START) )); }

pending=0
for entry in "${STAGES[@]}"; do
  name="${entry%%|*}"; rest="${entry#*|}"; cap="${rest%%|*}"; cmd="${rest#*|}"
  [ -f "$OUT/$name.done" ] && continue
  left=$(remaining)
  if [ "$left" -lt 90 ]; then pending=1; continue; fi
  t=$(( cap < left ? cap : left ))
  echo "== $name (cap ${t}s) $(date +%T) on $(hostname) job ${SLURM_JOB_ID:-local} chain $CHAIN"
  timeout --signal=TERM --kill-after=20 "$t" bash -c "$cmd" > "$OUT/$name.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; rc=$?; CHILD=0
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    n=$(( $(cat "$OUT/$name.timeouts" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$OUT/$name.timeouts"
    echo "   timeout #$n"
    if [ "$n" -ge "$MAX_TIMEOUTS" ]; then echo "TIMEOUT" > "$OUT/$name.done"; else pending=1; fi
  else
    echo "rc=$rc" > "$OUT/$name.done"; echo "   rc=$rc"
  fi
done

write_summary

if [ "$pending" -eq 1 ]; then resubmit; else echo "all stages done"; touch "$OUT/ALL_DONE"; fi
