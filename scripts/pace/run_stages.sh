#!/bin/bash
# Preemption-safe stage runner for PACE's free embers QOS.
#
# On embers, a job is guaranteed its first hour. After that a priority job can take the GPU,
# and the embers job is CANCELLED, not requeued, with about 30 s warning. So:
#   * each job asks for <= 55 min, inside the guarantee (shorter jobs also backfill sooner);
#   * the work is split into small stages (one fixture head, one test group, one benchmark
#     shape), and each finished stage writes a .done marker. Finished stages are never re-run;
#   * at startup a job queues its successor with --dependency=afterany:<this job>, so the
#     chain continues even if this job is killed outright with no chance to react. When every
#     stage is done, the spare successor is cancelled;
#   * on SIGTERM it stops the running stage (left not-done), writes the summary and exits;
#   * each stage runs in its own process group, and a timeout kills the whole group. Killing only
#     the stage's shell left its python running, still holding the (exclusive-mode) GPU, so every
#     later stage of that job failed with "CUDA-capable device(s) is/are busy or unavailable";
#   * a stage that hits its cap EA_MAX_TIMEOUTS times is recorded as TIMEOUT, so the chain cannot
#     loop forever. The chain is at most EA_MAX_CHAIN jobs long.
#
# Results: results/pace/run-<RUN>/<stage>.log, .done, .attempts, plus summary.md.
set -uo pipefail

REPO="${EA_REPO:-$HOME/scratch/exactattention}"
RUN="${EA_RUN:?set EA_RUN to a run name}"
CHAIN="${EA_CHAIN:-1}"
MAX_CHAIN="${EA_MAX_CHAIN:-8}"
BUDGET_S="${EA_BUDGET_S:-3000}"          # leave about 5 min of the 55-min limit for setup
MAX_TIMEOUTS="${EA_MAX_TIMEOUTS:-2}"
OUT="$REPO/results/pace/run-$RUN"
mkdir -p "$OUT"
cd "$REPO"
source "${EA_VENV:-$HOME/scratch/exactattn-venv}/bin/activate"
START=$(date +%s)
FIX="python scripts/fixture.py check"
PYT="python -m pytest -q -p no:cacheprovider"

# name | per-stage cap in seconds | command
STAGES=(
  "env|600|nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv && python -c \"import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.get_device_name(0))\""
  "fixture_cpp|600|$FIX --backend cpp"
  "fixture_torch_cuda_0_0|900|$FIX --backend torch:cuda --only 0_0"
  "fixture_torch_cuda_27_11|900|$FIX --backend torch:cuda --only 27_11"
  "tests_cpu_kernel|900|$PYT tests/test_cpu_kernel.py"
  "tests_torch_cuda|900|$PYT tests/test_torch_impl.py"
  "tests_triton_windowed|1500|EA_TRITON_DEVICE=cuda $PYT tests/test_triton.py -k 'matches_reference and False'"
  "tests_triton_tc|1500|EA_TRITON_DEVICE=cuda $PYT tests/test_triton.py -k 'matches_reference and True'"
  "tests_triton_rest|1500|EA_TRITON_DEVICE=cuda $PYT tests/test_triton.py -k 'not matches_reference'"
  "fixture_triton_0_0|1500|$FIX --backend triton:cuda --only 0_0"
  "fixture_triton_27_11|1500|$FIX --backend triton:cuda --only 27_11"
  "fixture_tritontc_0_0|1500|$FIX --backend tritontc:cuda --only 0_0"
  "fixture_tritontc_27_11|1500|$FIX --backend tritontc:cuda --only 27_11"
  "fixture_tritontcpv_0_0|1500|$FIX --backend tritontcpv:cuda --only 0_0"
  "fixture_tritontcpv_27_11|1500|$FIX --backend tritontcpv:cuda --only 27_11"
  "bench_n1024|900|python scripts/bench_decode.py --rows 16 --keys 1024 --out $OUT/bench_n1024.md"
  "bench_n4096|1200|python scripts/bench_decode.py --rows 16 --keys 4096 --out $OUT/bench_n4096.md"
  "bench_n16384|1800|python scripts/bench_decode.py --rows 16 --keys 16384 --out $OUT/bench_n16384.md"
)

# EA_PLAN=flash: ExactAttention vs FlashInfer / FlashAttention at serving shapes
# (scripts/bench_vs_flashinfer.py). FlashInfer JIT-compiles its kernels on first use into
# ~/.cache/flashinfer, which persists across jobs, so a preempted warmup is not lost.
BF="python scripts/bench_vs_flashinfer.py"
if [ "${EA_PLAN:-gpu_check}" = "flash" ]; then
  # FlashInfer JIT needs nvcc matching torch's CUDA (13.2); PACE modules stop at 12.8, so use the
  # pip nvidia-cuda-nvcc in the venv (pip install "nvidia-cuda-nvcc>=13.2,<13.3").
  SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  if [ -x "$SP/nvidia/cu13/bin/nvcc" ]; then
    export CUDA_HOME="$SP/nvidia/cu13" PATH="$SP/nvidia/cu13/bin:$PATH"
  fi
  STAGES=(
    "env|600|nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv && which nvcc; python -c \"import torch, triton, flashinfer; print(torch.__version__, torch.version.cuda, triton.__version__, flashinfer.__version__, torch.cuda.get_device_name(0))\""
    "tests_segments|1200|EA_TRITON_DEVICE=cuda $PYT tests/test_triton.py -k 'segments or slicing or pv'"
    "fixture_tritontcpv_27_11|1500|$FIX --backend tritontcpv:cuda --only 27_11"
    "flashinfer_warmup|2400|$BF --batch 1 --keys 1024 --drafts 1,8 --modes int64 --iters 2 --out $OUT/warmup.md"
    "decode_b1|1500|$BF --batch 1 --keys 1024,4096,16384,65536 --drafts 1 --out $OUT/decode_b1.md"
    "decode_b8|1500|$BF --batch 8 --keys 1024,4096,16384 --drafts 1 --modes tc,tcpv --out $OUT/decode_b8.md"
    "decode_b32|1500|$BF --batch 32 --keys 1024,4096,16384 --drafts 1 --modes tc,tcpv --out $OUT/decode_b32.md"
    "verify_t8|1500|$BF --batch 1,8 --keys 1024,4096,16384 --drafts 8 --modes tc,tcpv --out $OUT/verify_t8.md"
  )
fi

stage_name() { echo "${1%%|*}"; }
count_pending() {
  local n=0 e
  for e in "${STAGES[@]}"; do [ -f "$OUT/$(stage_name "$e").done" ] || n=$((n + 1)); done
  echo "$n"
}

write_summary() {
  {
    echo "# PACE run $RUN"; echo
    echo "Updated $(date '+%F %T') by job ${SLURM_JOB_ID:-local} (chain $CHAIN) on $(hostname)."; echo
    echo "| stage | result | attempts | last lines |"; echo "|---|---|---|---|"
    for e in "${STAGES[@]}"; do
      local name; name=$(stage_name "$e")
      local res="pending"; [ -f "$OUT/$name.done" ] && res=$(cat "$OUT/$name.done")
      local att; att=$(cat "$OUT/$name.attempts" 2>/dev/null || echo 0)
      local last; last=$(tail -n 2 "$OUT/$name.log" 2>/dev/null | tr '\n|' ' /' | cut -c1-160)
      echo "| $name | $res | $att | $last |"
    done
    echo; cat "$OUT"/bench_n*.md 2>/dev/null | grep -E '^\| [0-9]' | sed 's/^/bench: /'
    for f in "$OUT"/decode_*.md "$OUT"/verify_*.md; do [ -f "$f" ] && { echo; tail -n +5 "$f"; }; done
  } > "$OUT/summary.md"
}

submit_next() {   # $1 = extra sbatch args (e.g. a dependency)
  if [ "$CHAIN" -ge "$MAX_CHAIN" ]; then echo "chain: at EA_MAX_CHAIN=$MAX_CHAIN, no successor"; return; fi
  sbatch --parsable -A "${SLURM_JOB_ACCOUNT:-gts-agarg35}" -q embers $1 \
    --export=ALL,EA_RUN="$RUN",EA_CHAIN=$((CHAIN + 1)),EA_PLAN="${EA_PLAN:-gpu_check}" \
    -o "$HOME/scratch/ea_jobs/exactattn-$RUN-%j.out" "$REPO/scripts/pace/gpu_check.sbatch" 2>/dev/null
}

# Nothing left: mark done and stop. Covers a successor that starts after its predecessor
# already finished everything.
if [ "$(count_pending)" -eq 0 ]; then write_summary; touch "$OUT/ALL_DONE"; echo "all stages already done"; exit 0; fi

# Queue the successor now, so the chain survives even a hard kill of this job.
NEXT=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  NEXT=$(submit_next "--dependency=afterany:$SLURM_JOB_ID")
  [ -n "$NEXT" ] && echo "chain: pre-queued successor job $NEXT (chain $((CHAIN + 1)))"
fi

CHILD=0
on_term() {
  echo "SIGTERM at $(date) - stopping stage (left not-done)"
  [ "$CHILD" -ne 0 ] && kill -TERM -- -"$CHILD" 2>/dev/null
  write_summary
  if [ -z "$NEXT" ] && [ -n "${SLURM_JOB_ID:-}" ]; then NEXT=$(submit_next ""); echo "chain: submitted $NEXT"; fi
  exit 143
}
trap on_term TERM

remaining() { echo $(( BUDGET_S - ($(date +%s) - START) )); }

for e in "${STAGES[@]}"; do
  name="${e%%|*}"; rest="${e#*|}"; cap="${rest%%|*}"; cmd="${rest#*|}"
  [ -f "$OUT/$name.done" ] && continue
  left=$(remaining)
  [ "$left" -lt 90 ] && continue
  t=$(( cap < left ? cap : left ))
  echo $(( $(cat "$OUT/$name.attempts" 2>/dev/null || echo 0) + 1 )) > "$OUT/$name.attempts"
  echo "== $name (cap ${t}s) $(date +%T) on $(hostname) job ${SLURM_JOB_ID:-local} chain $CHAIN"
  rm -f "$OUT/$name.killed"
  setsid bash -c "$cmd" > "$OUT/$name.log" 2>&1 &        # new session: pgid = pid
  CHILD=$!
  ( sleep "$t"; touch "$OUT/$name.killed"; kill -TERM -- -"$CHILD" 2>/dev/null
    sleep 20; kill -KILL -- -"$CHILD" 2>/dev/null ) &
  WD=$!
  wait "$CHILD"; rc=$?
  kill "$WD" 2>/dev/null; wait "$WD" 2>/dev/null
  kill -KILL -- -"$CHILD" 2>/dev/null                     # reap anything the stage left behind
  CHILD=0
  if [ -f "$OUT/$name.killed" ]; then
    n=$(( $(cat "$OUT/$name.timeouts" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$OUT/$name.timeouts"
    echo "   timeout #$n"
    [ "$n" -ge "$MAX_TIMEOUTS" ] && echo "TIMEOUT" > "$OUT/$name.done"
  else
    echo "rc=$rc" > "$OUT/$name.done"; echo "   rc=$rc"
  fi
  write_summary   # checkpoint the summary after every stage
done

write_summary
if [ "$(count_pending)" -eq 0 ]; then
  touch "$OUT/ALL_DONE"; echo "all stages done"
  [ -n "$NEXT" ] && scancel "$NEXT" && echo "chain: cancelled spare successor $NEXT"
else
  echo "chain: $(count_pending) stages pending; successor ${NEXT:-none} continues"
fi
