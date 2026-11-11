# Design log

Decisions, measurements and dead ends, kept in order. Newest last.

## D1. Exact spec, certified fast path (Ziv's strategy)

There were two ways to bound the accumulator width:

1. **Put truncation in the spec**, for example ReproBLAS-style binned sums on a global grid.
   This is cheaper, but the spec then depends on window parameters, and it is no longer
   "the" correctly rounded attention.
2. **Keep the spec exact, and certify a windowed fast path.** Kernels carry rigorous error
   bounds and return a value only when the bounds decide the final rounding. Otherwise they
   fall back.

Option 2 was chosen. The intermediate state may depend on the partition; the returned bits
cannot. The spec is free of implementation parameters, and the window width becomes a pure
performance knob, traded against fallback rate.

## D2. Scale-free weights

`w_j = RN_FP24∞(2^{t_j})` has an unbounded exponent, so no max subtraction appears in the
spec. Implementations use running integer reference exponents. Merging two partials is an
exact power-of-two rescale plus counted truncation (SPEC §5.2). This is what makes split-KV
merges order-free.

## D3. Anchor each N channel at its largest term (measured)

The first model anchored channel `c` of `N` at `R + max_j lead(v_jc)`, where `R` is the
leading exponent of the largest weight. That is only an upper bound on the largest term
`|w_j · v_jc|`, and it can be loose by the whole logit spread. That happens when the
dominant-weight key has `v_jc = 0` and a large `v` sits on a negligible key. Every real
term then falls below the window, and the row cannot be certified.

Measured row-fallback counts, out of 12 rows: d = 32, d_v = 24, n = 150, 3 splits, 16-key
tiles. "q/k" and "v" name the input distributions from `tests/gen.py`.

| q/k | v | 64-bit, anchored at R + max lead(v) | 64-bit, per-term anchor | 128-bit, per-term anchor |
|---|---|---|---|---|
| wide | wide | 10 | 0 | 0 |
| wide | dupes | 12 | 0 | 0 |
| outlier | wide | 4 | 0 | 0 |
| outlier | dupes | 3 | 0 | 0 |
| others | any | 0 | 0 | 0 |

The fix anchors each channel at `max_j (lead(w_j) + 1 + lead(v_jc))`, an upper bound for each
term's leading bit. It adds no work on a GPU: the per-key weight exponent and the
per-element `v` exponent are both already in registers.

## D4. Window widths on real activations (Q1 answered for one model)

Source: Qwen3-0.6B, 2048 tokens of WikiText-2, 2,688 rows for distributions, 150 rows through
the bit-accurate model ([results/width_study_qwen3-0.6b.md](../results/width_study_qwen3-0.6b.md)).

- **Score cancellation** is small: p99.9 is 8 bits, max 19.
- **Logit spread** per row is at most 70.5 (log2 units). So a 128-bit Z accumulator
  (24 + 71 + 11 bits) would be *exact* with no truncation at all.
- **Output cancellation:** p99.9 7.8 bits, max 17.5.
- **Certification cliff:** 44-bit windows give 0 fallbacks over 138,336 scores and 150
  rows; 40 bits gives 8/150; 36 bits gives 68/150.

Consequence: one int64 per accumulator leaves about 20 bits of margin. The full-model run
(917,504 rows, int64) had zero fallbacks.

- **Exponent spread inside a K row:** max 29 bits, p99 18; q rows max 19.
- **Spread of product exponents within one q·k:** max 33 bits.

This prices the int8-slice tensor-core path for scores (Q2): exactness everywhere would need
about 5–6 slices for k. A certified windowed approach (fewer slices plus a counted truncation
error) is the realistic route.

## D5. Accuracy: the fp32 logit is the remaining weak point

In the full-model run, 491 of 117M output elements are more than 2 bf16 ulp from float64
attention. All of them are heavily cancelling outputs (median `|o| / max|v|` = 1.5e-7). There,
the single fp32 rounding of `t_j` (SPEC step 2) is amplified by about 2^17. Standard fp32
attention has the same issue and is worse: 1,652 such elements, plus 3× more 1-ulp ones.

Spec v1 option: round `t_j` to a wider format, for example 48 bits, and evaluate `2^{t_j}`
from it. This costs one more limb in the exp2 argument and none in the accumulators.

## D6. Intel GPU route

- **What works:** PyTorch 2.6.0 XPU on Windows runs on the Arc 140V with the installed driver
  (32.0.101.6299). int64 shifts are arithmetic for negatives, and fp64 is supported.
- **What doesn't:**
  - PyTorch 2.15 XPU aborts on the first kernel launch with this driver.
  - WSL lacks Intel's compute runtime, and installing it needs sudo.
- **Approach taken:** `torch_impl.py` expresses the algorithm as integer tensor ops, so it
  runs unchanged on any device.
- **Next:** a driver update would enable PyTorch 2.15 with `triton-xpu`, which ships for
  Windows, so the fused Triton kernel could run on the iGPU too.

## D7. Fused Triton kernel, v0 (correctness first)

`triton_kernels.py` has two kernels in the FlashDecoding shape: split and combine. The split
kernel's grid is (query block, KV split, value-channel block). Every program recomputes
scores for its channel block, which costs about 4× score work at d_v = 128 with 32-wide
channel blocks. That is acceptable at decode, where scores are a small fraction of the bytes.

Scores, P·V and the final division are all int64 CUDA-core work held in 3D blocks. This is
**not** the performance design; tensor cores come in v1 (Q2, Q3). It is the correctness
oracle for v1, and it is portable: the same source should run on NVIDIA, Intel and AMD.

Tests run under `TRITON_INTERPRET=1`. Splits 1 through 7 with permuted merge orders are
bit-identical to the reference.

Bug found during testing:

- A merge order shorter than the actual split count made the combine kernel read past the
  partial buffer. The kernel flagged every row as uncertified rather than returning garbage,
  so certification caught a host-side bug.
- Fixed: the wrapper now always passes a permutation of the actual splits.

## Open questions

- **Q1. Real activations.** How wide must the score, Z and N windows be on real model
  activations (attention sinks, massive activations) to keep fallbacks under about 1e-6 per
  row? → Step 5: a Qwen/Llama study on CPU.
- **Q2. Exact scores on tensor cores.** bf16 q·k products are exact in 16 bits. An exact
  128-term sum needs about 16 + exponent spread + 7 bits.
  - Option A: int8 slices on integer tensor cores, with exact int32 partial sums.
  - Option B: per-key block exponents stored at KV-write time.
  The spread distribution on real K caches decides between them.
- **Q3. P·V on tensor cores.** `w` is 24 bits and `v` is a bf16 significand, but the
  exponents vary per key and per element. Converting a V tile to block fixed point loses
  bits for small elements; that loss is counted as truncation error and certified. How many
  slices does it take at a given fallback rate?
- **Q4. Fast certified exp2 in fp32-only arithmetic,** for GPUs with slow fp64. There are
  about 1.07e9 fp32 inputs in [0, 1), so the implementation can be tested exhaustively on CPU.
