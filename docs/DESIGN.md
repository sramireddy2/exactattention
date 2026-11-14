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

## D8. Scores on tensor cores via int8 slices (Q2 answered)

Each q row and each K row is stored as block fixed point: one exponent per row, plus `S`
signed 7-bit digits per element as int8 (`slices.py`). Each slice pair `(s, t)` is one int8
MMA with exact int32 accumulation (`|·| < 128 · 127² < 2^21`). Pairs are combined in int64
with shifts of `7(s+t)`; 3×4 slices give at most 2^56, which fits.

The truncation bound uses per-row L1 norms: `2^qlsb·‖k‖₁ + 2^klsb·‖q‖₁`, applied only to
rows that lost bits. It certifies `t = RN32(α·s)` exactly as the windowed path does. K slices
are computed once, when K is written to the cache.

Measured on 57,800 real scores ([results/slice_study_qwen3-0.6b.md](../results/slice_study_qwen3-0.6b.md)):

| slices | int8 MMA k-steps per score | uncertified |
|---|---|---|
| 3×3 | 9 | 9.2e-2 |
| 3×4 | 12 | 2.1e-4 |
| 4×4 | 16 | 2.1e-4 |
| 5×5 | 25 | 0 |

No certified score was ever wrong. At 3×4 the cost is 12 int8 k-steps, about 6 bf16-MMA
equivalents.

- **Prefill:** roughly 6× the QK^T tensor work.
- **Decode:** a row group of `g` queries (GQA group × draft length) reads 2 bytes per key
  element and does `12·g` int8 MACs on it. For g ≤ 16 that stays below the A100/H100 int8
  ridge point (about 300–600 ops/byte), so decode remains memory-bound.

The per-score fallback is required, not optional:

- In real layer 0, one K row in 1,000 has an exponent spread that needs 5 slices. Every
  later query attends to it.
- A row-level fallback would therefore fire on every row. The kernel recomputes only tiles
  that contain an uncertified score, on the windowed int64 path, and only those entries take
  the recomputed value.

Testing lesson: the first tensor-core tests passed **only through the fallback**; the
pure-int8 path certified nothing on synthetic outlier data. A `TC_FALLBACK=False` switch and
a compact-range test now prove the int8 path is exact by itself. On real head 27:11 it
certifies 32/32 rows bit-exactly with no fallback.

## D9. P·V on tensor cores: harder than scores (Q3, measured)

Weights are anchored once per row at the largest term (tensor-core friendly). V rows are
block fixed point. `N_c = Σ_j W_j·Vfix_jc` is computed as int8 slice MMAs, and the result is
certified as `RN_bf16(N_c / Z)` with Z exact.

Measured on 3,072 real outputs ([results/pv_slice_study_qwen3-0.6b.md](../results/pv_slice_study_qwen3-0.6b.md)):

| Sw × Sv | bound | uncertified |
|---|---|---|
| 4×3 | per-row | 3.2% |
| 4×3 | per-channel | 1.2% |
| 5×3 | either | 0.42% |
| 4×4 | per-channel | 0.78% |

Conclusions:

1. **V truncation dominates.** Exponent spread within a V row reaches 25 bits (D4), so
   3 digits lose bits. The current V bound charges `W_j + 1` to *every* channel of a key
   whose row lost any bits.
   - Next: a per-element loss mask, which is one more 0/1 int8 MMA.
   - Or: block exponents per (key tile, channel) instead of per key row.
2. **The fallback must be per (row, channel block), not per row.** At 0.4% per element and
   128 channels, a row-level fallback would fire on about 40% of rows.
3. **P·V costs more than QK** at 15–20 int8 k-steps against 12 for scores. Decode still has
   headroom below the int8 ridge point for g ≤ 16.

Status: studied in numpy, not yet in the Triton kernel. P·V is still int64 CUDA-core work there.

## D10. P·V on tensor cores, implemented: 5×3 row layout with per-element bounds

The bounds are tightened from D9. Weight loss is charged per channel (`lostW mask × |V|` MMA),
and V loss **per element** (`W × lostV mask` MMA plus a `vis × lostV` MMA). That costs 15
value MMAs plus 8 bound MMAs per (key, channel).

On 3,072 real outputs ([results/pv_slice_study_qwen3-0.6b.md](../results/pv_slice_study_qwen3-0.6b.md)):

| layout | uncertified |
|---|---|
| row-elem 5×3 | **0** |
| row-elem 4×3 | 7.8e-3 |
| row-chan 5×3 (D9) | 4.2e-3 |
| col-elem 4×3 (V exponent per tile × channel) | 4.2e-3 |

Per-tile column exponents do not help: V spread *along a channel* is not smaller than
within a row.

Implementation (`_tc_pv` in `triton_kernels.py`, `USE_TC_PV`):

1. Weights times `2^vlsb` are anchored per row at the tile's largest term, then truncated to 5
   unsigned 7-bit digits.
2. V is precomputed once per cache write (`tc_v_slices`).
3. The tile result is re-anchored to the partial format at `TNR = base + ANCH`, with
   `ANCH = 7·SW + 7·SV + log2(BLOCK_N)`. Every byte stays inside int64; the host asserts
   `ANCH + 1 < 63`, which caps 5×3 at BLOCK_N ≤ 32.
4. Merges and certification are unchanged.

Tests (Triton interpreter, 120 passing in total):

- **Real activations:** layer 27 rows are bit-exact with both int8 paths on, and every row
  certified.
- **Compact synthetic data, score fallback off:** 16/16 rows certified, bit-exact.
- **Adversarial synthetic data:** outlier scores with repeated V values, or "wide" V spanning
  48 bits of exponent. More rows are flagged here, never wrong ones.

The in-kernel second tier for uncertified (row, channel-block) pairs is D12.

## D11. First NVIDIA run, segments, and serving-shape benchmarks

**First hardware run** (PACE embers, RTX PRO 6000 Blackwell, sm_120, 2026-11-12). The C++,
torch:cuda, Triton int64 and Triton tensor-core-score paths reproduce both fixture SHA-256
hashes with 0 fallback rows, so the bits match the Intel laptop's. The P·V tensor-core path
did not compile: an int8 `tl.dot` needs a contraction size K ≥ 32 (`mma.m16n8k32`), and P·V
contracts over the key tile, which was 16. The CPU interpreter does not check this. Since the
tile size cannot change the output (SPEC §5.3), the launcher now raises `block_n` to 32 for
P·V. ANCH = 7·5 + 7·3 + 5 = 61 still fits int64.

**Segments.** A launch used to cover one KV head. Serving needs B sequences × H_kv heads per
step, so the split kernel takes a segment axis: grid axis 0 is (segment, row block), each
segment has its own R rows and n keys, stored one after another. A segment is one
(sequence, KV head), and its rows are the GQA group's query heads, times T for tree or draft
verification. The combine kernel already works per row. Batched output equals one launch per
segment bit for bit, flags included (`test_segments_equal_separate_launches`).

**Per-step q slicing on the GPU.** `tc_slices_device` gives the same digits as the numpy
slicer, so the per-step q slicing, and K/V slicing at cache write, need no host round trip and
can be captured in a CUDA graph. The launcher also builds the default merge order on the
device for the same reason.

**Benchmark** (`scripts/bench_vs_flashinfer.py`). FlashInfer's paged batch decode and prefill
(causal verify), FlashAttention-2 through PyTorch SDPA and flash-attn (if installed) are
compared with the three exact paths. Shapes: 16 query heads over 8 KV heads, d = 128, B ∈
{1, 8, 32}, n up to 64K, T ∈ {1, 8}. Each backend is timed in a CUDA graph. Every exact
result passes a correctness gate against fp32 attention: a certified row more than 2⁻⁸
relative off counts as WRONG.

## D12. Tier 2 on the device: int64 P·V for the pairs tensor cores cannot certify

Three tiers, each certified:

| tier | where | what | when |
|---|---|---|---|
| 1 | split + combine kernels | int8 tensor-core scores and P·V | always |
| 2 | the same two kernels, `RERUN=True` | int64 P·V (CUDA cores) for one (row, channel block) pair | that pair's final rounding failed and no split flagged the row |
| 3 | host | exact reference | anything still flagged |

Tier 2 is sound because every path encloses the same exact N, and a certified rounding of an
enclosure is the unique correctly rounded value. So it cannot change any bits; it only
certifies more pairs.

How it runs:

1. After tier 1, `NEED = combine_flag & (no split flag)` is formed on the device. Split flags are
   NaN/inf inputs, uncertified logits or weights, and overflow. int64 P·V fixes none of these,
   so those rows go straight to tier 3.
2. The split kernel runs again with int64 P·V over the same grid and splits. A program with no
   NEED row in its (row block, channel block) walks zero tiles. One that has some recomputes
   its N partials and overwrites them for the NEED rows only. Z partials are not rewritten:
   they are the same in both tiers.
3. The combine kernel runs again and rewrites the output and flags of NEED pairs only.

Both launches have fixed grids and no host sync, so the whole step stays CUDA-graph
capturable. With nothing flagged, the cost is two near-empty launches. Tier 2 uses the
caller's key tile, not the 32 that tier 1 needs for int8 MMAs. That keeps the int64
M×N×DV product in registers. The split boundaries are unchanged because the split length
is a multiple of both tiles.

Measured on "wide" V (48-bit exponent spread per row), the worst synthetic case
(`test_pv_tier2_recovers_uncertified_pairs`, Triton interpreter):

| data | rows flagged, tier 1 only | rows flagged, tiers 1 and 2 | int64 P·V alone | pairs rerun |
|---|---|---|---|---|
| normal q/k, wide V | 15 / 16 | **0** | 0 | 27 / 32 |
| outlier q/k, wide V | 16 / 16 | **0** | 0 | 31 / 32 |

Every row certified by tiers 1 and 2 is bit-exact against the reference. On real activations
tier 1 already certifies every row (D10), so tier 2 is a safety net there, not a hot path.

Not covered: uncertified *scores* after the windowed int64 recompute, and uncertified weights
(the exp2 margin). Both have been 0 on real rows. A wider-logit spec (v1) would be the next tier
for them.

## D13. First speed numbers, and the first optimization pass

**Baseline.** Run 1112-flash: RTX PRO 6000 Blackwell, 16 query heads over 8 KV heads, d = 128,
CUDA graphs. FlashInfer's JIT build failed in that run (fixed since), so FlashAttention-2
through PyTorch SDPA is the reference.

| B | n | FA2 (SDPA) | exact int64 | exact tc | exact tcpv |
|---|---|---|---|---|---|
| 1 | 1K | 0.011 ms | 20.4 ms | 2.71 ms | 2.78 ms |
| 1 | 16K | 0.035 ms | 322 ms | 32.8 ms | 10.1 ms |
| 1 | 64K | 0.193 ms | 1268 ms | 128 ms | 24.8 ms |
| 8 | 16K | 0.363 ms | 2507 ms | 490 ms | 440 ms |
| 32 | 4K | 0.361 ms | 2470 ms | 425 ms | 339 ms |

Every row certified by these runs was correct (the fp32 gate never fired). The exact kernels are
100–900× slower, which is far too slow to be memory-bound. The B = 1, n = 16K tcpv step, for
example, works out to hundreds of microseconds per 32-key tile per program. Suspects, from the
code:

1. **A full bf16 decode of every K tile, on every path.** That is a (32, 128) int64 mantissa
   and exponent pair, about 64 KB live per program, far beyond the register file, so it likely
   spills. The tensor-core paths only needed it for NaN/inf checks and for the rare windowed
   recompute. The same applies to V on the P·V tensor-core path.
2. **Per-step q slicing as about 40 small torch kernels.** At a few µs each in a graph, that
   alone is a floor above FA2's whole step.
3. **Every channel block recomputes every score.** With block_dv = 32 and d_v = 128 the score
   path runs 4 times.
4. **Per-element fp64 in the score error bound** (`l1 * 2^-lsb` per element).
5. **Padded rows.** Decode has only 2 rows per segment (GQA group 2), but `tl.dot` needs
   M ≥ 16, so 7/8 of the elementwise (row, key) work is padding. Not addressed yet.

**This pass:**

- K slices carry a per-key NaN/inf bit (bit 1 of `KLOST`), and V slices carry a per-element one
  (bit 1 of `VLOST`). The tensor-core paths no longer load or decode bf16 K or V. bf16 K is read
  only inside the windowed recompute, for tiles that have an uncertified score. The slicers
  zero NaN/inf elements before slicing, since those rows are flagged anyway.
- q is sliced inside the split kernel, once per program, from the bf16 rows it already loads
  (`_slice_q`, same digits as the numpy slicer). A decode step is now the layout copy, the split
  kernel, the combine kernel and tier 2's small ops.
- K slices store `l1 * 2^-lsb` (an exact power-of-two scaling), and q's is computed once per
  program, so the per-element bound is two selects and an add.
- `launch` takes `num_warps`. The benchmark times each exact mode over (block_dv, num_warps)
  configs, including block_dv = 128, which computes each score once. It reports the fastest
  config with the split kernel's register and spill counts. It also checks that every config
  gives the same bits on rows both certify, which SPEC §5.3 requires.
- `test_nan_inf_inputs_are_flagged` covers the new NaN/inf path in all three modes.

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
