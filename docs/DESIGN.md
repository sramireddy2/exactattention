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
   M ≥ 16, so 7/8 of the elementwise (row, key) work is padding. Addressed in D14.

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

## D14. Small row blocks: digit slices fill the MMA rows instead of padding

Suspect 5 of D13. In a decode step each segment (sequence × KV head) has `T × group` query rows:
2 for one token with GQA group 2. int8 `tl.dot` needs M ≥ 16, so the row block was 16, and 14 of
16 rows did all of the elementwise work (int64 accumulators, the certified RN32, exp2, weight
rounding, the P·V bounds) on padding.

The row block is now its own size, `block_m` (a power of two). It defaults to the segment's
rows, capped at 16. The MMAs get their 16 rows from the digit slices instead:

- **Scores.** q has SQ digits, and each is an independent int8 operand. With `PACK = 16 / block_m`,
  PACK digits are stacked along the MMA rows (row `jl·block_m + r` holds digit `g·PACK + jl` of
  query row r). One MMA per K digit then gives all of those digit products. They are unpacked
  with a reshape to (PACK, block_m, keys), shifted by `7(j + t)` in int64, and summed. The
  sum is exact, so the order of the terms does not matter.
- **P·V.** The SW weight digits, the lost-weight mask and the visibility mask become SW + 2
  "slots" of one packed weight operand. Each V operand (V digit b, |V digit b|, the lost-V mask,
  the nonzero mask) is multiplied by it once, and each sum takes the slots it needs. At
  block_m = 2, a tile's P·V takes 8 MMAs where it took 25.
- At block_m = 16, PACK = 1, and the code is the old one-digit-per-MMA layout.
- The combine kernel works on flattened rows and keeps 16-row blocks.

The elementwise work per tile is now `block_m × keys` instead of `16 × keys`, which is 8× less
at decode with group 2. The row block is a tiling choice, so SPEC §5.3 says it cannot change any
bit. `test_row_block_size_does_not_change_bits` checks bits and flags at block_m = 1, 2, 4, 8
against 16 in all three modes. The benchmark keeps a block_m = 16 config for tc and tcpv, both as
an A/B on the GPU and as the same check at serving shapes.

## D15. Weights without float64: integer exp2 with a proven bound (Q4)

The per-key loop used float64 in three places: exp2 for every weight, the int64 → fp64
conversion inside `_bitlen` (several per score), and `q · 2^qe` to build each fp32 logit. The
RTX PRO 6000 runs fp64 at 1/64 of the fp32 rate, and libdevice's exp2 alone is about 20 fp64
operations per weight.

**Weights.** w = RN24(2^t) for fp32 t, in int64 arithmetic whose products all have 32-bit
factors (`_exp2_weight`; `exp2.exp2_weight_int_np` is the same computation in numpy):

1. Split t = n + f from its bits. An fp32 t with |t| ≥ 2^-27 is a multiple of 2^-50, so
   F = f · 2^50 is an exact integer. For smaller |t|, RN24(2^t) = 1.
2. 2^f = 2^(j/1024) · 2^r, with j = F >> 40 and r < 2^-10. A 1024-entry table holds
   round(2^(j/1024) · 2^61). `test_exp2_table_is_correctly_rounded` checks every entry with
   exact integers: (2T − 1)^1024 < 2^(1024·62 + j) < (2T + 1)^1024.
3. 2^r − 1 = x + x²/2 + x³/6 + ρ, with x = r ln 2 < 2^-10.5 and ρ < 2^-46.7, in fixed point with
   unit 2^-42.
4. Y = T_j + T_j · P approximates 2^f · 2^61. M = round(Y / 2^38) is the 24-bit significand.

**Error bound.** Absolute error on 2^f, in units of 2^-42:

| Source | Bound |
|---|---|
| truncating r to 42 bits, ln 2 rounded to 31 bits, the floor in x | 2.70 |
| floors in x², x³, /2, /6 | 2.01 |
| dropped x⁴ term | 0.04 |
| × T_j ≤ 2 | ×2 |
| T_j's own rounding, T_j truncated to 31 bits for the product, last floor | 2.8 |

The total is under 13 · 2^-42, i.e. 13 · 2^19 units of Y. `EXP2_ERR` = 2^23 (1.26× margin). A
weight is certified when Y mod 2^38 is farther than that from 2^37: a half-integer of
2^f · 2^23 can then not lie between Y and the true value, and 2^f is irrational for f ≠ 0, so
there are no ties. Over random f plus every table boundary, the worst measured error is
2^21.9 units, below the bound of 2^22.7 (`test_exp2_fix_error_bound`). This is a proof, not an
assumption about a libm, unlike the old path (D7).

About 2^-14 of weights fall within the margin. Their tile is recomputed with the old float64
exp2, and a weight that path cannot certify flags its row as before. At 2 × 32 weights per
tile, about 0.4% of tiles take that branch.

**Other float64 removed from the per-key loop:**

- `_bitlen` converts the nonzero 32-bit half through uint32 → fp32, which runs at full rate.
  The fp32 exponent is the bit length, or one too many after a round-up, which is detected and
  corrected. `test_kernel_integer_helpers_match_numpy` checks every 2^k and 2^k − 1.
- fp32 logits and the combine kernel's outputs are built from their bits (`_f32_bits`).
- The score truncation bound adds per-key and per-row integer ceilings instead of rounding a
  float64 sum per element. The new bound is at most one unit looser, still an upper bound.

What float64 is left is per row or per key: q's L1 norm, and `_l1_ceil` on (N,) per tile.
`launch(exp2="f64")` keeps the old weight path. `test_integer_exp2_equals_float64_exp2` checks
that both paths give the same bits, and that the integer path flags no row the float64 path
certifies. The benchmark keeps an f64 config as the GPU A/B.

## D16. A step is its kernels: fused tier-2 work list, no fills

FA2's whole decode step is about 11 µs at B = 1. In a CUDA graph each extra node costs a
microsecond or two, so the small torch ops around the exact kernels add up:

- tier 2's work list `(OF != 0) & (FL.amax(0) == 0)` took five ops;
- zero-filled flag buffers and 1-element placeholder arguments took up to nine;
- the merge-order `arange` took one.

That is up to about fifteen tiny kernels per step.

- The tier-1 combine kernel now writes NEED itself. It already visits every split of each
  (row, channel block) pair, so it takes the maximum of their split flags on the way
  (`WRITE_NEED`) and stores `(flag != 0) & (split flags == 0)`.
- FL and OF are `torch.empty`: the split and combine grids write every entry.
- Placeholder arguments and the default merge order are made once per device and cached. They
  are never written.

A tcpv step is now four launches (split, combine, tier-2 split, tier-2 combine) plus buffer
allocation; int64 and tc steps are two. Two tests cover this:

- `test_fused_need_equals_torch_formula` checks NEED against the old formula, on data that has
  both NEED pairs and split-flagged pairs.
- `test_step_runs_no_fill_or_reduction_ops` profiles a step and fails on any fill, reduction,
  comparison or cast op. It fails on the previous code.

## D17. Windowed scores in channel chunks: no (M, N, D) tensor

Run 1113-flash (RTX PRO 6000, sm_120) timed passes D13 to D16 for the first time. The
integer-only path (int64) improved a lot:

- T = 1, B = 1, n = 1024: 20.4 ms → 0.222 ms (block_m 2, integer exp2). FlashInfer is 0.010 ms,
  so the gap went from about 1850× to 22×.
- On the same case, integer exp2 is 1.84× faster than float64 exp2 (0.408 ms).
- T = 8 (16 rows, block_m 16) got slower, 18.4 → 25.6 ms, with 10 KB of spills.

Every stage that builds a tensor-core kernel with the windowed score fallback (`TC_FALLBACK`)
timed out at 20 to 25 minutes, twice each. On the laptop, compiling offline for sm_120 with
Triton 3.9 and ptxas-blackwell (no GPU) shows why:

- `_windowed_logits` built the full (M, N, D) int64 product tensor. At 16 × 32 × 128 that is
  65,536 values, about 512 per thread.
- The tc kernel with the fallback took 116 s to compile and spilled 100 KB per thread. The tcpv
  one was still compiling after 6+ minutes. Without the fallback, tcpv compiled in 15 s.

The windowed dot product now runs D in chunks of DC channels, with DC sized for about 16 live
int64 products per thread (`WDC`, set by `launch`). It makes two passes: the leading exponent
over all d, then the windowed sum at the final base. The integer sums are exact, so acc and err
are the same as the one-shot form's, bit for bit, at any DC. Certification decisions are
unchanged, and so are the outputs. q and k chunks are reloaded from global memory (L1/L2 hits)
instead of being kept in registers. The int64 path's NaN/inf key check reads the exponent
bits directly instead of decoding the whole K tile.

Offline sm_120 compiles (Triton lowering + ptxas; spill stores per thread):

| kernel | before | after |
|---|---|---|
| tcpv, fallback on, block_m 16 | > 6 min, still compiling | 42 s, 1.5 KB |
| tc, fallback on, block_m 16 | 116 s, 100 KB | 17 s, 2.5 KB |
| int64, block_m 16 | 70 s, 88 KB | 9 s, 1.9 KB |
| int64, block_m 2 | 3 s, 340 B | 6 s, 0 B |

The tensor-core kernels still sit at 255 registers with 1–2.5 KB of spills: the next thing to
tune.

The CPU interpreter runs kernels without compiling them, so it cannot catch this kind of
failure. `scripts/sm120_compile_check.py` compiles every split-kernel config the GPU stages
build, with no GPU needed. It fails if a config takes longer than a time limit or spills more
than a byte limit. Run it before every PACE job. All nine configs compile in 5–55 s.

**Run 1114-flash (job 15619813) with this fix.** Every stage passed on the first attempt:

- the 26 GPU tests passed;
- the tcpv fixture for head 27_11 matched its SHA-256 with 0 fallback rows (it had 3 before
  tier 2);
- the benchmarks had 0 uncertified rows and no config changed bits.

Best exact time against FlashInfer and against the pre-D13 code (run 1112-flash):

| T | B | n | FlashInfer | best exact | ratio | before (1112) |
|---|---|---|---|---|---|---|
| 1 | 1 | 1K | 0.009 | 0.119 tcpv | 14× | 2.78 |
| 1 | 1 | 16K | 0.021 | 0.567 tcpv | 27× | 10.1 |
| 1 | 1 | 64K | 0.188 | 1.41 tcpv | 7.5× | 24.8 |
| 1 | 8 | 16K | 0.359 | 2.60 tcpv | 7.2× | 440 |
| 1 | 32 | 4K | 0.359 | 3.47 tcpv | 9.7× | 339 |
| 1 | 32 | 16K | 1.41 | 14.9 tcpv | 10.6× | out of memory |
| 8 | 1 | 16K | 0.021 | 6.44 tc | 311× | — |
| 8 | 8 | 4K | 0.037 | 6.89 tcpv | 185× | — |

(ms per step, CUDA graphs.) The A/Bs, each on the same tile config:

- **Packed rows (D14).** block_m 2 against the padded block_m 16, at T=1, B=8, n=16K: tcpv
  6.6 vs 30.8 ms, tc 7.4 vs 50.1 ms.
- **Integer exp2 (D15).**
  - tc, B=1, n=64K: 3.33 vs 6.06 ms with float64.
  - tc, B=8, n=16K: 7.38 vs 11.5 ms.
  - tcpv gains only about 5%, and int64 about 3%.
- **Chunked scores (this section).** int64 at T=8, B=1, n=1K: 25.6 → 1.09 ms.

Decode is now 7–35× FlashInfer. Verify steps (T = 8, 16 rows, PACK = 1) are the weak spot at
57–311×: they take about 12× the T = 1 time for 8× the rows, and spill 170–518 B.

## D18. Verify steps: four q digits, 32-bit addressing, 64-key tiles

A verify step has 16 real query rows per segment (8 draft tokens × GQA group 2), so it has no
padding for D14 to remove. Run 1114 put it at 57–311× FlashInfer.

**Tools.** `scripts/sm120_sass_profile.py` compiles one config for sm_120 with line info, using
Triton's pipeline, ptxas-blackwell and nvdisasm (no GPU). It finds the key loop, and counts one
iteration's SASS (one key tile) by kernel line. It reports these counts:

- instructions per warp per tile;
- spill loads and stores (LDL/STL) inside the loop;
- the rare branches (windowed recompute, float64 exp2), counted apart.

`scripts/profile_step.py` (a PACE stage) gives the device time of each launch in a step, and
the share of programs that tier 2 reruns.

**Diagnosis.** The time is not in the hot loop. For the best run-1114 verify config (tcpv,
block_m 16, block_dv 128, 8 warps), one 32-key tile was 10.2K instructions per warp. At one
program per SM, that is about 8 µs per tile at full issue. The measured time is 120–345 µs per
tile, 15–45× more. Decode tiles come within about 4× of their count. Three observations point
to the windowed score recompute (D17) running on most verify tiles:

- at T = 8, B = 1, n = 1K, the int64 mode, which always takes the windowed path, matches tcpv
  (1.09 against 1.07 ms);
- tc and tcpv take about the same time for verify (6.4 and 6.9 ms at B = 1, n = 16K), even though
  tcpv executes a quarter of tc's instructions;
- tier 2 is not the cause: on the benchmark's data no (row, channel block) pair needs it.

The cause is q's digit count. With SQ = 3, q keeps 21 bits below each row's largest element.
1.6% of the benchmark's standard-normal q rows (d = 128) have an element more than 2^13 below
the row maximum, and lose bits. A lost bit costs that row the bound qlost·|K|₁, and relative to
a score this is about 2^-15. That is far wider than an fp32 logit's 2^-24, so nearly every score
of that row is uncertified, and every tile of every program holding the row is recomputed in
int64. A 2-row decode block holds such a row 3.3% of the time; a 16-row verify block holds one
23% of the time. Each wave of programs then waits on its slowest ones.

**1. SQ = 4 (default).** q keeps 28 bits. Rows with lost bits fall to 0.012% (decode blocks
0.02%, verify blocks 0.18%). The fixed-point elements are below 2^28, so

  |acc| ≤ d · (2^28 − 1)² < 2^63 at d = 128.

`launch` asserts 7(sq + sk) + log2 d ≤ 63. The overflow check now reads |acc| ≥ 2^62 − err, so
it cannot wrap near that bound. The cost:
- decode: none, because PACK = 8 already holds 4 digits in one MMA group;
- verify (PACK = 1): 16 score MMAs per tile instead of 12.

**2. int64 addressing.** The loop's largest item was the K digit loads: 2.3K instructions per
warp, issued as single-byte loads (`LDG.E.U8`) with 64-bit address math, and the address
registers spilled. The cause was one cast. `kvl` was int64, so the loop variable, `nn` and
every per-key offset were int64 too. Now:

- key indices are int32 within a segment;
- each program moves the K, V, slice and metadata pointers to its segment's first key once;
- the digit-plane strides are int64 scalars.

Per-element address math is 32-bit and the loads vectorize. Nothing computed changes.

**3. Equal-weight digit products accumulate in the MMA (PACK = 1).**
- Q[s]·K[t] products with the same s + t share one int32 accumulator. Each product is below
  2^21, and at most four share a weight, so the sum stays below 2^23.
- Adjacent weights pair in int32: S_u + (S_(u+1) << 7) < 2^30.
- A tile then widens 4 int64 terms instead of SQ·SK.

The P·V weight digits are cut from two int32 halves of Wf, digits 0–3 and 4+.

**4. 64-key tiles for tensor-core P·V.** The int8 MMA output layout spreads 8 warps along the
key axis in 8-key steps. A 16 × 32 score tile has only 4 such slices, so warps 4–7 recomputed
warps 0–3: every score, logit and weight was computed twice. The per-tile merge into the running
(row, channel) partial also costs the same whatever the tile size, so a bigger tile pays it less
often per key.

The old `anch + 1 < 63` limit is now an exact bound:
- The digit pairs of a tile combine to Σ_keys Wf·Vfix, with Wf < 2^(7·SW) and |Vfix| < 2^(7·SV)
  (digits in [-127, 127]). So |acc| < block_n · 2^(7(SW+SV)) ≤ 2^ANCH.
- The error count is e < block_n · (2^(7·SW) + 2^(7·SV) + 1) < 2^(ANCH−5).
- acc ± e and 2e fit int64 when ANCH ≤ 62.
- That allows block_n = 64 at SW = 5, SV = 3. The int32 MMA partials stay below 64 · 127² < 2^20.

`test_pv_key_tile_size_does_not_change_bits` checks 64 against 32 for bits and flags. It
includes the bound's worst case: equal weights over values that all have the largest bf16
significand, with each channel's sign fixed so its sum grows with the tile. Tier 2 keeps 16-key
tiles, so its int64 (rows, keys, channels) products stay small.

**Static counts.** Offline sm_120 key-loop counts per program: before is 51bb44b (32-key
tiles, SQ = 3), after is this section's code. "Per 32 keys" means warps × count per warp ×
32 / block_n. These were compiled without the GPU's divisibility hints, so they overstate the
loads; D19 has the corrected counts.

| config (rows, block_dv / warps) | instructions per 32 keys, before → after | loop spill ops per 32 keys | stack |
|---|---|---|---|
| tcpv verify, m16, 128/8 | 81.4K → 36.6K | 6280 → 3028 | 2240 → 1960 B |
| tcpv verify, m16, 64/8 | 61.2K → 24.3K | 2072 → 872 | 704 → 984 B |
| tcpv decode, m2, 128/8 | 41.0K → 25.4K | 2840 → 708 | 1464 → 424 B |
| tcpv decode, m2, 64/8 | 28.5K → 17.2K | 560 → 76 | 312 → 96 B |
| tc verify, m16, 64/8 | 169K → 166K | 8704 → 8384 | 2064 → 1808 B |
| tc decode, m2, 32/4 | 30.6K → 31.0K | 0 → 0 | 0 B |

The new rows include SQ = 4's extra MMAs. The tc path changes little, because its P·V is int64
on CUDA cores. These counts leave out the windowed recompute, which was the main cost (above).
The GPU decides between 128/8 and 64/8 for verify: a 64-channel block computes each score twice,
but spills a quarter as much.

The benchmark adds A/Bs: `n32` (32-key tiles) and `q3` (three q digits).

## D19. Run 1115: tier 2 at 2 × 32 blocks, shared memory, offline hints

**Run 1115-flash (job 15642615, code 86a52a7).** Every stage passed:

- the tests passed;
- the tcpv fixture for head 27_11 matched its SHA-256 with 0 fallback rows;
- 0 rows were uncertified, and no config changed bits.

Verify steps were 1.2–3.3× faster than in run 1114 (ms, best exact config):

| T = 8 | FlashInfer | 1114 | 1115 |
|---|---|---|---|
| B = 1, n = 1K | 0.008 | 1.07 | 0.322 |
| B = 1, n = 4K | 0.013 | 1.26 | 0.425 |
| B = 1, n = 16K | 0.021 | 6.44 | 3.10 |
| B = 8, n = 1K | 0.017 | 1.55 | 0.570 |
| B = 8, n = 4K | 0.035 | 6.89 | 4.25 |
| B = 8, n = 16K | 0.364 | 20.6 | 16.6 |

The q3 A/B isolates D18's fourth q digit: tc at T = 8, B = 1, n = 16K took 6.15 ms with three
digits and 3.92 ms with four. Decode also improved, from 2.60 to 2.16 ms at B = 8, n = 16K and
from 14.9 to 10.5 ms at B = 32, n = 16K.

The run showed two problems.

**1. 64-key tiles did not launch.** At Triton's default pipelining depth of 3, the 64/8 and
128/8 configs needed 120–159 KB of shared memory, and sm_120 allows 99 KB per block.

The offline compile check missed this for two reasons:
- It never read the shared-memory size.
- It compiled without the GPU JIT's divisibility hints: Triton marks every pointer, and every
  size divisible by 16, at launch. Without them, loads are neither vectorized nor pipelined
  through shared memory, so the offline kernel was not the GPU's kernel.

`scripts/sm120_compile_check.py` now passes the same hints. Its shared-memory figures now
match the GPU's error messages to the byte (159,232 B and 120,320 B), and it fails any config
above 101,376 B. `launch` uses pipelining depth 2 for 64-key tensor-core tiles, which needs
68–85 KB. D18's static counts were made without the hints; the table below replaces them.

**2. Tier 2 dominated verify steps.** `profile_step` at T = 8, B = 1, n = 16K:

| launch | ms |
|---|---|
| tier-1 split | 0.67 |
| tier-1 combine | 0.20 |
| tier-2 split | 4.50 |
| tier-2 combine | 0.16 |

2 of 128 (row, channel block) pairs needed tier 2. Tier 2 ran on tier 1's grid, though, so each
of those pairs reran its whole 16-row × 128-channel program over all 16K keys with int64 P·V.

Tier 2 now has its own grid:
- 2-row blocks (`block_m_t2`, digit-packed scores as in decode);
- 32-channel blocks: the combine kernel, and so NEED and the combine flags, now uses channel
  blocks of at most 32 whatever the split kernel's `block_dv`;
- 16-key tiles and 4 warps.

A pair that needs tier 2 now costs its own 2 × 32 block, 32× less work at block_m 16 and
block_dv 128. Tier 2's split flags land in tier 1's coarser flag entries. Several tier-2
programs share an entry, so they merge with an atomic maximum, whose result does not depend on
their order. The tier-2 recovery test and the fused-NEED test now also run at block_dv 64, where
tier 2's blocks sit inside tier 1's.

Offline sm_120 key-loop counts with the GPU's hints, before (51bb44b: 32-key tiles, SQ = 3,
depth 3) and after (this code: 64-key tensor-core tiles at depth 2, SQ = 4). "Per 32 keys" is
warps × count per warp × 32 / block_n, per program:

| config (rows, block_dv / warps) | instructions per 32 keys, before → after | loop spill ops per 32 keys | shared |
|---|---|---|---|
| tcpv verify, m16, 128/8 | 62.7K → 22.7K | 2976 → 580 | 80 → 84 KB |
| tcpv verify, m16, 64/8 | 50.9K → 16.7K | 1768 → 280 | 63 → 67 KB |
| tcpv decode, m2, 128/8 | 27.8K → 13.7K | 536 → 32 | 85 → 85 KB |
| tcpv decode, m2, 64/8 | 19.3K → 9.4K | 328 → 0 | 68 → 68 KB |
| tc verify, m16, 64/8 | 166K → 161K | 8528 → 8144 | 31 → 33 KB |
| tier 2, old 16 × 128 → new 2 × 32 block | 308K → 26K | 55,040 → 0 | 39 → 25 KB |

The old tier-2 kernel ran at 128 registers with 12.6 KB of spills per thread. That is how 2
NEED pairs cost 4.5 ms. A NEED pair now reruns one 26K-instruction program with no spills,
instead of a 308K-instruction one.

**Run 1116-flash (job 15646338, code 4901277).** Every stage passed:

- the tests passed;
- the fixture for head 27_11 matched its SHA-256 with 0 fallback rows;
- 0 rows were uncertified, and no config changed bits.

Best exact time, in ms per step, against FlashInfer:

| T | B | n | FlashInfer | 1114 | 1115 | 1116 | ratio |
|---|---|---|---|---|---|---|---|
| 8 | 1 | 1K | 0.008 | 1.07 | 0.322 | 0.332 | 39× |
| 8 | 1 | 16K | 0.023 | 6.44 | 3.10 | 1.06 | 47× |
| 8 | 8 | 4K | 0.032 | 6.89 | 4.25 | 1.30 | 41× |
| 8 | 8 | 16K | 0.363 | 20.6 | 16.6 | 3.40 | 9.4× |
| 1 | 1 | 16K | 0.021 | 0.567 | 0.594 | 0.342 | 17× |
| 1 | 1 | 64K | 0.187 | 1.41 | 1.29 | 0.725 | 3.9× |
| 1 | 8 | 16K | 0.359 | 2.60 | 2.16 | 1.11 | 3.1× |
| 1 | 32 | 16K | 1.41 | 14.9 | 10.5 | 6.41 | 4.6× |

Verify steps are now 6× faster than in run 1114 at B = 1, n = 16K. The 64-key tiles win
almost everywhere.

`profile_step` breakdown at T = 8, B = 1, n = 16K:

| launch | ms |
|---|---|
| tier-1 split | 0.59 |
| tier-1 combine | 0.10 |
| tier-2 split | 0.33 |
| tier-2 combine | 0.07 |

With three q digits, the tier-1 split takes 1.76 ms instead of 0.59.

Tier 2 still costs 0.33 ms for 2 NEED pairs. Its grid has 16K programs (8 segments × 8 row
blocks × 64 splits × 4 channel blocks), and almost all of them only load and slice q, then
find nothing to do. That fixed cost is the next item. At decode it is 0.03–0.05 ms.

## D20. Empty tier-2 programs exit first

Tier 2's grid is fixed so that a step stays capturable in a CUDA graph. At T = 8, B = 1,
n = 16K that grid has 16K programs (8 segments × 8 two-row blocks × 64 splits × 4 channel
blocks), and run 1116 found 2 pairs that needed it. Every program still loaded, decoded and
sliced its q rows, including the float64 L1 norm, before it found its key range empty. That
cost 0.33 ms, 30% of the step. The tier-2 combine likewise ran its 96-step final rounding
for every block and then stored nothing.

Both tier-2 kernels now load their NEED entries first and wrap the rest in `if go:`, a runtime
branch. Tier 1 is unchanged: there `go` is a compile-time True. A program with no NEED pair now
costs one small load and a reduction. The outputs cannot change: such a program never wrote
anything before either, because all of its stores were masked by NEED.

Run 1117-flash (job 15658020) timed nothing. Its node's GPU could not open a CUDA context
(`device busy or unavailable`, error 999 from `cuDevicePrimaryCtxRetain`), so every stage
failed within seconds. The stage runner still marked them done, and the chain stopped.
`nvidia-smi` and `get_device_name` do not open a context, so the env stage had passed.

`run_stages.sh` now probes each GPU stage with a real allocation first, and again after a
failure that names the error. On an unusable GPU it:
- leaves the stage not-done;
- records the node in `bad_nodes`;
- moves the pre-queued successor off that node (`scontrol update ExcNodeList`), and every later
  submission excludes it;
- exits, so the successor reruns the stage elsewhere.

Run 1118-flash also timed nothing, but this time the probe was at fault. A node's first
`import torch` reads thousands of files off Lustre scratch with a cold cache. That took 93 s on
the login node (8.5 s once warm) and over 10 minutes on the compute nodes. The env stage hit its
600 s cap twice, and the probe's 120 s cap then marked four healthy nodes bad in a row. Two of
those jobs had also landed on L40S nodes: the sbatch partition list allows them, and submission
adds `gpu-v100` to any `-p` list.

The runner now:
- probes once per job, before its first GPU stage, with a cap of up to 1500 s. The probe also
  warms the cache for the stages after it;
- marks a node bad only on an explicit CUDA error. A timeout or a failed import is
  inconclusive, and the stage runs under its own cap;
- gives the env stage 1800 s;
- pins flash runs and their successors to the RTX PRO 6000
  (`-p gpu-rtxpro-blackwell --gres=gpu:rtx_pro_6000_blackwell:1`).

Run 1118b-flash timed the early exit (c7f48e4). Run 1119-flash, which times D21, ran at the
same time from a second checkout (`EA_REPO`). FlashInfer's times agree to 1% between the two
nodes. Best exact config, ms (× FlashInfer):

| shape | 1116 | 1118b (early exit) |
|---|---|---|
| decode B=1 16K | 0.342 (17×) | 0.277 (13.4×) |
| decode B=1 64K | 0.725 (3.9×) | 0.664 (3.5×) |
| decode B=8 16K | 1.11 (3.1×) | 1.057 (2.9×) |
| decode B=32 16K | 6.41 (4.6×) | 6.43 (4.6×) |
| verify T=8 B=1 16K | 1.06 | 0.935 (41.8×) |
| verify T=8 B=8 16K | 3.40 (9.4×) | 3.26 (9.0×) |

At decode no pair needs tier 2, and its two launches now take 0.004 ms. In verify T=8 B=1 16K,
2 of 256 tier-2 programs have work. Tier 2 still costs 0.23 ms there (split 0.137, combine
0.093), down from 0.33. What is left is the latency of the few live programs, not the grid.

## D21. Registers

Every tensor-core split kernel uses all 255 registers, so an 8-warp program has an SM to itself.
`scripts/sm120_sass_profile.py` measured where the pressure comes from (with the GPU's hints;
"spill ops" are spill loads and stores in the key loop, per 32 keys):

| variant | 16-row verify, tcpv 128/8 | 2-row decode, tcpv 128/8 | 16-row verify, tc 64/8 |
|---|---|---|---|
| baseline (4901277) | 580, 22.7K instructions | 32, 13.7K | 8144, 161K |
| windowed fallback compiled out | 500 | 0 | |
| 128-register cap, depth 1 | 3032 | 336, 15.3K | |
| int32 running exponent (kept) | 668, 23.2K | 8, 13.7K | 4048, 160K |
| plus int32 error counts (dropped) | 532, 23.9K | 12, 13.9K | 7168 |
| plus int32 weights (dropped) | 524, 24.7K | 12, 14.0K | 6480 |

The windowed fallback is not what holds the registers. The live state that does is the
(row, channel) running partial: three (16 × 128) int64 arrays, 48 registers per thread at 8
warps.

**Kept: the running exponent in int32.** NEG32 = −2^30 marks "empty", and the store widens it
to the int64 NEG, so the partial format the combine kernel reads is unchanged. Spill traffic
halves for tc and falls 4× at decode, for no extra instructions. ptxas places spills a little
worse for tcpv 128/8 verify; 64/8 improves.

**Dropped: error counts in int32.** This broke the tier-2 test. On wide-exponent V, tier-1
tensor-core counts legitimately pass 2^30, so a sound saturation has to flag the row, and a
split flag stops tier 2 from recovering it. Doing this soundly would need a "rerun me" marker
that sends the pair to tier 2. That costs more code than the 8 registers it saves.

**Dropped: int32 weights.** They saved nothing.

**Register cap, for the GPU to decide.** A 128-register cap lets two 8-warp programs share an SM
at pipelining depth 1 (16 KB of shared memory). At decode it costs about 40× the spill traffic
and 11% more instructions, for twice the warps to hide latency. `launch` takes `maxnreg`, and the
benchmark carries `r128` A/Bs for tcpv 128/8 and 64/8. A register cap is a tiling choice, so the
benchmark's bit check covers it. Tier 2 keeps its own registers.

Run 1119-flash (16a6460, compared with 1118b; all bits checked, 0 uncertified rows):

| shape | config | 1118b | 1119 |
|---|---|---|---|
| verify T=8 B=1 16K | tc 64/8 | 3.750 (spill 450 B) | 2.751 (222 B) |
| verify T=8 B=8 16K | tc 64/8 | 29.40 | 22.81 |
| verify T=8 B=1 16K | tcpv 128/8, best | 0.935 | 0.929 |
| verify T=8 B=8 16K | tcpv 128/8, best | 3.261 | 3.086 |
| decode B=1 16K | tcpv 128/8, best | 0.277 | 0.278 |
| decode B=32 16K | tcpv 128/8, best | 6.434 | 6.283 |

The int32 exponent pays where the spills were: tc is 22–27% faster. Best-config tcpv verify
at B=8 is 5% faster, and decode is flat. The 128-register cap loses everywhere except
tcpv 64/8 at verify B=8 16K (5.14 vs 5.27 ms), and that config is not the best one. Occupancy
is not the limit, so `r128` stays an option and not a default.

Where verify T=8 B=1 16K (0.929 ms) goes: tier-1 split 0.576, tier-1 combine 0.102,
tier-2 split 0.141, tier-2 combine 0.094. At decode B=1 16K the combine is 0.096 of 0.278 ms.
The combine costs about 0.1 ms whatever the batch, and even when only 2 programs have work, so
it is per-program latency, not bandwidth. The suspected cause is `_ratio_round`'s 96-step
bit-serial division, run twice per output, after a serial merge over the splits. That is the
next target.

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
- **Q4. Fast certified exp2 without fp64.** Answered in D15 with integer fixed point and a
  proven bound, not fp32. Open: whether the tile fallback rate (about 2^-14 of weights) matters
  at long context, and whether a 4096-entry table (4× tighter bound) is worth its cache space.
