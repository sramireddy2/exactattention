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

## D22. The combine kernel: short rounding, one output per thread, pipelined merge

Runs 1118b and 1119 put the combine kernel at about 0.1 ms per launch. That is 0.096 of
0.278 ms at decode B=1 16K, and two launches (0.196 ms) of verify's 0.929 ms. The time barely
depends on the work: B=8 decode merges 12 splits in 0.062 ms, B=1 merges 64 in 0.096, and the
tier-2 combine with only 2 live programs still takes 0.093. That is about 0.65 µs per split
plus a fixed 0.054 ms. Compiled offline (sm_120, the GPU's hints), the old kernel was 13.3K
SASS instructions per thread at 201 registers. The merge loop was 474 per split, and 12.8K
were outside the loop, almost all of it `_ratio_round`'s 96-step bit-serial division, run
twice (lower and upper bound) on 4 outputs per thread. With 16-row tiles, a B=1 step had only
4 combine programs of 4 warps, so nothing hid that chain.

**Rounding in P + 1 steps.** `_ratio_round` computes q1 = floor(a · 2^s2 / b), and the exponent
choice qe ≥ K − (P − 1) keeps q1 < 2^(P+1). So when s2 > P + 1, the first z = s2 − (P + 1)
long-division steps produce zero bits: floor(a · 2^z / b) = 0, so a < b and a · 2^z < b < 2^62.
Those steps only double the remainder, so `a << z` replaces them. At most P + 1 steps remain:
9 for bf16, 25 for fp32. Since z ≤ 61 whenever a ≥ 1, the shift never overflows, and the old
96-step cap is gone with it. `test_ratio_round_is_exact` checks (q, qe) against exact fractions
on 2048 cases per precision: long quotients, negative s2, subnormal clamping, exact quotients
and ties. Skipping one step too many fails it.

**4 × 32 tiles.** Combine programs now take 4 rows instead of 16. At 4 warps that is one output
per thread, and 4× as many programs. NEED and OF keep their 32-channel layout, so only the grid
changes.

**Pipelined merge.** The split loop is `tl.range(NSPLIT, num_stages=3)`. Triton turns the
partials' loads into cp.async copies through 6.3 KB of shared memory, so later splits load
while the current one merges.

Each change alters only the arithmetic schedule or the tiling, never a value, so the outputs
and flags are the same bits.

| combine, sm_120 SASS per thread | instructions | merge loop per split | registers |
|---|---|---|---|
| before (16 rows, 96 steps) | 13 304 | 474 | 201 |
| 16 rows, P + 1 steps | 2 808 | 463 | 154 |
| 4 rows, P + 1 steps | 936 | 210 | 60 |
| plus pipelined merge (kept) | 1 128 | 254 | 60 |

`sm120_compile_check.py` now also builds the combine kernel (tier 1 bf16 and fp32, and tier 2).

Run 1120-flash (1ee0524, compared with 1119; all bits checked, 0 uncertified rows, best tcpv
config, ms):

| shape | 1119 | 1120 | × FlashInfer |
|---|---|---|---|
| decode B=1 4K | 0.106 | 0.076 | 7.2× |
| decode B=1 16K | 0.278 | 0.205 | 9.9× |
| decode B=1 64K | 0.672 | 0.559 | 3.0× |
| decode B=8 16K | 1.072 | 1.008 | 2.8× |
| decode B=32 16K | 6.283 | 6.261 | 4.4× |
| verify T=8 B=1 4K | 0.330 | 0.297 | 23.8× |
| verify T=8 B=1 16K | 0.929 | 0.765 | 36.4× |
| verify T=8 B=8 4K | 1.103 | 0.998 | 28.8× |
| verify T=8 B=8 16K | 3.086 | 3.092 | 8.5× |

Per launch, the combine went from 0.096–0.102 ms to 0.020–0.025 ms (tier 1) and from 0.094 to
0.019 ms (tier 2); B=8 decode's went from 0.062 to 0.007. The saving is a fixed ~0.075 ms per
launch, so it shows at small batch and vanishes where the split kernel dominates (B=32, verify
B=8 16K). What is left of a 2-program tier-2 combine, 0.019 ms, is the serial merge over 64
splits.

Where verify T=8 B=1 16K (0.765 ms) goes now: tier-1 split 0.579, tier-1 combine 0.023,
tier-2 split 0.139, tier-2 combine 0.019. At decode B=1 16K: split 0.179 of 0.205. The split
kernel is now the whole cost. Its cheapest target is tier 2: 0.139 ms to rerun 2 blocks (D23).

## D23. Tier 2: a work list and 64-key tiles

Tier 2's grid already covered every KV split, so a rerun block walks only its split's keys. Its
cost comes from two other places.

- **Tile overhead.** At verify B=1 16K a split is 256 keys: 16 tiles of 16 keys in 0.139 ms.
  At verify B=8 4K it is 352 keys: 22 tiles in 0.189 ms. Both come to about 8.7 µs per tile.
  With 2 or 3 pairs there is about one 4-warp program per SM, so nothing hides latency. Offline
  SASS shows half of a 16-key tile's instructions (1644 of 3314 per warp) in `_tc_logits`, a
  per-tile cost that bigger tiles spread over more keys.
- **The grid.** It was every 2-row block × split × 32-channel block (24.6K programs at verify
  B=8), nearly all empty. Bigger, 8-warp programs would make those cost more.

**Work list.** The tier-1 combine appends each NEED pair, a (row, 32-channel block), to a list
`WL` with one atomic add per pair on a counter `CNT`. Tier 1's split program (0, 0, 0) zeroes
the counter, because a fill launch would break the step's no-fill rule (D16). Tier 2's split
grid is (programs, KV splits), with programs = ⌈2 · SMs / splits⌉, about two waves. Each
program strides over the list, so any count is covered and the grid stays fixed (graph-safe).
With no pairs, every program reads `CNT` and exits. The list's order depends on the atomics,
but each item writes only its own (split, row, channels) partials and takes an `atomic_max` of
its flag, so the bits do not depend on that order. The kernel body moved into
`_split_program`, which tier 1 calls once and tier 2 calls once per item.

**One row, 64-key tiles, 8 warps.** A list item is a single row, so tier 2 runs
`block_m_t2 = 1` (16 digit slots per MMA). Two flagged rows of a block are now two items,
which run in parallel. Tiles are 64 keys when the split length allows it (else 32), with
pipelining depth 2.

| tier-2 tile, sm_120 SASS | regs / spill | hot SASS per warp per key | warp-instructions per 32 keys |
|---|---|---|---|
| 2 rows, 16 keys, 4 warps (before) | 254 / 0 | 207 | 26.5K |
| 2 rows, 64 keys, 8 warps | 255 / 24 B | 45 | 11.4K |
| 1 row, 32 keys, 4 warps | 249 / 0 | 73 | 9.4K |
| 1 row, 64 keys, 8 warps (kept) | 255 / 0 | 36 | 9.3K |

`sm120_sass_profile.py` now follows the extra inlining level: nvdisasm prints one line-info
comment per level, and work-list loops are skipped when it looks for the key loop.
`test_fused_need_equals_torch_formula` checks that the list holds each NEED pair exactly once.
Under the interpreter the SM count is 2, so the tier-2 tests run several items per program.

Run 1121-flash (a6369f8; all bits checked, 0 uncertified rows, fixture SHA-256 match). The
tier-2 split launch fell from 0.139 to 0.051 ms at verify B=1 16K and from 0.189 to 0.062 ms at
verify B=8 4K. Decode B=32 16K fell from 6.26 to 4.54 ms; no profile case covers it, but tier 2
is the only kernel change between the runs, and its old grid there was 3,072 mostly empty
programs. Results by shape are in D25's table.

## D24. Tier-1 P·V: a dropped-channel fix, a top-digit weight-loss bound, equal-weight MMA sums

After D22 and D23, the tier-1 split kernel is almost the whole step. Offline sm_120 SASS put
`_tc_pv`, the tensor-core P·V, at 56% of a decode tile (1783 of 3203 instructions per warp) and
43% of a verify tile (2331 of 5437). Decode B=1 16K moves about 0.75 TB/s and B=32 about
0.7 TB/s, well under DRAM bandwidth, so the kernel is bound by instructions and their latency,
not by bytes.

**A soundness bug, fixed.** A V element far enough below its key's largest element (more than
2^21 below at SV = 3) loses every bit in block fixed point: all its digits are 0 and only the
lost mask records it. `_tc_pv` decided that a channel "has a term" from its nonzero digits alone.
So when every visible key of a tile had such an element in channel c, the tile returned an
exact-zero partial for c and dropped the V-loss error, and the output could be certified and
wrong. Example: V channel 0 = 1 and channel 1 = 2^-40 on every key gave a certified 0 for
channel 1, not 2^-40 (bf16 0x2b80), and tier 2 never saw the pair. Now a channel is empty only
when acc = e = 0, where its interval is exactly {0}; a lost element makes e ≥ 1. Tier 1 then
flags the pair, and tier 2's int64 P·V returns the exact bits. This needs exponent spreads of
more than 2^21 within a V row across a whole tile. The benchmark's normal data and the Qwen
fixture do not have them, and their bit checks passed, so earlier results stand.
`test_pv_fully_lost_v_channel_is_not_dropped` covers all three of `_tc_pv`'s packings.

**Weight loss from the top V digit.** The bound Σ lostW · |Vfix| needed one MMA on |digit b|
for every V digit. Each `tl.abs` of an int8 tile cost about 160 instructions to unpack, take the
absolute value and repack. For any digit signs,

  |Vfix| ≤ |top digit| · 2^(7(SV−1)) + 127 · (1 + 2^7 + … + 2^(7(SV−2))) = |top| · 2^(7(SV−1)) + 2^(7(SV−1)) − 1,

so the bound is now 2^(7(SV−1)) · Σ lostW · |top digit| plus nlost · (2^(7(SV−1)) − 1): one abs
MMA. It is looser by under 2^14 units per lost weight. V is anchored at each key's largest
element, near 2^20, so that is at most about 2^-6 of the key's largest |V|. The per-key total
stays below 2^(7·SV), so D18's int64 bounds are unchanged.

**Fewer int64 widenings.**
- One slot per MMA (16-row verify blocks): weight digit j times V digit b has weight 7(j + b).
  Products of equal weight now accumulate inside the int32 MMA, and adjacent weights pair in
  int32, as D18 does for scores. That is 4 int64 terms instead of 15. Each product sum is below
  64 · 127², at most 3 share a weight, and a pair stays below 2^31; a static assert checks this.
  The V-loss products (weight digits × lost mask, below 64 · 127) are summed in int32 three at a
  time.
- One packed group (2-row decode blocks): the V digits' MMA results are summed in their own
  layout, so the slots are unpacked once for acc, not once per digit. The two error MMAs share
  one unpack (`_slot_err`).

The exact sum and its error are the same in every packing. Only the weight-loss term changed.

`sm120_sass_profile.py` had stopped finding the verify key loop: ptxas moves a `program_id`
compare from the kernel's top level into it. The work-list filter now looks only for that loop's
own lines.

Key loop, sm_120 SASS with the GPU's hints. Each cell is instructions per warp per tile, then
warp-instructions per 32 keys, then spill ops per 32 keys (64-key tiles):

| config | before (a6369f8) | after |
|---|---|---|
| decode m2 128/8 | 3203; 12,812; 8 | 2522; 10,088; 8 |
| decode m2 64/8 | 2172; 8,688; 0 | 1840; 7,360; 4 |
| decode m2 32/4 | 3156; 6,312; 0 | 2807; 5,614; 0 |
| verify m16 128/8 | 5437; 21,748; 628 | 4619; 18,476; 408 |
| verify m16 32/4 | 6010; 12,020; 176 | 5579; 11,158; 124 |

`_tc_pv` went from 1783 to 1106 per decode tile and from 2331 to 1420 per verify tile. What is
left: `_tc_logits` (554 decode, 1296 verify), the merge of the tile partial into the running
one (about 290 decode, 834 verify; at verify that is about 100 instructions per element for two
general int64 rescales), and `_exp2_weight` (221, 423).

## D25. Runs 1121–1122: the split kernel is not bound by its instructions

Runs 1121-flash (a6369f8, D23) and 1122-flash (3cacec5, D24) ran at the same time on one node
(separate GPUs). Every stage passed: all bits were checked, 0 rows were uncertified, and the
fixture SHA-256 matched. Best exact config, ms:

| shape | 1120 | 1121 (D23) | 1122 (D24) | × FlashInfer (1122) |
|---|---|---|---|---|
| decode B=1 4K | 0.076 | 0.076 | 0.073 | 7.0× |
| decode B=1 16K | 0.205 | 0.198 | 0.193 | 9.3× |
| decode B=1 64K | 0.559 | 0.551 | 0.534 | 2.8× |
| decode B=8 16K | 1.008 | 0.967 | 0.950 | 2.6× |
| decode B=32 16K | 6.261 | 4.535 | 4.301 | 3.1× |
| verify T=8 B=1 4K | 0.297 | 0.303 | 0.303 | 24.1× |
| verify T=8 B=1 16K | 0.765 | 0.691 | 0.686 | 31.4× |
| verify T=8 B=8 4K | 0.998 | 0.888 | 0.859 | 24.7× |
| verify T=8 B=8 16K | 3.092 | 2.595 | 2.591 | 7.1× |

D24 cut the tier-1 key loop by 15–21% of its instructions, but the tier-1 split launch barely
moved: 0.176 → 0.170 ms at decode B=1 16K, 0.995 → 0.945 at decode B=8 16K, 0.581 → 0.586 at
verify B=1 16K and 0.778 → 0.763 at verify B=8 4K. Instruction count is not the limit. The
slices are about 1053 bytes per key and KV head (K: 4 × 128 digits and 20 bytes of metadata; V:
3 × 128 digits, the 128-byte lost plane and 9 bytes), against 512 for bf16 K and V:

- **Decode, B ≥ 8.** The split kernel moves about 1.2 TB/s (B=8 16K) and 1.0 TB/s (B=32 16K).
  FlashInfer reaches about 1.5 TB/s at B=32. B=8 16K is 4.08 waves of one program per SM, and
  per tile its full waves run near 1.5 TB/s. These shapes are close to bound by bytes and wave
  quantization, so the next levers there are bytes per key and the split count.
- **Decode, B=1 16K.** 512 programs of 4 tiles each in 2.72 waves, about 0.8 TB/s. A program
  costs about 57 µs, against 4 × 8.6 µs for tiles at B=8's rate, so per-program latency (load
  the first tile, prologue, store) is a large share.
- **Verify.** About 0.36 TB/s, and 30–45 µs per tile against about 3 µs to issue it. The
  suspect is register spills: about 800 bytes per thread at 255 registers. With 84 KB of shared
  memory for depth-2 pipelining, L1 has too little room to hold them, so they likely go to L2.

The installed Nsight Compute (2024.3, CUDA 12.6 module) predates sm_120. So run 1123 tests the
two hypotheses with benchmark A/Bs on tcpv 128/8: `s1` (pipelining depth 1 at 255 registers,
about half the shared memory) and `h` (half the KV splits).

**Two cheaper per-tile pieces, same bits.** They are kept, because they shorten the chains even
if they do not show in time.

- `_rescale` shifts right only (every caller passes a maximum minus its own exponent), so the
  left-shift branch and the s ≥ 63 cases go. For |acc| < 2^63 and 0 ≤ err < 2^63, clamping s at
  63 is exact: acc >> 63 is floor(acc / 2^s), and −((−err) >> 63) is ceil(err / 2^s), for every
  s ≥ 63. The result equals the old one bit for bit, and `test_rescale_is_exact` checks it
  against Python integers, including s = 0, s ≥ 63 and empty-partial shifts near 2^31. This
  serves the tile merge and the combine kernel's merge. The merge went from about 290 to 112
  instructions per decode tile, and from 834 to 232 per verify tile.
- V's NaN/inf flag is one byte per key (`VBAD`), not a second bit of every lost-plane element.
  The lost plane is then a 0/1 MMA operand as stored, with no `& 1`, and the per-tile reduction
  over channels is gone. Flags are reported per row, and a row is flagged if any visible key
  has a NaN/inf in any channel either way, so row flags do not change. A pair in a clean channel
  block of such a row is now split-flagged too, so tier 2 no longer reruns it; its row stays
  flagged.

| key loop, sm_120 SASS per warp per tile | D24 (3cacec5) | D25 |
|---|---|---|
| decode m2 128/8 | 2522 | 2128 |
| decode m2 64/8 | 1840 | 1618 |
| decode m2 32/4 | 2807 | 2600 |
| verify m16 128/8 | 4619 (408 spill ops / 32 keys) | 3832 (412) |
| verify m16 32/4 | 5579 (124) | 5214 (202) |

Run 1123-flash (bf0b59e; all bits checked, 0 uncertified rows, fixture SHA-256 match), same
node as 1121–1122. tcpv 128/8 in ms: base, then the `s1` and `h` A/Bs:

| shape | 1122 base | 1123 base | s1 (depth 1) | h (half splits) |
|---|---|---|---|---|
| decode B=1 16K | 0.193 | 0.179 | 0.179 | 0.187 |
| decode B=1 64K | 0.534 | 0.512 | 0.561 | 0.488 |
| decode B=8 16K | 0.950 | 0.919 | 1.061 | 1.089 |
| decode B=32 16K | 4.301 | 4.199 | 4.926 | 5.969 |
| verify T=8 B=1 16K | 0.686 | 0.660 | 0.628 | 0.645 |
| verify T=8 B=8 4K | 0.859 | 0.818 | 0.821 | 0.889 |
| verify T=8 B=8 16K | 2.591 | 2.352 | 2.498 | 2.610 |

The tier-1 split launch: 0.170 → 0.156 ms at decode B=1 16K, 0.945 → 0.909 at decode B=8 16K,
0.586 → 0.565 at verify B=1 16K and 0.763 → 0.716 at verify B=8 4K. So D25 gives 4–9%, more
than D24's 15–21% instruction cut did. That fits shorter dependency chains (the merge's
selects) mattering more than issue slots.

- **Spills are not verify's main cost.** Depth 1 frees about 40 KB for L1. It helps verify at
  B=1 by 5%, and costs time at B=8 16K and at large-batch decode, where the lost overlap of
  loads matters more. It does not close a 10× gap.
- **Per-program overhead is not the main cost either.** Half the splits is slower nearly
  everywhere: 42% at decode B=32 16K, which drops to 1.36 waves. It is faster only at B=1 64K,
  where each split is long anyway. More, shorter programs balance the last wave better, so a
  split count chosen for whole waves (and an `x2` A/B) is a candidate.

What is left is latency inside a tile with 8 warps per SM: long int64 chains and block-level
reductions. Without an sm_120 profiler, the next measurement is in-kernel timers (clock64 at
the tile's phase boundaries, written to a debug buffer).

## D26. Phase clocks in the split kernel

`launch(..., timers=True)` builds the tier-1 split kernel with the constexpr `TIMERS`. The
default build has no timer code; tier 2 never takes it. Each program writes 12 int64 to
`_LAST_DEBUG["timers"]`:
- SM cycles (`%clock64`) for the prologue;
- for each key tile, cycles of each phase, summed over the program's tiles: scores
  (`_tc_logits` and the windowed fallback), weights (exp2), the tile partial (Z and `_tc_pv`)
  and the merge;
- its tile count, and its key loop and whole-program cycles;
- its start and end in `%globaltimer` nanoseconds, which every SM shares;
- `%smid`.

Inline PTX reads each clock, with an int32 input taken from a block-wide reduction of the
phase's result. So the read cannot be scheduled before the phase ends, and the reduction's
barrier lines the warps up. That perturbs the kernel. Offline sm_120 SASS: +19 instructions per
decode tile (2147 against 2128), and +104 per verify tile (3936 against 3832) with more spills.
Shared memory is the same, so the loads are still pipelined.

`profile_step` runs the timers build next to the plain one for every tcpv case. It checks that
the output bits are equal, then reports:
- cycles and µs per tile for each phase, and the rest of the loop (load waits, loop control);
- prologue and epilogue per program;
- the clock rate (cycles over nanoseconds);
- the launch span, how busy the SMs were over it, and the most programs at once on one SM.

The benchmark also gains an `x2` A/B, twice the KV splits (at least 128 keys each), for the
wave-quantization question in D25.

**Run 1124-flash** (13ec240; all bits checked, 0 uncertified rows, fixture SHA-256 match). In
all 6 profiled tcpv cases the timers build gave the same output bits as the plain build, and
its launch span was 2–3% longer than the plain split launch. The clock ran at 2.27–2.32 GHz on
188 SMs. tcpv 128/8, n64, q4; µs per 64-key tile (cycles converted at the measured clock):

| | decode B=1 16K | decode B=8 16K | verify B=1 16K | verify B=8 4K |
|---|---|---|---|---|
| scores | 0.39 | 0.37 | 0.64 | 0.63 |
| weights | 0.63 | 0.60 | 1.66 | 1.63 |
| tile partial (Z, `_tc_pv`) | 2.67 | 2.88 | 4.79 | 4.76 |
| merge | 0.48 | 0.46 | 1.53 | 1.52 |
| rest of the loop | 6.18 | 5.12 | 15.14 | 13.57 |
| **tile total** | **10.3** | **9.4** | **23.7** | **22.1** |
| prologue per program | 4.0 | 4.0 | 13.4 | 13.3 |
| programs, tiles each | 512, 4.0 | 768, 21.3 | 512, 4.0 | 704, 5.8 |
| program mean (min–max) | 45.8 (38.7–81.7) | 206 (116–273) | 110 (95–461) | 143 (94–519) |
| SMs busy over the span | 79% | 90% | 53% | 73% |

- **One program per SM, always.** At 255 registers × 256 threads a program fills the register
  file, so each SM has 8 warps to hide latency with.
- **Over half of every tile is outside the four phases.** "Rest" runs from the end of one
  tile's merge to the start of the next tile's scores. That is where the pipeliner waits for
  the next tile's copies and issues the ones after. The `_tc_pv` partial is the largest timed
  phase (17–30%).
- **Decode is close to its bytes.** A tile is 64 × 1053 bytes = 67 KB. One SM's share of
  1.79 TB/s moves that in about 7.1 µs, and decode B=8 takes 9.4 µs per tile with the SMs 90%
  busy. Bytes per key stay the decode lever (D25).
- **Verify waits about twice as long as its bytes need.** Its tiles move the same 67 KB but
  take 22–24 µs. That is 8.5 µs of timed phases and 13.6–15.1 µs of rest, against about
  7.1 µs for the bytes. So verify's rest is not bandwidth alone.
- **Verify's prologue** (Q decode and digit slicing for 16 rows) is 13.3 µs per program, 9–12%
  of program time. Every KV split of a row block repeats it for the same rows.
- **Verify has stragglers.** At B=1 16K the mean program takes 110 µs but the slowest takes
  461 µs. The SMs are busy 53% of the span: about 299 µs of work per SM in a 569 µs launch.
  Balanced, 2.72 waves of mean programs would take about 330 µs. The candidates are data
  dependent: the windowed score fallback and the float64 exp2 fallback. The aggregates cannot
  tell which. The q3 build, with more uncertified scores, puts 90% of its time in rest and has
  programs from 91 to 836 µs. So a fallback's cost may land outside the clock meant to bound
  it.

The `x2` A/B (tcpv 128/8, ms):

| shape | base | x2 |
|---|---|---|
| decode B=1 4K | 0.068 | 0.078 |
| decode B=1 16K | 0.177 | 0.213 |
| decode B=1 64K | 0.510 | 0.566 |
| decode B=8 16K | 0.909 | 0.911 |
| decode B=32 4K | 1.136 | 1.032 |
| decode B=32 16K | 4.197 | 3.668 |
| verify T=8 B=1 16K | 0.658 | 0.686 |
| verify T=8 B=8 4K | 0.820 | 0.916 |
| verify T=8 B=8 16K | 2.355 | 2.236 |

Twice the splits helps at large batch: decode B=32 16K is 13% faster, now 2.6× FlashInfer. It
also helps verify B=8 16K, which is 5% faster (6.2×). It hurts at B=1, where programs are
already short. Best per shape in run 1124:
- decode: B=1 16K 0.177 ms (8.5×), B=8 16K 0.909 (2.5×), B=32 16K 3.668 (2.6×);
- verify: B=1 16K 0.617 with `s1` (27.9×), B=8 4K 0.820 (22.3×).

The next measurement is per-program counters in the timers build for the two fallbacks: tiles
that took the windowed score fallback, and tiles that took the float64 exp2 fallback. With a
dump of the slowest programs, it should show whether the fallbacks make the verify stragglers.

## D27. Fallback counters and the slowest programs

The timers build now writes 16 int64 per program. The four new columns, after the SM:
- the number of tiles that took the windowed score fallback;
- the number that took the float64 exp2 fallback;
- the longest tile, in cycles from the previous tile's end (or from the loop's start), so load
  waits count toward the tile that waited;
- the program's `pid_m` and `pid_s`.

Each counter is incremented inside its fallback's runtime branch, so it counts exactly the
tiles that ran it. The default build is unchanged.

For every tcpv case, `profile_step` reports:
- the fallback tile counts and how many programs had any;
- the mean program time with and without a fallback;
- the longest tile overall, against the median program's longest;
- the 8 slowest programs: their position, SM, start, tile counts, longest tile, and every phase
  summed over their tiles.

The decision this serves:
- **If the slow programs are the ones with fallbacks:** tier 1 can stop running fallbacks. It
  would flag those rows and let tier 2 rerun them exactly, which keeps tier 1 uniform and does
  not change the bits.
- **If they are not:** the cause is placement or memory, such as an SM shared with another
  launch, or a split whose keys straddle a slow region. The start times and SMs will show
  which.

**Run 1125-flash** (1f99a95; all bits checked, 0 uncertified rows, fixture SHA-256 match;
timers bits equal to the plain build in all 6 cases). Benchmark times match run 1124. The
stragglers are the windowed score fallback:

| tcpv 128/8 n64 q4 | decode B=1 16K | decode B=8 16K | verify B=1 16K | verify B=8 4K |
|---|---|---|---|---|
| tiles with the windowed fallback | 22 / 2048 | 154 / 16384 | 22 / 2048 | 34 / 4096 |
| tiles with float64 exp2 | 23 | 112 | 133 | 260 |
| program µs, with / without any fallback | 57.9 / 46.5 | 219 / 200 | 137 / 101 | 168 / 134 |
| longest tile, µs (median program's longest) | 38.4 (11.6) | 43.9 (11.1) | 213 (23.6) | 213 (25.1) |

- **Every one of the 8 slowest programs has a windowed tile**, in every case. At m16 that tile
  takes about 200 µs, against about 24 µs for a normal tile, because it recomputes all 16 rows
  × 64 keys exactly. At m2, with fewer rows, it costs about 30 µs.
- **The windowed tiles set verify's launch time.** At verify B=1 16K the slowest program has 2
  windowed tiles. It starts at 109 µs and runs 475 µs, so it ends at 584 µs, the end of the
  launch. Programs without a fallback average 101 µs. At 2.72 waves the launch would take
  about 0.30 ms without the windowed tiles, against 0.56 ms now. That is 1% of tiles costing
  about 45% of the tier-1 split launch.
- **The same tiles fall back in decode and verify** (for example pid_m 2, pid_s 33). The cause
  is in K, not Q: those keys' scores cannot be certified with 4 q digits on the tensor cores.
- **The fallback's time is not in the scores clock.** It lands in "rest": scores sum to only
  2–5 µs per slow program. So the scores clock does not bound the branch. Why is not settled.
  The longest-tile column spans whole tiles and is the reliable measure.
- **q3 explained.** With 3 q digits, 1033 of 2048 verify tiles take the windowed fallback. Its
  programs average 664 µs with a fallback and 97 µs without.
- **float64 exp2 is minor.** It costs about 7 µs per tile at m16 and about 4 µs at m2. At
  verify B=1 16K, 133 such tiles add about 5 µs per SM.

So tier 1 should not run the windowed fallback (D28): it should flag the rows and let tier 2
recompute them. Tier 2 runs one row per program (`block_m_t2 = 1`), so the same fallback there
costs 1/16 of the m16 work per tile. The cost moves to tier 2's work list: a flagged row
reruns every channel block over every KV split. Whether that is cheaper is an A/B question.

## D28. Deferring the windowed score recompute to tier 2

`launch(..., defer_scores=True)` builds tier 1 without the windowed score fallback. It applies
only with tensor-core scores, tensor-core P·V and tier 2 on; otherwise the flag is ignored.

**Tier 1.** A row with an uncertified tensor-core score on any visible key is *deferred*:
- The split kernel marks it in a new buffer `DFL (nsplit, R)`. Channel block 0 writes it, like Z,
  so every entry is written each step and nothing needs zeroing.
- The row's split flag is cleared. Its partials came from uncertified scores, so its tier-1
  flags mean nothing; tier 2 recomputes them. The float64-exp2 and large-score checks also skip
  uncertified scores, so an approximate score cannot raise a hard flag that would send the row to
  the host fallback.

**Combine.** The tier-1 combine reads `DFL` across the splits. A pair goes on tier 2's work list
when its rounding failed **or** its row was deferred, and no split hard-flagged it (as before).
Every channel block of a deferred row sees the mark, so the whole row is rerun.

**Tier 2.** Unchanged apart from two things:
- It keeps the windowed fallback (`TC_FALLBACK` is now per tier). At one row per program, a
  windowed tile is 1/16 of tier 1's m16 work.
- It rewrites the Z partials too, not only N, because a deferred row's tier-1 Z is not an
  enclosure. All channel blocks of a row compute the same Z with the same tiling, so the
  duplicate stores write the same bits.

Tier 2 detects bad inputs (NaN/inf q, k, v), overflow and uncertified weights itself, and
writes its flags with `atomic_max` into entries tier 1 left at 0. So the final flags cover every
cause. A certified result is the same bits on either path (SPEC §5.3).

**Two rerun scopes.** A deferred row is only wrong in the splits it was deferred in. In its
other splits the scores were certified, and tier 1's partials are valid enclosures (and their
flags stand).
- `defer_scores="row"` (or True) reruns the whole row: every split, with int64 P·V.
- `defer_scores="split"` reruns only the deferred splits. The tier-1 combine writes NEED = 2 for
  a deferred row (1 for a rounding failure, as before). A tier-2 program skips a NEED-2 item in
  any split where `DFL` is 0. The final combine then merges tier 2's partials for the deferred
  splits with tier 1's tensor-core P·V partials for the rest.

So the row scope does work ∝ deferred rows × all keys, and the split scope ∝ deferred
(row, split) pairs × one split's keys. The split scope can leave a row flagged that the row
scope would certify: its final rounding still uses tier 1's wider tensor-core P·V enclosures
in the splits it kept. Such a row stays flagged and correct, never wrong.

**Checks.**
- Test: real Qwen3 rows, run twice. At 2 q digits every row is deferred. At 3 digits with one
  q element of every other row scaled by 2^12, exactly those rows are deferred. In both cases
  and both scopes, at block_m 16 and 2, the outputs and flags equal the non-deferred launch's
  and certified rows equal the reference. Every row certifies.
- Compile check (sm_120): without the windowed code, tier 1 at m2 128/8 needs 190 registers and
  spills nothing, against 255 registers and 36 B of spill. At m16 128/8 it spills 452 B, against
  816 B.

**Run 1126** adds four A/Bs at 128/8 unless noted:
- `d`: row scope;
- `ds`: split scope;
- `dsx2`: split scope at twice the KV splits;
- `ds` at 64/8.

The benchmark logs tier 2's work-item count for every tcpv config. `profile_step` times each
tcpv case in split scope too: per-kernel times, and the tier-1 timers, where the windowed
column then counts deferred tiles.

Expected: verify B=1 16K tier 1 split from 0.56 ms toward 0.30 ms, plus tier 2's extra work.
Each deferred row adds 4 work items (128 channels / 32). Verify B=1 has 128 rows (8 KV heads ×
8 drafts × 2 query heads), so at most 512 items. The row scope runs each one over every KV
split, which may cost more than it saves. The split scope runs each one only over its deferred
splits: at most 22 tiles × 16 rows × 4 channel blocks, one split each.

**Run 1126-flash** (a737f84): every check passed: no config changed bits, no certified row was
wrong, fixture SHA-256 matched, and the deferred builds' certified bits equal the plain build's.
**Tier 1 sped up as predicted, but tier 2 cost far more than it saved**, so deferral loses
almost everywhere. Profile (ms per launch; plain build vs split scope):

| tcpv 128/8 n64 | tier-1 split | tier-2 split | step | tier-2 items |
|---|---|---|---|---|
| decode B=1 16K | 0.155 → 0.114 | 0.002 → 0.166 | 0.176 → 0.316 | 0 → 56 of 64 |
| decode B=8 16K | 0.913 → 0.816 | 0.002 → 1.646 | 0.924 → 2.478 | 0 → 480 of 512 |
| verify B=1 16K | 0.565 → 0.307 | 0.051 → 0.972 | 0.655 → 1.318 | 2 → 448 of 512 |
| verify B=1 16K, q3 | 1.700 → 0.285 | 0.051 → 1.344 | 1.792 → 1.669 | 2 → 448 of 512 |
| verify B=8 4K | 0.714 → 0.539 | 0.061 → 1.057 | 0.805 → 1.631 | 3 → 1473 of 4096 |

- **Tier 1 hit the prediction.** At verify B=1 16K the split launch went from 0.565 to 0.307 ms.
  The longest tile fell from 215 to 38 µs, and programs with and without a deferred tile now
  average 103 and 98 µs. With q3, tier 1 is 6× faster (1.700 → 0.285 ms) and faster than q4.
- **Almost every row gets deferred.** At verify B=1 16K, 22 deferred tiles mark 112 of 128
  rows. An uncertified score is a property of a key (D27: the same tiles fall back in decode
  and verify), so every row that sees that key is deferred.
- **Tier 2 runs those rows slowly.** Its grid is sized for a few items spread over all splits:
  about 2 × 188 / nsplit programs per split. In split scope the work sits in the deferred
  splits, so a handful of programs per split each run about 10 one-row items in sequence. In
  row scope every item covers every split: verify B=1 16K takes 6.8 ms, B=8 16K 42 ms.
- **Split scope leaves rows uncertified**, as expected: 1–5 rows at the large shapes, where
  the row scope and the plain build leave 0. They are flagged, never wrong.
- **Bench.** Deferral wins only at 1K keys, where tier 2 is cheap: verify B=1 1K 0.248 → 0.135
  ms (dsx2, now the best cell) and verify B=8 1K 0.467 → 0.437 (dsx2, but 128/8 n32 without
  deferral is still faster at 0.384). Everywhere else it is 1.3–12× slower. The best cells are
  otherwise unchanged from run 1125.

So deferral stays off by default (`defer_scores=False`). The useful result is the cause. The
windowed recompute costs 200 µs because it recomputes the whole 16 × 64 tile. The row counts
suggest the bad scores sit in a few key columns, but that is not yet measured. If they do,
recomputing only the key sub-blocks that contain one, in tier 1, should keep most of tier 1's
gain without involving tier 2 (D29). The first step is to count uncertified keys per
deferred tile.

## D29. Where the uncertified tensor-core scores sit

Before changing the windowed fallback, measure its input. The timers build now writes 24 int64
per program. Five new columns are summed over the program's tiles, taken right after the
tensor-core scores and before any fallback:
- key columns with at least one uncertified visible score;
- how many of those keys lost bits when K was sliced (`KLOST` bit 0);
- uncertified (row, key) scores;
- visible rows in those key columns, so the share of them that are uncertified;
- the most uncertified key columns in any one tile.

`profile_step` prints, per tcpv case:
- columns per tile that has any uncertified score, and the maximum;
- the share of those columns that are lossy keys;
- the share of their visible rows that are uncertified;
- how many keys in the cache lost bits when sliced.

The deferred-scope pass is dropped from the profile; D28 answered it.

What each outcome means for the fallback:
- **A few columns per tile, mostly lossy keys, most of their rows uncertified.** The cause is
  K's slicing. A windowed recompute over just those key columns, in sub-blocks of about 8
  keys, skips the rest of the tile. Lossy keys are known when the cache is written, so they
  could also get a 5th digit or an exact side path instead.
- **Many columns, or uncertified scores spread across rows.** The cause is in q, or in the
  bound itself, and slicing by key will not help. Look at the q rows and the error bound
  instead.

**Run 1127-profile** (26bb145; output bits of the timers build equal the plain build's in
every case, 0 uncertified rows). With 4 q digits the answer is the first outcome, without
exception:

| tcpv n64 q4 | windowed tiles | uncertified key columns | of them lossy keys | rows of those columns uncertified | lossy keys in the cache |
|---|---|---|---|---|---|
| decode B=1 16K (m2) | 22 | 22 (1 per tile, max 1) | 22 | 100% | 22 of 131,072 |
| decode B=8 16K (m2) | 154 | 154 (1 per tile, max 1) | 154 | 100% | 154 of 1,048,576 |
| verify B=1 16K (m16) | 22 | 22 (1 per tile, max 1) | 22 | 100% | 22 of 131,072 |
| verify B=1 16K (m16, n32) | 22 of 4096 | 22 (1 per tile, max 1) | 22 | 100% | 22 of 131,072 |
| verify B=8 4K (m16) | 34 | 34 (1 per tile, max 1) | 34 | 100% | 34 of 262,144 |

- **Every windowed tile is one lossy key.** Each key that lost bits when sliced into 4 int8
  digits makes exactly one tile fall back. Every row that sees that key is uncertified, and no
  other key in the tile is. About 1.7 keys in 10,000 are lossy here, and that is all of the
  windowed work at q4.
- **So the windowed recompute does 64× the work it needs.** It recomputes all 64 key columns
  of a tile to fix one.
- **q3 is the other outcome.** 1033 of 2048 tiles have about 63 uncertified columns each
  (max 64), but only 6% of the visible rows in those columns. That is q losing bits at 3
  digits, spread across keys. Slicing by key cannot help it, so q stays at 4 digits.

A 5th K digit for lossy keys does not fit: 7 × (4 + 5) + 7 = 70 bits exceeds int64's 63. But
lossy keys are marked when the cache is written, so tier 1 can recompute just their columns:
16 rows × 1 key exactly, in place of 16 × 64 (D30).

## D30. Recomputing uncertified scores by key column

D29 showed that at 4 q digits each windowed tile has exactly one uncertified key column. So
tier 1 now counts the key columns that hold an uncertified visible score. With up to
`col_fallback` of them (default 4), it recomputes just those keys:
- `_windowed_col` is `_windowed_logits` for one key j. It works on (rows, channels) products
  instead of (rows, keys, channels). Its integer sums per (row, key) are the same, so it gives
  the same bits as the whole-tile form's column j.
- Columns are taken lowest key first. Each one replaces only the uncertified scores in its
  column, as the whole-tile form does.
- A tile with more uncertified columns recomputes the whole tile as before. So does every tile
  when `col_fallback=0`. That covers q3, where tiles have about 63 such columns (D29).

A certified result is the same bits either way, and so is every flag. A score that neither form
can certify stays uncertified in both, with the same value.

**Branch layout matters to ptxas.** The two recomputes are exclusive, and the obvious code nests
the whole-tile one in an `else`. For 16-row blocks, ptxas then caps the split kernel at 128
registers: 21 KB of spills instead of 816 B. Running the whole-tile recompute after the
column branch, gated on the column count, keeps 255 registers and 936 B. Decode's 2-row blocks
are the other way round. The sequential layout spills 2.5 KB and needs 95 KB of shared memory;
the nested one spills 48 B and needs 84.7 KB, as before. So the layout is chosen by `BLOCK_M`,
a compile-time constant. Every build in `sm120_compile_check` passes. The two checked against
pre-D30 code are close: tcpv m16 128/8 spills 936 B (was 816), tc m16 1860 B (was 1732).

Checks:
- `test_column_score_recompute_changes_no_bits` (block_m 16 and 2) compares outputs and flags
  with `col_fallback=0`, and certified rows with the reference.
  - "few": 4 keys made lossy (one element 2^-40), two of them in one 32-key tile. Without any
    recompute every row is flagged; with the column path every row is certified.
  - "many": 2 q digits, so the tiles take the whole-tile path.
- The timers build adds two columns: key columns recomputed one at a time, and the tiles they
  were in. `profile_step` splits windowed tiles into by-column and whole-tile.
- The bench adds an A/B config "w" (128/8 n64, `col_fallback=0`): the old whole-tile fallback.

Expected: verify B=1 16K's tier 1 should fall from 0.565 ms toward the 0.307 ms it took with
no score recompute at all (D28). Decode should gain a little: its 22 or 154 lossy keys each
cost a whole-tile recompute.

**Run 1128** (9e41d4e). All checks passed:
- the GPU test segments;
- the Qwen3 fixture head, bit-identical (ALL MATCH, 0 fallback rows);
- the timers build, whose output bits equal the plain build's in every profile case;
- 0 uncertified rows in every benchmark cell.

At q4, every windowed tile took the column path, with no whole-tile recompute: 22, 22, 34 and
154 tiles of one column each. At q3, 1024 of 1033 tiles took the whole-tile path, as designed.

Tier-1 split kernel (profile, ms). Run 1126's plain build is the "before":

| case | run 1126 | run 1128 |
|---|---|---|
| decode B=1 16K | 0.155 | 0.128 |
| decode B=8 16K | 0.913 | 0.843 |
| verify B=1 16K | 0.565 | 0.314 |
| verify B=8 4K | 0.714 | 0.568 |

A program with a fallback tile now takes 113 µs against 107 µs without one, for verify B=1
16K. In run 1127 such a tile alone cost about 200 µs.

Whole step in the same run (bench, CUDA graph, 128/8 n64 q4; default against "w", the
whole-tile recompute):

| case | 1K | 4K | 16K | 64K |
|---|---|---|---|---|
| verify B=1 | 0.111 / 0.275 | 0.121 / 0.284 | 0.432 / 0.652 | |
| verify B=8 | 0.220 / 0.460 | 0.675 / 0.804 | 2.075 / 2.352 | |
| decode B=1 | 0.041 / 0.061 | 0.047 / 0.068 | 0.146 / 0.173 | 0.467 / 0.508 |
| decode B=8 | 0.076 / 0.086 | 0.262 / 0.291 | 0.851 / 0.916 | |
| decode B=32 | 0.337 / 0.362 | 1.118 / 1.153 | 4.129 / 4.201 | |

The best exact config against FlashInfer is now:
- verify B=1: 9.1×, 8.8× and 19.5× at 1K, 4K and 16K (0.078, 0.110 and 0.404 ms);
- decode B=1: 3.6×, 4.5×, 7.3× and 2.4× at 1K to 64K.

**Where verify B=1 16K's tier 1 goes now.** Its 512 programs run one per SM (255 registers)
in about 2.7 waves. The SMs are busy 88% of the launch span. Per 64-key tile:

| part | µs |
|---|---|
| scores | 1.5 |
| weights | 1.8 |
| tile partial | 4.5 |
| merge | 1.3 |
| rest of the loop (load waits) | 14.5 |

The rest of the loop is 53% of program time, and the prologue (q slicing, first loads) another
12%. The score fallback no longer matters. Tier 2 is 0.062 ms for 2 work items, 15% of the
step.

## D31. What the key loop waits on: Nsight Compute

The phase clocks put 53% of verify B=1 16K's tier-1 program time in the key loop, outside the
four timed phases. They cannot say which loads that is, or whether DRAM, L2 or latency is the
limit. A rough count says it is not DRAM bandwidth:
- Per 64-key tile, tier 1 reads 4 K digit planes (32 KB), 3 V digit planes (24 KB) and the
  V-lost mask (8 KB), about 64 KB.
- For the step's 8 KV heads × 16K keys, that is about 128 MB in 0.314 ms: about 0.4 TB/s, a
  fraction of the card's DRAM bandwidth.
- FlashInfer reads 64 MB of bf16 K and V. That fits in the 128 MB L2 across steps.

`scripts/ncu_step.py` runs Nsight Compute 2025.2 on one step per case. It is free with PACE's
NVHPC 25.5 module tree. It profiles the step's two split-kernel launches (tier 1, tier 2), on
decode B=1 16K, verify B=1 16K, verify B=8 4K and decode B=8 16K. It records:
- duration, DRAM, L2 and L1 throughput, DRAM bytes read, and the hit rates;
- achieved occupancy, registers, warp cycles per issued instruction, and scheduler stats;
- the sampled warp-stall breakdown;
- the 20 SASS lines with the most stall samples.

The cache is not flushed between replay passes (`--cache-control none`), so L2 holds what
earlier passes left, as in a decode loop. Raw reports (`.ncu-rep`) and CSVs stay in the run
directory. `EA_PLAN=ncu` runs only the environment stage and this one.

What each answer would point to:
- **Long-scoreboard stalls on the digit-plane loads, low DRAM throughput:** latency with one
  program per SM. Deeper pipelining of the K and V planes, or fewer bytes per key (packing the
  V-lost mask into bits: 8 KB → 1 KB per tile).
- **DRAM throughput near peak:** bytes. Cut the planes read per tile.
- **Barrier or MMA-wait stalls:** the tile's own dependency chain, not memory.
- **A large prologue share in the hot lines:** q slicing per program, which is repeated per KV
  split.

**Run 1129-ncu** (259b6a5). ncu could read the GPU counters on embers, and all 4 cases
profiled, with 0 uncertified and 0 wrong. ncu runs at its locked base clock, 1.85 GHz instead
of about 2.33, so durations are about 1.3× the benchmark's.

Two exports failed inside the job: the details page and the source page. ncu's embedded Python
hit an ImportError under the job's venv and module paths. Re-importing the saved reports on the
login node with a clean environment worked, and `ncu_step.py` now imports that way.

Tier 1, verify B=1 16K (418 µs under ncu):
- **Memory is not the limit.** DRAM throughput is 30% (483 GB/s), the L2 hit rate 76%, and
  SM throughput 25%.
- **Latency is.** One block per SM, limited by 255 registers and 84 KB of shared memory, gives
  2 warps per scheduler. Only 0.26 of them are eligible on average, so 81% of cycles issue
  nothing. A warp waits 10.6 cycles per issued instruction.
- **Stall reasons:** short scoreboard 33%, long scoreboard 16%, fixed-latency wait 15%,
  math-pipe throttle 7%, barrier 6%.

Where the stalls are, from the source page's samples:
- **float64 work: about 21% of tier 1's stall samples** (float64 instructions and their
  selects). sm_120 runs float64 at a small fraction of float32 speed, and these are
  short-scoreboard waits.
  - 9.3% is per tile, run once per warp per tile: `_l1_ceil(kl1s)`, the integer bound on each
    key's slicing error, recomputed in float64 on every tile from the stored float64 L1 norm.
  - 11.8% is per program: q's L1 norm in `_slice_q`, a float64 sum over 128 channels per row,
    plus its ceiling. It is redone by every KV split's program: 64 times for the same rows here.
  - 1.4% is the float64 exp2 fallback (133 tiles).
- **Spills:** STL and LDL hold 6.4% of samples (76–160 B of spill).
- **Tensor-core waits:** IMMA holds 7%.
- **The rest is the int64 arithmetic.** IADD, SHF, SEL and LOP3 together are 44% of the
  instructions issued.

Decode B=8 16K (898 µs) is different. L2 hits only 24%. The stalls are fixed-latency wait 19%,
long scoreboard 17%, short scoreboard 14% and LSU throttle 10%. One instruction holds 11% of
all samples: an `ISETP` per tile waiting on a global load (long scoreboard). The many 8-byte
`LDGSTS.E.64` copies add LSU throttle. float64 is only 3% here: a 2-row block spreads the
per-key work over fewer rows.

Tier 2 is tiny in decode (2.7 µs). In verify B=1 16K it takes 70 µs under ncu, waiting mostly on
short scoreboard and fixed latency.

Next (D32): take float64 out of the hot path.
- Store each key's integer L1 ceiling and its too-big bit when the cache is written. These are
  the same values `_l1_ceil` computes, so outputs and flags are unchanged.
- Bound q's L1 norm in integers instead of float64. Each element is an int8-range mantissa
  shifted by at most 7 × sq bits, so the exact integer sum fits int64.

## D32. No float64 in the score error bound

The score error bound is qlost·|K|₁ + klost·|q|₁, each scaled by 2^-lsb and rounded up to an
integer. D31 found its two float64 parts holding about 21% of verify B=1 16K's tier-1 stall
samples. Both are now integer.

**Keys: made at cache write.** `tc_slices` and `tc_slices_device` used to store the scaled L1
norm (float64), and every tile recomputed `ceil(l1s · (1 + 2^-40)) + 1` from it. The cache now
stores that integer, `KERR` (int64), plus a too-big bit as bit 2 of the per-key flag word. It is
computed by `_l1_ceil_host` with the same float64 operations, in numpy or torch to match the
writer. So the kernel sees the same integers: outputs and flags are unchanged. Per tile the
kernel loads 8 bytes per key either way, and no longer does any float64 work.

**q: an integer bound in `_slice_q`.** It was a float64 sum of |q|·2^e over the row times
(1 + 2^-40), then a float64 ceiling, plus 1. It is now exact fixed point:
- Each element am·2^(e − lsb) is held with QF = 20 fraction bits. An element smaller than one
  unit is rounded up to one unit.
- An element is below 2^(7·sq + 20). 128 of them sum below 2^55.
- The bound is the sum rounded up to an integer. That is the exact ceiling, or one more when
  the round-ups cross an integer.

It is never larger than the old float64 bound, which was at least the exact ceiling plus 1. A
smaller error interval can only keep a certified score certified, with the same bits. So
results can only gain certified scores, never lose them.

`qbig` is now always false: the bound is below 2^35.

Checks:
- `test_q_l1_bound_is_tight_integer` (every input kind, sq 3 and 4) runs `_slice_q` in a probe
  kernel. The bound must lie between the exact value's ceiling (in Python fractions) and one
  more, and must not exceed the old float64 bound. The inputs include an all-zero row, a lone
  subnormal, and one large element among tiny ones.
- The device and numpy key slicers may differ by one in the stored bound. They already differed
  in l1 by about 1e-12 relative (summation order).
- All 25 builds in `sm120_compile_check` pass, with slightly less spilling: tcpv m16 128/8
  880 B (was 936), m16 32/4 1232 B (was 1536).

Expected: up to about a fifth fewer stall samples in verify B=1 16K's tier 1, part of it in
every program's prologue. Decode gains less: its float64 share was 3%.

**Run 1130** (65514ed). All checks passed:
- the GPU test segments, on the second attempt (the first hit the stage timeout);
- the Qwen3 fixture head, bit-identical (ALL MATCH, 0 fallback rows);
- the timers build, whose output bits equal the plain build's in every profile case;
- 0 uncertified rows in every default-config benchmark cell.

Fallback counts are the same as run 1128's: 22, 22, 34 and 154 windowed tiles, each with one
column, and the same float64 exp2 tiles. So the speedup comes from the bound's arithmetic
alone.

Tier-1 split kernel (profile, ms):

| case | run 1126 | run 1128 | run 1130 |
|---|---|---|---|
| decode B=1 16K | 0.155 | 0.128 | 0.119 |
| decode B=8 16K | 0.913 | 0.843 | 0.828 |
| verify B=1 16K | 0.565 | 0.314 | 0.236 |
| verify B=8 4K | 0.714 | 0.568 | 0.436 |

Verify tier 1 is about 24% faster, more than the 21% float64 share of stall samples. Decode
moves 2–7%, as expected from its 3% share.

Verify B=1 16K per 64-key tile (µs), run 1128 → run 1130:

| part | run 1128 | run 1130 |
|---|---|---|
| scores | 1.5 | 1.32 |
| weights | 1.8 | 1.63 |
| tile partial | 4.5 | 4.38 |
| merge | 1.3 | 1.34 |
| rest of the loop | 14.5 | 11.03 |
| prologue (per program) | 12% of program | 2.04 (2.5%) |

The program mean is 82 µs (was about 107). The prologue gain is q's L1 bound. Most of the
per-tile gain shows up in "rest of the loop", where the float64 waits of the key bound landed.

Whole step (bench, CUDA graph, 128/8 n64 q4), run 1128 → run 1130:

| case | 1K | 4K | 16K | 64K |
|---|---|---|---|---|
| verify B=1 | 0.111 → 0.087 | 0.121 → 0.096 | 0.432 → 0.354 | |
| verify B=8 | 0.220 → 0.172 | 0.675 → 0.555 | 2.075 → 1.762 | |
| decode B=1 | 0.041 → 0.037 | 0.047 → 0.043 | 0.146 → 0.142 | 0.467 → 0.452 |
| decode B=8 | 0.076 → 0.068 | 0.262 → 0.248 | 0.851 → 0.833 | |
| decode B=32 | 0.337 → 0.313 | 1.118 → 1.058 | 4.129 → 3.962 | |

The best exact config against FlashInfer is now:
- verify B=1: 6.9×, 6.9× and 15.9× at 1K, 4K and 16K (0.058, 0.085 and 0.329 ms);
- verify B=8: 8.8×, 12.3× and 4.5×;
- decode B=1: 3.2×, 4.1×, 7.0× and 2.3× at 1K to 64K.

What is left in verify B=1 16K's step (0.330 ms profiled):
- tier 1 is 0.236 ms, and 54% of its program time is still the rest of the loop;
- tier 2 is 0.056 ms for 2 work items, plus 0.015 ms to combine. Together that is 21% of the
  step, up from 15%.

Decode B=8 16K barely moved. D31 put its time in a per-tile `ISETP` waiting on a global load,
with a 24% L2 hit rate, not in float64.

**Run 1131-ncu** (3e71506): Nsight Compute again, on the D32 build, with the same 4 cases as
run 1129. All 4 cases had 0 uncertified and 0 wrong. Durations are under ncu's locked clock.

| tier 1 | run 1129 | run 1131 |
|---|---|---|
| verify B=1 16K: µs | 418 | 314 |
| verify B=1 16K: DRAM throughput | 30% | 40% |
| verify B=1 16K: eligible warps per scheduler | 0.26 | 0.37 |
| verify B=1 16K: float64 share of stall samples | about 21% | 1.9% |
| verify B=1 16K: short-scoreboard stalls | 33% | 8% |
| verify B=8 4K: µs | 717 | 546 |
| decode B=1 16K: µs (DRAM) | 158 (57%) | 151 (60%) |
| decode B=8 16K: µs (DRAM) | 898 (79%) | 870 (82%) |

Verify issues about 4% fewer instructions but takes 25% less time: D32 removed latency, not
work.

**Verify tier 1 has no single hot spot left.**
- The stalls are spread out: fixed-latency wait 17%, long scoreboard 17%, math-pipe throttle
  9%, short scoreboard 8%, barrier 6%.
- The hottest SASS line holds 2.5% of samples.
- The int64 arithmetic (IADD, SHF, SEL, LOP3) is now the bulk of both instructions and stalls.
- IMMA holds 9.7%.
- The limit is still one 255-register block per SM: 2 warps per scheduler, with 74% of
  cycles issuing nothing.

**Decode B=8 16K is bound by DRAM bytes.** It runs at 82% of DRAM throughput (1.30 TB/s), so
only fewer bytes per key can speed it up. Per key, tier 1 reads:
- 512 B of K digits (4 planes);
- 384 B of V digits (3 planes);
- 128 B of lost-V mask (one int8 0/1 per element);
- about 26 B of per-key metadata.

That is about 1050 B per key. FlashInfer reads 512 B of bf16 K and V per key, which matches
the 2.3× gap at this shape.

The hottest lines fit this:
- the `ISETP` (11.9%) and an `STS.U16` (4.5%) wait on plain synchronous global loads;
- the `LDGSTS` copies wait on LSU throttle (8%).

The `STS.U16` most likely stages the lost-V mask into shared memory as an MMA operand. This
is inferred from the SASS; the run has no source-line mapping.

**Verify tier 2 is latency.**
- B=1 16K: 65 µs for 2 work items. Only 128 of its 384 programs do any work, each over one
  row's 256 keys in its KV split, which is about 16 µs per 64-key tile on the int64 path.
- DRAM is idle (0.2%) and L2 hits 99.8%.
- The top line (7.2%) is a `LOP3` waiting on a global load.

Next (D33): stop reading the lost-V mask when a tile has no lost V element. Store a per-key
"has lost V" bit at cache write, next to `VBAD`. Only tiles with such a key load the 8 KB
mask and run its MMA. That removes about 12% of decode's bytes per key, if lost V elements
are as rare as lossy keys (about 1.7e-4 of keys). The first step is to measure that rate on
the bench and fixture caches. Packing the mask into bits instead (128 B to 16 B per key)
saves the same bytes, but needs an unpack before the MMA.

## D33. Read a key's lost-V row only if it has a lost element

The lost-V mask is one int8 (0 or 1) per V element: 128 B of the ~1050 B that tier 1 reads
per key. Decode B=8 16K runs at 82% of DRAM throughput (D32), so these bytes cost time.

**How rare are lost V elements?** With 3 V digits, measured with `slice_rows` at 3 digits:

| V rows | lost elements | keys with one | 64-key tiles with one |
|---|---|---|---|
| bench (randn bf16, 262144 keys) | 1.3e-4 | 1.6% | 66% |
| Qwen3 layer 0 head 0 (2048 keys) | 1.8e-4 | 1.5% | 53% |
| Qwen3 layer 27 head 11 (2048 keys) | 1.9e-4 | 2.4% | 75% |

So the per-tile skip proposed above would rarely fire: most tiles hold such a key. A per-key
skip does fire: about 98% of keys have an all-zero row.

**Change.** The V flag word (`VBAD`, one int8 per key) gains bit 1: the key has a lost element.
It is made at cache write by `tc_v_slices` and `tc_v_slices_device`, with bit 0 still the
NaN/inf bit. `_tc_pv` loads the flag word first, and loads a key's mask row only when bit 1 is
set. The other rows come back as the masked load's 0, which is what they hold. So every MMA
operand is unchanged: same bits, flags and partials. A predicated-off load moves no bytes, so
the mask costs about 2.6 B per key instead of 128 (about 12% of tier 1's DRAM bytes).

The mask row's load address now depends on the flag word's load. The plain build already
issued the mask as a synchronous load (D32's `STS.U16` waiting on long scoreboard), so this
adds no new wait.

Check: `test_lost_v_rows_read_only_for_flagged_keys`, at block_m 16, 4 and 2, which are the
three P·V layouts. One key in 5 has an element scaled by 2^20, which makes its small elements
lose bits.
- The stored bit 1 must equal "the row has a lost element".
- Forcing bit 1 on for every key (read every row) must give the same output, flags, and P·V
  partials N and their error widths.
- Clearing bit 1 everywhere must change the error widths. This shows the rows really are read
  through the bit.

All 25 builds in `sm120_compile_check` pass. tcpv m16 128/8 spills 836 B (was 880); tcpv m2
128/8 spills 68 B.

Expected: decode B=8 16K tier 1 down by up to about 10% (0.828 ms), and decode B=32 likewise.
B=1 decode runs at 60% of DRAM and gains less. Verify is latency-bound and should not change.

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
