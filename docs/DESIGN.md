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
