# ExactAttention specification, v0

This document defines the value an ExactAttention kernel must return. It defines no algorithm.
Any implementation, on any hardware, with any tiling, split count, page layout, batch composition
or key order, must return exactly these bits. If it cannot certify that it has, it must fall back
to a slower path that can.

## 1. Number formats

A *finite binary floating-point number* with precision `p` is `x = M · 2^E` with integer `M`,
`|M| < 2^p`, integer `E`.

| format | p  | min quantum exponent `qmin` | max exponent `emax` | max finite |
|--------|----|------------------------------|---------------------|------------|
| bf16   | 8  | -133                         | 127                 | (2^8 - 1) · 2^120  |
| fp32   | 24 | -149                         | 127                 | (2^24 - 1) · 2^104 |
| FP24∞  | 24 | none (unbounded)             | none (unbounded)    | none |

`RN_F(x)` is IEEE 754 round-to-nearest, ties-to-even, into format `F` with gradual underflow.
It is defined for every real `x`, and in this spec it is only ever applied to rationals.
For bf16 and fp32, a result whose magnitude exceeds max finite after rounding is an error, not
an infinity; see §4. `FP24∞` has 24 significand bits and an unbounded exponent, so it can neither
overflow nor underflow. It exists only as an intermediate.

All inputs are finite. NaN or infinity in any input is an error.

## 2. Inputs

For one query row:

- `q ∈ bf16^d`: the query vector.
- `S`: a finite set of key indices (the mask). Each `j ∈ S` contributes a key `k_j ∈ bf16^d`
  and a value `v_j ∈ bf16^{d_v}`.
- `α ∈ fp32`: the base-2 logit scale. Standard attention uses `α = RN_fp32(log2(e) / sqrt(d))`.
  The scale is an input, so callers can use any softmax temperature, but it is fixed per call.
- An output format `F_out ∈ {bf16, fp32}`.

## 3. Definition

For each `j ∈ S`:

1. **Exact score.** `s_j = Σ_{i=1..d} q_i · k_{j,i}`, computed exactly. It is a dyadic rational.
2. **Rounded base-2 logit.** `t_j = RN_fp32(α · s_j)`. This is one rounding of the exact product.
3. **Weight.** `w_j = RN_FP24∞(2^{t_j})`. Ties cannot occur (§5.1), so `w_j` is well defined.

Then, exactly:

4. `Z = Σ_{j∈S} w_j` and `N_c = Σ_{j∈S} w_j · v_{j,c}` for each channel `c = 1..d_v`.
5. **Output.** `o_c = RN_{F_out}(N_c / Z)`. If `S` is empty, `o = +0`. Every zero result is
   encoded as `+0`, including `N_c = 0` and a tiny negative quotient that underflows. The
   sign of zero carries no information here, and fixing it removes a source of mismatch.

No max is subtracted. Every `w_j` has an unbounded exponent, so the definition is invariant to
scaling. Real implementations subtract a running max only for range, and §5.2 shows this
cannot change the result.

## 4. Domain and errors

An implementation must report an error, and must not return a value, when:

- any input is NaN or infinite;
- `|α · s_j|` rounds above fp32 max finite for some `j`, so `t_j` would be infinite.

The output itself cannot overflow. `N_c / Z` is a convex combination of the `v_{j,c}`, so
`|N_c / Z| ≤ max_j |v_{j,c}|`, and rounding a value at most bf16 max finite cannot exceed it.

## 5. Properties

### 5.1 No ties in step 3

`t_j` is a dyadic rational. Write it in lowest terms as `a / 2^k`.

- If `k = 0`, then `t_j` is an integer, `2^{t_j}` is a power of two, and it is exact in FP24∞.
- If `k > 0`, then `2^{t_j}` is irrational. Suppose instead `2^{a/2^k} = u/v` in lowest terms.
  Then `2^a · v^{2^k} = u^{2^k}`. Comparing 2-adic valuations gives
  `a + 2^k·ν(v) = 2^k·ν(u)`, so `2^k | a`. That contradicts lowest terms when `a` is odd.

An irrational number is never a rounding midpoint, because midpoints are dyadic. So
`RN_FP24∞(2^{t_j})` never needs the tie rule, and any method that determines `2^{t_j}` closely
enough decides it. Interval arithmetic with rising precision (Ziv's strategy) always terminates.

### 5.2 Scale freedom

For any integer `R`, `RN_FP24∞(2^{t - R}) = 2^{-R} · RN_FP24∞(2^t)`, because rounding to a fixed
precision with an unbounded exponent commutes with multiplication by a power of two.

So an implementation may compute weights relative to any integer reference exponent: a running
max, a per-split max, a per-tile max. When two partial results are combined, the reference shift
is an exact power of two. Only the final ratio `N_c / Z` is rounded, and the common factor
cancels.

### 5.3 The invariance theorem

`o` depends only on `q`, `α`, `F_out` and the multiset `{(k_j, v_j) : j ∈ S}`.

*Proof.* Steps 1–3 depend only on `(q, α, k_j)`. Step 4 uses exact sums, which are associative
and commutative. Step 5 is a function of `(N, Z)`. ∎

**Corollaries.** In each case below, two computations attend the same multiset, so the bits are
identical:

- **Key order and KV splits.** Any partition of `S` into splits, any merge tree, and any number
  of CTAs, warps or SMs.
- **Paging.** Any page size or page-table layout.
- **Batch invariance.** Other rows in the batch do not enter the definition.
- **Prefill equals decode.** A token's row attends to the same set either way.
- **Speculative and tree verification equal sequential decoding.** A draft node at depth `δ`
  attends to the prefix plus its ancestors. Sequential decoding of the accepted path attends to
  the same set at that position. Bit-identical attention outputs follow for every node of any
  draft tree, of any shape or draft length.
- **Tensor parallelism.** Attention heads are independent, so any head sharding gives the same
  result.
- **Hardware.** No step references hardware rounding behaviour.

These are properties of attention alone. A whole-model guarantee also needs exact or
order-fixed GEMM, normalisation and sampling. Those are out of scope for v0.

## 6. Relation to standard attention

With `α = log2(e) / sqrt(d)`, standard softmax attention is `o = Σ_j softmax(s/√d)_j · v_j`,
and `softmax(s/√d)_j = 2^{α s_j} / Σ 2^{α s_j}`. ExactAttention rounds twice before the final
step: `t_j` to fp32, and `w_j` to 24 bits. Each relative perturbation is at most `2^-24`,
up to the factor `|t_j|` in `t_j`'s absolute error. So for logits of moderate size, ExactAttention
agrees with exact real attention to well below one bf16 ulp. Each intermediate step is at least
as accurate as the matching step of an fp32-accumulating FlashAttention kernel:

| step | ExactAttention | typical FlashAttention |
|------|----------------|------------------------|
| score | exact | fp32 accumulation |
| logit | one rounding | scale, then round again |
| weight | correctly rounded | approximate `ex2` |
| sums | exact | fp32 sums |

Unlike FlashAttention, it is also reproducible.

## 7. Certified implementations

The intended fast path, used on GPU:

1. Accumulate `Z` and `N` in fixed-width windows of 64 or 128 bits. Use a running integer
   reference exponent, and drop or truncate terms that fall below the window.
2. Carry rigorous error bounds for everything dropped or truncated.
3. At the end, compute the interval `[N_lo/Z_hi, N_hi/Z_lo]`. If both endpoints round to the
   same `F_out` value, that value equals `o_c` by definition, and the implementation returns it.
4. Otherwise, fall back to a wider or fully exact computation for that row.

Step 3 makes the fast path correct by construction, even though its intermediate state
depends on the partition. The same strategy applies to `s_j` in step 1 and to `w_j` in step 3.
