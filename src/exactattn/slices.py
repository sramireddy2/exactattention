"""int8-slice representation for tensor-core scores (DESIGN Q2).

A bf16 vector x (one q row, or one K row) becomes block fixed point. Take the leading
exponent E of its largest element, keep B = 7*S bits below E + 1, and split the magnitude
into S signed 7-bit digits (int8, in [-127, 127]):

    x_i ~= sum_s  X[s, i] * 2^(lsb + 7*s),     lsb = E + 1 - 7*S

Bits below lsb are truncated toward zero. The truncation error is bounded by
|x_i - x~_i| < 2^lsb, and only for elements that lost bits.

Then q.k = sum over slice pairs (s, t) of 2^(7(s+t)) * (Q[s] . K[t]). Each Q[s] . K[t] is
an int8 x int8 dot product with exact int32 accumulation (|.| < d * 127^2 < 2^21 for
d <= 128). That is exactly what integer tensor cores compute (mma.s8 / DPAS int8).

Pairs whose combined weight sits far below the result can be dropped. Their worst-case
contribution goes into the error bound. The bound then certifies t = RN32(alpha * s), as in
the windowed CPU path. Certified scores are exact by definition; the rest fall back.
"""

from __future__ import annotations

import numpy as np

from .reference import decode_bf16_array

DIGIT = 7
DMAX = 127


def slice_rows(bits: np.ndarray, S: int):
    """bits: (n, d) bf16 bit patterns. Returns (X: (S, n, d) int8 digits, lsb: (n,) int64,
    l1: (n,) float64 upper bound of sum |x_i|, lost: (n, d) bool)."""
    m, e = decode_bf16_array(bits)
    nz = m != 0
    am = np.abs(m)
    lead = np.where(nz, e + np.floor(np.log2(np.maximum(am, 1))).astype(np.int64), -(1 << 40))
    E = lead.max(axis=1)
    lsb = E + 1 - DIGIT * S
    sh = e - lsb[:, None]                                   # shift that brings m to units of 2^lsb
    fix = np.where(sh >= 0, am << np.clip(sh, 0, 62), am >> np.clip(-sh, 0, 62))
    fix = np.where(-sh > 62, 0, fix)
    lost = nz & (sh < 0) & ((fix << np.clip(-sh, 0, 62)) != am)
    sign = np.sign(m)
    X = np.zeros((S,) + bits.shape, dtype=np.int8)
    rem = fix.copy()
    for s in range(S):
        X[s] = (sign * (rem & DMAX)).astype(np.int8)
        rem >>= DIGIT
    assert not rem.any(), "magnitude exceeded S digits (E computed wrong)"
    l1 = (am * np.exp2(e.astype(np.float64))).sum(axis=1) * (1 + 2.0 ** -40)
    return X, lsb, l1, lost


def sliced_scores(qX, qlsb, ql1, qlost, kX, klsb, kl1, klost, keep_pairs=None):
    """Exact integer part S_int and a rigorous error bound, per key, in units of
    2^(qlsb + klsb_j).

    qX: (Sq, d) for one query row; kX: (Sk, n, d). keep_pairs: set of (s, t) pairs to
    compute (default all). Returns (acc int64 (n,), err float64 (n,), n_mma)."""
    Sq, Sk = qX.shape[0], kX.shape[0]
    pairs = [(s, t) for s in range(Sq) for t in range(Sk)] if keep_pairs is None else sorted(keep_pairs)
    # Exact combination in Python ints: with many slices it needs more than 64 bits. A GPU
    # combine uses two words, or keeps Sq + Sk small enough for int64.
    acc = np.zeros(kX.shape[1], dtype=object)
    for s, t in pairs:
        dot = kX[t].astype(np.int32) @ qX[s].astype(np.int32)    # what one int8 MMA k-step yields
        acc = acc + np.array([int(x) << (DIGIT * (s + t)) for x in dot.tolist()], dtype=object)
    pset = set(pairs)
    dropped = 0.0
    for s in range(Sq):
        q_l1 = float(np.abs(qX[s].astype(np.int64)).sum())     # |Q[s] . K[t]| <= ||Q[s]||_1 * 127
        for t in range(Sk):
            if (s, t) not in pset:
                dropped += q_l1 * DMAX * 2.0 ** (DIGIT * (s + t))
    # Truncation: |q k - q~ k~| <= |dq| |k| + |q~| |dk|, with |dq_i| < 2^qlsb only where bits
    # were lost. Bound it in units of 2^(qlsb + klsb):
    #   sum_i |dq_i| |k_i| <= 2^qlsb * ||k||_1                       -> ||k||_1 / 2^klsb units
    #   sum_i |q~_i| |dk_i| <= 2^klsb * ||q||_1                       -> ||q||_1 / 2^qlsb units
    err = np.zeros(kX.shape[1])
    if qlost.any():
        err += kl1 / np.exp2(klsb.astype(np.float64))
    err += np.where(klost.any(axis=1), ql1 / np.exp2(float(qlsb)), 0.0)
    err += dropped
    # Integer bound: 0 when nothing was truncated or dropped, otherwise rounded up with slack.
    err_i = np.where(err > 0, np.ceil(err * (1 + 2.0 ** -40)) + 1, 0).astype(np.int64)
    return acc, err_i, len(pairs)


def triangle_pairs(Sq: int, Sk: int, depth: int) -> set:
    """Keep slice pairs whose significance is within `depth` digits of the top pair."""
    top = (Sq - 1) + (Sk - 1)
    return {(s, t) for s in range(Sq) for t in range(Sk) if s + t >= top - depth}
