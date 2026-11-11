"""Correctly rounded weights w = RN_FP24inf(2**t) for fp32 t (SPEC §3 step 3).

Two implementations:

* `exp2_weight_oracle`: rigorous. It uses mpmath interval arithmetic and raises the precision
  until the enclosing interval of 2**f decides the rounding (Ziv's strategy). It always
  terminates, because 2**f is irrational for non-integer dyadic f (SPEC §5.1).
* `exp2_weight_fast_np`: vectorised float64 evaluation. Inputs whose float64 result is too
  close to a rounding midpoint to be safe are flagged and recomputed by the oracle. Its
  soundness rests on one stated assumption: the platform's exp2 has relative error below
  2**-44. Libms are typically within 1 double ulp, about 2**-52. The fast path is checked
  against the oracle in tests.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from mpmath import iv
from mpmath.libmp import to_man_exp

P = 24  # weight precision (FP24inf)


def _split(t: float) -> tuple[int, float]:
    n = math.floor(t)
    f = t - n  # exact for any fp32 t, since the double has enough bits
    return n, f


def _rn_int_of_dyadic(m: int, e: int) -> int:
    """floor(m*2**e + 1/2) for an exact dyadic."""
    if e >= 0:
        return m << e
    sh = -e
    return (m + (1 << (sh - 1))) >> sh


@lru_cache(maxsize=1 << 20)
def exp2_weight_oracle(t: float, p: int = P) -> tuple[int, int]:
    """Return (M, E) with 2**(p-1) <= M < 2**p and M*2**E = RN_p(2**t), unbounded exponent."""
    if not math.isfinite(t):
        raise ValueError("t must be finite")
    n, f = _split(t)
    if f == 0.0:
        return (1 << (p - 1), n - (p - 1))
    prec = 64
    while True:
        old = iv.prec
        try:
            iv.prec = prec
            y = iv.exp(iv.mpf(f) * iv.log(2))  # encloses 2**f, which lies in (1, 2)
            lo_raw, hi_raw = y._mpi_
        finally:
            iv.prec = old
        lo_m, lo_e = to_man_exp(lo_raw, signed=True)
        hi_m, hi_e = to_man_exp(hi_raw, signed=True)
        q_lo = _rn_int_of_dyadic(lo_m, lo_e + (p - 1))
        q_hi = _rn_int_of_dyadic(hi_m, hi_e + (p - 1))
        if q_lo == q_hi:
            q = q_lo
            e = n - (p - 1)
            if q == (1 << p):
                q >>= 1
                e += 1
            return (q, e)
        prec *= 2
        if prec > 1 << 16:
            raise RuntimeError("exp2 oracle failed to converge (should be impossible)")


# Distance from the nearest midpoint, in units of the final ulp. If |frac - 0.5| exceeds
# this, a float64 exp2 with relative error below 2**-44 cannot have crossed the midpoint:
# the absolute error in units of the 24-bit ulp is below 2**-44 * 2**24 * 2 = 2**-19.
_SAFE_MARGIN = 2.0 ** -18


def exp2_weight_fast_np(t: np.ndarray, p: int = P) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised weights. t is a float32 array (any shape).

    Returns int64 arrays (M, E) with M in [2**(p-1), 2**p).
    """
    t64 = np.asarray(t, dtype=np.float32).astype(np.float64)
    if not np.all(np.isfinite(t64)):
        raise ValueError("t must be finite")
    n = np.floor(t64)
    f = t64 - n
    y = np.exp2(f)                       # in [1, 2]
    scaled = y * float(1 << (p - 1))     # in [2**(p-1), 2**p]
    fl = np.floor(scaled)
    frac = scaled - fl                    # exact, since scaled has at most 53 significant bits
    q = (fl + (frac >= 0.5)).astype(np.int64)
    e = n.astype(np.int64) - (p - 1)
    carry = q == (1 << p)
    q = np.where(carry, q >> 1, q)
    e = np.where(carry, e + 1, e)
    exact_int = f == 0.0
    q = np.where(exact_int, 1 << (p - 1), q)
    e = np.where(exact_int, n.astype(np.int64) - (p - 1), e)
    risky = (~exact_int) & (np.abs(frac - 0.5) < _SAFE_MARGIN)
    if np.any(risky):
        idx = np.flatnonzero(risky)
        tf = t64.reshape(-1)
        qf, ef = q.reshape(-1), e.reshape(-1)
        for i in idx:
            qi, ei = exp2_weight_oracle(float(tf[i]), p)
            qf[i], ef[i] = qi, ei
    return q, e
