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
from mpmath import iv, mp, mpf

P = 24  # weight precision (FP24inf)


def _raw_to_man_exp(raw) -> tuple[int, int]:
    """Signed (man, exp) of a raw mpmath mpf tuple (sign, man, exp, bc)."""
    sign, man, exp, _ = raw
    if not man and exp:
        raise ValueError("non-finite interval endpoint")
    return (-int(man) if sign else int(man)), int(exp)


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
        lo_m, lo_e = _raw_to_man_exp(lo_raw)
        hi_m, hi_e = _raw_to_man_exp(hi_raw)
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


# ---------------------------------------------------------------------------
# Integer exp2 for the GPU kernels (DESIGN D15)
# ---------------------------------------------------------------------------
#
# The kernels compute w = RN24(2^t) without float64, which runs at 1/64 rate on consumer GPUs.
# t = n + f with n = floor(t), and f is exact as F = f * 2^50 (an fp32 t with |t| >= 2^-27 is a
# multiple of 2^-50; smaller |t| give w = 1). Then
#     2^f = 2^(j/1024) * 2^r,   j = F >> 40,   r = (F mod 2^40) * 2^-50 < 2^-10,
# with 2^(j/1024) from a 1024-entry table and 2^r = 1 + x + x^2/2 + x^3/6 + O(x^4), x = r ln 2,
# in fixed point with int64 products of 32-bit factors (exact). Y approximates 2^f * 2^61 with
# |Y - 2^f * 2^61| <= 13 * 2^19 (error analysis in DESIGN D15); EXP2_ERR rounds that up. The
# 24-bit weight is round(Y / 2^38), certified when Y mod 2^38 is more than EXP2_ERR from 2^37.

EXP2_TBITS = 10
EXP2_ERR = 1 << 23


def _ln2_scaled(shift: int) -> int:
    with mp.workprec(200):
        return int(mp.nint(mp.log(2) * (mpf(2) ** shift)))



EXP2_L31 = _ln2_scaled(31)          # round(ln2 * 2^31) < 2^31


@lru_cache(maxsize=1)
def exp2_table() -> np.ndarray:
    """T[j] = round(2^(j/1024) * 2^61), int64."""
    with mp.workprec(200):
        vals = [int(mp.nint(mp.power(2, mpf(j) / (1 << EXP2_TBITS)) * (mpf(2) ** 61)))
                for j in range(1 << EXP2_TBITS)]
    return np.array(vals, dtype=np.int64)


def exp2_fix_np(F: np.ndarray) -> np.ndarray:
    """Y ~ 2^(F / 2^50) * 2^61 for int64 0 <= F < 2^50: the kernels' integer arithmetic, op for op."""
    F = np.asarray(F, dtype=np.int64)
    T = exp2_table()
    j = F >> 40
    R = (F & ((1 << 40) - 1)) >> 8
    X = (R * EXP2_L31) >> 31
    X2 = (X * X) >> 42
    X3 = (X2 * X) >> 42
    P = X + (X2 >> 1) + X3 // 6
    Tj = T[j]
    return Tj + (((Tj >> 31) * P) >> 11)


def exp2_split_np(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, F) with t = n + F * 2^-50 for fp32 t, using integer ops on its bits (|t| < 2^-27 maps to
    (0, 0), whose weight 1 is RN24(2^t))."""
    b = np.asarray(t, dtype=np.float32).view(np.int32).astype(np.int64)
    ex = (b >> 23) & 0xFF
    m = (b & 0x7FFFFF) | np.where(ex != 0, 0x800000, 0)
    e = np.where(ex != 0, ex - 150, -149)
    ms = np.where(b < 0, -m, m)
    tiny = e < -50
    sr = np.clip(-e, 0, 50)
    n_neg = ms >> sr
    F = (ms - (n_neg << sr)) << np.clip(50 + e, 0, 50)
    n_pos = ms << np.clip(e, 0, 8)
    n = np.where(e >= 0, n_pos, np.where(tiny, 0, n_neg))
    F = np.where((e >= 0) | tiny, 0, F)
    return n, F


def exp2_weight_int_np(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer-only weights (M, E, risky) as the kernels compute them; risky entries are not
    certified (the kernels recompute those tiles another way)."""
    n, F = exp2_split_np(t)
    Y = exp2_fix_np(F)
    M = (Y + (1 << 37)) >> 38
    risky = np.abs((Y & ((1 << 38) - 1)) - (1 << 37)) <= EXP2_ERR
    carry = M == (1 << 24)
    return np.where(carry, M >> 1, M), n - 23 + carry, risky
