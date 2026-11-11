"""Exact reference implementation of ExactAttention v0 (docs/SPEC.md).

This is the ground truth every other implementation is tested against. It aims to be obviously
correct, not fast: every sum is an exact Python integer, every rounding goes through
`formats.round_rational`, and every weight comes from the rigorous interval oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exp2 import exp2_weight_oracle
from .formats import (
    BF16,
    FP32,
    ExactAttentionDomainError,
    decode_fp32,
    dyadic_to_float,
    encode_bf16,
    encode_fp32,
    round_dyadic,
    round_rational,
)

# The reference shifts Python ints by exponent differences. Past this spread it refuses
# rather than allocate gigantic integers. This is an implementation limit, not part of the
# spec, and real inputs come nowhere near it.
MAX_EXPONENT_SPREAD = 1 << 20


def decode_bf16_array(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised bf16 decode -> (m, e) int64 arrays, value m * 2**e."""
    b = np.asarray(bits, dtype=np.uint16).astype(np.int64)
    exp = (b >> 7) & 0xFF
    if np.any(exp == 0xFF):
        raise ExactAttentionDomainError("bf16 NaN or infinity in input")
    frac = b & 0x7F
    sign = np.where(b & 0x8000, -1, 1)
    normal = exp != 0
    m = sign * np.where(normal, frac | 0x80, frac)
    e = np.where(normal, exp - 134, -133)
    return m, e


def exact_dot(qm, qe, km, ke) -> tuple[int, int]:
    """Exact sum_i q_i * k_i as (M, E). Inputs are int64 arrays of decoded bf16."""
    pm = (qm * km)            # |pm| < 2**16, exact in int64
    pe = (qe + ke)
    nz = pm != 0
    if not np.any(nz):
        return (0, 0)
    pm, pe = pm[nz], pe[nz]
    emin = int(pe.min())
    if int(pe.max()) - emin > MAX_EXPONENT_SPREAD:
        raise ExactAttentionDomainError("score exponent spread exceeds reference limit")
    total = 0
    for m, e in zip(pm.tolist(), pe.tolist()):
        total += m << (e - emin)
    return (total, emin)


@dataclass
class RowTrace:
    s: list[tuple[int, int]]        # exact scores (M, E)
    t: list[float]                  # fp32 logits, as exactly-fp32 Python floats
    w: list[tuple[int, int]]        # weights (M, E), M in [2**23, 2**24)
    Z: tuple[int, int]              # exact denominator (M, E)
    N: list[tuple[int, int]]        # exact numerators per channel (M, E)
    out: np.ndarray                 # output bits (uint16) or float32


def _logit(alpha_m: int, alpha_e: int, s: tuple[int, int]) -> float:
    sm, se = s
    if sm == 0 or alpha_m == 0:
        return 0.0
    tm, te = round_dyadic(alpha_m * sm, alpha_e + se, FP32)  # raises on overflow
    return dyadic_to_float(tm, te)


def trace_row(q_bits, k_bits, v_bits, alpha, out: str = "bf16") -> RowTrace:
    """Compute one query row exactly, returning every intermediate."""
    q_bits = np.asarray(q_bits, dtype=np.uint16)
    k_bits = np.asarray(k_bits, dtype=np.uint16).reshape(-1, q_bits.shape[-1])
    v_bits = np.asarray(v_bits, dtype=np.uint16)
    n = k_bits.shape[0]
    dv = v_bits.shape[-1] if v_bits.ndim == 2 else 0
    v_bits = v_bits.reshape(n, dv)
    if out not in ("bf16", "fp32"):
        raise ValueError("out must be 'bf16' or 'fp32'")
    fmt = BF16 if out == "bf16" else FP32
    alpha_m, alpha_e = decode_fp32(np.float32(alpha))

    qm, qe = decode_bf16_array(q_bits)
    km, ke = decode_bf16_array(k_bits)
    vm, ve = decode_bf16_array(v_bits)

    s = [exact_dot(qm, qe, km[j], ke[j]) for j in range(n)]
    t = [_logit(alpha_m, alpha_e, sj) for sj in s]
    w = [exp2_weight_oracle(tj) for tj in t]

    if n == 0:
        zero = np.zeros(dv, dtype=np.uint16 if out == "bf16" else np.float32)
        return RowTrace(s, t, w, (0, 0), [(0, 0)] * dv, zero)

    wE = [e for _, e in w]
    w_emin = min(wE)
    if max(wE) - w_emin > MAX_EXPONENT_SPREAD:
        raise ExactAttentionDomainError("weight exponent spread exceeds reference limit")
    Zint = 0
    for (m, e) in w:
        Zint += m << (e - w_emin)

    vm_l = vm.tolist()
    ve_l = ve.tolist()
    N: list[tuple[int, int]] = []
    outs = []
    for c in range(dv):
        terms = [(w[j][0] * vm_l[j][c], w[j][1] + ve_l[j][c]) for j in range(n)]
        terms = [(m, e) for m, e in terms if m != 0]
        if not terms:
            N.append((0, 0))
            outs.append(np.uint16(0) if out == "bf16" else np.float32(0.0))
            continue
        base = min(e for _, e in terms)
        Nint = 0
        for m, e in terms:
            Nint += m << (e - base)
        N.append((Nint, base))
        # N/Z = Nint * 2**base / (Zint * 2**w_emin)
        sh = base - w_emin
        if sh >= 0:
            num, den = Nint << sh, Zint
        else:
            num, den = Nint, Zint << -sh
        om, oe = round_rational(num, den, fmt)
        if om == 0:
            outs.append(np.uint16(0) if out == "bf16" else np.float32(0.0))
        else:
            outs.append(encode_bf16(om, oe) if out == "bf16" else encode_fp32(om, oe))
    out_arr = np.array(outs, dtype=np.uint16 if out == "bf16" else np.float32)
    return RowTrace(s, t, w, (Zint, w_emin), N, out_arr)


def attention_row(q_bits, k_bits, v_bits, alpha, out: str = "bf16") -> np.ndarray:
    return trace_row(q_bits, k_bits, v_bits, alpha, out).out


def attention(q_bits, k_bits, v_bits, alpha, mask=None, out: str = "bf16") -> np.ndarray:
    """Batched reference. q: (r, d), k: (n, d), v: (n, dv), mask: (r, n) bool or None."""
    q_bits = np.asarray(q_bits, dtype=np.uint16)
    k_bits = np.asarray(k_bits, dtype=np.uint16)
    v_bits = np.asarray(v_bits, dtype=np.uint16)
    r = q_bits.shape[0]
    dv = v_bits.shape[1]
    res = np.zeros((r, dv), dtype=np.uint16 if out == "bf16" else np.float32)
    for i in range(r):
        sel = np.arange(k_bits.shape[0]) if mask is None else np.flatnonzero(mask[i])
        res[i] = attention_row(q_bits[i], k_bits[sel], v_bits[sel], alpha, out)
    return res


def default_alpha(d: int) -> np.float32:
    """alpha = RN_fp32(log2(e) / sqrt(d)), computed exactly with mpmath."""
    from mpmath import log, mp, mpf, sqrt

    with mp.workprec(200):
        val = 1 / (log(2) * sqrt(mpf(d)))  # log2(e) = 1/ln 2
        # Round to fp32 exactly: convert to a rational m/2**k at high precision, then RN.
        man, exp = mp.mpf(val).man_exp
    m, e = round_dyadic(int(man), int(exp), FP32)
    return encode_fp32(m, e)
