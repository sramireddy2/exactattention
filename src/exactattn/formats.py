"""Binary floating-point formats and exact rounding, built on Python integers.

Every value here is an exact dyadic `m * 2**e` (ints) or an exact rational `num / den`.
Nothing goes through hardware floating point except the final conversion to a numpy scalar,
and that conversion is exact because the value is already representable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ExactAttentionDomainError(ValueError):
    """An input is outside the spec's domain (NaN/inf, or an infinite intermediate)."""


@dataclass(frozen=True)
class Format:
    name: str
    p: int            # significand precision in bits, including the hidden bit
    qmin: int | None  # smallest quantum exponent (ulp of subnormals); None = unbounded
    emax: int | None  # exponent of the largest binade; None = unbounded

    @property
    def max_finite(self) -> tuple[int, int] | None:
        if self.emax is None:
            return None
        return ((1 << self.p) - 1, self.emax - self.p + 1)


BF16 = Format("bf16", 8, -133, 127)
FP32 = Format("fp32", 24, -149, 127)
FP24_UNBOUNDED = Format("fp24inf", 24, None, None)


def round_rational(num: int, den: int, fmt: Format) -> tuple[int, int]:
    """RN-even of num/den into fmt. Returns (m, e) with value m * 2**e and |m| < 2**p.

    Zero comes back as (0, 0). Raises ExactAttentionDomainError on overflow.
    """
    if den <= 0:
        if den == 0:
            raise ZeroDivisionError("round_rational: zero denominator")
        num, den = -num, -den
    if num == 0:
        return (0, 0)
    sign = -1 if num < 0 else 1
    a = -num if num < 0 else num

    # k = floor(log2(a/den)), so that 2**k <= a/den < 2**(k+1).
    k = a.bit_length() - den.bit_length()
    if k >= 0:
        if a < (den << k):
            k -= 1
    else:
        if (a << -k) < den:
            k -= 1

    qe = k - (fmt.p - 1)
    if fmt.qmin is not None and qe < fmt.qmin:
        qe = fmt.qmin

    # q = round(a / den / 2**qe)
    if qe >= 0:
        n, d = a, den << qe
    else:
        n, d = a << -qe, den
    q, r = divmod(n, d)
    twice_r = r << 1
    if twice_r > d or (twice_r == d and (q & 1)):
        q += 1

    if q == (1 << fmt.p):  # rounding carried into the next binade
        q >>= 1
        qe += 1

    if fmt.emax is not None:
        mf_m, mf_e = fmt.max_finite  # type: ignore[misc]
        # Compare q * 2**qe with mf_m * 2**mf_e.
        if _cmp_dyadic(q, qe, mf_m, mf_e) > 0:
            raise ExactAttentionDomainError(f"{fmt.name} overflow")
    return (sign * q, qe)


def round_dyadic(m: int, e: int, fmt: Format) -> tuple[int, int]:
    """RN-even of the exact dyadic m * 2**e into fmt."""
    if e >= 0:
        return round_rational(m << e, 1, fmt)
    return round_rational(m, 1 << -e, fmt)


def _cmp_dyadic(m1: int, e1: int, m2: int, e2: int) -> int:
    if e1 >= e2:
        a, b = m1 << (e1 - e2), m2
    else:
        a, b = m1, m2 << (e2 - e1)
    return (a > b) - (a < b)


# ---------------------------------------------------------------------------
# Decoding bit patterns into exact dyadics
# ---------------------------------------------------------------------------

def decode_bf16(bits: int) -> tuple[int, int]:
    """bf16 bit pattern -> (m, e), value m * 2**e. Raises on NaN/inf."""
    bits = int(bits) & 0xFFFF
    sign = -1 if bits & 0x8000 else 1
    exp = (bits >> 7) & 0xFF
    frac = bits & 0x7F
    if exp == 0xFF:
        raise ExactAttentionDomainError("bf16 NaN or infinity in input")
    if exp == 0:
        return (sign * frac, -133)
    return (sign * (frac | 0x80), exp - 127 - 7)


def decode_fp32(x) -> tuple[int, int]:
    """fp32 value (np.float32 or a Python float that is exactly fp32) -> (m, e)."""
    bits = int(np.asarray(x, dtype=np.float32).view(np.uint32))
    sign = -1 if bits & 0x80000000 else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ExactAttentionDomainError("fp32 NaN or infinity in input")
    if exp == 0:
        return (sign * frac, -149)
    return (sign * (frac | 0x800000), exp - 127 - 23)


# ---------------------------------------------------------------------------
# Encoding exact dyadics back into bit patterns
# ---------------------------------------------------------------------------

def dyadic_to_float(m: int, e: int) -> float:
    """Exact conversion of a representable fp32/bf16 dyadic to a Python float."""
    return math.ldexp(m, e)


def encode_fp32(m: int, e: int) -> np.float32:
    x = dyadic_to_float(m, e)
    f = np.float32(x)
    assert float(f) == x, "value is not fp32-representable"
    return f


def encode_bf16(m: int, e: int) -> np.uint16:
    f = encode_fp32(m, e)
    bits = int(np.asarray(f).view(np.uint32))
    assert bits & 0xFFFF == 0, "value is not bf16-representable"
    return np.uint16(bits >> 16)


# ---------------------------------------------------------------------------
# numpy helpers for producing test inputs
# ---------------------------------------------------------------------------

def float_to_bf16_bits(x) -> np.ndarray:
    """RN-even conversion of float32-castable values to bf16 bit patterns (uint16).

    The input goes through float32 first. That conversion is itself RN-even, so the result
    is double-rounded for float64 inputs. That's fine for generating test data, but it is not
    the spec's rounding, so never use it inside a kernel path.
    """
    f = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(f)):
        raise ExactAttentionDomainError("non-finite value")
    u = f.view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    out = rounded.astype(np.uint16)
    # Rounding can overflow to inf (exp field 0xFF). Reject that.
    if np.any((out & 0x7F80) == 0x7F80):
        raise ExactAttentionDomainError("bf16 overflow when generating input")
    return out


def bf16_bits_to_float32(bits) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return b.view(np.float32)
