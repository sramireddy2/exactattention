import math

import numpy as np
import pytest

from exactattn.formats import (
    BF16,
    FP32,
    ExactAttentionDomainError,
    decode_bf16,
    decode_fp32,
    float_to_bf16_bits,
    round_dyadic,
    round_rational,
)


def _rn32_via_numpy(x: float):
    with np.errstate(over="ignore"):
        return np.float32(x)


@pytest.mark.parametrize("seed", range(5))
def test_round_rational_matches_numpy_fp32(seed):
    rng = np.random.default_rng(seed)
    # Doubles spanning fp32 normal, subnormal and overflow ranges, plus values near ties.
    mags = np.exp2(rng.uniform(-160, 130, size=4000))
    xs = mags * rng.choice([-1.0, 1.0], size=mags.shape)
    ties = np.float32(rng.standard_normal(1000)).astype(np.float64)
    ties = ties + np.spacing(np.float32(ties)).astype(np.float64) / 2  # exact midpoints
    for x in np.concatenate([xs, ties]):
        x = float(x)
        num, den = x.as_integer_ratio()
        expect = _rn32_via_numpy(x)
        if math.isinf(expect):
            with pytest.raises(ExactAttentionDomainError):
                round_rational(num, den, FP32)
            continue
        m, e = round_rational(num, den, FP32)
        assert math.ldexp(m, e) == float(expect), x


@pytest.mark.parametrize("seed", range(3))
def test_round_rational_bf16_matches_bit_trick(seed):
    rng = np.random.default_rng(seed)
    xs = np.float32(rng.standard_normal(5000) * np.exp2(rng.uniform(-135, 100, size=5000)))
    bits = float_to_bf16_bits(xs)  # single rounding from exact fp32 values
    for x, b in zip(xs.tolist(), bits.tolist()):
        num, den = float(x).as_integer_ratio()
        m, e = round_rational(num, den, BF16)
        if m == 0:
            assert b & 0x7FFF == 0
            continue
        assert math.ldexp(m, e) == math.ldexp(*decode_bf16(b)), x


def test_decode_roundtrip_fp32():
    rng = np.random.default_rng(0)
    xs = np.float32(rng.standard_normal(1000) * np.exp2(rng.uniform(-149, 127, size=1000)))
    for x in xs.tolist():
        m, e = decode_fp32(np.float32(x))
        assert math.ldexp(m, e) == x


def test_decode_rejects_nonfinite():
    with pytest.raises(ExactAttentionDomainError):
        decode_bf16(0x7F80)  # +inf
    with pytest.raises(ExactAttentionDomainError):
        decode_bf16(0x7FC0)  # NaN
    with pytest.raises(ExactAttentionDomainError):
        decode_fp32(np.float32("nan"))


def test_round_dyadic_simple():
    assert math.ldexp(*round_dyadic(3, -1, FP32)) == 1.5
    # 2**24 + 1 is not fp32-representable. It is a tie, and rounds to even (2**24).
    m, e = round_dyadic((1 << 24) + 1, 0, FP32)
    assert math.ldexp(m, e) == float(1 << 24)
    m, e = round_dyadic((1 << 24) + 3, 0, FP32)
    assert math.ldexp(m, e) == float((1 << 24) + 4)
