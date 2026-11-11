import numpy as np
import pytest

from exactattn import (
    ExactAttentionDomainError,
    attention,
    attention_row,
    bf16_bits_to_float32,
    default_alpha,
    trace_row,
)
from gen import bf16


def _float64_attention(q, k, v, alpha):
    qf = bf16_bits_to_float32(q).astype(np.float64)
    kf = bf16_bits_to_float32(k).astype(np.float64)
    vf = bf16_bits_to_float32(v).astype(np.float64)
    s = kf @ qf * float(alpha)
    p = np.exp2(s - s.max())
    return (p @ vf) / p.sum()


def test_default_alpha():
    a = default_alpha(128)
    assert a == np.float32(np.log2(np.e) / np.sqrt(128))


def test_single_key_returns_value_exactly():
    rng = np.random.default_rng(0)
    q, k, v = bf16(rng, 64), bf16(rng, (1, 64)), bf16(rng, (1, 64))
    out = attention_row(q, k, v, default_alpha(64))
    assert np.array_equal(out, v[0])


def test_identical_keys_average_values():
    rng = np.random.default_rng(1)
    q = bf16(rng, 32)
    k = np.repeat(bf16(rng, (1, 32)), 4, axis=0)
    v = bf16(rng, (4, 32))
    out = attention_row(q, k, v, default_alpha(32), out="fp32")
    vf = bf16_bits_to_float32(v).astype(np.float64)
    # Average of 4 bf16 values: exact in float64, then rounded once to fp32.
    assert np.array_equal(out, np.float32(vf.mean(axis=0)))


def test_empty_set_is_zero():
    rng = np.random.default_rng(2)
    q = bf16(rng, 16)
    out = attention_row(q, np.zeros((0, 16), np.uint16), np.zeros((0, 8), np.uint16), default_alpha(16))
    assert np.array_equal(out, np.zeros(8, np.uint16))


def test_rejects_nan():
    rng = np.random.default_rng(3)
    q = bf16(rng, 16)
    k = bf16(rng, (3, 16))
    k[1, 2] = 0x7FC0
    with pytest.raises(ExactAttentionDomainError):
        attention_row(q, k, bf16(rng, (3, 16)), default_alpha(16))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", ["normal", "outlier"])
def test_accuracy_against_float64(seed, kind):
    rng = np.random.default_rng(seed)
    d, n = 64, 96
    q, k, v = bf16(rng, d, kind), bf16(rng, (n, d), kind), bf16(rng, (n, d))
    alpha = default_alpha(d)
    ref64 = _float64_attention(q, k, v, alpha)
    out32 = attention_row(q, k, v, alpha, out="fp32").astype(np.float64)
    scale = np.abs(bf16_bits_to_float32(v)).max()
    # Two 2**-24 roundings per weight plus float64 noise: far below bf16 resolution.
    assert np.max(np.abs(out32 - ref64)) <= scale * 2.0 ** -18
    out16 = bf16_bits_to_float32(attention_row(q, k, v, alpha)).astype(np.float64)
    # bf16 output is within one bf16 ulp of the float64 result.
    ulp = np.spacing(np.abs(ref64).astype(np.float32)).astype(np.float64) * 2.0 ** 16
    assert np.all(np.abs(out16 - ref64) <= ulp + scale * 2.0 ** -18)


def test_bf16_out_is_rounding_of_exact_ratio():
    """bf16 output must be a single rounding of N/Z, never bf16(fp32(...))."""
    rng = np.random.default_rng(7)
    q, k, v = bf16(rng, 16), bf16(rng, (40, 16)), bf16(rng, (40, 8))
    tr = trace_row(q, k, v, default_alpha(16), out="bf16")
    from fractions import Fraction

    Zm, Ze = tr.Z
    for c, (Nm, Ne) in enumerate(tr.N):
        ratio = Fraction(Nm) * Fraction(2) ** Ne / (Fraction(Zm) * Fraction(2) ** Ze)
        got = float(bf16_bits_to_float32(tr.out[c]))
        # got is a nearest bf16 value: no other bf16 value is strictly closer.
        lo = np.float32(got)
        up = bf16_bits_to_float32(np.uint16(int(tr.out[c]) + 1))
        dn = bf16_bits_to_float32(np.uint16(int(tr.out[c]) - 1)) if tr.out[c] & 0x7FFF else lo
        for nb in (float(up), float(dn)):
            assert abs(Fraction(got) - ratio) <= abs(Fraction(nb) - ratio)


def test_batched_mask_matches_rows():
    rng = np.random.default_rng(8)
    Q, K, V = bf16(rng, (3, 16)), bf16(rng, (10, 16)), bf16(rng, (10, 8))
    mask = rng.random((3, 10)) < 0.6
    out = attention(Q, K, V, default_alpha(16), mask=mask)
    for i in range(3):
        sel = np.flatnonzero(mask[i])
        assert np.array_equal(out[i], attention_row(Q[i], K[sel], V[sel], default_alpha(16)))
