"""Fused Triton kernel vs the exact reference.

Without a GPU this runs under TRITON_INTERPRET=1, Triton's CPU interpreter. Set EA_TRITON_DEVICE
(for example "cuda" or "xpu") to run the same tests on hardware.
"""

import os

DEVICE = os.environ.get("EA_TRITON_DEVICE", "cpu")
if DEVICE == "cpu":
    os.environ["TRITON_INTERPRET"] = "1"

import zlib

import numpy as np  # noqa: E402
import pytest  # noqa: E402

triton = pytest.importorskip("triton")

from exactattn import attention_row, default_alpha  # noqa: E402
from exactattn.triton_kernels import attention_triton  # noqa: E402
from gen import KINDS, bf16  # noqa: E402


@pytest.mark.parametrize("use_tc", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("vkind", ["normal", "wide", "dupes"])
def test_matches_reference(kind, vkind, use_tc):
    rng = np.random.default_rng(zlib.crc32(f"{kind}/{vkind}".encode()))
    d, dv, R, n = 64, 32, 5, 90
    Q, K, V = bf16(rng, (R, d), kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), vkind)
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    got, flags = attention_triton(Q, K, V, alpha, kv_len, splits=3, block_m=16, block_n=16, block_dv=16,
                                  merge_order=[2, 0, 1], device=DEVICE, fallback=False, use_tc=use_tc)
    assert (flags == 0).sum() >= R - 1, flags
    for r in range(R):
        if flags[r] == 0:
            L = int(kv_len[r])
            assert np.array_equal(got[r], attention_row(Q[r], K[:L], V[:L], alpha)), r


def test_split_counts_and_merge_orders_identical():
    rng = np.random.default_rng(42)
    d, dv, R, n = 128, 32, 4, 200
    Q, K, V = bf16(rng, (R, d), "outlier"), bf16(rng, (n, d), "outlier"), bf16(rng, (n, dv))
    kv_len = np.array([200, 150, 77, 1])
    alpha = default_alpha(d)
    base, f0 = attention_triton(Q, K, V, alpha, kv_len, splits=1, block_m=4, block_n=32, device=DEVICE,
                                fallback=False)
    assert not f0.any()
    for splits, order in ((2, [1, 0]), (5, [4, 2, 0, 3, 1]), (7, None)):
        got, f = attention_triton(Q, K, V, alpha, kv_len, splits=splits, block_m=2, block_n=16,
                                  merge_order=order, device=DEVICE, fallback=False)
        assert not f.any() and np.array_equal(got, base), splits
    for r in range(R):
        L = int(kv_len[r])
        assert np.array_equal(base[r], attention_row(Q[r], K[:L], V[:L], alpha))


def test_fp32_output():
    rng = np.random.default_rng(9)
    d, dv, R, n = 32, 16, 3, 50
    Q, K, V = bf16(rng, (R, d)), bf16(rng, (n, d)), bf16(rng, (n, dv), "wide")
    kv_len = np.array([50, 20, 3])
    alpha = default_alpha(d)
    got, flags = attention_triton(Q, K, V, alpha, kv_len, out="fp32", splits=2, block_m=4, block_n=16,
                                  block_dv=16, device=DEVICE, fallback=False)
    assert not flags.any()
    for r in range(R):
        L = int(kv_len[r])
        ref = attention_row(Q[r], K[:L], V[:L], alpha, "fp32")
        assert np.array_equal(got[r].view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("kind", ["normal", "wide", "outlier"])
def test_tensor_core_fallback_path(kind):
    """1x1 slices certify almost nothing, so most tiles take the windowed recompute. The bits
    must still match the reference exactly."""
    rng = np.random.default_rng(zlib.crc32(kind.encode()))
    d, dv, R, n = 64, 16, 4, 70
    Q, K, V = bf16(rng, (R, d), kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv))
    kv_len = np.array([70, 33, 16, 1])
    alpha = default_alpha(d)
    got, flags = attention_triton(Q, K, V, alpha, kv_len, splits=2, block_m=16, block_n=16, block_dv=16,
                                  device=DEVICE, fallback=False, use_tc=True, sq=1, sk=1)
    assert not flags.any()
    for r in range(R):
        L = int(kv_len[r])
        assert np.array_equal(got[r], attention_row(Q[r], K[:L], V[:L], alpha))

def test_tensor_core_path_alone_is_exact():
    """With the windowed recompute disabled, every certified row comes purely from int8 tensor-core
    products. Inputs have a compact exponent range (as real q/K rows mostly do, per
    results/slice_study), so 3x4 slices hold them exactly. Every row must certify and match
    the reference."""
    from exactattn.formats import float_to_bf16_bits

    rng = np.random.default_rng(77)

    def compact(shape):
        return float_to_bf16_bits(rng.choice([-1.0, 1.0], size=shape) * np.exp2(rng.uniform(-6, 1, size=shape)))

    d, dv, R, n = 128, 16, 16, 120
    Q, K, V = compact((R, d)), compact((n, d)), bf16(rng, (n, dv))
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    got, flags = attention_triton(Q, K, V, alpha, kv_len, splits=2, block_m=16, block_n=16, block_dv=16,
                                  device=DEVICE, fallback=False, use_tc=True, sq=3, sk=4, tc_fallback=False)
    assert not flags.any(), flags
    for r in range(R):
        L = int(kv_len[r])
        assert np.array_equal(got[r], attention_row(Q[r], K[:L], V[:L], alpha))