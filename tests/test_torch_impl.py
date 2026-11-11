"""Tensor-op implementation on every available torch device vs the exact reference."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from exactattn import attention_row, default_alpha  # noqa: E402
from exactattn.torch_impl import TorchStats, attention_torch  # noqa: E402
from gen import KINDS, bf16  # noqa: E402

DEVICES = ["cpu"]
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICES.append("xpu")
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("vkind", ["normal", "wide", "dupes"])
def test_matches_reference(device, kind, vkind):
    rng = np.random.default_rng(hash((kind, vkind)) % 2**32)
    d, dv, R, n = 64, 32, 6, 150
    Q, K, V = bf16(rng, (R, d), kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), vkind)
    mask = rng.random((R, n)) < 0.8
    mask[:, 0] = True
    alpha = default_alpha(d)
    st = TorchStats()
    got, flags = attention_torch(Q, K, V, alpha, mask, device=device, tile=32, row_chunk=4,
                                 tile_order_seed=7, stats=st, fallback=False)
    assert (flags == 0).sum() >= R - 1, f"too many flagged rows: {flags}"
    for r in range(R):
        if flags[r] == 0:
            sel = np.flatnonzero(mask[r])
            assert np.array_equal(got[r], attention_row(Q[r], K[sel], V[sel], alpha)), (device, r)


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_output_and_tile_order(device):
    rng = np.random.default_rng(3)
    d, dv, R, n = 128, 64, 4, 300
    Q, K, V = bf16(rng, (R, d), "outlier"), bf16(rng, (n, d), "outlier"), bf16(rng, (n, dv))
    mask = np.tril(np.ones((R, n), bool), k=n - R)
    alpha = default_alpha(d)
    a, fa = attention_torch(Q, K, V, alpha, mask, device=device, out="fp32", tile=64, fallback=False)
    b, fb = attention_torch(Q, K, V, alpha, mask, device=device, out="fp32", tile=17, tile_order_seed=11,
                            fallback=False)
    assert not fa.any() and not fb.any()
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    for r in range(R):
        sel = np.flatnonzero(mask[r])
        ref = attention_row(Q[r], K[sel], V[sel], alpha, "fp32")
        assert np.array_equal(a[r].view(np.uint32), ref.view(np.uint32))
