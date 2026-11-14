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
import triton.language as tl  # noqa: E402

from exactattn.triton_kernels import _bitlen, _exp2_weight, _f32_bits, attention_triton  # noqa: E402
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

@pytest.mark.parametrize("kind", ["normal", "outlier", "dupes"])
@pytest.mark.parametrize("vkind", ["normal", "wide", "dupes"])
def test_tensor_core_pv_matches_reference(kind, vkind):
    """P·V on int8 tensor cores (5x3 slices, per-element bounds). Every row the kernel certifies must
    equal the exact reference; rows it cannot certify are flagged, never wrong."""
    rng = np.random.default_rng(zlib.crc32(f"pv/{kind}/{vkind}".encode()))
    d, dv, R, n = 64, 32, 16, 100
    Q, K, V = bf16(rng, (R, d), kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), vkind)
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    got, flags = attention_triton(Q, K, V, alpha, kv_len, splits=2, block_m=16, block_n=16, block_dv=16,
                                  device=DEVICE, fallback=False, use_tc=True, use_tc_pv=True)
    # Synthetic outlier/dupes data is harder than real activations (wide weight spreads, exact
    # ties), so only require that the test is not vacuous. The real-data bar is in the next test.
    # "wide" V spans ~48 bits of exponent per row (real V rows: <= 25), far beyond 3 digits, so
    # most of those rows are legitimately uncertified: only correctness is required there.
    if vkind != "wide":
        assert (flags == 0).sum() >= R // 2, flags
    for r in np.flatnonzero(flags == 0):
        L = int(kv_len[r])
        assert np.array_equal(got[r], attention_row(Q[r], K[:L], V[:L], alpha)), r


def test_tensor_core_scores_and_pv_alone_are_exact():
    """Both int8 paths on, the windowed score fallback off, compact-range inputs: every row must
    certify, and match the reference bit for bit."""
    from exactattn.formats import float_to_bf16_bits

    rng = np.random.default_rng(91)

    def compact(shape):
        return float_to_bf16_bits(rng.choice([-1.0, 1.0], size=shape) * np.exp2(rng.uniform(-6, 1, size=shape)))

    d, dv, R, n = 128, 32, 16, 90
    Q, K, V = compact((R, d)), compact((n, d)), compact((n, dv))
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    got, flags = attention_triton(Q, K, V, alpha, kv_len, splits=3, block_m=16, block_n=16, block_dv=32,
                                  device=DEVICE, fallback=False, use_tc=True, tc_fallback=False, use_tc_pv=True)
    assert not flags.any(), flags
    for r in range(R):
        L = int(kv_len[r])
        assert np.array_equal(got[r], attention_row(Q[r], K[:L], V[:L], alpha))

def test_tensor_core_pv_on_real_activations():
    """Real Qwen3-0.6B rows (fixtures/): both int8 paths, tensor-core certification must hold on
    nearly every row, bit-exact. (DESIGN D10: 5x3 P·V slices left 0 of 3,072 real outputs uncertified.)"""
    import os

    z = np.load(os.path.join(os.path.dirname(__file__), "..", "fixtures", "qwen3_0.6b_heads.npz"))
    q, k, v, alpha = z["q_27_11"], z["k_27_11"], z["v_27_11"], np.float32(z["alpha_27_11"])
    rows = np.arange(300, 332)
    kv_len = rows + 1
    n = int(kv_len.max())
    got, flags = attention_triton(q[rows], k[:n], v[:n, :32], alpha, kv_len, splits=2, block_m=16, block_n=16,
                                  block_dv=32, device=DEVICE, fallback=False, use_tc=True, use_tc_pv=True)
    assert (flags == 0).sum() >= len(rows) - 1, flags
    for i in np.flatnonzero(flags == 0):
        r = int(rows[i])
        assert np.array_equal(got[i], attention_row(q[r], k[: r + 1], v[: r + 1, :32], alpha)), r

@pytest.mark.parametrize("kind", ["normal", "outlier"])
def test_pv_tier2_recovers_uncertified_pairs(kind):
    """Tier 2 (DESIGN D12): pairs that tensor-core P·V cannot certify are recomputed on the device
    with int64 P·V. "wide" V (48-bit exponent spread per row) defeats 3 V digits on most rows, so
    tier 1 alone flags many; with tier 2 the flags fall to what the int64 path alone leaves, and
    every certified row is still bit-exact."""
    rng = np.random.default_rng(zlib.crc32(f"tier2/{kind}".encode()))
    d, dv, R, n = 64, 64, 16, 100
    Q, K, V = bf16(rng, (R, d), kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), "wide")
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    kw = dict(splits=2, block_m=16, block_n=16, block_dv=32, device=DEVICE, fallback=False, use_tc=True)
    _, f1 = attention_triton(Q, K, V, alpha, kv_len, use_tc_pv=True, pv_tier2=False, **kw)
    got, f2 = attention_triton(Q, K, V, alpha, kv_len, use_tc_pv=True, **kw)
    _, f64 = attention_triton(Q, K, V, alpha, kv_len, **kw)
    assert f1.sum() >= R // 4, f1            # not vacuous: tier 1 really leaves work for tier 2
    assert not (f2 & ~f1).any()               # tier 2 never flags a row tier 1 certified
    assert f2.sum() <= f64.sum() < f1.sum(), (f1, f2, f64)
    for r in np.flatnonzero(f2 == 0):
        L = int(kv_len[r])
        assert np.array_equal(got[r], attention_row(Q[r], K[:L], V[:L], alpha)), r


@pytest.mark.parametrize("mode", ["int64", "tc", "tcpv"])
def test_segments_equal_separate_launches(mode):
    """Batched launch over G segments (sequence x KV head) gives, row for row, the bits and flags
    of one launch per segment, and certified rows match the reference."""
    import torch

    from exactattn.formats import float_to_bf16_bits
    from exactattn.triton_kernels import launch, tc_slices, tc_v_slices

    rng = np.random.default_rng(zlib.crc32(f"segments/{mode}".encode()))
    G, r_seg, n_seg, d, dv = 3, 5, 70, 64, 32

    def compact(shape):
        return float_to_bf16_bits(rng.choice([-1.0, 1.0], size=shape) * np.exp2(rng.uniform(-6, 1, size=shape)))

    Q, K, V = compact((G, r_seg, d)), compact((G, n_seg, d)), compact((G, n_seg, dv))
    kv_len = rng.integers(1, n_seg + 1, size=(G, r_seg)).astype(np.int32)
    alpha = default_alpha(d)
    dev = torch.device(DEVICE)

    def t16(x):
        return torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).to(dev)

    def kw(q, k, v):
        out = {}
        if mode != "int64":
            out.update(q_tc=tc_slices(q, 3, dev), k_tc=tc_slices(k, 4, dev))
        if mode == "tcpv":
            out.update(v_tc=tc_v_slices(v, 3, dev))
        return out

    maxlen = int(kv_len.max())
    OUT, FL, OF = launch(t16(Q), t16(K), t16(V), torch.from_numpy(kv_len).to(dev), maxlen, alpha, splits=2,
                         block_m=16, block_n=32, block_dv=32,
                         **kw(Q.reshape(-1, d), K.reshape(-1, d), V.reshape(-1, dv)))
    got = (OUT.cpu().numpy() & 0xFFFF).astype(np.uint16)
    flags = np.maximum(FL.amax(dim=(0, 2)).cpu().numpy(), OF.amax(dim=1).cpu().numpy()).reshape(G, r_seg)
    for g in range(G):
        o, fl, of = launch(t16(Q[g]), t16(K[g]), t16(V[g]), torch.from_numpy(kv_len[g]).to(dev), maxlen, alpha,
                           splits=2, block_m=16, block_n=32, block_dv=32, **kw(Q[g], K[g], V[g]))
        f1 = np.maximum(fl.amax(dim=(0, 2)).cpu().numpy(), of.amax(dim=1).cpu().numpy())
        assert np.array_equal(flags[g], f1), g
        assert np.array_equal(got[g], (o.cpu().numpy() & 0xFFFF).astype(np.uint16)), g
        for r in range(r_seg):
            if flags[g, r] == 0:
                L = int(kv_len[g, r])
                assert np.array_equal(got[g, r], attention_row(Q[g, r], K[g, :L], V[g, :L], alpha)), (g, r)
    assert (flags == 0).sum() >= G * r_seg - 1, flags


@pytest.mark.parametrize("kind", KINDS)
def test_device_q_slicing_matches_numpy(kind):
    """tc_slices_device (per-step q slicing on the GPU) equals the numpy slicer exactly; l1 is a
    float upper bound whose summation order may differ, so it is compared to a relative 1e-12."""
    import torch

    from exactattn.triton_kernels import tc_slices, tc_slices_device, tc_v_slices, tc_v_slices_device

    rng = np.random.default_rng(zlib.crc32(f"qslice/{kind}".encode()))
    x = bf16(rng, (37, 128), kind)
    x[3] = 0                                     # all-zero row
    for S in (3, 4):
        ref = tc_slices(x, S, torch.device("cpu"))
        got = tc_slices_device(torch.from_numpy(x.view(np.int16)).to(DEVICE), S)
        assert torch.equal(got[0].cpu(), ref[0]) and torch.equal(got[1].cpu(), ref[1])
        assert torch.equal(got[3].cpu(), ref[3])
        assert torch.allclose(got[2].cpu(), ref[2], rtol=1e-12, atol=0)
        vref = tc_v_slices(x, S, torch.device("cpu"))
        vgot = tc_v_slices_device(torch.from_numpy(x.view(np.int16)).to(DEVICE), S)
        for a_, b_ in zip(vgot, vref):
            assert torch.equal(a_.cpu(), b_)


def test_chunked_slicing_equals_one_piece():
    """Whole-cache slicing runs in row chunks to bound memory; the digits must not change."""
    import torch

    from exactattn.triton_kernels import _slice_rows_chunked, _slice_rows_device

    rng = np.random.default_rng(5)
    x = torch.from_numpy(bf16(rng, (53, 64), "outlier").view(np.int16)).to(DEVICE)
    for a_, b_ in zip(_slice_rows_chunked(x, 4, chunk_elems=7 * 64), _slice_rows_device(x, 4)):
        assert torch.equal(a_, b_)


@pytest.mark.parametrize("mode", ["int64", "tc", "tcpv"])
def test_nan_inf_inputs_are_flagged(mode):
    """A NaN key or an inf value must flag exactly the rows that can see it, in every mode. The
    tensor-core paths take this from flags made at cache write, not from decoding K and V."""
    rng = np.random.default_rng(17)
    d, dv, R, n = 64, 32, 16, 80
    Q, K, V = bf16(rng, (R, d), "normal"), bf16(rng, (n, d), "normal"), bf16(rng, (n, dv), "normal")
    kv_len = np.arange(R) * 5 + 3                              # 3 .. 78
    alpha = default_alpha(d)
    kw = dict(splits=2, block_m=16, block_n=16, block_dv=16, device=DEVICE, fallback=False,
              use_tc=mode != "int64", use_tc_pv=mode == "tcpv")
    _, clean = attention_triton(Q, K, V, alpha, kv_len, **kw)
    Kb, Vb = K.copy(), V.copy()
    Kb[40, 7] = 0x7FC0                                         # NaN in key 40
    Vb[60, 3] = 0x7F80                                         # +inf in value 60
    _, f = attention_triton(Q, Kb, Vb, alpha, kv_len, **kw)
    sees = kv_len > 40
    assert f[sees].all(), (f, kv_len)
    assert np.array_equal(f[~sees], clean[~sees])


@pytest.mark.parametrize("mode", ["int64", "tc", "tcpv"])
def test_row_block_size_does_not_change_bits(mode):
    """Small row blocks (decode: 2 query heads per KV head) pack digit slices into the 16 MMA rows
    instead of padding (DESIGN D14). The row block is a tiling choice, so every size must give the
    same bits and the same flags."""
    rng = np.random.default_rng(zlib.crc32(f"blockm/{mode}".encode()))
    d, dv, R, n = 64, 64, 16, 90
    Q, K, V = bf16(rng, (R, d), "outlier"), bf16(rng, (n, d), "normal"), bf16(rng, (n, dv), "wide")
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    kw = dict(splits=2, block_n=16, block_dv=32, device=DEVICE, fallback=False,
              use_tc=mode != "int64", use_tc_pv=mode == "tcpv")
    ref, f_ref = attention_triton(Q, K, V, alpha, kv_len, block_m=16, **kw)
    for bm in (1, 2, 4, 8):
        got, f = attention_triton(Q, K, V, alpha, kv_len, block_m=bm, **kw)
        assert np.array_equal(f, f_ref), (bm, f, f_ref)
        assert np.array_equal(got[f == 0], ref[f == 0]), bm
    for r in np.flatnonzero(f_ref == 0):
        L = int(kv_len[r])
        assert np.array_equal(ref[r], attention_row(Q[r], K[:L], V[:L], alpha)), r


@triton.jit
def _helper_probe(T, X, Q, QE, OM, OE, OR, OB, OF, NB: tl.constexpr):
    i = tl.arange(0, NB)
    m, e, r = _exp2_weight(tl.load(T + i), X)
    tl.store(OM + i, m)
    tl.store(OE + i, e)
    tl.store(OR + i, r.to(tl.int32))
    q = tl.load(Q + i)
    tl.store(OB + i, _bitlen(q))
    qe = tl.load(QE + i)
    tl.store(OF + i, _f32_bits(q & 0xFFFFFF, qe, (q & 1) != 0, qe > 104))


def test_kernel_integer_helpers_match_numpy():
    """_exp2_weight equals exp2.exp2_weight_int_np (M, E and risky, bit for bit), _bitlen is exact,
    and _f32_bits builds the same fp32 as numpy."""
    import torch

    from exactattn.exp2 import exp2_weight_int_np
    from exactattn.triton_kernels import _exp2_table_dev

    rng = np.random.default_rng(11)
    NB = 4096
    ts = np.float32(np.concatenate([rng.uniform(-60, 60, NB - 96), rng.uniform(-1e-6, 1e-6, 64),
                                    np.float32(2.0) ** -np.arange(20, 52)]))
    pw = np.concatenate([np.int64(1) << np.arange(63), (np.int64(1) << np.arange(1, 63)) - 1])
    qs = np.concatenate([pw, rng.integers(0, 1 << 62, NB - len(pw) - 1), [0]]).astype(np.int64)
    q24 = qs & 0xFFFFFF
    qes = np.where(q24 >= 1 << 23, rng.integers(-149, 106, NB), -149).astype(np.int64)
    dev = torch.device(DEVICE)
    outs = [torch.empty(NB, dtype=dt, device=dev) for dt in (torch.int64, torch.int64, torch.int32, torch.int64, torch.int32)]
    _helper_probe[(1,)](torch.from_numpy(ts).to(dev), _exp2_table_dev(dev), torch.from_numpy(qs).to(dev),
                torch.from_numpy(qes).to(dev), *outs, NB=NB)
    M, E, R, B, Fb = [o.cpu().numpy() for o in outs]
    m_ref, e_ref, r_ref = exp2_weight_int_np(ts)
    assert np.array_equal(M, m_ref) and np.array_equal(E, e_ref) and np.array_equal(R != 0, r_ref)
    assert np.array_equal(B, np.array([int(q).bit_length() for q in qs]))
    with np.errstate(over="ignore"):
        ref = (q24.astype(np.float64) * np.exp2(qes.astype(np.float64))).astype(np.float32)
    ref = np.where((qs & 1) != 0, -ref, ref)
    assert np.array_equal(Fb, ref.view(np.int32))


@pytest.mark.parametrize("mode", ["int64", "tc", "tcpv"])
def test_integer_exp2_equals_float64_exp2(mode):
    """Integer exp2 (default) and float64 exp2 give the same bits; integer exp2 certifies at
    least as many rows (its uncertain weights fall back to float64)."""
    rng = np.random.default_rng(zlib.crc32(f"exp2/{mode}".encode()))
    d, dv, R, n = 64, 32, 16, 120
    Q, K, V = bf16(rng, (R, d), "outlier"), bf16(rng, (n, d), "normal"), bf16(rng, (n, dv), "normal")
    kv_len = rng.integers(1, n + 1, size=R)
    alpha = default_alpha(d)
    kw = dict(splits=2, block_n=16, block_dv=16, device=DEVICE, fallback=False,
              use_tc=mode != "int64", use_tc_pv=mode == "tcpv")
    a, fa = attention_triton(Q, K, V, alpha, kv_len, exp2="int", **kw)
    b, fb = attention_triton(Q, K, V, alpha, kv_len, exp2="f64", **kw)
    assert not (fa & ~fb).any()
    ok = (fa == 0) & (fb == 0)
    assert np.array_equal(a[ok], b[ok])
    for r in np.flatnonzero(fa == 0):
        L = int(kv_len[r])
        assert np.array_equal(a[r], attention_row(Q[r], K[:L], V[:L], alpha)), r
