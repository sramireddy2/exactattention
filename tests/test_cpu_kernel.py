"""C++ kernel vs the exact reference. Runs wherever the kernel builds (Linux/WSL, gcc)."""

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from exactattn import attention_row, default_alpha
from exactattn import cpu
from exactattn.formats import BF16, FP32, ExactAttentionDomainError, round_rational
from gen import KINDS, bf16

pytestmark = pytest.mark.skipif(not cpu.available(), reason="C++ kernel not built (needs gcc)")


@pytest.mark.parametrize("seed", range(4))
def test_round_ratio_matches_python(seed):
    rng = np.random.default_rng(seed)
    for _ in range(3000):
        abits, bbits = int(rng.integers(1, 128)), int(rng.integers(1, 128))
        a = int(rng.integers(0, 2**62)) << (abits - 62) if abits > 62 else int(rng.integers(1, 2**abits))
        b = int(rng.integers(1, 2**62)) << (bbits - 62) if bbits > 62 else int(rng.integers(1, 2**bbits))
        a = max(a, 1) | (int(rng.integers(0, 2)))
        E = int(rng.integers(-320, 200))
        neg = bool(rng.integers(0, 2))
        for fmt, is32 in ((FP32, True), (BF16, False)):
            got, ov = cpu.round_ratio(a, b, E, neg, is32)
            num = a << E if E >= 0 else a
            den = b if E >= 0 else b << -E
            try:
                m, e = round_rational(-num if neg else num, den, fmt)
                assert not ov
                expect = math.ldexp(m, e)
                assert got == expect or (got == 0 and expect == 0), (a, b, E, neg, is32)
            except ExactAttentionDomainError:
                assert ov, (a, b, E, neg, is32)


def _rows_case(rng, kind, vkind, rows, n, d, dv):
    return bf16(rng, (rows, d), kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), vkind)


@given(seed=st.integers(0, 2**32 - 1), kind=st.sampled_from(KINDS),
       vkind=st.sampled_from(["normal", "wide", "dupes"]), n=st.integers(1, 300),
       splits=st.integers(1, 9), tile=st.sampled_from([1, 7, 32, 64, 128]),
       acc=st.sampled_from([64, 128]), out=st.sampled_from(["bf16", "fp32"]))
def test_certified_rows_match_reference(seed, kind, vkind, n, splits, tile, acc, out):
    rng = np.random.default_rng(seed)
    d, dv = 64, 32
    Q, K, V = _rows_case(rng, kind, vkind, 3, n, d, dv)
    alpha = default_alpha(d)
    got, flags = cpu.attention_cpu(Q, K, V, alpha, out=out, acc_bits=acc, tile=tile, splits=splits,
                                   order_seed=seed, fallback=False)
    for r in range(3):
        if flags[r] == 0:
            assert np.array_equal(got[r], attention_row(Q[r], K, V, alpha, out))


def test_schedule_invariance_many_schedules():
    rng = np.random.default_rng(11)
    d, dv, n = 128, 64, 700
    Q, K, V = _rows_case(rng, "outlier", "normal", 8, n, d, dv)
    alpha = default_alpha(d)
    base, flags = cpu.attention_cpu(Q, K, V, alpha, fallback=False)
    assert not flags.any()
    for splits in (2, 3, 7, 16, 64):
        for tile in (1, 5, 64, 256):
            for acc in (64, 128):
                got, fl = cpu.attention_cpu(Q, K, V, alpha, acc_bits=acc, tile=tile, splits=splits,
                                            order_seed=splits * 1000 + tile, fallback=False)
                assert not fl.any() and np.array_equal(got, base)
    for r in range(2):
        assert np.array_equal(base[r], attention_row(Q[r], K, V, alpha))


def test_tree_mask_csr_equals_sequential():
    """Tree verification through a CSR mask: each node attends prefix + ancestors. Each row
    must equal sequential decoding of its root-to-node path."""
    rng = np.random.default_rng(5)
    d, dv, P, T = 64, 32, 300, 15
    parent = [-1] + [int(rng.integers(0, i)) for i in range(1, T)]
    Kp, Vp = bf16(rng, (P, d)), bf16(rng, (P, dv))
    Kt, Vt, Qt = bf16(rng, (T, d)), bf16(rng, (T, dv)), bf16(rng, (T, d))
    K, V = np.concatenate([Kp, Kt]), np.concatenate([Vp, Vt])
    rows = []
    for i in range(T):
        path, x = [], i
        while x != -1:
            path.append(P + x)
            x = parent[x]
        rows.append(np.concatenate([np.arange(P), np.array(path)]))  # ancestors in reverse
    row_ptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    key_idx = np.concatenate(rows).astype(np.int64)
    alpha = default_alpha(d)
    tree, flags = cpu.attention_cpu(Qt, K, V, alpha, row_ptr, key_idx, splits=4, tile=64, fallback=False)
    assert not flags.any()
    for i in range(T):
        seq_idx = np.sort(rows[i])  # sequential layout: prefix, then path in depth order
        seq, f2 = cpu.attention_cpu(Qt[i:i + 1], K[seq_idx], V[seq_idx], alpha, splits=1, tile=128,
                                    fallback=False)
        assert not f2.any()
        assert np.array_equal(tree[i], seq[0])
        assert np.array_equal(tree[i], attention_row(Qt[i], K[seq_idx], V[seq_idx], alpha))


def test_nonfinite_input_raises():
    rng = np.random.default_rng(1)
    Q, K, V = _rows_case(rng, "normal", "normal", 2, 10, 16, 8)
    K[3, 1] = 0x7FC0
    with pytest.raises(ExactAttentionDomainError):
        cpu.attention_cpu(Q, K, V, default_alpha(16))
