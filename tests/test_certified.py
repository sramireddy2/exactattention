"""Invariance properties of the certified windowed algorithm.

Every test compares returned bits with the exact reference. The model's intermediate state
depends on the schedule; the output must not.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from exactattn import attention_row, default_alpha
from exactattn.certified import Config, Stats, split_attention_row
from gen import KINDS, bf16


def _random_partition(rng, n, max_splits):
    perm = rng.permutation(n)
    k = int(rng.integers(1, max_splits + 1))
    cuts = np.sort(rng.choice(np.arange(1, n), size=min(k - 1, max(n - 1, 0)), replace=False)) if n > 1 else []
    return [perm[a:b] for a, b in zip([0, *cuts], [*cuts, n])]


@given(seed=st.integers(0, 2**32 - 1), kind=st.sampled_from(KINDS),
       vkind=st.sampled_from(["normal", "wide", "dupes"]),
       n=st.integers(1, 160), out=st.sampled_from(["bf16", "fp32"]),
       acc_bits=st.sampled_from([48, 64, 128]))
def test_any_split_any_order_matches_reference(seed, kind, vkind, n, out, acc_bits):
    rng = np.random.default_rng(seed)
    d, dv = 32, 16
    q, k, v = bf16(rng, d, kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), vkind)
    alpha = default_alpha(d)
    ref = attention_row(q, k, v, alpha, out)
    splits = _random_partition(rng, n, 7)
    order = rng.permutation(len(splits))
    tile = int(rng.choice([1, 3, 16, 64]))
    got = split_attention_row(q, k, v, alpha, splits, Config(acc_bits=acc_bits), out, order, tile=tile)
    assert np.array_equal(got, ref)


def test_two_schedules_identical_and_counts_fallbacks():
    """Same row under two very different schedules: same bits. Reports fallback rates."""
    rng = np.random.default_rng(1234)
    d, dv, n = 64, 32, 512
    stats64, stats128 = Stats(), Stats()
    for trial in range(6):
        kind = KINDS[trial % len(KINDS)]
        q, k, v = bf16(rng, d, kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv))
        alpha = default_alpha(d)
        ref = attention_row(q, k, v, alpha)
        a = split_attention_row(q, k, v, alpha, [np.arange(n)], Config(64), stats=stats64, tile=128)
        chunks = [np.arange(i, min(i + 37, n)) for i in range(0, n, 37)][::-1]
        b = split_attention_row(q, k, v, alpha, chunks, Config(128), stats=stats128, tile=5)
        assert np.array_equal(a, ref) and np.array_equal(b, ref)
    print(f"\n64-bit:  {stats64}\n128-bit: {stats128}")


def _random_tree(rng, n_nodes):
    parent = [-1]
    for i in range(1, n_nodes):
        parent.append(int(rng.integers(0, i)))
    return parent


def _ancestors(parent, i):
    out = []
    while i != -1:
        out.append(i)
        i = parent[i]
    return out[::-1]


@pytest.mark.parametrize("seed", range(6))
def test_tree_verify_equals_sequential_decode(seed):
    """A draft node's tree-verify row equals the decode row at its position, bit for bit.

    Tree verify: prefix split across CTAs (FlashDecoding-style), then a frontier tile of
    the node's ancestors, gathered out of order from the tree buffer.
    Sequential decode: one contiguous position-ordered sequence, split differently.
    """
    rng = np.random.default_rng(seed)
    d, dv, P, T = 32, 16, 200, 9
    kind = KINDS[seed % len(KINDS)]
    alpha = default_alpha(d)
    k_pre, v_pre = bf16(rng, (P, d), kind), bf16(rng, (P, dv))
    k_tree, v_tree = bf16(rng, (T, d), kind), bf16(rng, (T, dv))
    q_tree = bf16(rng, (T, d), kind)
    parent = _random_tree(rng, T)
    K = np.concatenate([k_pre, k_tree])
    V = np.concatenate([v_pre, v_tree])
    for node in range(T):
        path = _ancestors(parent, node)  # root .. node
        # Tree-verify schedule: 4 prefix splits, then the frontier (ancestors, reversed).
        prefix_splits = np.array_split(np.arange(P), 4)
        frontier = np.array([P + i for i in path[::-1]])
        tv = split_attention_row(q_tree[node], K, V, alpha, [*prefix_splits, frontier],
                                 merge_order=[4, 2, 0, 3, 1], tile=32)
        # Sequential decode schedule: the accepted path laid out contiguously after the
        # prefix, processed in 64-key position-aligned chunks.
        seq_idx = np.concatenate([np.arange(P), np.array([P + i for i in path])])
        Kseq, Vseq = K[seq_idx], V[seq_idx]
        chunks = [np.arange(i, min(i + 64, len(seq_idx))) for i in range(0, len(seq_idx), 64)]
        sd = split_attention_row(q_tree[node], Kseq, Vseq, alpha, chunks, tile=64)
        assert np.array_equal(tv, sd)
        assert np.array_equal(tv, attention_row(q_tree[node], Kseq, Vseq, alpha))


@pytest.mark.parametrize("acc_bits", [40, 44, 48])
def test_certification_is_sound_under_tight_margins(acc_bits):
    """Soundness: with mid-width windows the error intervals are tight, so an unsound bound
    would certify a wrong answer. Count the rows the certified path returned without a
    fallback; each must match the reference exactly."""
    rng = np.random.default_rng(acc_bits)
    d, dv, n = 32, 24, 150
    certified = 0
    for trial in range(40):
        kind = KINDS[trial % len(KINDS)]
        q, k, v = bf16(rng, d, kind), bf16(rng, (n, d), kind), bf16(rng, (n, dv), ["normal", "wide"][trial % 2])
        alpha = default_alpha(d)
        stats = Stats()
        splits = np.array_split(rng.permutation(n), int(rng.integers(1, 6)))
        got = split_attention_row(q, k, v, alpha, splits, Config(acc_bits, acc_bits, key_log2=8), "bf16",
                                  rng.permutation(len(splits)), stats, tile=int(rng.integers(1, 40)))
        assert np.array_equal(got, attention_row(q, k, v, alpha))
        certified += stats.row_fallbacks == 0
    assert certified >= 10, f"only {certified} certified rows: test too weak"


@pytest.mark.parametrize("seed", range(4))
def test_narrow_accumulators_still_exact_via_fallback(seed):
    """Tiny windows force frequent fallbacks. Results must still be exact."""
    rng = np.random.default_rng(seed)
    d, dv, n = 32, 16, 120
    q, k, v = bf16(rng, d, "wide"), bf16(rng, (n, d), "wide"), bf16(rng, (n, dv), "wide")
    alpha = default_alpha(d)
    stats = Stats()
    cfg = Config(acc_bits=34, score_bits=34, key_log2=8)
    got = split_attention_row(q, k, v, alpha, np.array_split(rng.permutation(n), 5), cfg, stats=stats, tile=7)
    assert np.array_equal(got, attention_row(q, k, v, alpha))
    assert stats.score_fallbacks + stats.row_fallbacks > 0
