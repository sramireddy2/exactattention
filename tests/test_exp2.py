import math

import numpy as np
import pytest
from mpmath import mp, mpf, power

from exactattn.exp2 import exp2_weight_fast_np, exp2_weight_oracle


def _rn24_highprec(t: float) -> tuple[int, int]:
    """Independent check: 400-bit evaluation, then RN to 24 bits."""
    with mp.workprec(400):
        y = power(2, mpf(t))
        n = math.floor(t)
        scaled = y / mpf(2) ** n * 2 ** 23  # in [2**23, 2**24]
        q = int(mp.floor(scaled + mpf("0.5")))
    e = n - 23
    if q == 1 << 24:
        q >>= 1
        e += 1
    return q, e


def test_integer_t_is_exact_power_of_two():
    for n in [-200, -1, 0, 1, 5, 127, 300]:
        m, e = exp2_weight_oracle(float(n))
        assert m == 1 << 23 and e == n - 23


@pytest.mark.parametrize("seed", range(3))
def test_oracle_matches_highprec(seed):
    rng = np.random.default_rng(seed)
    ts = np.float32(rng.uniform(-80, 80, size=400))
    for t in ts.tolist():
        assert exp2_weight_oracle(t) == _rn24_highprec(t), t


def test_oracle_range_of_significand():
    for t in np.float32(np.linspace(-3, 3, 997)).tolist():
        m, e = exp2_weight_oracle(t)
        assert (1 << 23) <= m < (1 << 24)
        # Within half an ulp of the true value.
        assert abs(math.ldexp(m, e) - 2.0 ** t) <= math.ldexp(1, e) * 0.5000001


@pytest.mark.parametrize("seed", range(3))
def test_fast_matches_oracle(seed):
    rng = np.random.default_rng(100 + seed)
    parts = [
        rng.uniform(-60, 60, size=20000),
        rng.uniform(-1e-6, 1e-6, size=2000),            # f near 0, near 1
        np.arange(-40, 40) + rng.uniform(-1e-7, 1e-7, size=80),
        rng.uniform(-1e4, 1e4, size=2000),             # large |t|, few fraction bits
    ]
    ts = np.float32(np.concatenate(parts))
    qm, qe = exp2_weight_fast_np(ts)
    for t, m, e in zip(ts.tolist(), qm.tolist(), qe.tolist()):
        assert (m, e) == exp2_weight_oracle(t), t


# ---- integer exp2 used by the GPU kernels (DESIGN D15) ----

def test_exp2_table_is_correctly_rounded():
    """T[j] = round(2^(61 + j/1024)), checked with exact integers: (2T -/+ 1)^1024 brackets
    2^(1024 * 62 + j), i.e. T - 1/2 < 2^(61 + j/1024) < T + 1/2."""
    from exactattn.exp2 import EXP2_TBITS, exp2_table

    N = 1 << EXP2_TBITS
    for j, T in enumerate(exp2_table().tolist()):
        target = 1 << (N * 62 + j)
        assert (2 * T - 1) ** N < target < (2 * T + 1) ** N, j


def test_ln2_constant():
    from exactattn.exp2 import EXP2_L31

    with mp.workprec(200):
        assert abs(mpf(EXP2_L31) - mp.log(2) * 2 ** 31) <= 0.5
    assert EXP2_L31 < 2 ** 31


def test_exp2_fix_error_bound():
    """|Y - 2^f * 2^61| stays inside the analysed bound 13 * 2^19 (<= EXP2_ERR) on random f, on
    every table boundary and at the ends of [0, 1)."""
    from exactattn.exp2 import EXP2_ERR, exp2_fix_np

    rng = np.random.default_rng(7)
    edges = np.array([(j << 40) + d for j in range(1024) for d in (0, 1, (1 << 40) - 1)], dtype=np.int64)
    F = np.concatenate([rng.integers(0, 1 << 50, size=6000, dtype=np.int64), edges,
                        np.array([0, 1, (1 << 50) - 1], dtype=np.int64)])
    Y = exp2_fix_np(F)
    worst = 0
    with mp.workprec(160):
        for f_, y in zip(F.tolist(), Y.tolist()):
            err = abs(mpf(y) - mp.power(2, mpf(f_) / 2 ** 50 + 61))
            worst = max(worst, float(err))
    assert worst <= 13 * 2 ** 19 <= EXP2_ERR, worst
    print(f"worst |Y - 2^f 2^61| = {worst:.0f} = 2^{math.log2(worst):.2f} units of 2^-61")


@pytest.mark.parametrize("seed", range(2))
def test_int_weights_match_oracle(seed):
    """Every weight the integer path certifies equals the oracle; it certifies nearly all."""
    from exactattn.exp2 import exp2_weight_int_np

    rng = np.random.default_rng(300 + seed)
    parts = [
        rng.uniform(-60, 60, size=20000),
        rng.uniform(-1e-6, 1e-6, size=2000),
        np.float32(2.0) ** rng.uniform(-40, -20, size=1000) * rng.choice([-1, 1], size=1000),
        np.arange(-40, 40) + rng.uniform(-1e-7, 1e-7, size=80),
        rng.uniform(-1e4, 1e4, size=2000),
        np.array([0.0, -0.0, 1.0, -1.0, 2.0 ** -27, -(2.0 ** -27), 2.0 ** -28, np.nextafter(np.float32(1), 0)]),
    ]
    ts = np.float32(np.concatenate(parts))
    M, E, risky = exp2_weight_int_np(ts)
    assert risky.mean() < 1e-3, risky.mean()
    for t, m, e, r in zip(ts.tolist(), M.tolist(), E.tolist(), risky.tolist()):
        if not r:
            assert (m, e) == exp2_weight_oracle(t), t
