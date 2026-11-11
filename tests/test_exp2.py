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
