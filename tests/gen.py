"""Random bf16 test-data generators, covering typical, wide-range and adversarial inputs."""

from __future__ import annotations

import numpy as np

from exactattn.formats import float_to_bf16_bits


def bf16(rng: np.random.Generator, shape, kind: str = "normal", scale: float = 1.0) -> np.ndarray:
    """Return a uint16 array of bf16 bit patterns."""
    if kind == "normal":
        x = rng.standard_normal(shape) * scale
    elif kind == "wide":
        # Log-uniform magnitudes over about 2**-40 .. 2**8, random signs: large exponent spreads.
        mag = np.exp2(rng.uniform(-40, 8, size=shape))
        x = mag * rng.choice([-1.0, 1.0], size=shape)
    elif kind == "outlier":
        # Mostly small values plus a few large outliers. This mimics attention sinks and
        # massive activations.
        x = rng.standard_normal(shape) * 0.05 * scale
        mask = rng.random(shape) < 0.02
        x = np.where(mask, rng.standard_normal(shape) * 30 * scale, x)
    elif kind == "dupes":
        # Few distinct values, to exercise ties and exact cancellation.
        vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]) * scale
        x = rng.choice(vals, size=shape)
    elif kind == "subnormal":
        x = rng.standard_normal(shape) * 2.0 ** -130
    else:
        raise ValueError(kind)
    return float_to_bf16_bits(x)


KINDS = ["normal", "wide", "outlier", "dupes", "subnormal"]
