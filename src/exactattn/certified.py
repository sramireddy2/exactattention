"""Certified windowed ExactAttention: a bit-accurate model of the GPU algorithm.

The kernels do not use unbounded integers. Each partial result (one tile, one KV split, one
shared prefix) is a `Partial`:

  * an integer reference exponent `R`: the leading-bit exponent of the largest weight seen;
  * fixed-point accumulators for Z and N_c, `W` bits wide, with a fixed number of fraction
    bits below the reference;
  * an error count per accumulator. The true value lies in [acc, acc + err] units, where one
    unit is 2**(ref - F). Every truncation and every rescale adds to the count rigorously.

Partials merge in any order and any tree shape. Merging aligns to the larger reference
exponent, which is an exact power-of-two rescale apart from the counted truncation (SPEC Â§5.2).
`finalize` rounds both ends of the error interval. If they agree, the result equals the spec
value by definition (SPEC Â§7). If they don't, the row is recomputed exactly and counted as a
fallback.

The intermediate state depends on the partition. The *returned bits* never do, and the tests
check exactly that.

Bit widths are asserted throughout, so the model cannot quietly use more precision than a
64- or 128-bit hardware accumulator has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exp2 import exp2_weight_fast_np, exp2_weight_oracle
from .formats import BF16, FP32, decode_fp32, dyadic_to_float, encode_bf16, encode_fp32, round_dyadic, round_rational
from .reference import decode_bf16_array, exact_dot



@dataclass(frozen=True)
class Config:
    acc_bits: int = 64      # width W of the Z / N accumulators (signed)
    score_bits: int = 64    # width of the windowed score accumulator (signed)
    head_dim_log2: int = 8  # score headroom: up to 2**8 = 256 products per dot
    key_log2: int = 20      # accumulator headroom: up to 2**key_log2 keys per row

    @property
    def fz(self) -> int:   # fraction bits for Z: each weight is < 2**(F+1) units
        return self.acc_bits - 2 - self.key_log2

    @property
    def fn(self) -> int:   # fraction bits for N: each |w*v| is < 2**(F+1) units
        return self.acc_bits - 2 - self.key_log2

    @property
    def fs(self) -> int:   # fraction bits for scores: each |product| is < 2**F units
        return self.score_bits - 1 - self.head_dim_log2


@dataclass
class Stats:
    score_fallbacks: int = 0
    weight_oracle_calls: int = 0
    row_fallbacks: int = 0
    rows: int = 0
    scores: int = 0


def _check_width(value: int, err: int, bits: int):
    assert abs(value) + err < (1 << (bits - 1)), "accumulator overflow: headroom violated"


def _floor_shift(m: int, sh: int) -> tuple[int, bool]:
    """floor(m * 2**sh) and whether bits were lost."""
    if sh >= 0:
        return m << sh, False
    s = -sh
    q = m >> s
    return q, (q << s) != m


def _rescale(acc: int, err: int, s: int) -> tuple[int, int]:
    """Re-express [acc, acc+err] in units 2**s times larger (s >= 0), rigorously."""
    if s == 0:
        return acc, err
    q = acc >> s
    lost = (q << s) != acc
    e = -(-err >> s)  # ceil(err / 2**s)
    return q, e + (1 if lost else 0)


# ---------------------------------------------------------------------------
# Scores and logits (SPEC steps 1 and 2)
# ---------------------------------------------------------------------------

def windowed_logits(q_bits, k_bits, alpha, cfg: Config, stats: Stats) -> list[float]:
    """Certified t_j = RN32(alpha * s_j) from a windowed score accumulator."""
    qm, qe = decode_bf16_array(q_bits)
    km, ke = decode_bf16_array(k_bits)
    am, ae = decode_fp32(np.float32(alpha))
    d = qm.shape[0]
    assert d <= (1 << cfg.head_dim_log2)
    P = km * qm[None, :]                 # |P| < 2**16
    E = ke + qe[None, :]
    nz = P != 0
    # Leading-bit exponent of each product: E + bitlen(|P|) - 1.
    absP = np.abs(P)
    bl = np.where(nz, np.floor(np.log2(np.maximum(absP, 1))).astype(np.int64) + 1, 0)
    lead = np.where(nz, E + bl - 1, np.iinfo(np.int64).min // 2)
    Lmax = lead.max(axis=1)              # per key
    out: list[float] = []
    for j in range(k_bits.shape[0]):
        stats.scores += 1
        if not nz[j].any() or am == 0:
            out.append(0.0)
            continue
        # unit u = 2**(Lmax + 1 - fs); each |term| < 2**fs units.
        base = int(Lmax[j]) + 1 - cfg.fs
        acc = 0
        err = 0
        for p, e in zip(P[j].tolist(), E[j].tolist()):
            if p == 0:
                continue
            v, lost = _floor_shift(p, e - base)
            acc += v
            err += 1 if lost else 0
        _check_width(acc, err, cfg.score_bits)
        # s in [acc, acc+err] * 2**base, so t = RN32(alpha*s) at both ends.
        try:
            lo = round_dyadic(am * acc, ae + base, FP32)
            hi = round_dyadic(am * (acc + err), ae + base, FP32)
        except Exception:
            lo, hi = None, None
        if lo is not None and dyadic_to_float(*lo) == dyadic_to_float(*hi):
            out.append(dyadic_to_float(*lo))
        else:
            stats.score_fallbacks += 1
            sm, se = exact_dot(qm, qe, km[j], ke[j])
            tm, te = round_dyadic(am * sm, ae + se, FP32)
            out.append(dyadic_to_float(tm, te))
    return out


# ---------------------------------------------------------------------------
# Partial states (SPEC step 4, windowed)
# ---------------------------------------------------------------------------

@dataclass
class Partial:
    dv: int
    count: int = 0
    R: int | None = None                      # leading-bit exponent of the max weight
    Z: int = 0
    Zerr: int = 0
    NR: list[int | None] = field(default_factory=list)  # per-channel lead exponent of max |w*v|
    N: list[int] = field(default_factory=list)
    Nerr: list[int] = field(default_factory=list)
    # Weights of every key, kept only so the fallback path can recompute exactly. A GPU
    # implementation recomputes from q, K, V instead.
    keys: list = field(default_factory=list)

    def __post_init__(self):
        if not self.NR:
            self.NR = [None] * self.dv
            self.N = [0] * self.dv
            self.Nerr = [0] * self.dv


def tile_partial(weights: list[tuple[int, int]], vm: np.ndarray, ve: np.ndarray,
                 cfg: Config, key_ids=None) -> Partial:
    """Accumulate one tile of keys. weights[j] = (M, E) with M in [2**23, 2**24)."""
    n = len(weights)
    dv = vm.shape[1] if vm.ndim == 2 else 0
    p = Partial(dv)
    if n == 0:
        return p
    p.count = n
    tops = [E + 23 for (_, E) in weights]
    R = max(tops)
    p.R = R
    fz, fn = cfg.fz, cfg.fn
    # Z: unit 2**(R - fz)
    for (M, E) in weights:
        v, lost = _floor_shift(M, E - (R - fz))
        p.Z += v
        p.Zerr += 1 if lost else 0
    _check_width(p.Z, p.Zerr, cfg.acc_bits)
    vm_l, ve_l = vm.tolist(), ve.tolist()
    for c in range(dv):
        # Lead exponent of each term |w_j * v_jc| is at most tops[j] + lead(v_jc) + 1. Using the
        # bound tops[j] + lead(v_jc) + 1 as the per-term lead keeps every |term| < 2**(NR+1).
        leads = [tops[j] + 1 + ve_l[j][c] + (abs(vm_l[j][c]).bit_length() - 1) for j in range(n) if vm_l[j][c] != 0]
        if not leads:
            continue
        NR = max(leads)
        p.NR[c] = NR
        base = NR - fn  # unit for N_c; each |w*v| < 2**(NR+1) = 2**(fn+1) units
        acc = 0
        err = 0
        for j in range(n):
            m = vm_l[j][c]
            if m == 0:
                continue
            M, E = weights[j]
            t, lost = _floor_shift(M * m, E + ve_l[j][c] - base)
            acc += t
            err += 1 if lost else 0
        _check_width(acc, err, cfg.acc_bits)
        p.N[c] = acc
        p.Nerr[c] = err
    if key_ids is not None:
        p.keys = list(key_ids)
    return p


def merge(a: Partial, b: Partial, cfg: Config) -> Partial:
    """Combine two partials over disjoint key sets. Order and tree shape don't matter for
    the certified result."""
    assert a.dv == b.dv
    if a.R is None:
        return b
    if b.R is None:
        return a
    out = Partial(a.dv)
    out.count = a.count + b.count
    assert out.count <= (1 << cfg.key_log2), "key headroom exceeded"
    R = max(a.R, b.R)
    out.R = R
    za, ea = _rescale(a.Z, a.Zerr, R - a.R)
    zb, eb = _rescale(b.Z, b.Zerr, R - b.R)
    out.Z, out.Zerr = za + zb, ea + eb
    _check_width(out.Z, out.Zerr, cfg.acc_bits)
    for c in range(a.dv):
        va, vb = a.NR[c], b.NR[c]
        if va is None and vb is None:
            continue
        ref = max(x for x in (va, vb) if x is not None)
        out.NR[c] = ref
        acc, err = 0, 0
        for part, v in ((a, va), (b, vb)):
            if v is None:
                continue
            x, e = _rescale(part.N[c], part.Nerr[c], ref - v)
            acc += x
            err += e
        _check_width(acc, err, cfg.acc_bits)
        out.N[c], out.Nerr[c] = acc, err
    out.keys = a.keys + b.keys
    return out


def _ratio_bounds(Nlo: int, Nhi: int, Zlo: int, Zhi: int):
    """Interval of N/Z over N in [Nlo, Nhi], Z in [Zlo, Zhi] with Zlo > 0, as rationals
    (num, den) for the low and high end."""
    if Nlo >= 0:
        return (Nlo, Zhi), (Nhi, Zlo)
    if Nhi <= 0:
        return (Nlo, Zlo), (Nhi, Zhi)
    return (Nlo, Zlo), (Nhi, Zlo)


def finalize(p: Partial, cfg: Config, out: str, exact_fallback, stats: Stats) -> np.ndarray:
    """Certified final rounding. exact_fallback() -> output array (exact reference)."""
    fmt = BF16 if out == "bf16" else FP32
    dtype = np.uint16 if out == "bf16" else np.float32
    stats.rows += 1
    res = np.zeros(p.dv, dtype=dtype)
    if p.R is None:
        return res
    zunit = p.R - cfg.fz
    Zlo, Zhi = p.Z, p.Z + p.Zerr
    assert Zlo > 0
    for c in range(p.dv):
        if p.NR[c] is None:
            continue  # exactly zero
        nunit = p.NR[c] - cfg.fn
        Nlo, Nhi = p.N[c], p.N[c] + p.Nerr[c]
        (ln, ld), (hn, hd) = _ratio_bounds(Nlo, Nhi, Zlo, Zhi)
        sh = nunit - zunit
        r_lo = _round_scaled(ln, ld, sh, fmt)
        r_hi = _round_scaled(hn, hd, sh, fmt)
        if r_lo != r_hi:
            stats.row_fallbacks += 1
            return exact_fallback()
        m, e = r_lo
        if m == 0:
            continue
        res[c] = encode_bf16(m, e) if out == "bf16" else encode_fp32(m, e)
    return res


def _round_scaled(num: int, den: int, sh: int, fmt):
    """RN(num * 2**sh / den), with zero canonicalised."""
    if num == 0:
        return (0, 0)
    if sh >= 0:
        r = round_rational(num << sh, den, fmt)
    else:
        r = round_rational(num, den << -sh, fmt)
    return (0, 0) if r[0] == 0 else r


# ---------------------------------------------------------------------------
# Convenience drivers
# ---------------------------------------------------------------------------

def row_weights(q_bits, k_bits, alpha, cfg: Config, stats: Stats):
    t = windowed_logits(q_bits, k_bits, alpha, cfg, stats)
    tm = np.asarray(t, dtype=np.float32)
    if tm.size == 0:
        return []
    M, E = exp2_weight_fast_np(tm)
    return list(zip(M.tolist(), E.tolist()))


def split_attention_row(q_bits, k_bits, v_bits, alpha, splits, cfg: Config = Config(),
                        out: str = "bf16", merge_order=None, stats: Stats | None = None,
                        tile: int = 64) -> np.ndarray:
    """Process the keys in `splits` (a list of index arrays partitioning range(n)), each in
    tiles of `tile`, then merge the split partials in `merge_order` (a permutation, or None).

    This is the split-KV / FlashDecoding schedule, modelled bit-accurately.
    """
    from .reference import attention_row

    stats = stats if stats is not None else Stats()
    k_bits = np.asarray(k_bits, dtype=np.uint16)
    v_bits = np.asarray(v_bits, dtype=np.uint16)
    vm, ve = decode_bf16_array(v_bits)
    w = row_weights(q_bits, k_bits, alpha, cfg, stats)
    parts = []
    for idx in splits:
        acc = Partial(v_bits.shape[1])
        for t0 in range(0, len(idx), tile):
            ids = list(idx[t0:t0 + tile])
            tp = tile_partial([w[i] for i in ids], vm[ids], ve[ids], cfg, ids)
            acc = merge(acc, tp, cfg)
        parts.append(acc)
    order = merge_order if merge_order is not None else range(len(parts))
    total = Partial(v_bits.shape[1])
    for i in order:
        total = merge(total, parts[i], cfg)
    return finalize(total, cfg, out, lambda: attention_row(q_bits, k_bits, v_bits, alpha, out), stats)
