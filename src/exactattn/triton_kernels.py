"""Fused split-KV ExactAttention in Triton: one source for NVIDIA, Intel and AMD GPUs.

Two kernels, in the FlashDecoding shape:

  _ea_split_kernel    grid (query blocks, KV splits, value-channel blocks). Each program walks
                      its share of the keys in tiles. Per tile it computes windowed int64
                      scores, the certified logit t = RN32(alpha*s), the weight w = RN24(2^t)
                      and fixed-point partials with rigorous error counts, merged into a
                      running partial. The split's partial is written out.
  _ea_combine_kernel  grid (query blocks, value-channel blocks). Merges the split partials in a
                      caller-given order (any permutation gives the same bits), then does the
                      certified final rounding.

Rows the kernel cannot certify are flagged, and the host recomputes them exactly.

Tested on CPU through TRITON_INTERPRET=1 against the exact reference. Same algorithm as
torch_impl.py (DESIGN.md). The weights w = RN24(2^t) use integer fixed-point exp2 with a proven
error bound (exp2.py, DESIGN D15), so the per-key loop needs no float64, which consumer GPUs run
at 1/64 rate. A tile with a weight that integer exp2 cannot certify is recomputed with float64
exp2 (margin 2^-26, assuming relative error below 2^-50: CUDA libdevice is within 1 ulp, Intel
SYCL within 4 ulp, numpy too); exp2="f64" uses that path for every tile.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .exp2 import EXP2_ERR, EXP2_L31, exp2_table
from .formats import decode_fp32
from .reference import attention_row
from .slices import slice_rows

NEG = tl.constexpr(-(1 << 60))  # "empty" sentinel for reference exponents
NEG32 = tl.constexpr(-(1 << 30))  # the same, for exponents held in int32 (DESIGN D21)
QF = tl.constexpr(20)  # fraction bits of _slice_q's integer |q|_1 bound (DESIGN D32)
QF_ULP = tl.constexpr((1 << 20) - 1)  # 2^QF - 1, for rounding up
_LAST_DEBUG: dict = {}  # last call's intermediate buffers, for debugging
SCORE_FS = 63 - 1 - 7  # d <= 128
_L31 = tl.constexpr(EXP2_L31)
_EXP2_ERR = tl.constexpr(EXP2_ERR)


# ---------------------------------------------------------------------------
# Integer helpers (exact, device-independent)
# ---------------------------------------------------------------------------

@triton.jit
def _bitlen(x):
    """Exact bit length of int64 x >= 0 (0 -> 0). Uses a uint32 -> fp32 conversion of the
    nonzero 32-bit half (full rate; no float64): its exponent is the bit length, or one more if
    rounding to 24 bits carried into the next power of two, which the last step undoes."""
    hi = x >> 32
    v = tl.where(hi > 0, hi, x & 0xFFFFFFFF)                           # < 2^32
    f = v.to(tl.uint32).to(tl.float32)
    e = ((f.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 126
    b = tl.where(v > 0, e, 0).to(tl.int64)
    sb = tl.minimum(tl.maximum(b - 1, 0), 32)                         # v < 2^32 may round up to 2^32
    b = tl.where((v > 0) & ((v >> sb) == 0), b - 1, b)
    return b + tl.where(hi > 0, 32, 0)


@triton.jit
def _f32_bits(q, qe, neg, ovf):
    """int32 bits of the fp32 (-1)^neg * q * 2^qe, with q < 2^24 and q >= 2^23 unless qe == -149
    or q == 0 (the form RN32 produces). ovf gives inf. Exact, no float64."""
    mag = tl.where(q == 0, 0, ((qe + 149) << 23) + q)
    mag = tl.where(ovf & (q != 0), 0x7F800000, mag)
    return (mag | (neg.to(tl.int64) << 31)).to(tl.int32)


@triton.jit
def _floor_shift(m, sh):
    """(floor(m * 2^sh), lost). Callers guarantee left shifts do not overflow."""
    left = tl.minimum(tl.maximum(sh, 0), 62)
    s = tl.minimum(tl.maximum(-sh, 0), 63)
    sr = tl.minimum(s, 62)
    vr = m >> sr
    vr = tl.where(s >= 63, tl.where(m < 0, -1, 0), vr)
    lost_r = tl.where(s >= 63, m != 0, (vr << sr) != m)
    v = tl.where(sh >= 0, m << left, vr)
    lost = tl.where(sh >= 0, False, lost_r)
    return v, lost


@triton.jit
def _rescale(acc, err, s):
    """[acc, acc + err] re-expressed in units 2^s larger: (floor(acc / 2^s),
    ceil(err / 2^s) + [acc lost bits]), for s >= 0 and err >= 0 (every caller shifts by a maximum
    minus its own exponent). Clamping s at 63 is exact: for |acc| < 2^63 and 0 <= err < 2^63,
    acc >> 63 is floor(acc / 2^s) and -((-err) >> 63) is ceil(err / 2^s) for every s >= 63, so
    no s >= 63 case and no left-shift branch are needed (DESIGN D25)."""
    sc = tl.minimum(s, 63).to(tl.int64)
    v = acc >> sc
    lost = (v << sc) != acc
    return v, (-((-err) >> sc)) + lost.to(tl.int64)


@triton.jit
def _decode_bf16(b):
    b = b.to(tl.int64) & 0xFFFF
    ex = (b >> 7) & 0xFF
    fr = b & 0x7F
    sign = tl.where((b & 0x8000) != 0, -1, 1)
    normal = ex != 0
    m = sign * tl.where(normal, fr | 0x80, fr)
    e = tl.where(normal, ex - 134, -133)
    return m, e, ex == 0xFF


@triton.jit
def _rn32_product(am, ae, acc, base):
    """RN32(am * acc * 2^(ae + base)), exact. am is a signed int (24 bits). Returns (t, overflow)."""
    sgn = tl.where(acc < 0, -1, 1) * tl.where(am < 0, -1, 1)
    A = tl.abs(acc)
    a = tl.abs(am).to(tl.int64)
    X = (A >> 32) * a
    Y = (A & 0xFFFFFFFF) * a
    C = X + (Y >> 32)
    Yl = Y & 0xFFFFFFFF
    E0 = ae + base
    blC = _bitlen(C)
    bl = tl.where(C > 0, blC + 32, _bitlen(Yl))
    K = bl - 1 + E0
    qe = tl.maximum(K - 23, -149)
    s = qe - E0 - 1
    shi = tl.minimum(tl.maximum(s - 32, 0), 62)
    q_big = tl.where(s - 32 > 62, 0, C >> shi)
    st_big = tl.where(s - 32 > 62, (C != 0) | (Yl != 0),
                      ((C & ((tl.full(C.shape, 1, tl.int64) << shi) - 1)) != 0) | (Yl != 0))
    slo = tl.minimum(tl.maximum(s, 0), 31)
    q_small = (C << (32 - slo)) | (Yl >> slo)
    st_small = (Yl & ((tl.full(Yl.shape, 1, tl.int64) << slo) - 1)) != 0
    q_left = Yl << tl.minimum(tl.maximum(-s, 0), 62)
    q1 = tl.where(s >= 32, q_big, tl.where(s >= 1, q_small, q_left))
    sticky = tl.where(s >= 32, st_big, tl.where(s >= 1, st_small, False))
    rnd = q1 & 1
    q = q1 >> 1
    up = (rnd == 1) & (sticky | ((q & 1) == 1))
    q = q + up.to(tl.int64)
    carry = q == (1 << 24)
    q = tl.where(carry, q >> 1, q)
    qe = tl.where(carry, qe + 1, qe)
    q = tl.where(bl == 0, 0, q)
    overflow = (bl != 0) & (qe + _bitlen(q) - 1 > 127)
    t = _f32_bits(q, qe, sgn < 0, overflow).to(tl.float32, bitcast=True)
    return t, overflow


@triton.jit
def _ratio_round(a, b, sh, P: tl.constexpr, QMIN: tl.constexpr):
    """RN(a / b * 2^sh), a >= 0, 0 < b < 2^62. Returns (q, qe).

    q1 = floor(a * 2^s2 / b) has at most P + 1 bits (qe >= K - (P - 1)), so when s2 > P + 1 its
    first z = s2 - (P + 1) long-division steps give zero bits: floor(a * 2^z / b) = 0, so a < b
    and a * 2^z < b < 2^62. Those steps only double the remainder, so one shift replaces them, and
    at most P + 1 steps are left (DESIGN D22)."""
    k = _bitlen(a) - _bitlen(b)
    kp = tl.minimum(tl.maximum(k, 0), 62)
    kn = tl.minimum(tl.maximum(-k, 0), 62)
    one = tl.full(b.shape, 1, tl.int64)
    ceil_b = (b >> kn) + ((b & ((one << kn) - 1)) != 0).to(tl.int64)
    less = tl.where(k >= 0, (a >> kp) < b, a < ceil_b)
    k = tl.where(less, k - 1, k)
    K = k + sh
    qe = tl.maximum(K - (P - 1), QMIN)
    s2 = sh - qe + 1
    ip = a // b
    r = a - ip * b
    sn = tl.minimum(tl.maximum(-s2, 0), 62)
    q_neg = ip >> sn
    st_neg = (r != 0) | ((ip & ((one << sn) - 1)) != 0)
    z = tl.maximum(s2 - (P + 1), 0)
    q_pos = tl.where(z > 0, 0, ip)
    rr = tl.where(z > 0, a << tl.minimum(z, 62), r)
    n_it = s2 - z
    for i in tl.static_range(P + 1):
        act = n_it > i
        r2 = rr << 1
        bit = r2 >= b
        r2 = tl.where(bit, r2 - b, r2)
        rr = tl.where(act, r2, rr)
        q_pos = tl.where(act, (q_pos << 1) | bit.to(tl.int64), q_pos)
    st_pos = rr != 0
    q1 = tl.where(s2 >= 0, q_pos, q_neg)
    sticky = tl.where(s2 >= 0, st_pos, st_neg)
    rnd = q1 & 1
    q = q1 >> 1
    up = (rnd == 1) & (sticky | ((q & 1) == 1))
    q = q + up.to(tl.int64)
    carry = q == (1 << P)
    q = tl.where(carry, q >> 1, q)
    qe = tl.where(carry, qe + 1, qe)
    q = tl.where(a == 0, 0, q)
    return q, qe


# ---------------------------------------------------------------------------
# Scores: windowed int64 (CUDA cores) and int8-sliced (tensor cores)
# ---------------------------------------------------------------------------

@triton.jit
def _qk_chunk(Q, qoff, rmask, K, nn, nmask, c, D: tl.constexpr, DC: tl.constexpr):
    """Channels [c, c + DC) of the exact products q_m,d * k_n,d: mantissa P and exponent E, (M, N, DC)."""
    dc = c + tl.arange(0, DC)
    qm, qe, _qb = _decode_bf16(tl.load(Q + qoff[:, None] + dc[None, :], mask=rmask[:, None], other=0))
    km, ke, _kb = _decode_bf16(tl.load(K + nn[:, None] * D + dc[None, :], mask=nmask[:, None], other=0))
    return qm[:, None, :] * km[None, :, :], qe[:, None, :] + ke[None, :, :]


@triton.jit
def _windowed_logits(Q, qoff, rmask, K, nn, nmask, AM, AE, D: tl.constexpr, DC: tl.constexpr,
                     FS: tl.constexpr):
    """Certified t = RN32(alpha * q.k) from a windowed int64 dot product. Returns (t, uncertified).

    DC channels at a time, in two passes (the leading exponent over all d, then the windowed sum),
    so no (M, N, D) tensor is ever live (DESIGN D17). The integer sums are exact, so acc and err
    are the same as the one-shot form's, bit for bit, whatever DC is. q and k chunks are reloaded
    from global memory (L1/L2 hits) rather than kept in registers."""
    lmax = tl.full((qoff.shape[0], nn.shape[0]), NEG, tl.int64)
    for c in range(0, D, DC):
        P, E = _qk_chunk(Q, qoff, rmask, K, nn, nmask, c, D, DC)
        lead = tl.where(P != 0, E + _bitlen(tl.abs(P)) - 1, NEG)
        lmax = tl.maximum(lmax, tl.max(lead, axis=2))                  # (M, N)
    base = lmax + 1 - FS
    acc = tl.zeros((qoff.shape[0], nn.shape[0]), tl.int64)
    err = tl.zeros((qoff.shape[0], nn.shape[0]), tl.int64)
    for c in range(0, D, DC):
        P, E = _qk_chunk(Q, qoff, rmask, K, nn, nmask, c, D, DC)
        nzp = P != 0
        term, lost = _floor_shift(P, E - base[:, :, None])
        acc += tl.sum(tl.where(nzp, term, 0), axis=2)
        err += tl.sum((lost & nzp).to(tl.int64), axis=2)
    anyp = lmax != NEG
    big = (tl.abs(acc) + err) >= (1 << 62)
    t_lo, ov1 = _rn32_product(AM, AE, acc, base)
    t_hi, ov2 = _rn32_product(AM, AE, acc + err, base)
    t = tl.where(anyp, t_lo, 0.0)
    unc = anyp & ((t_lo != t_hi) | ov1 | ov2 | big)
    return t, unc


@triton.jit
def _qk_col(Q, qoff, rmask, K, j, c, D: tl.constexpr, DC: tl.constexpr):
    """_qk_chunk for the single key j: channels [c, c + DC) of q_m,d * k_j,d, as (M, DC)."""
    dc = c + tl.arange(0, DC)
    qm, qe, _qb = _decode_bf16(tl.load(Q + qoff[:, None] + dc[None, :], mask=rmask[:, None], other=0))
    km, ke, _kb = _decode_bf16(tl.load(K + j * D + dc))
    return qm * km[None, :], qe + ke[None, :]


@triton.jit
def _windowed_col(Q, qoff, rmask, K, j, AM, AE, D: tl.constexpr, DC: tl.constexpr, FS: tl.constexpr):
    """_windowed_logits for one key j: (t, uncertified), each (M,). The same exact integer sums per
    (row, key), so the same bits as the whole-tile form's column j (DESIGN D30), from (M, DC)
    products instead of (M, N, DC)."""
    lmax = tl.full((qoff.shape[0],), NEG, tl.int64)
    for c in range(0, D, DC):
        P, E = _qk_col(Q, qoff, rmask, K, j, c, D, DC)
        lead = tl.where(P != 0, E + _bitlen(tl.abs(P)) - 1, NEG)
        lmax = tl.maximum(lmax, tl.max(lead, axis=1))                  # (M,)
    base = lmax + 1 - FS
    acc = tl.zeros((qoff.shape[0],), tl.int64)
    err = tl.zeros((qoff.shape[0],), tl.int64)
    for c in range(0, D, DC):
        P, E = _qk_col(Q, qoff, rmask, K, j, c, D, DC)
        nzp = P != 0
        term, lost = _floor_shift(P, E - base[:, None])
        acc += tl.sum(tl.where(nzp, term, 0), axis=1)
        err += tl.sum((lost & nzp).to(tl.int64), axis=1)
    anyp = lmax != NEG
    big = (tl.abs(acc) + err) >= (1 << 62)
    t_lo, ov1 = _rn32_product(AM, AE, acc, base)
    t_hi, ov2 = _rn32_product(AM, AE, acc + err, base)
    t = tl.where(anyp, t_lo, 0.0)
    unc = anyp & ((t_lo != t_hi) | ov1 | ov2 | big)
    return t, unc


@triton.jit
def _col_recompute(t, unc, ucol, nuc, Q, qoff, rmask, K, nn, AM, AE, D: tl.constexpr, DC: tl.constexpr,
                   FS: tl.constexpr):
    """Replace the uncertified scores in the nuc key columns marked in ucol with windowed ones,
    one key at a time (DESIGN D30)."""
    left = ucol
    for _c in range(0, nuc):
        j = tl.min(tl.where(left != 0, nn, 2147483647), axis=0)
        t_c, unc_c = _windowed_col(Q, qoff, rmask, K, j, AM, AE, D, DC, FS)
        sel = (nn == j)[None, :] & unc
        t = tl.where(sel, t_c[:, None], t)
        unc = tl.where(sel, unc_c[:, None], unc)
        left = tl.where(nn == j, 0, left)
    return t, unc


@triton.jit
def _tile_recompute(t, unc, Q, qoff, rmask, K, nn, kmax, AM, AE, D: tl.constexpr, DC: tl.constexpr,
                    FS: tl.constexpr):
    """Replace every uncertified score of the tile with a windowed one."""
    t_w, unc_w = _windowed_logits(Q, qoff, rmask, K, nn, nn < kmax, AM, AE, D, DC, FS)
    return tl.where(unc, t_w, t), unc & unc_w


@triton.jit
def _q_digit(fix, neg, s: tl.constexpr):
    d = (fix >> (7 * s)) & 127
    return tl.where(neg, -d, d).to(tl.int8)


@triton.jit
def _q_group(fix, neg, g: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
             SQ: tl.constexpr):
    """MMA operand for q digit group g: digits j = g*PACK .. g*PACK + PACK-1 stacked along the rows
    (row jl*BM + r is digit g*PACK + jl of row r; digits past SQ are zero). With PACK = 16 // BM a
    decode block of 2 real rows still fills the 16 rows int8 MMA needs, with digits instead of
    padding."""
    if PACK == 1:
        a = _q_digit(fix, neg, g)
    else:
        tl.static_assert(PACK * BM >= 16)
        j = g * PACK + tl.arange(0, PACK)[:, None, None]
        d = (fix[None, :, :] >> (7 * tl.minimum(j, 8))) & 127
        d = tl.where(j < SQ, tl.where(neg[None, :, :], -d, d), 0)
        a = tl.reshape(d, (PACK * BM, D)).to(tl.int8)
    return a


@triton.jit
def _slot_sum(R, g: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr,
              SH0: tl.constexpr):
    """Unpack an MMA result over a packed operand: sum over slots j in [LO, HI) of group g of
    R_j << (7 (j - LO) + SH0), as int64 (BM, X). Exact."""
    if PACK == 1:
        out = R.to(tl.int64) << (7 * (g - LO) + SH0)
    else:
        R3 = tl.reshape(R, (PACK, BM, R.shape[1])).to(tl.int64)
        j = g * PACK + tl.arange(0, PACK)
        m = (j >= LO) & (j < HI)
        sh = tl.where(m, 7 * (j - LO) + SH0, 0).to(tl.int64)
        out = tl.sum(tl.where(m[:, None, None], R3 << sh[:, None, None], 0), axis=0)
    return out


@triton.jit
def _slot_err(R, PACK: tl.constexpr, BM: tl.constexpr, SW: tl.constexpr, SHA: tl.constexpr):
    """_tc_pv's error unpack for one packed group holding every slot (PACK >= SW + 2): the sum of
    R_j << 7j over the weight-digit slots j < SW, R_SW << SHA (lost weight x |top V digit|) and
    R_(SW+1) (visible x lost V), as int64 (BM, X). Exact."""
    R3 = tl.reshape(R, (PACK, BM, R.shape[1])).to(tl.int64)
    j = tl.arange(0, PACK)
    m = j <= SW + 1
    sh = tl.where(j < SW, 7 * j, tl.where(j == SW, SHA, 0)).to(tl.int64)
    return tl.sum(tl.where(m[:, None, None], R3 << sh[:, None, None], 0), axis=0)


@triton.jit
def _slice_q(qm, qe, SQ: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr, D: tl.constexpr):
    """slices.slice_rows for the program's q rows, in registers (same digits as the numpy and
    torch slicers). Done once per program, so a decode step needs no separate slicing launches.
    Returns up to 4 packed digit groups (_q_group; unused ones are zeros), lsb, an integer upper
    bound on |q|_1 * 2^-lsb, lost."""
    am = tl.abs(qm)
    nz = am != 0
    lead = tl.where(nz, qe + _bitlen(am) - 1, NEG)
    E = tl.max(lead, axis=1)
    lsb = E + 1 - 7 * SQ
    fix, lost = _floor_shift(am, qe - lsb[:, None])
    lost = lost & nz
    neg = qm < 0
    NG: tl.constexpr = (SQ + PACK - 1) // PACK
    z = tl.zeros((PACK * BM, D), tl.int8)
    q0 = _q_group(fix, neg, 0, PACK, BM, D, SQ)
    q1 = z
    q2 = z
    q3 = z
    if NG > 1:
        q1 = _q_group(fix, neg, 1, PACK, BM, D, SQ)
    if NG > 2:
        q2 = _q_group(fix, neg, 2, PACK, BM, D, SQ)
    if NG > 3:
        q3 = _q_group(fix, neg, 3, PACK, BM, D, SQ)
    # |q|_1 * 2^-lsb, rounded up to an integer, in integer arithmetic (DESIGN D32): each element
    # am * 2^(qe - lsb) in fixed point with QF fraction bits, the tiny ones rounded up to one
    # unit. Each is below 2^(7 SQ + QF) and D <= 128 of them sum below 2^(7 SQ + QF + 7) <= 2^55.
    # The result is at most ceil(|q|_1 2^-lsb + D 2^-QF), never more than the float64 bound it
    # replaces, so no score that bound certified becomes uncertified.
    sh = qe - lsb[:, None] + QF
    up = am << tl.minimum(tl.maximum(sh, 0), 62)
    dn = ((am - 1) >> tl.minimum(tl.maximum(-sh, 0), 63)) + 1                 # ceil(am 2^sh), am >= 1
    term = tl.where(nz, tl.where(sh >= 0, up, dn), 0)
    qerr = (tl.sum(term, axis=1) + QF_ULP) >> QF
    qlost = tl.max(lost.to(tl.int32), axis=1) != 0
    return q0, q1, q2, q3, lsb, qerr, qlost


@triton.jit
def _tc_logits(q0, q1, q2, q3, qlsb, qerr, qbig, qlost, KX, nn, nmask, KST, KLSB, KERR, KLOST,
               AM, AE, D: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, PACK: tl.constexpr,
               BM: tl.constexpr):
    """Certified t = RN32(alpha * q.k) from int8 slices (slices.py) on tensor cores.

    q.k = 2^(qlsb + klsb) * (sum_{s,t} 2^(7(s+t)) Q[s].K[t] + truncation), where each Q[s].K[t] is
    an exact int8 x int8 -> int32 MMA (PACK q digits share one MMA, _q_group). The truncation
    bound comes from per-row L1 norms scaled by 2^-lsb, as integer ceilings: K's (KERR, with the
    too-big bit 4 of KLOST) made at cache write, q's (qerr, qbig) by _slice_q. KX and the
    per-key pointers start at the segment,
    nn is int32 within it and KST is the digit-plane stride (int64 scalar), so the per-element
    address math is 32-bit. Returns (t, uncertified, key has NaN/inf)."""
    NG: tl.constexpr = (SQ + PACK - 1) // PACK
    dd = tl.arange(0, D)
    if PACK == 1:
        # One digit per MMA: products of equal weight u = s + t accumulate inside the int32 MMA
        # (|Q[s].K[t]| < d * 127^2 < 2^21, at most 4 per weight: < 2^23), and adjacent weights
        # pair up in int32 (S_u + S_(u+1) << 7 < 2^30) before widening. Exact, and 4 int64
        # terms instead of SQ * SK.
        tl.static_assert(SQ + SK - 1 <= 8)
        S0 = tl.zeros((BM, nn.shape[0]), tl.int32)
        S1 = S0
        S2 = S0
        S3 = S0
        S4 = S0
        S5 = S0
        S6 = S0
        S7 = S0
        for t_ in tl.static_range(SK):
            kt = tl.trans(tl.load((KX + t_ * KST) + (nn[:, None] * D + dd[None, :]), mask=nmask[:, None], other=0))
            for s in tl.static_range(SQ):
                qs = q0
                if s == 1:
                    qs = q1
                if s == 2:
                    qs = q2
                if s == 3:
                    qs = q3
                if s + t_ == 0:
                    S0 = tl.dot(qs, kt, S0, out_dtype=tl.int32)
                if s + t_ == 1:
                    S1 = tl.dot(qs, kt, S1, out_dtype=tl.int32)
                if s + t_ == 2:
                    S2 = tl.dot(qs, kt, S2, out_dtype=tl.int32)
                if s + t_ == 3:
                    S3 = tl.dot(qs, kt, S3, out_dtype=tl.int32)
                if s + t_ == 4:
                    S4 = tl.dot(qs, kt, S4, out_dtype=tl.int32)
                if s + t_ == 5:
                    S5 = tl.dot(qs, kt, S5, out_dtype=tl.int32)
                if s + t_ == 6:
                    S6 = tl.dot(qs, kt, S6, out_dtype=tl.int32)
                if s + t_ == 7:
                    S7 = tl.dot(qs, kt, S7, out_dtype=tl.int32)
        acc = (S0 + (S1 << 7)).to(tl.int64)
        if SQ + SK > 3:
            acc += (S2 + (S3 << 7)).to(tl.int64) << 14
        if SQ + SK > 5:
            acc += (S4 + (S5 << 7)).to(tl.int64) << 28
        if SQ + SK > 7:
            acc += (S6 + (S7 << 7)).to(tl.int64) << 42
    else:
        acc = tl.zeros((BM, nn.shape[0]), tl.int64)
        for t_ in tl.static_range(SK):
            kt = tl.trans(tl.load((KX + t_ * KST) + (nn[:, None] * D + dd[None, :]), mask=nmask[:, None], other=0))
            acc += _slot_sum(tl.dot(q0, kt, out_dtype=tl.int32), 0, PACK, BM, 0, SQ, 7 * t_)
            if NG > 1:
                acc += _slot_sum(tl.dot(q1, kt, out_dtype=tl.int32), 1, PACK, BM, 0, SQ, 7 * t_)
            if NG > 2:
                acc += _slot_sum(tl.dot(q2, kt, out_dtype=tl.int32), 2, PACK, BM, 0, SQ, 7 * t_)
            if NG > 3:
                acc += _slot_sum(tl.dot(q3, kt, out_dtype=tl.int32), 3, PACK, BM, 0, SQ, 7 * t_)
    klsb = tl.load(KLSB + nn, mask=nmask, other=0)
    kerr = tl.load(KERR + nn, mask=nmask, other=0)
    kw = tl.load(KLOST + nn, mask=nmask, other=0)
    klost = (kw & 1) != 0
    kbad = (kw & 2) != 0
    kbig = (kw & 4) != 0
    # Truncation bound qlost*|K|_1 + klost*|q|_1 (pre-scaled), as the sum of per-key and per-row
    # integer ceilings, so it is all integer work. The key ceilings are made at cache write
    # (DESIGN D32): on sm_120 the float64 ceiling here cost 9% of tier 1's stall samples.
    err = tl.where(qlost[:, None], kerr[None, :], 0) + tl.where(klost[None, :], qerr[:, None], 0)
    base = qlsb[:, None] + klsb[None, :]
    big = (qlost[:, None] & kbig[None, :]) | (klost[None, :] & qbig[:, None])
    big = big | (tl.abs(acc) >= (1 << 62) - err)              # err <= 2^60 + 2: no overflow
    t_lo, ov1 = _rn32_product(AM, AE, acc - err, base)
    t_hi, ov2 = _rn32_product(AM, AE, acc + err, base)
    unc = (t_lo != t_hi) | ov1 | ov2 | big
    return t_lo, unc, kbad


@triton.jit
def _pv_group(Wlo, Whi, lw, vis, g: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr, SW: tl.constexpr):
    """Packed weight operand for slot group g (rows jl*BM + r for slot j = g*PACK + jl): slot j < SW
    is weight digit j, slot SW the lost-weight mask, slot SW + 1 the visibility mask, later slots
    zero. One MMA per V operand then covers PACK slots. The weight comes as two int32 halves,
    Wlo = digits 0-3 and Whi = digits 4 and up, so the digit cuts are 32-bit shifts."""
    if PACK == 1:
        if g < 4:
            a = (Wlo >> (7 * g)) & 127
        elif g < SW:
            a = (Whi >> (7 * (g - 4))) & 127
        elif g == SW:
            a = lw.to(tl.int32)
        else:
            a = vis.to(tl.int32)
    else:
        tl.static_assert(PACK * BM >= 16)
        j = g * PACK + tl.arange(0, PACK)[:, None, None]
        dlo = (Wlo[None, :, :] >> (7 * tl.minimum(j, 3))) & 127
        dhi = (Whi[None, :, :] >> (7 * tl.minimum(tl.maximum(j - 4, 0), 3))) & 127
        dig = tl.where(j < 4, dlo, dhi)
        a = tl.where(j < SW, dig, tl.where(j == SW, lw[None, :, :].to(tl.int32),
                                            tl.where(j == SW + 1, vis[None, :, :].to(tl.int32), 0)))
        a = tl.reshape(a, (PACK * BM, Wlo.shape[1]))
    return a.to(tl.int8)


@triton.jit
def _tc_pv(Mw, Ew, vis, VX, VLSB, VLOST, VBAD, nn, nmask, cc, VST, FN, DV: tl.constexpr, SW: tl.constexpr,
           SV: tl.constexpr, ANCH: tl.constexpr, PACK: tl.constexpr, BM: tl.constexpr):
    """One tile's P·V on tensor cores (DESIGN D9/D10, D24). Returns (TNR, TN, TNE) in the partial
    format: value N_c lies in [TN, TN + TNE] * 2^(TNR - FN).

    V rows are block fixed point (per-key exponent vlsb, SV signed 7-bit digits, a per-element
    lost-bits mask), made once at cache write. Weights times 2^vlsb are anchored per row at the
    tile's largest term and truncated to SW unsigned 7-bit digits. Each digit pair is an exact
    int8 MMA, and more small MMAs bound the truncation per element:
      weight loss  sum_j lostW_mj * |Vfix_jc|  <=  2^(7(SV-1)) * sum_j lostW_mj * |top V digit_jc|
                                                   + nlost_m * (2^(7(SV-1)) - 1)
      V loss       sum_j (W_mj + 1) * vlost_jc      (weight digits and visibility x lost-V mask).
    The first holds for any digit signs: |Vfix| <= |top| * 2^(7(SV-1)) + 127 * (1 + 2^7 + ...).
    The weight digits and both masks are slots of one packed operand (_pv_group), PACK slots per
    MMA, so small row blocks do not waste the MMA rows. A channel with acc = e = 0 encloses exactly
    zero and gets an empty partial; a lost V element makes e >= 1, so it is never dropped."""
    vlsb = tl.load(VLSB + nn, mask=nmask, other=0)
    G = Ew + vlsb[None, :]                                                 # term = M * Vfix * 2^G
    gtop = tl.max(tl.where(vis, G + 24, NEG), axis=1)                      # (M,)
    base = gtop - 7 * SW
    Wf, lw = _floor_shift(Mw, G - base[:, None])                           # Wf < 2^(7*SW)
    Wf = tl.where(vis, Wf, 0)
    lw = lw & vis
    tl.static_assert(SW <= 8)
    Wlo = (Wf & 0xFFFFFFF).to(tl.int32)                                    # digits 0-3
    Whi = (Wf >> 28).to(tl.int32)                                          # digits 4 .. SW-1
    NGW: tl.constexpr = (SW + 2 + PACK - 1) // PACK     # slots: SW weight digits, lost mask, vis mask
    SHA: tl.constexpr = 7 * (SV - 1)                    # weight of the top V digit
    BN: tl.constexpr = nn.shape[0]
    # The int32 MMA sums below are exact: a digit-product sum is below BN * 127^2, at most SV of
    # them share a weight, and a pair S_u + (S_(u+1) << 7) stays below 2^31.
    tl.static_assert(BN * 16129 * SV * 129 < 2147483648)
    vw = tl.load(VBAD + nn, mask=nmask, other=0)          # (N,) bit 0: NaN/inf, bit 1: a lost V element
    vbad = (vw & 1) != 0
    # Only keys with a lost element read their mask row (DESIGN D33): about 2% of keys, while the
    # row is 128 of the key's ~1050 bytes. Any other key's row is all zero, as the masked load gives.
    vl = tl.load(VLOST + nn[:, None] * DV + cc[None, :], mask=(nmask & ((vw & 2) != 0))[:, None], other=0)
    nlost = tl.sum(lw.to(tl.int32), axis=1).to(tl.int64)                   # (M,) lost weights
    if PACK == 1:
        # One slot per MMA. Weight digit j times V digit b has weight 7(j + b): products of equal
        # weight u accumulate inside the int32 MMA, and adjacent weights pair up in int32 before
        # widening, as in _tc_logits (DESIGN D18): 4 int64 terms instead of SW * SV.
        tl.static_assert(SW + SV - 1 <= 8)
        S0 = tl.zeros((BM, cc.shape[0]), tl.int32)
        S1 = S0
        S2 = S0
        S3 = S0
        S4 = S0
        S5 = S0
        S6 = S0
        S7 = S0
        ea = S0
        for b in tl.static_range(SV):
            vb = tl.load((VX + b * VST) + (nn[:, None] * DV + cc[None, :]), mask=nmask[:, None], other=0)
            for j in tl.static_range(SW):
                a = _pv_group(Wlo, Whi, lw, vis, j, PACK, BM, SW)
                if j + b == 0:
                    S0 = tl.dot(a, vb, S0, out_dtype=tl.int32)
                if j + b == 1:
                    S1 = tl.dot(a, vb, S1, out_dtype=tl.int32)
                if j + b == 2:
                    S2 = tl.dot(a, vb, S2, out_dtype=tl.int32)
                if j + b == 3:
                    S3 = tl.dot(a, vb, S3, out_dtype=tl.int32)
                if j + b == 4:
                    S4 = tl.dot(a, vb, S4, out_dtype=tl.int32)
                if j + b == 5:
                    S5 = tl.dot(a, vb, S5, out_dtype=tl.int32)
                if j + b == 6:
                    S6 = tl.dot(a, vb, S6, out_dtype=tl.int32)
                if j + b == 7:
                    S7 = tl.dot(a, vb, S7, out_dtype=tl.int32)
            if b == SV - 1:                                                # lost weight x |top V digit|
                ea = tl.dot(_pv_group(Wlo, Whi, lw, vis, SW, PACK, BM, SW), tl.abs(vb), out_dtype=tl.int32)
        acc = (S0 + (S1 << 7)).to(tl.int64)
        if SW + SV > 3:
            acc += (S2 + (S3 << 7)).to(tl.int64) << 14
        if SW + SV > 5:
            acc += (S4 + (S5 << 7)).to(tl.int64) << 28
        if SW + SV > 7:
            acc += (S6 + (S7 << 7)).to(tl.int64) << 42
        # V loss: visible x lost V, then weight digit j x lost V (each below BN * 127) summed in
        # int32 three digits at a time.
        e = (ea.to(tl.int64) << SHA) + tl.dot(_pv_group(Wlo, Whi, lw, vis, SW + 1, PACK, BM, SW), vl,
                                              out_dtype=tl.int32).to(tl.int64)
        t3 = tl.zeros((BM, cc.shape[0]), tl.int32)
        for j in tl.static_range(SW):
            t3 += tl.dot(_pv_group(Wlo, Whi, lw, vis, j, PACK, BM, SW), vl, out_dtype=tl.int32) << (7 * (j % 3))
            if j % 3 == 2 or j == SW - 1:
                e += t3.to(tl.int64) << (21 * (j // 3))
                t3 = tl.zeros((BM, cc.shape[0]), tl.int32)
    elif NGW == 1:
        # Every slot in one MMA group (decode): add the V digits' MMA results in their own layout,
        # then unpack the slots once for acc and once for e.
        a = _pv_group(Wlo, Whi, lw, vis, 0, PACK, BM, SW)
        X = tl.zeros((PACK * BM, cc.shape[0]), tl.int64)
        ra = tl.zeros((PACK * BM, cc.shape[0]), tl.int32)
        for b in tl.static_range(SV):
            vb = tl.load((VX + b * VST) + (nn[:, None] * DV + cc[None, :]), mask=nmask[:, None], other=0)
            X += tl.dot(a, vb, out_dtype=tl.int32).to(tl.int64) << (7 * b)
            if b == SV - 1:
                ra = tl.dot(a, tl.abs(vb), out_dtype=tl.int32)
        acc = _slot_sum(X, 0, PACK, BM, 0, SW, 0)
        rv = tl.dot(a, vl, out_dtype=tl.int32)
        slot = (tl.arange(0, PACK * BM) // BM)[:, None]
        e = _slot_err(tl.where(slot == SW, ra, rv), PACK, BM, SW, SHA)
    else:
        acc = tl.zeros((BM, cc.shape[0]), tl.int64)
        e = tl.zeros((BM, cc.shape[0]), tl.int64)
        for b in tl.static_range(SV):
            vb = tl.load((VX + b * VST) + (nn[:, None] * DV + cc[None, :]), mask=nmask[:, None], other=0)
            for g in tl.static_range(NGW):
                if g * PACK < SW:                                          # weight digits x V digit b
                    acc += _slot_sum(tl.dot(_pv_group(Wlo, Whi, lw, vis, g, PACK, BM, SW), vb, out_dtype=tl.int32),
                                     g, PACK, BM, 0, SW, 7 * b)
                if b == SV - 1 and g * PACK <= SW and SW < g * PACK + PACK:   # lost weight x |top V digit|
                    e += _slot_sum(tl.dot(_pv_group(Wlo, Whi, lw, vis, g, PACK, BM, SW), tl.abs(vb),
                                          out_dtype=tl.int32), g, PACK, BM, SW, SW + 1, SHA)
        for g in tl.static_range(NGW):
            if g * PACK < SW or g * PACK + PACK > SW + 1:
                rv = tl.dot(_pv_group(Wlo, Whi, lw, vis, g, PACK, BM, SW), vl, out_dtype=tl.int32)
                if g * PACK < SW:                                          # weight digits x lost V
                    e += _slot_sum(rv, g, PACK, BM, 0, SW, 0)
                if g * PACK + PACK > SW + 1:                               # visible x lost V
                    e += _slot_sum(rv, g, PACK, BM, SW + 1, SW + 2, 0)
    e += nlost[:, None] * ((1 << SHA) - 1)
    # True N in [acc - e, acc + e] * 2^base, with |acc| < 2^ANCH (ANCH = 7*SW + 7*SV + log2 BLOCK_N).
    # Re-anchor at TNR = base + ANCH, i.e. units 2^(base + ANCH - FN): a right shift by
    # sr = ANCH - FN (counted truncation) or an exact left shift. acc = e = 0 maps to 0, 0.
    chan = (acc != 0) | (e != 0)
    lo = acc - e
    wid = 2 * e
    sr = ANCH - FN
    TN_r, TNE_r = _rescale(lo, wid, tl.full(lo.shape, 0, tl.int64) + tl.maximum(sr, 0))
    sl = tl.minimum(tl.maximum(-sr, 0), 62)
    TN = tl.where(sr >= 0, TN_r, lo << sl)
    TNE = tl.where(sr >= 0, TNE_r, wid << sl)
    TNR = tl.where(chan, (base[:, None] + ANCH).to(tl.int32), NEG32)
    return TNR, TN, TNE, vbad


# ---------------------------------------------------------------------------
# Weights w = RN24(2^t)
# ---------------------------------------------------------------------------

@triton.jit
def _exp2_weight(t, EXP2T):
    """(M, E, risky): w = M * 2^E = RN24(2^t), M in [2^23, 2^24), in integer arithmetic
    (exp2.exp2_weight_int_np is the same computation in numpy; DESIGN D15 has the error bound).
    risky marks weights it cannot certify."""
    b = t.to(tl.int32, bitcast=True)
    ex = (b >> 23) & 0xFF
    m = (b & 0x7FFFFF) | tl.where(ex != 0, 0x800000, 0)
    e = tl.where(ex != 0, ex - 150, -149).to(tl.int64)
    ms = tl.where(b < 0, -m, m).to(tl.int64)                           # t = ms * 2^e
    tiny = e < -50                                                     # |t| < 2^-27: w = 1
    sr = tl.minimum(tl.maximum(-e, 0), 50)
    n_neg = ms >> sr                                                   # floor(t)
    F = (ms - (n_neg << sr)) << tl.minimum(tl.maximum(50 + e, 0), 50)  # frac(t) * 2^50, exact
    n_pos = ms << tl.minimum(tl.maximum(e, 0), 8)
    n = tl.where(e >= 0, n_pos, tl.where(tiny, 0, n_neg))
    F = tl.where((e >= 0) | tiny, 0, F)
    j = F >> 40
    R = (F & 0xFFFFFFFFFF) >> 8                                        # r * 2^42, r < 2^-10
    X = (R * _L31) >> 31                                               # x = r ln2, * 2^42
    X2 = (X * X) >> 42
    X3 = (X2 * X) >> 42
    P = X + (X2 >> 1) + X3 // 6                                        # (2^r - 1) * 2^42
    Tj = tl.load(EXP2T + j)                                            # 2^(j/1024) * 2^61
    Y = Tj + (((Tj >> 31) * P) >> 11)                                  # 2^f * 2^61
    M = (Y + (1 << 37)) >> 38
    risky = tl.abs((Y & ((1 << 38) - 1)) - (1 << 37)) <= _EXP2_ERR
    carry = M == (1 << 24)
    M = tl.where(carry, M >> 1, M)
    return M, n - 23 + carry.to(tl.int64), risky


@triton.jit
def _exp2_weight_f64(t, EXP2_MARGIN: tl.constexpr):
    """The same weights from float64 exp2 (assumes relative error below 2^-50)."""
    td = t.to(tl.float64)
    nfl = tl.floor(td)
    f = td - nfl
    scaled = tl.exp2(f) * 8388608.0
    fl = tl.floor(scaled)
    frac = scaled - fl
    Mw = fl.to(tl.int64) + (frac >= 0.5).to(tl.int64)
    Ew = nfl.to(tl.int64) - 23
    carry = Mw == (1 << 24)
    Mw = tl.where(carry, Mw >> 1, Mw)
    Ew = tl.where(carry, Ew + 1, Ew)
    exact = f == 0.0
    Mw = tl.where(exact, 1 << 23, Mw)
    Ew = tl.where(exact, nfl.to(tl.int64) - 23, Ew)
    risky = (~exact) & (tl.abs(frac - 0.5) < EXP2_MARGIN)
    return Mw, Ew, risky


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _clk(dep):
    """%clock64 (SM cycles), read after the int32 scalar dep is computed (DESIGN D26)."""
    return tl.inline_asm_elementwise("mov.u64 $0, %clock64;", "=l,r", [dep], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _gtime(dep):
    """%globaltimer (ns, the same on every SM), read after dep."""
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [dep], dtype=tl.int64, is_pure=False,
                                     pack=1)


@triton.jit
def _smid(dep):
    return tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [dep], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _split_program(pid_m, pid_s, pid_c, Q, K, V, KVLEN, ROWS, AM, AE,
                   PR, PZ, PZE, PNR, PN, PNE, FLAG,
                   SPLIT_LEN, FZ, FN,
                   KX, KLSB, KERR, KLOST, NKEYS, VX, VLSB, VLOST, VBAD,
                   R_SEG, NK_SEG, NRB, FL_NCB, TM, DFL,
                   D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_DV: tl.constexpr, FS: tl.constexpr, EXP2_MARGIN: tl.constexpr,
                   USE_TC: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, TC_FALLBACK: tl.constexpr,
                   USE_TC_PV: tl.constexpr, SW: tl.constexpr, SV: tl.constexpr, ANCH: tl.constexpr,
                   RERUN: tl.constexpr, PACK: tl.constexpr, EXP2T, EXP2_INT: tl.constexpr,
                   WDC: tl.constexpr, FL_DIV: tl.constexpr, TIMERS: tl.constexpr, DEFER: tl.constexpr,
                   COLMAX: tl.constexpr, CDC: tl.constexpr):
    # One program's work: row block pid_m (segment-major), KV split pid_s, channel block pid_c.
    # Segments: G independent problems (e.g. sequence x KV head), each with R_SEG query rows
    # attending its own NK_SEG keys. Rows and keys are stored segment after segment, so
    # ROWS = G * R_SEG and NKEYS = G * NK_SEG. Each program handles one segment's row block.
    if TIMERS:
        # Diagnostic phase clocks (DESIGN D26): each read waits on a block reduction of its
        # phase's result, so it cannot be scheduled before the phase ends.
        c0 = _clk(pid_m + pid_s)
        g0 = _gtime(pid_m + pid_s)
        T1 = tl.zeros((), tl.int64)
        T2 = T1
        T3 = T1
        T4 = T1
        NT = T1
        WF = T1                     # tiles that took a windowed score fallback (DESIGN D27, D30)
        CF = T1                     # key columns recomputed one at a time (DESIGN D30)
        CT = T1                     # ... in this many tiles (the rest of WF took the whole tile)
        EF = T1                     # tiles that took the float64 exp2 fallback
        MX = T1                     # longest tile, cycles from the previous tile's end
        UC = T1                     # key columns with an uncertified tensor-core score (DESIGN D29)
        UL = T1                     # ... of which the key lost bits when sliced (KLOST)
        UE = T1                     # uncertified (row, key) scores
        UV = T1                     # visible rows in those key columns
        UX = T1                     # most uncertified key columns in one tile
    seg = pid_m // NRB
    lrows = (pid_m % NRB) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = lrows < R_SEG
    rows = seg * R_SEG + lrows
    # Pointers start at the segment's first key, and key indices are int32 within it, so the
    # per-element address math is 32-bit (int64 offsets cost several instructions per element).
    kb0 = seg.to(tl.int64) * NK_SEG
    K += kb0 * D
    V += kb0 * DV
    KX += kb0 * D
    KLSB += kb0
    KERR += kb0
    KLOST += kb0
    VX += kb0 * DV
    VLSB += kb0
    VLOST += kb0 * DV
    VBAD += kb0
    KST = (kb0 * 0 + NKEYS) * D                                            # digit-plane strides
    VST = (kb0 * 0 + NKEYS) * DV
    kvl = tl.load(KVLEN + rows, mask=rmask, other=0)
    dd = tl.arange(0, D)
    cc = pid_c * BLOCK_DV + tl.arange(0, BLOCK_DV)
    qb = tl.load(Q + rows[:, None] * D + dd[None, :], mask=rmask[:, None], other=0)
    qm, qe, qbad = _decode_bf16(qb)
    flag = tl.max(qbad.to(tl.int32), axis=1)
    if DEFER and not RERUN:
        # Rows with a tensor-core score left uncertified (DESIGN D28): tier 2 recomputes them.
        dfl = tl.zeros((BLOCK_M,), tl.int32)
    if USE_TC:
        q0, q1, q2, q3, qlsb, qerr, qlost = _slice_q(qm, qe, SQ, PACK, BLOCK_M, D)
        qbig = qerr < 0                                       # never: qerr < 2^35 (_slice_q)
    PRr = tl.full((BLOCK_M,), NEG, tl.int64)
    PZr = tl.zeros((BLOCK_M,), tl.int64)
    PZEr = tl.zeros((BLOCK_M,), tl.int64)
    # The (row, channel) running partial is the loop's largest live state. Its exponent is held
    # in int32 (DESIGN D21); the value and the error count need int64.
    PNRr = tl.full((BLOCK_M, BLOCK_DV), NEG32, tl.int32)
    PNr = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
    PNEr = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
    k0 = pid_s * SPLIT_LEN
    kmax = tl.max(kvl, axis=0)
    k1 = tl.minimum(k0 + SPLIT_LEN, kmax)
    if TIMERS:
        if USE_TC:
            cA = _clk(kmax + tl.max(qlsb, axis=0).to(tl.int32))
        else:
            cA = _clk(kmax)
        tP = cA
    for n0 in range(k0, k1, BLOCK_N):
        if TIMERS:
            tA = _clk(n0)
        nn = n0 + tl.arange(0, BLOCK_N)
        vis = (nn[None, :] < kvl[:, None]) & rmask[:, None]              # (M, N)
        # ---- scores ----
        if USE_TC:
            # Tensor cores: int8 slice products, certified. The bf16 K tile is read and decoded
            # only for tiles that contain an uncertified score (windowed recompute); NaN/inf
            # keys come from a flag made at cache write.
            t, unc, kbad = _tc_logits(q0, q1, q2, q3, qlsb, qerr, qbig, qlost, KX, nn, nn < kmax, KST,
                                      KLSB, KERR, KLOST, AM, AE, D, SQ, SK, PACK, BLOCK_M)
            flag = tl.maximum(flag, tl.max((vis & kbad[None, :]).to(tl.int32), axis=1))
            if TIMERS:
                # Where the uncertified scores sit, before any fallback (DESIGN D29).
                uu = (unc & vis).to(tl.int32)
                ucol = tl.max(uu, axis=0)                                  # (N,)
                klb = tl.load(KLOST + nn, mask=nn < kmax, other=0) & 1
                nuc = tl.sum(ucol, axis=0).to(tl.int64)
                UC += nuc
                UL += tl.sum(ucol * klb, axis=0).to(tl.int64)
                UE += tl.sum(tl.sum(uu, axis=1), axis=0).to(tl.int64)
                UV += tl.sum(ucol * tl.sum(vis.to(tl.int32), axis=0), axis=0).to(tl.int64)
                UX = tl.maximum(UX, nuc)
            if DEFER and not RERUN:
                usc = (unc & vis).to(tl.int32)
                dfl = tl.maximum(dfl, tl.max(usc, axis=1))
                if TIMERS:
                    WF += tl.max(tl.max(usc, axis=1), axis=0).to(tl.int64)
            if TC_FALLBACK:
                # Exact recompute of the uncertified scores. Nearly always they sit in one key column,
                # a key that lost bits when sliced (DESIGN D29), so a tile with up to COLMAX such
                # columns recomputes them one key at a time, and only a tile with more recomputes
                # the whole tile (DESIGN D30). Each recomputed (row, key) gets the same windowed
                # bits either way. The two branch layouts differ only in what ptxas makes of them:
                # nested keeps small (decode) blocks' spills low, but caps 16-row blocks at 128
                # registers, so those run the whole-tile recompute after, not inside, this branch.
                if COLMAX > 0:
                    ucol = tl.max((unc & vis).to(tl.int32), axis=0)             # (N,)
                    nuc = tl.sum(ucol, axis=0)
                    if BLOCK_M < 16:
                        if nuc > 0:
                            if nuc <= COLMAX:
                                t, unc = _col_recompute(t, unc, ucol, nuc, Q, rows * D, rmask, K, nn, AM, AE,
                                                        D, CDC, FS)
                                if TIMERS:
                                    CF += nuc.to(tl.int64)
                                    CT += 1
                            else:
                                t, unc = _tile_recompute(t, unc, Q, rows * D, rmask, K, nn, kmax, AM, AE, D, WDC, FS)
                            if TIMERS:
                                WF += 1
                    else:
                        if (nuc > 0) & (nuc <= COLMAX):
                            t, unc = _col_recompute(t, unc, ucol, nuc, Q, rows * D, rmask, K, nn, AM, AE, D,
                                                    CDC, FS)
                            if TIMERS:
                                CF += nuc.to(tl.int64)
                                CT += 1
                                WF += 1
                        whole = nuc > COLMAX
                else:
                    whole = tl.max(tl.max((unc & vis).to(tl.int32), axis=1), axis=0) > 0
            if TC_FALLBACK and (COLMAX == 0 or BLOCK_M >= 16):
                if whole:
                    t, unc = _tile_recompute(t, unc, Q, rows * D, rmask, K, nn, kmax, AM, AE, D, WDC, FS)
                    if TIMERS:
                        WF += 1
        else:
            kb = tl.load(K + nn[:, None] * D + dd[None, :], mask=(nn < kmax)[:, None], other=0).to(tl.int32)
            kbad = tl.max((((kb >> 7) & 0xFF) == 0xFF).to(tl.int32), axis=1)  # (N,) NaN/inf key
            flag = tl.maximum(flag, tl.max((vis & (kbad != 0)[None, :]).to(tl.int32), axis=1))
            t, unc = _windowed_logits(Q, rows * D, rmask, K, nn, nn < kmax, AM, AE, D, WDC, FS)
            if TIMERS:
                WF += 1
        if TIMERS:
            tB = _clk(tl.max(tl.max(t.to(tl.int32, bitcast=True), axis=1), axis=0))
        # ---- weights ----
        if EXP2_INT:
            Mw, Ew, risky = _exp2_weight(t, EXP2T)
            if tl.max(tl.max((risky & vis).to(tl.int32), axis=1), axis=0) > 0:
                # Rare (about 2^-14 of weights): float64 exp2 for this tile.
                Mw2, Ew2, risky2 = _exp2_weight_f64(t, EXP2_MARGIN)
                Mw = tl.where(risky, Mw2, Mw)
                Ew = tl.where(risky, Ew2, Ew)
                risky = risky & risky2
                if TIMERS:
                    EF += 1
        else:
            Mw, Ew, risky = _exp2_weight_f64(t, EXP2_MARGIN)
        badw = risky | (tl.abs(t) >= 1073741824.0)
        if DEFER and not RERUN:
            badw = badw & ~unc        # a deferred score's weight is recomputed with it in tier 2
        else:
            badw = badw | unc
        flag = tl.maximum(flag, tl.max((vis & badw).to(tl.int32), axis=1))
        if TIMERS:
            tC = _clk(tl.max(tl.max(Ew, axis=1), axis=0).to(tl.int32))
        # ---- tile partial ----
        top = tl.where(vis, Ew + 23, NEG)
        TR = tl.max(top, axis=1)                                           # (M,)
        zt, zl = _floor_shift(Mw, Ew - (TR[:, None] - FZ))
        TZ = tl.sum(tl.where(vis, zt, 0), axis=1)
        TZE = tl.sum((zl & vis).to(tl.int64), axis=1)
        if USE_TC_PV:
            TNR, TN, TNE, vbad = _tc_pv(Mw, Ew, vis, VX, VLSB, VLOST, VBAD, nn, nn < kmax, cc, VST, FN, DV,
                                        SW, SV, ANCH, PACK, BLOCK_M)
            flag = tl.maximum(flag, tl.max((vis & vbad[None, :]).to(tl.int32), axis=1))
        else:
            vb = tl.load(V + nn[:, None] * DV + cc[None, :], mask=(nn < kmax)[:, None], other=0)
            vm, ve, vbad = _decode_bf16(vb)                                # (N, BDV)
            flag = tl.maximum(flag, tl.max((vis & (tl.max(vbad.to(tl.int32), axis=1) != 0)[None, :]).to(tl.int32), axis=1))
            lv = tl.where(vm != 0, ve + _bitlen(tl.abs(vm)) - 1, NEG)
            valid = vis[:, :, None] & (vm != 0)[None, :, :]                # (M, N, BDV)
            L = tl.where(valid, top[:, :, None] + 1 + lv[None, :, :], NEG)
            TNR = tl.max(L, axis=1)                                        # (M, BDV)
            TNR = tl.where(TNR > NEG // 2, TNR, NEG32).to(tl.int32)
            nt, nl = _floor_shift(Mw[:, :, None] * vm[None, :, :],
                                  Ew[:, :, None] + ve[None, :, :] - (TNR[:, None, :] - FN))
            TN = tl.sum(tl.where(valid, nt, 0), axis=1)
            TNE = tl.sum((nl & valid).to(tl.int64), axis=1)
        if TIMERS:
            tD = _clk(tl.max(tl.max(TNR, axis=1), axis=0) + tl.max(TZ, axis=0).to(tl.int32))
        # ---- merge into the running partial ----
        nR = tl.maximum(PRr, TR)
        a1, e1 = _rescale(PZr, PZEr, nR - PRr)
        a2, e2 = _rescale(TZ, TZE, nR - TR)
        PRr = nR
        PZr = a1 + a2
        PZEr = e1 + e2
        nNR = tl.maximum(PNRr, TNR)
        b1, f1 = _rescale(PNr, PNEr, (nNR - PNRr).to(tl.int64))
        b2, f2 = _rescale(TN, TNE, (nNR - TNR).to(tl.int64))
        PNRr = nNR
        PNr = b1 + b2
        PNEr = f1 + f2
        flag = tl.maximum(flag, ((tl.abs(PZr) + PZEr) >= (1 << 62)).to(tl.int32))
        flag = tl.maximum(flag, tl.max(((tl.abs(PNr) + PNEr) >= (1 << 62)).to(tl.int32), axis=1))
        if TIMERS:
            tE = _clk(tl.max(tl.max(PNEr, axis=1), axis=0).to(tl.int32) + tl.max(flag, axis=0))
            T1 += tB - tA
            T2 += tC - tB
            T3 += tD - tC
            T4 += tE - tD
            NT += 1
            MX = tl.maximum(MX, tE - tP)
            tP = tE
    if TIMERS:
        cB = _clk(tl.max(tl.max(PNEr, axis=1), axis=0).to(tl.int32) + tl.max(flag, axis=0))
    off = pid_s * ROWS + rows
    wmask = rmask
    if not RERUN:
        zmask = rmask & ((rows * 0 + pid_c) == 0)  # only channel block 0 writes Z
        tl.store(PR + off, PRr, mask=zmask)
        tl.store(PZ + off, PZr, mask=zmask)
        tl.store(PZE + off, PZEr, mask=zmask)
        if DEFER:
            # A deferred row's partials and flags in this split came from uncertified scores.
            # Tier 2 recomputes this split of the row for every channel block (and, in row scope,
            # every other split too) and writes its own flags, so tier 1 leaves the entry
            # unflagged and marks it in DFL (DESIGN D28).
            tl.store(DFL + off, dfl, mask=zmask)
            flag = tl.where(dfl != 0, 0, flag)
    elif DEFER:
        # Tier 2 rewrites Z as well when rows may be deferred: tier 1's Z of such a row is not an
        # enclosure. Every channel block of a row computes the same Z, so the duplicate stores
        # write the same bits. Without deferral, tier 2 keeps tier 1's Z and rewrites only N.
        tl.store(PR + off, PRr, mask=rmask)
        tl.store(PZ + off, PZr, mask=rmask)
        tl.store(PZE + off, PZEr, mask=rmask)
    o2 = off[:, None] * DV + cc[None, :]
    tl.store(PNR + o2, tl.where(PNRr > NEG32, PNRr.to(tl.int64), NEG), mask=wmask[:, None])
    tl.store(PN + o2, PNr, mask=wmask[:, None])
    tl.store(PNE + o2, PNEr, mask=wmask[:, None])
    # FLAG has FL_NCB channel blocks of tier 1's width; tier 2 runs FL_DIV narrower blocks per
    # entry, so it takes the maximum (atomics: the result does not depend on their order).
    fo = FLAG + (pid_s * ROWS + rows) * FL_NCB + pid_c // FL_DIV
    if RERUN:
        tl.atomic_max(fo, flag, mask=wmask)
    else:
        tl.store(fo, flag, mask=wmask)
    if TIMERS:
        # Row per program: prologue, scores, weights, tile partial, merge (summed over tiles),
        # tiles, key loop, whole program (cycles), start and end (ns), SM, then (DESIGN D27)
        # windowed-fallback tiles, float64-exp2 tiles, longest tile (cycles), pid_m, pid_s, then
        # (DESIGN D29) uncertified key columns, of them lossy keys, uncertified scores, visible rows
        # in those columns, most uncertified columns in one tile, then (DESIGN D30) key columns
        # recomputed one at a time and the tiles they were in.
        cZ = _clk(tl.max(flag, axis=0))
        g1 = _gtime(tl.max(flag, axis=0))
        sm = _smid(tl.max(flag, axis=0))
        tb = TM + ((pid_m * tl.num_programs(1) + pid_s) * tl.num_programs(2) + pid_c).to(tl.int64) * 24
        tl.store(tb + 0, cA - c0)
        tl.store(tb + 1, T1)
        tl.store(tb + 2, T2)
        tl.store(tb + 3, T3)
        tl.store(tb + 4, T4)
        tl.store(tb + 5, NT)
        tl.store(tb + 6, cB - cA)
        tl.store(tb + 7, cZ - c0)
        tl.store(tb + 8, g0)
        tl.store(tb + 9, g1)
        tl.store(tb + 10, sm.to(tl.int64))
        tl.store(tb + 11, WF)
        tl.store(tb + 12, EF)
        tl.store(tb + 13, MX)
        tl.store(tb + 14, (pid_m + 0 * WF).to(tl.int64))
        tl.store(tb + 15, (pid_s + 0 * WF).to(tl.int64))
        tl.store(tb + 16, UC)
        tl.store(tb + 17, UL)
        tl.store(tb + 18, UE)
        tl.store(tb + 19, UV)
        tl.store(tb + 20, UX)
        tl.store(tb + 21, CF)
        tl.store(tb + 22, CT)


@triton.jit
def _ea_split_kernel(Q, K, V, KVLEN, ROWS, AM, AE,
                     PR, PZ, PZE, PNR, PN, PNE, FLAG,
                     SPLIT_LEN, FZ, FN,
                     KX, KLSB, KERR, KLOST, NKEYS, VX, VLSB, VLOST, VBAD,
                     R_SEG, NK_SEG, NRB, FL_NCB, CNT, WL, NCCB, TM, DFL, NEED,
                     D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_DV: tl.constexpr, FS: tl.constexpr, EXP2_MARGIN: tl.constexpr,
                     USE_TC: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, TC_FALLBACK: tl.constexpr,
                     USE_TC_PV: tl.constexpr, SW: tl.constexpr, SV: tl.constexpr, ANCH: tl.constexpr,
                     RERUN: tl.constexpr, PACK: tl.constexpr, EXP2T, EXP2_INT: tl.constexpr,
                     WDC: tl.constexpr, FL_DIV: tl.constexpr, RESET_CNT: tl.constexpr, TIMERS: tl.constexpr,
                     DEFER: tl.constexpr, DEFER_SPLIT: tl.constexpr, COLMAX: tl.constexpr, CDC: tl.constexpr):
    pid_s = tl.program_id(1)
    if RERUN:
        # Tier 2 (DESIGN D12, D23): the tier-1 combine appended each (row, channel block) pair to
        # rerun to the work list WL, and CNT counts them. The grid is fixed (graph-safe) and small:
        # programs stride over the list, one row x one channel block per item, for this KV split.
        # The list's order depends on atomics, but each item writes only its own partials, so the
        # bits do not.
        cnt = tl.load(CNT)
        for it in range(tl.program_id(0), cnt, tl.num_programs(0)):
            item = tl.load(WL + it)
            go = True
            if DEFER_SPLIT:
                # NEED 2 marks a deferred row (DESIGN D28). Its tier-1 partials are enclosures in
                # every split it was not deferred in, so only the deferred splits are recomputed.
                # The item's row is item // NCCB (tier 2 runs one row per program).
                go = (tl.load(NEED + item) != 2) | (tl.load(DFL + pid_s * ROWS + item // NCCB) != 0)
            if go:
                _split_program(item // NCCB, pid_s, item % NCCB, Q, K, V, KVLEN, ROWS, AM, AE, PR, PZ, PZE, PNR, PN, PNE,
                               FLAG, SPLIT_LEN, FZ, FN, KX, KLSB, KERR, KLOST, NKEYS, VX, VLSB, VLOST, VBAD, R_SEG, NK_SEG,
                               NRB,
                               FL_NCB, TM, DFL, D, DV, BLOCK_M, BLOCK_N, BLOCK_DV, FS, EXP2_MARGIN, USE_TC, SQ, SK,
                               TC_FALLBACK, USE_TC_PV, SW, SV, ANCH, RERUN, PACK, EXP2T, EXP2_INT, WDC, FL_DIV, False,
                               DEFER, COLMAX, CDC)
    else:
        if RESET_CNT:
            # Tier 1 runs first in a step: zero tier 2's work-list count here, not with a fill launch.
            if (tl.program_id(0) == 0) & (pid_s == 0) & (tl.program_id(2) == 0):
                tl.store(CNT, 0)
        _split_program(tl.program_id(0), pid_s, tl.program_id(2), Q, K, V, KVLEN, ROWS, AM, AE, PR, PZ, PZE, PNR, PN,
                       PNE, FLAG, SPLIT_LEN, FZ, FN, KX, KLSB, KERR, KLOST, NKEYS, VX, VLSB, VLOST, VBAD, R_SEG, NK_SEG, NRB,
                       FL_NCB, TM, DFL, D, DV, BLOCK_M, BLOCK_N, BLOCK_DV, FS, EXP2_MARGIN, USE_TC, SQ, SK,
                       TC_FALLBACK, USE_TC_PV, SW, SV, ANCH, RERUN, PACK, EXP2T, EXP2_INT, WDC, FL_DIV, TIMERS, DEFER,
                       COLMAX, CDC)


@triton.jit
def _ea_combine_kernel(PR, PZ, PZE, PNR, PN, PNE, ORDER, NSPLIT, ROWS, FZ, FN, OUT, OFLAG, NEED, FLAG, FL_NCB, CNT, WL,
                       DFL, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
                       P: tl.constexpr, QMIN: tl.constexpr, OUT_FP32: tl.constexpr,
                       RERUN: tl.constexpr, WRITE_NEED: tl.constexpr, FL_DIV: tl.constexpr, DEFER: tl.constexpr):
    # Tier 1 with WRITE_NEED also writes tier 2's work (DESIGN D12, D16, D23): NEED = this
    # combine's flag set and no split flagged the pair (FLAG), fused here instead of torch ops, and
    # each such pair appended to the work list WL (CNT counts them; tier 1's split zeroed it).
    # With DEFER, a row tier 1 deferred (DFL, DESIGN D28) is rerun whatever this combine's flag.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < ROWS
    cc = pid_c * BLOCK_DV + tl.arange(0, BLOCK_DV)
    wmask = rmask
    go = True
    if RERUN:
        # Tier 2's combine: only programs with a NEED pair merge and round again (DESIGN D20).
        wmask = (tl.load(NEED + rows * tl.num_programs(1) + pid_c, mask=rmask, other=0) != 0) & rmask
        go = tl.max(wmask.to(tl.int32), axis=0) > 0
    if go:
        R = tl.full((BLOCK_M,), NEG, tl.int64)
        Z = tl.zeros((BLOCK_M,), tl.int64)
        ZE = tl.zeros((BLOCK_M,), tl.int64)
        NR = tl.full((BLOCK_M, BLOCK_DV), NEG, tl.int64)
        N = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
        NE = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
        flag = tl.zeros((BLOCK_M,), tl.int32)
        sflag = tl.zeros((BLOCK_M,), tl.int32)
        dflag = tl.zeros((BLOCK_M,), tl.int32)
        # Pipelined: the partials of later splits load (cp.async) while this one merges.
        for i in tl.range(NSPLIT, num_stages=3):
            s = tl.load(ORDER + i)
            off = s * ROWS + rows
            if WRITE_NEED:
                sflag = tl.maximum(sflag, tl.load(FLAG + off * FL_NCB + pid_c // FL_DIV, mask=rmask, other=0))
                if DEFER:
                    dflag = tl.maximum(dflag, tl.load(DFL + off, mask=rmask, other=0))
            sR = tl.load(PR + off, mask=rmask, other=NEG)
            sZ = tl.load(PZ + off, mask=rmask, other=0)
            sZE = tl.load(PZE + off, mask=rmask, other=0)
            o2 = off[:, None] * DV + cc[None, :]
            sNR = tl.load(PNR + o2, mask=rmask[:, None], other=NEG)
            sN = tl.load(PN + o2, mask=rmask[:, None], other=0)
            sNE = tl.load(PNE + o2, mask=rmask[:, None], other=0)
            nR = tl.maximum(R, sR)
            a1, e1 = _rescale(Z, ZE, nR - R)
            a2, e2 = _rescale(sZ, sZE, nR - sR)
            R = nR
            Z = a1 + a2
            ZE = e1 + e2
            nNR = tl.maximum(NR, sNR)
            b1, f1 = _rescale(N, NE, nNR - NR)
            b2, f2 = _rescale(sN, sNE, nNR - sNR)
            NR = nNR
            N = b1 + b2
            NE = f1 + f2
            flag = tl.maximum(flag, ((tl.abs(Z) + ZE) >= (1 << 62)).to(tl.int32))
            flag = tl.maximum(flag, tl.max(((tl.abs(N) + NE) >= (1 << 62)).to(tl.int32), axis=1))
        has = R > NEG // 2
        chan = NR > NEG // 2
        zunit = R - FZ
        Zlo = Z
        Zhi = Z + ZE
        Nlo = N
        Nhi = N + NE
        sh = tl.where(chan, (NR - FN) - zunit[:, None], 0)
        lo_den = tl.where(Nlo >= 0, Zhi[:, None], Zlo[:, None])
        hi_den = tl.where(Nhi <= 0, Zhi[:, None], Zlo[:, None])
        lo_den = tl.maximum(lo_den, 1)
        hi_den = tl.maximum(hi_den, 1)
        ql, qel = _ratio_round(tl.abs(Nlo), lo_den, sh, P, QMIN)
        qh, qeh = _ratio_round(tl.abs(Nhi), hi_den, sh, P, QMIN)
        sl = tl.where(Nlo < 0, -1, 1)
        shh = tl.where(Nhi < 0, -1, 1)
        same = ((ql == 0) & (qh == 0)) | ((ql == qh) & (qel == qeh) & (sl == shh)) | (~chan)
        flag = tl.maximum(flag, (has & (tl.min(same.to(tl.int32), axis=1) == 0)).to(tl.int32))
        flag = tl.maximum(flag, (has & (Zlo <= 0)).to(tl.int32))
        # fp32 bits of sl * ql * 2^qel (ql < 2^P, normalized unless qel == QMIN), built exactly.
        q24 = ql << (24 - P)
        qe24 = qel - (24 - P)
        bits = _f32_bits(q24, qe24, Nlo < 0, qe24 > 104)
        bits = tl.where(chan & has[:, None] & (ql != 0), bits, 0)
        if not OUT_FP32:
            bits = (bits >> 16) & 0xFFFF
        tl.store(OUT + rows[:, None] * DV + cc[None, :], bits, mask=wmask[:, None])
        tl.store(OFLAG + rows * tl.num_programs(1) + pid_c, flag, mask=wmask)
        if WRITE_NEED:
            pair = rows * tl.num_programs(1) + pid_c
            nd = ((flag != 0) | (dflag != 0)) & (sflag == 0) & rmask
            tl.store(NEED + pair, tl.where(dflag != 0, 2, 1) * nd.to(tl.int32), mask=rmask)
            slot = tl.atomic_add(CNT + rows * 0, 1, mask=nd)     # one slot per pair, any order
            tl.store(WL + slot, pair, mask=nd)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

def _bad_np(bits: np.ndarray) -> np.ndarray:
    return ((np.asarray(bits, dtype=np.uint16) >> 7) & 0xFF) == 0xFF          # NaN or inf


def _l1_ceil_host(l1s, xp):
    """(ceil(l1s * (1 + 2^-40)) + 1 or 0, too big): the integer upper bound the score error
    bound uses for a key, from its scaled L1 norm l1s = l1 * 2^-lsb (float64). xp is numpy or
    torch. Made once at cache write (DESIGN D32); the same float64 operations the kernel did per
    tile before, so the same integers."""
    big = l1s > 2.0 ** 59
    if xp is np:
        e = np.where(l1s > 0, np.ceil(np.minimum(l1s, 2.0 ** 59) * (1.0 + 2.0 ** -40)) + 1.0, 0.0)
        return e.astype(np.int64), big
    e = torch.where(l1s > 0, torch.ceil(torch.clamp(l1s, max=2.0 ** 59) * (1.0 + 2.0 ** -40)) + 1.0,
                    torch.zeros_like(l1s))
    return e.to(torch.int64), big


def _k_meta_np(X, lsb, l1, lost, bad, dev) -> list:
    """Kernel format for sliced K rows: [X, lsb, kerr, lost | bad << 1 | big << 2]. kerr is an
    integer upper bound on l1 * 2^-lsb, what the score error bound uses (big: too large for it),
    and the NaN/inf bit spares the kernel a bf16 decode of K."""
    lsb = lsb.astype(np.int64)
    with np.errstate(over="ignore", invalid="ignore"):
        l1s = np.where(l1 > 0, l1 * np.exp2(-lsb.astype(np.float64)), 0.0)
    kerr, big = _l1_ceil_host(l1s.astype(np.float64), np)
    w = (lost.any(axis=1).astype(np.int32) | (bad.any(axis=1).astype(np.int32) << 1)
         | (big.astype(np.int32) << 2))
    return [torch.from_numpy(np.ascontiguousarray(X)).to(dev), torch.from_numpy(lsb).to(dev),
            torch.from_numpy(kerr).to(dev), torch.from_numpy(w).to(dev)]


def tc_slices(bits: np.ndarray, S: int, dev) -> list:
    """int8 slices of K rows, as device tensors: [X, lsb, kerr, lost | NaN/inf << 1 | big << 2] (_k_meta_np).
    Done once when the KV cache is written. (q is sliced inside the kernel.)"""
    bad = _bad_np(bits)
    X, lsb, l1, lost = slice_rows(np.where(bad, 0, bits).astype(np.uint16), S)   # bad rows are flagged anyway
    return _k_meta_np(X, lsb, l1, lost, bad, dev)


def _slice_rows_device(bits: torch.Tensor, S: int):
    """slices.slice_rows as torch ops on device-resident rows (int16 view of bf16, (R, d)).
    Returns X (S, R, d) int8, lsb (R,) int64, l1 (R,) float64, lost (R, d) bool. NaN/inf rows
    give arbitrary digits, but the kernel flags those rows from the raw bits anyway."""
    b = bits.to(torch.int32) & 0xFFFF
    ex = (b >> 7) & 0xFF
    frac = b & 0x7F
    normal = ex != 0
    am = torch.where(normal, frac | 0x80, frac).to(torch.int64)
    e = torch.where(normal, ex - 134, torch.full_like(ex, -133)).to(torch.int64)
    neg = (b & 0x8000) != 0
    nz = am != 0
    blen = torch.floor(torch.log2(am.clamp(min=1).to(torch.float64))).to(torch.int64)  # exact: am < 2^8
    lead = torch.where(nz, e + blen, torch.full_like(e, -(1 << 40)))
    E = lead.amax(dim=1)
    lsb = E + 1 - 7 * S
    sh = e - lsb[:, None]
    up = am << sh.clamp(0, 62)
    dn = am >> (-sh).clamp(0, 62)
    fix = torch.where(sh >= 0, up, dn)
    fix = torch.where(-sh > 62, torch.zeros_like(fix), fix)
    lost = nz & (sh < 0) & ((fix << (-sh).clamp(0, 62)) != am)
    X = torch.empty((S,) + tuple(bits.shape), dtype=torch.int8, device=bits.device)
    rem = fix
    for s in range(S):
        dig = rem & 127
        X[s] = torch.where(neg, -dig, dig).to(torch.int8)
        rem = rem >> 7
    l1 = (am.to(torch.float64) * torch.exp2(e.to(torch.float64))).sum(dim=1) * (1 + 2.0 ** -40)
    return X, lsb, l1, lost


def _slice_rows_chunked(bits: torch.Tensor, S: int, chunk_elems: int = 1 << 26):
    """_slice_rows_device in row chunks. Rows are sliced independently, so this is exact; it
    bounds the int64 temporaries (about 10 live, 8 bytes each per element) for whole KV caches.
    A 32 x 8-head x 16K-key cache in one piece needed over 40 GB."""
    R = bits.shape[0]
    step = max(1, chunk_elems // max(1, bits.shape[1]))
    if R <= step:
        return _slice_rows_device(bits, S)
    X = torch.empty((S,) + tuple(bits.shape), dtype=torch.int8, device=bits.device)
    lsb = torch.empty(R, dtype=torch.int64, device=bits.device)
    l1 = torch.empty(R, dtype=torch.float64, device=bits.device)
    lost = torch.empty(tuple(bits.shape), dtype=torch.bool, device=bits.device)
    for a in range(0, R, step):
        X[:, a:a + step], lsb[a:a + step], l1[a:a + step], lost[a:a + step] = _slice_rows_device(bits[a:a + step], S)
    return X, lsb, l1, lost


def _bad_dev(bits: torch.Tensor) -> torch.Tensor:
    return ((bits.to(torch.int32) >> 7) & 0xFF) == 0xFF


def tc_slices_device(bits: torch.Tensor, S: int) -> list:
    """tc_slices for K rows already on the device (cache write without a host round trip)."""
    bad = _bad_dev(bits)
    X, lsb, l1, lost = _slice_rows_chunked(torch.where(bad, torch.zeros_like(bits), bits), S)
    l1s = torch.where(l1 > 0, l1 * torch.exp2(-lsb.clamp(-2000, 2000).to(torch.float64)), torch.zeros_like(l1))
    kerr, big = _l1_ceil_host(l1s, torch)
    w = (lost.any(dim=1).to(torch.int32) | (bad.any(dim=1).to(torch.int32) << 1)
         | (big.to(torch.int32) << 2))
    return [X, lsb, kerr, w]


def tc_v_slices_device(v: torch.Tensor, S: int) -> list:
    """tc_v_slices for V rows already on the device."""
    bad = _bad_dev(v)
    X, lsb, _, lost = _slice_rows_chunked(torch.where(bad, torch.zeros_like(v), v), S)
    w = bad.any(dim=1).to(torch.int8) | (lost.any(dim=1).to(torch.int8) << 1)
    return [X, lsb, lost.to(torch.int8).contiguous(), w.contiguous()]


def tc_v_slices(v_bits: np.ndarray, S: int, dev) -> list:
    """int8 slices of V rows for tensor-core P·V: [X (S, n, dv) int8, lsb (n,) int64,
    lost (n, dv) int8 0/1, per-key flags (n,) int8]. Done once when the KV cache is written.
    The lost plane is an MMA operand as stored. Flag bit 0: the key has a NaN/inf (per key, since
    flags are per row, DESIGN D25). Bit 1: the key has a lost element, so the kernel reads only
    those keys' lost rows (DESIGN D33)."""
    bad = _bad_np(v_bits)
    X, lsb, _, lost = slice_rows(np.where(bad, 0, v_bits).astype(np.uint16), S)
    w = bad.any(axis=1).astype(np.int8) | (lost.any(axis=1).astype(np.int8) << 1)
    return [torch.from_numpy(np.ascontiguousarray(X)).to(dev), torch.from_numpy(lsb.astype(np.int64)).to(dev),
            torch.from_numpy(np.ascontiguousarray(lost.astype(np.int8))).to(dev),
            torch.from_numpy(np.ascontiguousarray(w)).to(dev)]


_EXP2_TABLES: dict = {}
_DEV_CACHE: dict = {}


def _dummies(dev) -> dict:
    """1-element placeholders for kernel arguments a mode does not read, made once per device so
    a step allocates and fills nothing it does not need. Never written."""
    key = ("dummy", str(dev))
    if key not in _DEV_CACHE:
        _DEV_CACHE[key] = {dt: torch.zeros(1, dtype=dt, device=dev)
                           for dt in (torch.int8, torch.int32, torch.int64, torch.float64)}
    return _DEV_CACHE[key]


def _sm_count(dev) -> int:
    """Streaming multiprocessors on a CUDA/ROCm device; 32 for other GPUs (it only sizes tier 2's
    grid), and 2 for the CPU interpreter, so tests run tier 2's work-list loop over several items
    per program."""
    key = ("sms", str(dev))
    if key not in _DEV_CACHE:
        d = torch.device(dev)
        _DEV_CACHE[key] = (torch.cuda.get_device_properties(d).multi_processor_count if d.type == "cuda"
                           else 2 if d.type == "cpu" else 32)
    return _DEV_CACHE[key]


def _arange_dev(n: int, dev) -> torch.Tensor:
    key = ("arange", str(dev), n)
    if key not in _DEV_CACHE:
        _DEV_CACHE[key] = torch.arange(n, dtype=torch.int64, device=dev)
    return _DEV_CACHE[key]


def _exp2_table_dev(dev):
    """exp2.exp2_table() on the device, made once per device (outside any graph capture)."""
    key = str(dev)
    if key not in _EXP2_TABLES:
        _EXP2_TABLES[key] = torch.from_numpy(exp2_table()).to(dev)
    return _EXP2_TABLES[key]


def launch(Qt, Kt, Vt, KL, maxlen: int, alpha, *, out: str = "bf16", splits: int = 4, block_m: int | None = None,
           block_n: int = 16, block_dv: int = 32, merge_order=None, exp2_margin: float = 2.0 ** -26,
           q_tc=None, k_tc=None, tc_fallback: bool = True, v_tc=None, sw: int = 5, pv_tier2: bool = True,
           sq: int = 4, num_warps: int = 4, exp2: str = "int", num_stages: int | None = None,
           maxnreg: int | None = None, timers: bool = False, defer_scores=False, col_fallback: int = 4):
    """Device-level launch with no host transfers and no sync.

    Qt (R, d), Kt (n, d), Vt (n, dv) are int16 views of bf16 on the device; KL (R,) is int32.
    Batched form: Qt (G, R, d), Kt (G, n, d), Vt (G, n, dv), KL (G, R) for G independent
    segments (e.g. sequence x KV head, with a GQA group's query heads as the R rows); the
    tc slices are then made from the flattened (G*R, d) and (G*n, d) rows.
    Pass k_tc (from tc_slices) to use tensor-core scores, with q cut into sq digits inside the
    kernel (q_tc is accepted for compatibility; only its digit count is used), and v_tc (from
    tc_v_slices) for tensor-core P·V. With pv_tier2, (row, channel block) pairs that tensor-core P·V leaves
    uncertified are recomputed on the device with int64 P·V (DESIGN D12). block_m (query rows per
    program, a power of two) defaults to the segment's row count, capped at 16: a decode step with
    2 query heads per KV head then runs 2-row blocks, and the tensor-core paths fill the 16 MMA rows
    with digit slices instead of padding (DESIGN D14). exp2="int" computes weights with integer
    exp2 and float64 only for tiles it cannot certify (DESIGN D15); "f64" uses float64 throughout.
    defer_scores (DESIGN D28) moves the windowed score recompute out of tier 1: a row with an
    uncertified tensor-core score is rerun by tier 2, one row per program; True or "row" reruns
    the whole row, "split" only the KV splits it was deferred in. It needs tensor-core scores and
    P·V with tier 2. A certified result is the same bits; "split" may leave a row flagged that
    "row" would certify (its other splits keep tier 1's tensor-core P·V enclosures).
    col_fallback (DESIGN D30): a tile whose uncertified tensor-core scores sit in at most this many
    key columns recomputes just those keys; more, or 0, recomputes the whole tile. Same bits.
    Returns (OUT int32 (R, dv) or
    (G, R, dv), split_flags, combine_flags), all on the device; flags are per flattened row."""
    batched = Qt.dim() == 3
    G = Qt.shape[0] if batched else 1
    r_seg, d = Qt.shape[-2:]
    n_seg, dv = Vt.shape[-2:]
    assert Kt.shape[-2] == n_seg and KL.numel() == G * r_seg
    Qt, Kt, Vt, KL = Qt.reshape(G * r_seg, d), Kt.reshape(G * n_seg, d), Vt.reshape(G * n_seg, dv), KL.reshape(-1)
    R, n = G * r_seg, G * n_seg
    assert d <= 128 and (d & (d - 1)) == 0, "d must be a power of two <= 128"
    assert dv % block_dv == 0
    if block_m is None:
        block_m = min(16, triton.next_power_of_2(max(1, r_seg)))
    assert block_m >= 1 and (block_m & (block_m - 1)) == 0, "block_m must be a power of two"
    pack = max(1, 16 // block_m)   # digit slots per MMA: PACK * block_m >= 16 rows
    assert exp2 in ("int", "f64")
    exp2t = _exp2_table_dev(Qt.device)
    dev = Qt.device
    key_log2 = max(1, int(np.ceil(np.log2(max(2, maxlen)))))
    fz = fn = 64 - 3 - key_log2
    if v_tc is not None:
        # P·V contracts over the key tile, and int8 MMAs on NVIDIA need K >= 32 (mma.m16n8k32).
        # The CPU interpreter does not enforce this. The tile size cannot change the output
        # (SPEC §5.3), so raising it is free.
        block_n = max(block_n, 32)
    split_len = max(block_n, -(-max(maxlen, 1) // splits))
    split_len = -(-split_len // block_n) * block_n
    nsplit = max(1, -(-max(maxlen, 1) // split_len))
    # Tier 2 (DESIGN D23) reruns one row x one 32-channel block per work item with int64 P·V. A
    # rerun has few programs to hide latency with, so it takes 64-key tiles at 8 warps: a 16-key
    # tile spent half its instructions on per-tile tensor-core score overhead. Tiles must not
    # cross a split, and any tile size gives the same bits (SPEC §5.3).
    block_n_t2 = 64 if split_len % 64 == 0 else 32 if split_len % 32 == 0 else 16
    block_m_t2 = 1
    am, ae = decode_fp32(np.float32(alpha))
    dz = _dummies(dev)
    use_tc = k_tc is not None
    if use_tc:
        tc = list(k_tc)
        sk = k_tc[0].shape[0]
        sq = q_tc[0].shape[0] if q_tc is not None else sq
        assert 1 <= sq <= 4
        # Each fixed-point element is below 2^(7 sq) or 2^(7 sk), so |acc| < d * 2^(7(sq + sk))
        # must fit int64. sq = 4 keeps 28 bits of q below each row's largest element: with 21,
        # 1.6% of normal q rows lost bits, which leaves their scores uncertified (DESIGN D18).
        assert 7 * (sq + sk) + int(np.ceil(np.log2(d))) <= 63, f"q/k slices too wide for int64: sq={sq}, sk={sk}"
    else:
        tc = [dz[torch.int8], dz[torch.int64], dz[torch.float64], dz[torch.int32]]
        sq, sk = 1, 1
    use_tc_pv = v_tc is not None
    if use_tc_pv:
        vtc, sv = list(v_tc), v_tc[0].shape[0]
    else:
        vtc, sv, sw = [dz[torch.int8], dz[torch.int64], dz[torch.int8], dz[torch.int8]], 1, 1
    anch = 7 * sw + 7 * sv + int(np.ceil(np.log2(block_n)))
    if use_tc_pv:
        # The digit pairs of a tile combine to sum_keys Wf * Vfix with Wf < 2^(7 sw) and
        # |Vfix| < 2^(7 sv), so |acc| < block_n * 2^(7(sw + sv)) <= 2^anch. The error count is
        # e < block_n * (2^(7 sw) + 2^(7 sv) + 1) < 2^(anch - 5), so acc +- e and 2e fit int64 when
        # anch <= 62 (DESIGN D18). That allows 64-key tiles at sw = 5, sv = 3.
        assert anch <= 62, f"P·V slices too wide for int64: sw={sw}, sv={sv}, block_n={block_n}"
    i64 = dict(dtype=torch.int64, device=dev)
    PR_ = torch.empty((nsplit, R), **i64)
    PZ_ = torch.empty((nsplit, R), **i64)
    PZE_ = torch.empty((nsplit, R), **i64)
    PNR_ = torch.empty((nsplit, R, dv), **i64)
    PN_ = torch.empty((nsplit, R, dv), **i64)
    PNE_ = torch.empty((nsplit, R, dv), **i64)
    ncb = dv // block_dv
    FL = torch.empty((nsplit, R, ncb), dtype=torch.int32, device=dev)   # every entry is written
    z32 = dz[torch.int32]

    # Software-pipelining depth of the split kernel's key loop. Each stage buffers a tile's operands
    # in shared memory, so 64-key tensor-core tiles may need fewer than Triton's default.
    # Triton's default of 3 makes 64-key tensor-core tiles need 120-160 KB, over sm_120's 99 KB.
    if num_stages is None and use_tc_pv and block_n >= 64:
        num_stages = 2
    stages = {} if num_stages is None else {"num_stages": num_stages}
    if maxnreg is not None:
        # A register cap for the tier-1 split kernel: 128 lets two 8-warp programs share an SM
        # (more latency hiding, more spills). A tiling choice, so it cannot change bits.
        stages["maxnreg"] = maxnreg

    tier2 = use_tc_pv and pv_tier2
    # Combine channel blocks (at most 32): NEED, OF and tier 2's work items use them (DESIGN D19).
    cbdv = min(block_dv, 32)
    nccb = dv // cbdv
    # Tier 2's work list: CNT items in WL, each a flattened (row, combine channel block) pair.
    CNT = torch.empty(1, dtype=torch.int32, device=dev) if tier2 else z32
    WL = torch.empty(R * nccb, dtype=torch.int32, device=dev) if tier2 else z32
    # Deferred score recompute (DESIGN D28): DFL marks, per split and row, a row tier 1 left to tier 2.
    assert defer_scores in (False, True, "row", "split"), defer_scores
    defer = bool(defer_scores and tier2 and use_tc and tc_fallback)
    defer_split = defer and defer_scores == "split"
    DFL = torch.empty((nsplit, R), dtype=torch.int32, device=dev) if defer else z32   # all written

    # Diagnostic phase clocks of the tier-1 split kernel (DESIGN D26): 24 int64 per program.
    TMB = (torch.zeros((G * triton.cdiv(r_seg, block_m) * nsplit * (dv // block_dv), 24), dtype=torch.int64,
                       device=dev) if timers else None)

    def split(rerun, bn, pv, v_args, sw_, sv_, anch_, bm, bdv, nw, grid0, stg):
        nrb_ = triton.cdiv(r_seg, bm)
        # Windowed-score channel chunk (DESIGN D17): about 16 int64 products per thread live at once.
        wdc = max(1, min(d, (512 * nw) // (bm * bn)))
        cdc = max(1, min(d, (512 * nw) // bm))          # the same for one key column (DESIGN D30)
        grid = (grid0 or G * nrb_, nsplit, 1 if rerun else dv // bdv)
        tm = TMB if (timers and not rerun) else dz[torch.int64]
        return _ea_split_kernel[grid](
            Qt, Kt, Vt, KL, R, am, ae, PR_, PZ_, PZE_, PNR_, PN_, PNE_, FL,
            split_len, fz, fn, *tc, n, *v_args, r_seg, n_seg, nrb_, ncb, CNT, WL, nccb, tm, DFL,
            NEED if (rerun and defer_split) else z32, D=d, DV=dv,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_DV=bdv, FS=SCORE_FS,
            EXP2_MARGIN=exp2_margin, USE_TC=use_tc, SQ=sq, SK=sk, TC_FALLBACK=tc_fallback and (rerun or not defer),
            USE_TC_PV=pv, SW=sw_, SV=sv_, ANCH=anch_, RERUN=rerun, PACK=max(1, 16 // bm),
            EXP2T=exp2t, EXP2_INT=(exp2 == "int"), WDC=wdc, FL_DIV=block_dv // bdv,
            RESET_CNT=tier2 and not rerun, TIMERS=bool(timers and not rerun), DEFER=defer,
            DEFER_SPLIT=bool(rerun and defer_split), COLMAX=int(col_fallback), CDC=cdc, num_warps=nw, **stg)

    ck = split(False, block_n, use_tc_pv, vtc, sw, sv, anch, block_m, block_dv, num_warps, 0, stages)
    # Registers and spills of the compiled split kernel (None under the CPU interpreter).
    _LAST_DEBUG.update(n_regs=getattr(ck, "n_regs", None), n_spills=getattr(ck, "n_spills", None))
    if merge_order is None:
        ORD = _arange_dev(nsplit, dev)   # cached: no host copy and no launch per step
    else:
        # Keep the caller's relative order for splits that exist, then append any missing ones,
        # so the combine kernel always gets a permutation of range(nsplit).
        given = [int(i) for i in merge_order if 0 <= int(i) < nsplit]
        order = np.array(given + [i for i in range(nsplit) if i not in given], dtype=np.int64)
        assert sorted(order.tolist()) == list(range(nsplit))
        ORD = torch.from_numpy(order.astype(np.int64)).to(dev)
    OUT = torch.empty((R, dv), dtype=torch.int32, device=dev)
    OF = torch.empty((R, nccb), dtype=torch.int32, device=dev)   # every entry is written
    p, qmin = (8, -133) if out == "bf16" else (24, -149)

    # Combine rows per program (it works on flattened rows, not segments). 4 rows x 32 channels is
    # one output per thread at 4 warps: the merge and the rounding are latency-bound chains, so
    # more programs with shorter per-thread work beat fewer, wider ones (DESIGN D22).
    cbm = 4

    def combine(need, rerun, write_need):
        _ea_combine_kernel[(triton.cdiv(R, cbm), nccb)](PR_, PZ_, PZE_, PNR_, PN_, PNE_, ORD, nsplit, R, fz, fn,
                                                        OUT, OF, need, FL, ncb, CNT, WL, DFL, DV=dv, BLOCK_M=cbm,
                                                        BLOCK_DV=cbdv,
                                                        P=p, QMIN=qmin, OUT_FP32=(out == "fp32"),
                                                        RERUN=rerun, WRITE_NEED=write_need, FL_DIV=block_dv // cbdv,
                                                        DEFER=defer,
                                                        num_warps=4)

    NEED = torch.empty((R, nccb), dtype=torch.int32, device=dev) if tier2 else None
    combine(NEED if tier2 else z32, False, tier2)
    if tier2:
        # Tier 2 (DESIGN D12): a pair whose final rounding failed, with no split-level flag (bad
        # input, uncertified logit or weight, overflow: int64 P·V cannot fix those), gets its N
        # recomputed with int64 P·V and is combined again. Every path encloses the same exact N,
        # so a certified result is the same bits whichever tier produced it. Fixed grids and no
        # host sync: it costs two near-empty launches when nothing is flagged. The tier-1 combine
        # wrote NEED = (combine flag) & (no split flag) and the work list, so tier 2 adds no other
        # device work.
        assert split_len % block_n_t2 == 0
        # Tier 2's split grid is (programs, KV splits) whatever the batch, about two waves, and
        # each program strides over the work list (DESIGN D23). A grid over every row block would
        # be mostly empty programs, and at large batch their launch cost would exceed the work.
        progs = max(1, min(R * nccb, -(-2 * _sm_count(dev) // nsplit)))
        split(True, block_n_t2, False, [dz[torch.int8], dz[torch.int64], dz[torch.int8], dz[torch.int8]], 1, 1, 0,
              block_m_t2, cbdv, 8, progs, {"num_stages": 2})
        combine(NEED, True, False)
    _LAST_DEBUG.update(timers=TMB, nsplit=nsplit, defer_flags=DFL if defer else None)
    _LAST_DEBUG.update(block_m=block_m, block_m_t2=block_m_t2, pack=pack, split_flags=FL, combine_flags=OF, tier2_need=NEED, tier2_cnt=CNT, tier2_wl=WL, PR=PR_, PZ=PZ_, PZE=PZE_, PNR=PNR_, PN=PN_, PNE=PNE_,
                       fz=fz, fn=fn)
    return (OUT.view(G, r_seg, dv) if batched else OUT), FL, OF


def attention_triton(q_bits, k_bits, v_bits, alpha, kv_len, out: str = "bf16", splits: int = 4,
                     block_m: int | None = None, block_n: int = 16, block_dv: int | None = None,
                     merge_order=None, device: str = "cpu", fallback: bool = True,
                     exp2_margin: float = 2.0 ** -26, use_tc: bool = False, sq: int = 4, sk: int = 4,
                     k_slices=None, tc_fallback: bool = True, use_tc_pv: bool = False, sw: int = 5,
                     sv: int = 3, pv_tier2: bool = True, exp2: str = "int", defer_scores=False,
                     col_fallback: int = 4):
    """numpy-level API. q: (R, d), k: (n, d), v: (n, dv) bf16 bits (uint16). Row r attends keys
    [0, kv_len[r]): causal, decode, prefill and chunked prefill all have this shape.
    Returns (out, flags); flagged rows are recomputed exactly when fallback=True."""
    dev = torch.device(device)
    dv = v_bits.shape[1]
    block_dv = block_dv or min(dv, 32)
    kv_len = np.asarray(kv_len, dtype=np.int32)

    def t16(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint16).view(np.int16)).to(dev)

    k_tc = None
    if use_tc:
        if k_slices is not None:
            kX, klsb, kl1, klost = k_slices
            k_tc = _k_meta_np(kX, klsb, kl1, klost, _bad_np(k_bits), dev)   # caller-made slices
        else:
            k_tc = tc_slices(k_bits, sk, dev)
    OUT, FL, OF = launch(t16(q_bits), t16(k_bits), t16(v_bits), torch.from_numpy(kv_len).to(dev),
                         int(kv_len.max(initial=0)), alpha, out=out, splits=splits, block_m=block_m,
                         block_n=block_n, block_dv=block_dv, merge_order=merge_order, exp2_margin=exp2_margin,
                         sq=sq, k_tc=k_tc, tc_fallback=tc_fallback,
                         v_tc=tc_v_slices(v_bits, sv, dev) if use_tc_pv else None, sw=sw, pv_tier2=pv_tier2,
                         exp2=exp2, defer_scores=defer_scores, col_fallback=col_fallback)
    flags = np.maximum(FL.amax(dim=(0, 2)).cpu().numpy(), OF.amax(dim=1).cpu().numpy()).astype(np.uint8)
    o = OUT.cpu().numpy()
    res = (o.astype(np.uint32).view(np.float32) if out == "fp32" else (o & 0xFFFF).astype(np.uint16)).copy()
    if fallback:
        for r in np.flatnonzero(flags):
            L = int(kv_len[r])
            res[r] = attention_row(q_bits[r], k_bits[:L], v_bits[:L], alpha, out)
    return res, flags