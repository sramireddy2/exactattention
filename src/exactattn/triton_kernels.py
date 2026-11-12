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
torch_impl.py (DESIGN.md), with one change: the exp2 margin check is tighter (2^-26) and
assumes the device's float64 exp2 has relative error below 2^-50. That holds for CUDA
libdevice (1 ulp), Intel SYCL (<= 4 ulp) and numpy. DESIGN Q4 tracks replacing it with our
own exp2.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .formats import decode_fp32
from .reference import attention_row
from .slices import slice_rows

NEG = tl.constexpr(-(1 << 60))  # "empty" sentinel for reference exponents
_LAST_DEBUG: dict = {}  # last call's intermediate buffers, for debugging
SCORE_FS = 63 - 1 - 7  # d <= 128


# ---------------------------------------------------------------------------
# Integer helpers (exact, device-independent)
# ---------------------------------------------------------------------------

@triton.jit
def _bitlen(x):
    """Exact bit length of int64 x >= 0 (0 -> 0)."""
    f = x.to(tl.float64)
    e = ((f.to(tl.int64, bitcast=True) >> 52) & 0x7FF) - 1022
    b = tl.where(x > 0, e, 0)
    sb = tl.minimum(tl.maximum(b - 1, 0), 62)
    b = tl.where((x > 0) & ((x >> sb) == 0), b - 1, b)
    return b


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
    """[acc, acc + err] re-expressed in units 2^s larger (s >= 0)."""
    v, lost = _floor_shift(acc, -s)
    sc = tl.minimum(tl.maximum(s, 0), 62)
    mask = (tl.full(err.shape, 1, tl.int64) << sc) - 1
    e = tl.where(s >= 63, (err > 0).to(tl.int64), (err >> sc) + ((err & mask) != 0).to(tl.int64))
    return v, e + lost.to(tl.int64)


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
    t = (q.to(tl.float64) * tl.exp2(qe.to(tl.float64))).to(tl.float32) * sgn.to(tl.float32)
    return t, overflow


@triton.jit
def _ratio_round(a, b, sh, P: tl.constexpr, QMIN: tl.constexpr, N_ITER: tl.constexpr):
    """RN(a / b * 2^sh), a >= 0, 0 < b < 2^62. Returns (q, qe)."""
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
    q_pos = ip
    rr = r
    for i in tl.static_range(N_ITER):
        act = s2 > i
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
def _windowed_logits(qm, qe, km, ke, AM, AE, FS: tl.constexpr):
    """Certified t = RN32(alpha * q.k) from a windowed int64 dot product. Returns (t, uncertified)."""
    P = qm[:, None, :] * km[None, :, :]                               # (M, N, D)
    E = qe[:, None, :] + ke[None, :, :]
    nzp = P != 0
    lead = tl.where(nzp, E + _bitlen(tl.abs(P)) - 1, NEG)
    Lmax = tl.max(lead, axis=2)                                        # (M, N)
    base = Lmax + 1 - FS
    term, lost = _floor_shift(P, E - base[:, :, None])
    acc = tl.sum(tl.where(nzp, term, 0), axis=2)
    err = tl.sum((lost & nzp).to(tl.int64), axis=2)
    anyp = tl.max(nzp.to(tl.int32), axis=2) != 0
    big = (tl.abs(acc) + err) >= (1 << 62)
    t_lo, ov1 = _rn32_product(AM, AE, acc, base)
    t_hi, ov2 = _rn32_product(AM, AE, acc + err, base)
    t = tl.where(anyp, t_lo, 0.0)
    unc = anyp & ((t_lo != t_hi) | ov1 | ov2 | big)
    return t, unc


@triton.jit
def _tc_logits(QX, KX, rows, rmask, nn, nmask, ROWS, NKEYS, QLSB, QL1, QLOST, KLSB, KL1, KLOST,
               AM, AE, D: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr):
    """Certified t = RN32(alpha * q.k) from int8 slices (slices.py) on tensor cores.

    q.k = 2^(qlsb + klsb) * (sum_{s,t} 2^(7(s+t)) Q[s].K[t] + truncation), where each Q[s].K[t] is
    an exact int8 x int8 -> int32 MMA. The truncation bound comes from per-row L1 norms."""
    dd = tl.arange(0, D)
    acc = tl.zeros((rows.shape[0], nn.shape[0]), tl.int64)
    for t_ in tl.static_range(SK):
        kt = tl.load(KX + (t_ * NKEYS + nn[:, None]) * D + dd[None, :], mask=nmask[:, None], other=0)
        for s_ in tl.static_range(SQ):
            qs = tl.load(QX + (s_ * ROWS + rows[:, None]) * D + dd[None, :], mask=rmask[:, None], other=0)
            dot = tl.dot(qs, tl.trans(kt), out_dtype=tl.int32)            # (M, N), exact
            acc += dot.to(tl.int64) << (7 * (s_ + t_))
    qlsb = tl.load(QLSB + rows, mask=rmask, other=0)
    ql1 = tl.load(QL1 + rows, mask=rmask, other=0.0)
    qlost = tl.load(QLOST + rows, mask=rmask, other=0) != 0
    klsb = tl.load(KLSB + nn, mask=nmask, other=0)
    kl1 = tl.load(KL1 + nn, mask=nmask, other=0.0)
    klost = tl.load(KLOST + nn, mask=nmask, other=0) != 0
    errf = tl.where(qlost[:, None], kl1[None, :] * tl.exp2(-klsb[None, :].to(tl.float64)), 0.0) \
        + tl.where(klost[None, :], ql1[:, None] * tl.exp2(-qlsb[:, None].to(tl.float64)), 0.0)
    big = errf > 2.0 ** 60
    err = tl.where(errf > 0, tl.ceil(tl.minimum(errf, 2.0 ** 60) * (1.0 + 2.0 ** -40)) + 1.0, 0.0).to(tl.int64)
    base = qlsb[:, None] + klsb[None, :]
    big = big | ((tl.abs(acc) + err) >= (1 << 62))
    t_lo, ov1 = _rn32_product(AM, AE, acc - err, base)
    t_hi, ov2 = _rn32_product(AM, AE, acc + err, base)
    unc = (t_lo != t_hi) | ov1 | ov2 | big
    return t_lo, unc


@triton.jit
def _tc_pv(Mw, Ew, vis, VX, VLSB, VLOST, nn, nmask, cc, NKEYS, FN, DV: tl.constexpr, SW: tl.constexpr,
           SV: tl.constexpr, ANCH: tl.constexpr):
    """One tile's P·V on tensor cores (DESIGN D9/D10). Returns (TNR, TN, TNE) in the partial format:
    value N_c lies in [TN, TN + TNE] * 2^(TNR - FN).

    V rows are block fixed point (per-key exponent vlsb, SV signed 7-bit digits, a per-element
    lost-bits mask), made once at cache write. Weights times 2^vlsb are anchored per row at the
    tile's largest term and truncated to SW unsigned 7-bit digits. Each digit pair is an exact
    int8 MMA, and two more small MMAs bound the truncation per element:
      weight loss  sum_j lostW_mj * |Vfix_jc|       (lost-weight mask x |V| digits)
      V loss       sum_j (W_mj + 1) * vlost_jc      (weight digits x lost-V mask)."""
    vlsb = tl.load(VLSB + nn, mask=nmask, other=0)
    G = Ew + vlsb[None, :]                                                 # term = M * Vfix * 2^G
    gtop = tl.max(tl.where(vis, G + 24, NEG), axis=1)                      # (M,)
    base = gtop - 7 * SW
    Wf, lw = _floor_shift(Mw, G - base[:, None])                           # Wf < 2^(7*SW)
    Wf = tl.where(vis, Wf, 0)
    lw = lw & vis
    acc = tl.zeros((Mw.shape[0], cc.shape[0]), tl.int64)
    e = tl.zeros((Mw.shape[0], cc.shape[0]), tl.int64)
    nz = tl.zeros((nn.shape[0], cc.shape[0]), tl.int32)
    for b in tl.static_range(SV):
        vb = tl.load(VX + (b * NKEYS + nn[:, None]) * DV + cc[None, :], mask=nmask[:, None], other=0)
        nz = nz | (vb != 0).to(tl.int32)
        e += tl.dot(lw.to(tl.int8), tl.abs(vb), out_dtype=tl.int32).to(tl.int64) << (7 * b)
        for s in tl.static_range(SW):
            ws = ((Wf >> (7 * s)) & 127).to(tl.int8)
            acc += tl.dot(ws, vb, out_dtype=tl.int32).to(tl.int64) << (7 * (s + b))
    vl = tl.load(VLOST + nn[:, None] * DV + cc[None, :], mask=nmask[:, None], other=0)
    for s in tl.static_range(SW):
        ws = ((Wf >> (7 * s)) & 127).to(tl.int8)
        e += tl.dot(ws, vl, out_dtype=tl.int32).to(tl.int64) << (7 * s)
    visb = vis.to(tl.int8)
    e += tl.dot(visb, vl, out_dtype=tl.int32).to(tl.int64)
    chan = tl.dot(visb, nz.to(tl.int8), out_dtype=tl.int32) > 0           # channel has any term
    # True N in [acc - e, acc + e] * 2^base, with |acc| < 2^ANCH (ANCH = 7*SW + 7*SV + log2 BLOCK_N).
    # Re-anchor at TNR = base + ANCH, i.e. units 2^(base + ANCH - FN): a right shift by
    # sr = ANCH - FN (counted truncation) or an exact left shift.
    lo = acc - e
    wid = 2 * e
    sr = ANCH - FN
    TN_r, TNE_r = _rescale(lo, wid, tl.full(lo.shape, 0, tl.int64) + tl.maximum(sr, 0))
    sl = tl.minimum(tl.maximum(-sr, 0), 62)
    TN = tl.where(sr >= 0, TN_r, lo << sl)
    TNE = tl.where(sr >= 0, TNE_r, wid << sl)
    TNR = tl.where(chan, base[:, None] + ANCH, NEG)
    TN = tl.where(chan, TN, 0)
    TNE = tl.where(chan, TNE, 0)
    return TNR, TN, TNE


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _ea_split_kernel(Q, K, V, KVLEN, ROWS, AM, AE,
                     PR, PZ, PZE, PNR, PN, PNE, FLAG,
                     SPLIT_LEN, FZ, FN,
                     QX, KX, QLSB, QL1, QLOST, KLSB, KL1, KLOST, NKEYS, VX, VLSB, VLOST,
                     D: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_DV: tl.constexpr, FS: tl.constexpr, EXP2_MARGIN: tl.constexpr,
                     USE_TC: tl.constexpr, SQ: tl.constexpr, SK: tl.constexpr, TC_FALLBACK: tl.constexpr,
                     USE_TC_PV: tl.constexpr, SW: tl.constexpr, SV: tl.constexpr, ANCH: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_c = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < ROWS
    kvl = tl.load(KVLEN + rows, mask=rmask, other=0).to(tl.int64)
    dd = tl.arange(0, D)
    cc = pid_c * BLOCK_DV + tl.arange(0, BLOCK_DV)
    qb = tl.load(Q + rows[:, None] * D + dd[None, :], mask=rmask[:, None], other=0)
    qm, qe, qbad = _decode_bf16(qb)
    flag = tl.max(qbad.to(tl.int32), axis=1)
    PRr = tl.full((BLOCK_M,), NEG, tl.int64)
    PZr = tl.zeros((BLOCK_M,), tl.int64)
    PZEr = tl.zeros((BLOCK_M,), tl.int64)
    PNRr = tl.full((BLOCK_M, BLOCK_DV), NEG, tl.int64)
    PNr = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
    PNEr = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
    k0 = pid_s * SPLIT_LEN
    kmax = tl.max(kvl, axis=0)
    k1 = tl.minimum(k0 + SPLIT_LEN, kmax)
    for n0 in range(k0, k1, BLOCK_N):
        nn = n0 + tl.arange(0, BLOCK_N)
        vis = (nn[None, :] < kvl[:, None]) & rmask[:, None]              # (M, N)
        kb = tl.load(K + nn[:, None] * D + dd[None, :], mask=(nn < kmax)[:, None], other=0)
        km, ke, kbad = _decode_bf16(kb)                                    # (N, D)
        flag = tl.maximum(flag, tl.max((vis & (tl.max(kbad.to(tl.int32), axis=1) != 0)[None, :]).to(tl.int32), axis=1))
        # ---- scores ----
        if USE_TC:
            # Tensor cores: int8 slice products, certified; windowed recompute only for
            # tiles that contain an uncertified score.
            t, unc = _tc_logits(QX, KX, rows, rmask, nn, nn < kmax, ROWS, NKEYS, QLSB, QL1, QLOST,
                                KLSB, KL1, KLOST, AM, AE, D, SQ, SK)
            if TC_FALLBACK and tl.max(tl.max((unc & vis).to(tl.int32), axis=1), axis=0) > 0:
                t_w, unc_w = _windowed_logits(qm, qe, km, ke, AM, AE, FS)
                t = tl.where(unc, t_w, t)
                unc = unc & unc_w
        else:
            t, unc = _windowed_logits(qm, qe, km, ke, AM, AE, FS)
        # ---- weights ----
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
        unc = unc | risky | (tl.abs(td) >= 1073741824.0)
        flag = tl.maximum(flag, tl.max((vis & unc).to(tl.int32), axis=1))
        # ---- tile partial ----
        top = tl.where(vis, Ew + 23, NEG)
        TR = tl.max(top, axis=1)                                           # (M,)
        zt, zl = _floor_shift(Mw, Ew - (TR[:, None] - FZ))
        TZ = tl.sum(tl.where(vis, zt, 0), axis=1)
        TZE = tl.sum((zl & vis).to(tl.int64), axis=1)
        vb = tl.load(V + nn[:, None] * DV + cc[None, :], mask=(nn < kmax)[:, None], other=0)
        vm, ve, vbad = _decode_bf16(vb)                                    # (N, BDV)
        flag = tl.maximum(flag, tl.max((vis & (tl.max(vbad.to(tl.int32), axis=1) != 0)[None, :]).to(tl.int32), axis=1))
        if USE_TC_PV:
            TNR, TN, TNE = _tc_pv(Mw, Ew, vis, VX, VLSB, VLOST, nn, nn < kmax, cc, NKEYS, FN, DV, SW, SV, ANCH)
        else:
            lv = tl.where(vm != 0, ve + _bitlen(tl.abs(vm)) - 1, NEG)
            valid = vis[:, :, None] & (vm != 0)[None, :, :]                # (M, N, BDV)
            L = tl.where(valid, top[:, :, None] + 1 + lv[None, :, :], NEG)
            TNR = tl.max(L, axis=1)                                        # (M, BDV)
            nt, nl = _floor_shift(Mw[:, :, None] * vm[None, :, :],
                                  Ew[:, :, None] + ve[None, :, :] - (TNR[:, None, :] - FN))
            TN = tl.sum(tl.where(valid, nt, 0), axis=1)
            TNE = tl.sum((nl & valid).to(tl.int64), axis=1)
        # ---- merge into the running partial ----
        nR = tl.maximum(PRr, TR)
        a1, e1 = _rescale(PZr, PZEr, nR - PRr)
        a2, e2 = _rescale(TZ, TZE, nR - TR)
        PRr = nR
        PZr = a1 + a2
        PZEr = e1 + e2
        nNR = tl.maximum(PNRr, TNR)
        b1, f1 = _rescale(PNr, PNEr, nNR - PNRr)
        b2, f2 = _rescale(TN, TNE, nNR - TNR)
        PNRr = nNR
        PNr = b1 + b2
        PNEr = f1 + f2
        flag = tl.maximum(flag, ((tl.abs(PZr) + PZEr) >= (1 << 62)).to(tl.int32))
        flag = tl.maximum(flag, tl.max(((tl.abs(PNr) + PNEr) >= (1 << 62)).to(tl.int32), axis=1))
    off = pid_s * ROWS + rows
    zmask = rmask & ((rows * 0 + pid_c) == 0)  # only channel block 0 writes Z
    tl.store(PR + off, PRr, mask=zmask)
    tl.store(PZ + off, PZr, mask=zmask)
    tl.store(PZE + off, PZEr, mask=zmask)
    o2 = off[:, None] * DV + cc[None, :]
    tl.store(PNR + o2, PNRr, mask=rmask[:, None])
    tl.store(PN + o2, PNr, mask=rmask[:, None])
    tl.store(PNE + o2, PNEr, mask=rmask[:, None])
    tl.store(FLAG + (pid_s * ROWS + rows) * tl.num_programs(2) + pid_c, flag, mask=rmask)


@triton.jit
def _ea_combine_kernel(PR, PZ, PZE, PNR, PN, PNE, ORDER, NSPLIT, ROWS, FZ, FN, OUT, OFLAG,
                       DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
                       P: tl.constexpr, QMIN: tl.constexpr, N_ITER: tl.constexpr, OUT_FP32: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < ROWS
    cc = pid_c * BLOCK_DV + tl.arange(0, BLOCK_DV)
    R = tl.full((BLOCK_M,), NEG, tl.int64)
    Z = tl.zeros((BLOCK_M,), tl.int64)
    ZE = tl.zeros((BLOCK_M,), tl.int64)
    NR = tl.full((BLOCK_M, BLOCK_DV), NEG, tl.int64)
    N = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
    NE = tl.zeros((BLOCK_M, BLOCK_DV), tl.int64)
    flag = tl.zeros((BLOCK_M,), tl.int32)
    for i in range(NSPLIT):
        s = tl.load(ORDER + i)
        off = s * ROWS + rows
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
    ql, qel = _ratio_round(tl.abs(Nlo), lo_den, sh, P, QMIN, N_ITER)
    qh, qeh = _ratio_round(tl.abs(Nhi), hi_den, sh, P, QMIN, N_ITER)
    sl = tl.where(Nlo < 0, -1, 1)
    shh = tl.where(Nhi < 0, -1, 1)
    same = ((ql == 0) & (qh == 0)) | ((ql == qh) & (qel == qeh) & (sl == shh)) | (~chan)
    flag = tl.maximum(flag, (has & (tl.min(same.to(tl.int32), axis=1) == 0)).to(tl.int32))
    flag = tl.maximum(flag, (has & (Zlo <= 0)).to(tl.int32))
    val = (ql.to(tl.float64) * tl.exp2(qel.to(tl.float64))) * sl.to(tl.float64)
    val = tl.where(chan & has[:, None] & (ql != 0), val, 0.0).to(tl.float32)
    bits = val.to(tl.int32, bitcast=True)
    if not OUT_FP32:
        bits = (bits >> 16) & 0xFFFF
    tl.store(OUT + rows[:, None] * DV + cc[None, :], bits, mask=rmask[:, None])
    tl.store(OFLAG + rows * tl.num_programs(1) + pid_c, flag, mask=rmask)


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------

def tc_slices(bits: np.ndarray, S: int, dev) -> list:
    """int8 slices of q or K rows, as device tensors: [X, lsb, l1, lost]. For K, do this once when
    the KV cache is written."""
    X, lsb, l1, lost = slice_rows(bits, S)
    return [torch.from_numpy(np.ascontiguousarray(X)).to(dev), torch.from_numpy(lsb.astype(np.int64)).to(dev),
            torch.from_numpy(l1.astype(np.float64)).to(dev),
            torch.from_numpy(lost.any(axis=1).astype(np.int32)).to(dev)]


def tc_v_slices(v_bits: np.ndarray, S: int, dev) -> list:
    """int8 slices of V rows for tensor-core P·V: [X (S, n, dv) int8, lsb (n,) int64, lost (n, dv) int8].
    Done once when the KV cache is written."""
    X, lsb, _, lost = slice_rows(v_bits, S)
    return [torch.from_numpy(np.ascontiguousarray(X)).to(dev), torch.from_numpy(lsb.astype(np.int64)).to(dev),
            torch.from_numpy(np.ascontiguousarray(lost.astype(np.int8))).to(dev)]


def launch(Qt, Kt, Vt, KL, maxlen: int, alpha, *, out: str = "bf16", splits: int = 4, block_m: int = 16,
           block_n: int = 16, block_dv: int = 32, merge_order=None, exp2_margin: float = 2.0 ** -26,
           q_tc=None, k_tc=None, tc_fallback: bool = True, v_tc=None, sw: int = 5):
    """Device-level launch with no host transfers and no sync.

    Qt (R, d), Kt (n, d), Vt (n, dv) are int16 views of bf16 on the device; KL (R,) is int32.
    Pass q_tc/k_tc (from tc_slices) to use tensor-core scores. Returns (OUT int32 (R, dv),
    split_flags, combine_flags), all on the device."""
    R, d = Qt.shape
    n, dv = Vt.shape
    assert d <= 128 and (d & (d - 1)) == 0, "d must be a power of two <= 128"
    assert dv % block_dv == 0
    dev = Qt.device
    key_log2 = max(1, int(np.ceil(np.log2(max(2, maxlen)))))
    fz = fn = 64 - 3 - key_log2
    if v_tc is not None:
        # P·V contracts over the key tile, and int8 MMAs on NVIDIA need K >= 32 (mma.m16n8k32).
        # The CPU interpreter does not enforce this. The tile size cannot change the output
        # (SPEC §5.3), so raising it is free.
        block_n = max(block_n, 32)
    split_len = max(block_n,-(-max(maxlen, 1) // splits))
    split_len = -(-split_len // block_n) * block_n
    nsplit = max(1, -(-max(maxlen, 1) // split_len))
    am, ae = decode_fp32(np.float32(alpha))
    use_tc = q_tc is not None
    if use_tc:
        tc = [q_tc[0], k_tc[0], q_tc[1], q_tc[2], q_tc[3], k_tc[1], k_tc[2], k_tc[3]]
        sq, sk = q_tc[0].shape[0], k_tc[0].shape[0]
    else:
        z8 = torch.zeros(1, dtype=torch.int8, device=dev)
        z64 = torch.zeros(1, dtype=torch.int64, device=dev)
        zf = torch.zeros(1, dtype=torch.float64, device=dev)
        z32 = torch.zeros(1, dtype=torch.int32, device=dev)
        tc, sq, sk = [z8, z8, z64, zf, z32, z64, zf, z32], 1, 1
    use_tc_pv = v_tc is not None
    if use_tc_pv:
        vtc, sv = list(v_tc), v_tc[0].shape[0]
    else:
        z8 = torch.zeros(1, dtype=torch.int8, device=dev)
        vtc, sv, sw = [z8, torch.zeros(1, dtype=torch.int64, device=dev), z8], 1, 1
    anch = 7 * sw + 7 * sv + int(np.ceil(np.log2(block_n)))
    if use_tc_pv:
        # Largest tile partial: block_n * 127 * 127 * 2^(7(sw+sv-2)) (all digit pairs combined) must fit int64.
        assert anch + 1 < 63, f"P·V slices too wide for int64: sw={sw}, sv={sv}, block_n={block_n}"
    i64 = dict(dtype=torch.int64, device=dev)
    PR_ = torch.empty((nsplit, R), **i64)
    PZ_ = torch.empty((nsplit, R), **i64)
    PZE_ = torch.empty((nsplit, R), **i64)
    PNR_ = torch.empty((nsplit, R, dv), **i64)
    PN_ = torch.empty((nsplit, R, dv), **i64)
    PNE_ = torch.empty((nsplit, R, dv), **i64)
    ncb = dv // block_dv
    FL = torch.zeros((nsplit, R, ncb), dtype=torch.int32, device=dev)
    grid = (triton.cdiv(R, block_m), nsplit, ncb)
    _ea_split_kernel[grid](Qt, Kt, Vt, KL, R, am, ae, PR_, PZ_, PZE_, PNR_, PN_, PNE_, FL,
                           split_len, fz, fn, *tc, n, *vtc, D=d, DV=dv, BLOCK_M=block_m, BLOCK_N=block_n,
                           BLOCK_DV=block_dv, FS=SCORE_FS, EXP2_MARGIN=exp2_margin,
                           USE_TC=use_tc, SQ=sq, SK=sk, TC_FALLBACK=tc_fallback,
                           USE_TC_PV=use_tc_pv, SW=sw, SV=sv, ANCH=anch)
    if merge_order is None:
        order = np.arange(nsplit)
    else:
        # Keep the caller's relative order for splits that exist, then append any missing ones,
        # so the combine kernel always gets a permutation of range(nsplit).
        given = [int(i) for i in merge_order if 0 <= int(i) < nsplit]
        order = np.array(given + [i for i in range(nsplit) if i not in given], dtype=np.int64)
    assert sorted(order.tolist()) == list(range(nsplit))
    ORD = torch.from_numpy(order.astype(np.int64)).to(dev)
    OUT = torch.empty((R, dv), dtype=torch.int32, device=dev)
    OF = torch.zeros((R, ncb), dtype=torch.int32, device=dev)
    p, qmin = (8, -133) if out == "bf16" else (24, -149)
    _ea_combine_kernel[(triton.cdiv(R, block_m), ncb)](PR_, PZ_, PZE_, PNR_, PN_, PNE_, ORD, nsplit, R, fz, fn,
                                                       OUT, OF, DV=dv, BLOCK_M=block_m, BLOCK_DV=block_dv,
                                                       P=p, QMIN=qmin, N_ITER=96, OUT_FP32=(out == "fp32"))
    _LAST_DEBUG.update(split_flags=FL, combine_flags=OF, PR=PR_, PZ=PZ_, PZE=PZE_, PNR=PNR_, PN=PN_, PNE=PNE_,
                       fz=fz, fn=fn)
    return OUT, FL, OF


def attention_triton(q_bits, k_bits, v_bits, alpha, kv_len, out: str = "bf16", splits: int = 4,
                     block_m: int = 16, block_n: int = 16, block_dv: int | None = None,
                     merge_order=None, device: str = "cpu", fallback: bool = True,
                     exp2_margin: float = 2.0 ** -26, use_tc: bool = False, sq: int = 3, sk: int = 4,
                     k_slices=None, tc_fallback: bool = True, use_tc_pv: bool = False, sw: int = 5,
                     sv: int = 3):
    """numpy-level API. q: (R, d), k: (n, d), v: (n, dv) bf16 bits (uint16). Row r attends keys
    [0, kv_len[r]): causal, decode, prefill and chunked prefill all have this shape.
    Returns (out, flags); flagged rows are recomputed exactly when fallback=True."""
    dev = torch.device(device)
    dv = v_bits.shape[1]
    block_dv = block_dv or min(dv, 32)
    kv_len = np.asarray(kv_len, dtype=np.int32)

    def t16(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint16).view(np.int16)).to(dev)

    q_tc = k_tc = None
    if use_tc:
        q_tc = tc_slices(q_bits, sq, dev)
        if k_slices is not None:
            kX, klsb, kl1, klost = k_slices
            k_tc = [torch.from_numpy(np.ascontiguousarray(kX)).to(dev), torch.from_numpy(klsb.astype(np.int64)).to(dev),
                    torch.from_numpy(kl1.astype(np.float64)).to(dev),
                    torch.from_numpy(klost.any(axis=1).astype(np.int32)).to(dev)]
        else:
            k_tc = tc_slices(k_bits, sk, dev)
    OUT, FL, OF = launch(t16(q_bits), t16(k_bits), t16(v_bits), torch.from_numpy(kv_len).to(dev),
                         int(kv_len.max(initial=0)), alpha, out=out, splits=splits, block_m=block_m,
                         block_n=block_n, block_dv=block_dv, merge_order=merge_order, exp2_margin=exp2_margin,
                         q_tc=q_tc, k_tc=k_tc, tc_fallback=tc_fallback,
                         v_tc=tc_v_slices(v_bits, sv, dev) if use_tc_pv else None, sw=sw)
    flags = np.maximum(FL.amax(dim=(0, 2)).cpu().numpy(), OF.amax(dim=1).cpu().numpy()).astype(np.uint8)
    o = OUT.cpu().numpy()
    res = (o.astype(np.uint32).view(np.float32) if out == "fp32" else (o & 0xFFFF).astype(np.uint16)).copy()
    if fallback:
        for r in np.flatnonzero(flags):
            L = int(kv_len[r])
            res[r] = attention_row(q_bits[r], k_bits[:L], v_bits[:L], alpha, out)
    return res, flags