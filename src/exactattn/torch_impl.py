"""Certified ExactAttention written as integer tensor ops, for any torch device (CPU, XPU, CUDA).

This is the data-parallel form of the algorithm. Every step is an elementwise or reduction op
on int64 or float64 tensors, so the same code runs on an Intel GPU (XPU), an NVIDIA GPU or a
CPU. Each op is exact integer arithmetic, or a float64 op whose result is checked
(certified) before use, so the output bits do not depend on the device.

It is not a fused kernel. It exists to show bit-identity across hardware, and to serve as an
executable spec for the fused kernels. Window parameters differ slightly from the C++
kernel's; certification makes the outputs identical regardless.

Uncertified rows are flagged and recomputed by the exact reference on the host.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .exp2 import exp2_weight_oracle
from .formats import decode_fp32
from .reference import attention_row

I64 = torch.int64
NEG = -(1 << 60)  # "empty" sentinel for reference exponents
SCORE_FS = 63 - 1 - 7  # score fraction bits: |acc| < 2**62 for d <= 128


@dataclass
class TorchStats:
    rows: int = 0
    flagged_rows: int = 0
    oracle_weights: int = 0


def _bitlen(x: torch.Tensor) -> torch.Tensor:
    """Exact bit length of non-negative int64 values (0 -> 0)."""
    xf = x.to(torch.float64)
    b = torch.where(x > 0, torch.floor(torch.log2(torch.clamp(xf, min=1.0))).to(I64) + 1, torch.zeros_like(x))
    b = torch.clamp(b, 0, 63)
    # Fix float rounding at boundaries: require 2**(b-1) <= x < 2**b.
    too_big = (x > 0) & (b > 0) & ((x >> torch.clamp(b - 1, 0, 62)) == 0)
    b = torch.where(too_big, b - 1, b)
    too_small = (x > 0) & (b < 63) & ((x >> torch.clamp(b, 0, 62)) != 0)
    return torch.where(too_small, b + 1, b)


def _floor_shift(m: torch.Tensor, sh: torch.Tensor):
    """floor(m * 2**sh) and lost-bits flags. Callers guarantee left shifts don't overflow."""
    left = torch.clamp(sh, 0, 62)
    s = torch.clamp(-sh, 0, 63)
    sr = torch.clamp(s, 0, 62)
    v_right = m >> sr
    v_right = torch.where(s >= 63, torch.where(m < 0, -torch.ones_like(m), torch.zeros_like(m)), v_right)
    lost_right = torch.where(s >= 63, m != 0, (v_right << sr) != m)
    v = torch.where(sh >= 0, m << left, v_right)
    lost = torch.where(sh >= 0, torch.zeros_like(lost_right), lost_right)
    return v, lost


def _rescale(acc, err, s):
    """[acc, acc+err] in units 2**s times larger (s >= 0)."""
    v, lost = _floor_shift(acc, -s)
    sc = torch.clamp(s, 0, 62)
    mask = (torch.ones_like(err) << sc) - 1
    e = torch.where(s >= 63, (err > 0).to(I64), (err >> sc) + ((err & mask) != 0).to(I64))
    return v, e + lost.to(I64)


def _decode_bf16(bits: torch.Tensor):
    b = bits.to(I64) & 0xFFFF
    exp = (b >> 7) & 0xFF
    frac = b & 0x7F
    sign = torch.where((b & 0x8000) != 0, -1, 1).to(I64)
    normal = exp != 0
    m = sign * torch.where(normal, frac | 0x80, frac)
    e = torch.where(normal, exp - 134, torch.full_like(exp, -133))
    return m, e, exp == 0xFF


def _round_two_word(C, Yl, E0, p, qmin, emax):
    """RN of (C * 2**32 + Yl) * 2**E0 into a format with precision p. C, Yl >= 0, C < 2**56,
    Yl < 2**32. Returns (q, qe, overflow); q == 0 means zero."""
    blC = _bitlen(C)
    bl = torch.where(C > 0, blC + 32, _bitlen(Yl))
    K = bl - 1 + E0
    qe = torch.maximum(K - (p - 1), torch.full_like(K, qmin))
    drop = qe - E0  # bits to drop from the integer value
    # q1 = floor(value / 2**(drop - 1)), with a sticky flag for the dropped remainder.
    s = drop - 1
    s_hi = torch.clamp(s - 32, 0, 62)
    q_big = C >> s_hi
    st_big = ((C & ((torch.ones_like(C) << s_hi) - 1)) != 0) | (Yl != 0)
    q_big = torch.where(s - 32 > 62, torch.zeros_like(q_big), q_big)
    st_big = torch.where(s - 32 > 62, (C != 0) | (Yl != 0), st_big)
    s_lo = torch.clamp(s, 0, 31)
    q_small = (C << (32 - s_lo)) | (Yl >> s_lo)
    st_small = (Yl & ((torch.ones_like(Yl) << s_lo) - 1)) != 0
    # s <= 0: the value fits in at most p + 1 bits (C == 0), so shift left.
    q_left = Yl << torch.clamp(-s, 0, 62)
    q1 = torch.where(s >= 32, q_big, torch.where(s >= 1, q_small, q_left))
    sticky = torch.where(s >= 32, st_big, torch.where(s >= 1, st_small, torch.zeros_like(st_small)))
    rnd = q1 & 1
    q = q1 >> 1
    up = (rnd == 1) & (sticky | ((q & 1) == 1))
    q = q + up.to(I64)
    carry = q == (1 << p)
    q = torch.where(carry, q >> 1, q)
    qe = torch.where(carry, qe + 1, qe)
    zero = bl == 0
    q = torch.where(zero, torch.zeros_like(q), q)
    top = qe + _bitlen(q) - 1
    overflow = (~zero) & (top > emax)
    return q, qe, overflow


def _rn32_product(am: int, ae: int, acc: torch.Tensor, base: torch.Tensor):
    """t = RN32(am * acc * 2**(ae + base)), exact, as a float32 tensor. Also returns an
    overflow mask."""
    sign = torch.sign(acc) * (1 if am >= 0 else -1)
    A = torch.abs(acc)
    a = abs(am)
    hi = A >> 32
    lo = A & 0xFFFFFFFF
    X = hi * a
    Y = lo * a
    C = X + (Y >> 32)
    Yl = Y & 0xFFFFFFFF
    q, qe, ov = _round_two_word(C, Yl, ae + base, 24, -149, 127)
    t = (q.to(torch.float64) * torch.exp2(qe.to(torch.float64))).to(torch.float32)
    return t * sign.to(torch.float32), ov


def _ratio_round(a, b, sh, p, qmin):
    """RN(a / b * 2**sh) for a >= 0, 0 < b < 2**62 (int64 tensors). Returns (q, qe)."""
    k = _bitlen(a) - _bitlen(b)
    kpos = torch.clamp(k, 0, 62)
    kneg = torch.clamp(-k, 0, 62)
    ceil_b = (b >> kneg) + ((b & ((torch.ones_like(b) << kneg) - 1)) != 0).to(I64)
    less = torch.where(k >= 0, (a >> kpos) < b, a < ceil_b)
    k = torch.where(less, k - 1, k)
    K = k + sh
    qe = torch.maximum(K - (p - 1), torch.full_like(K, qmin))
    s2 = sh - qe + 1
    ip = torch.div(a, b, rounding_mode="floor")
    r = a - ip * b
    # Case s2 < 0: q1 = ip >> -s2.
    sn = torch.clamp(-s2, 0, 62)
    q_neg = ip >> sn
    st_neg = (r != 0) | ((ip & ((torch.ones_like(ip) << sn) - 1)) != 0)
    # Case s2 >= 0: generate s2 fraction bits by restoring division.
    q_pos = ip.clone()
    rr = r.clone()
    n_iter = int(torch.clamp(s2, min=0).max().item()) if s2.numel() else 0
    for i in range(n_iter):
        act = s2 > i
        r2 = rr << 1
        bit = r2 >= b
        r2 = torch.where(bit, r2 - b, r2)
        rr = torch.where(act, r2, rr)
        q_pos = torch.where(act, (q_pos << 1) | bit.to(I64), q_pos)
    st_pos = rr != 0
    q1 = torch.where(s2 >= 0, q_pos, q_neg)
    sticky = torch.where(s2 >= 0, st_pos, st_neg)
    rnd = q1 & 1
    q = q1 >> 1
    up = (rnd == 1) & (sticky | ((q & 1) == 1))
    q = q + up.to(I64)
    carry = q == (1 << p)
    q = torch.where(carry, q >> 1, q)
    qe = torch.where(carry, qe + 1, qe)
    q = torch.where(a == 0, torch.zeros_like(q), q)
    return q, qe


def attention_torch(q_bits: np.ndarray, k_bits: np.ndarray, v_bits: np.ndarray, alpha, mask: np.ndarray,
                    device: str = "cpu", out: str = "bf16", tile: int = 64, row_chunk: int = 64,
                    tile_order_seed: int | None = None, stats: TorchStats | None = None,
                    fallback: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """q: (R, d), k: (n, d), v: (n, dv) bf16 bits; mask: (R, n) bool. Returns (out, flags)."""
    dev = torch.device(device)
    R, d = q_bits.shape
    n, dv = v_bits.shape
    assert d <= 128, "SCORE_FS assumes d <= 128"
    p, qmin = (8, -133) if out == "bf16" else (24, -149)
    am, ae = decode_fp32(np.float32(alpha))
    key_log2 = max(1, int(np.ceil(np.log2(max(2, mask.sum(axis=1).max(initial=1))))))
    fz = 64 - 3 - key_log2
    fn = 64 - 3 - key_log2
    km, ke, kbad = _decode_bf16(torch.from_numpy(k_bits.astype(np.int64)).to(dev))
    vm, ve, vbad = _decode_bf16(torch.from_numpy(v_bits.astype(np.int64)).to(dev))
    lead_v = torch.where(vm != 0, ve + _bitlen(torch.abs(vm)) - 1, torch.full_like(ve, NEG))
    res = np.zeros((R, dv), dtype=np.uint16 if out == "bf16" else np.float32)
    flags = np.zeros(R, dtype=np.uint8)
    rng = np.random.default_rng(tile_order_seed) if tile_order_seed is not None else None
    for r0 in range(0, R, row_chunk):
        r1 = min(R, r0 + row_chunk)
        Rb = r1 - r0
        qm, qe_, qbad = _decode_bf16(torch.from_numpy(q_bits[r0:r1].astype(np.int64)).to(dev))
        mrows = torch.from_numpy(mask[r0:r1]).to(dev)
        active = np.flatnonzero(mask[r0:r1].any(axis=0))
        if active.size == 0:
            continue
        kmax = int(active.max()) + 1
        bad = qbad.any(dim=1)
        # Partial state per row.
        PR = torch.full((Rb,), NEG, dtype=I64, device=dev)
        PZ = torch.zeros(Rb, dtype=I64, device=dev)
        PZe = torch.zeros(Rb, dtype=I64, device=dev)
        PNR = torch.full((Rb, dv), NEG, dtype=I64, device=dev)
        PN = torch.zeros((Rb, dv), dtype=I64, device=dev)
        PNe = torch.zeros((Rb, dv), dtype=I64, device=dev)
        flag = torch.zeros(Rb, dtype=torch.bool, device=dev)
        tiles = list(range(0, kmax, tile))
        if rng is not None:
            rng.shuffle(tiles)
        for t0 in tiles:
            t1 = min(kmax, t0 + tile)
            msk = mrows[:, t0:t1]
            if not bool(msk.any()):
                continue
            # --- scores (windowed int64) ---
            P = qm[:, None, :] * km[None, t0:t1, :]                    # (Rb, T, d)
            E = qe_[:, None, :] + ke[None, t0:t1, :]
            nzp = P != 0
            lead = torch.where(nzp, E + _bitlen(torch.abs(P)) - 1, torch.full_like(E, NEG))
            Lmax = lead.max(dim=2).values                               # (Rb, T)
            base = Lmax + 1 - SCORE_FS
            term, lost = _floor_shift(P, E - base[..., None])
            term = torch.where(nzp, term, torch.zeros_like(term))
            acc = term.sum(dim=2)
            err = (lost & nzp).sum(dim=2)
            anyp = nzp.any(dim=2)
            flag |= (msk & anyp & ((torch.abs(acc) + err) >= (1 << 62))).any(dim=1)
            t_lo, ov1 = _rn32_product(am, ae, acc, base)
            t_hi, ov2 = _rn32_product(am, ae, acc + err, base)
            t = torch.where(anyp, t_lo, torch.zeros_like(t_lo))
            unc = anyp & ((t_lo != t_hi) | ov1 | ov2)
            flag |= (msk & unc).any(dim=1)
            # --- weights w = RN24(2**t) ---
            td = t.to(torch.float64)
            nfl = torch.floor(td)
            f = td - nfl
            scaled = torch.exp2(f) * 8388608.0
            fl = torch.floor(scaled)
            frac = scaled - fl
            M = fl.to(I64) + (frac >= 0.5).to(I64)
            Ew = nfl.to(I64) - 23
            carry = M == (1 << 24)
            M = torch.where(carry, M >> 1, M)
            Ew = torch.where(carry, Ew + 1, Ew)
            exact = f == 0
            M = torch.where(exact, torch.full_like(M, 1 << 23), M)
            Ew = torch.where(exact, nfl.to(I64) - 23, Ew)
            flag |= (msk & (torch.abs(td) >= 2.0 ** 30)).any(dim=1)
            risky = msk & (~exact) & (torch.abs(frac - 0.5) < 2.0 ** -18)
            if bool(risky.any()):
                idx = risky.nonzero().cpu().numpy()
                tv = t.cpu().numpy()
                Mc, Ec = M.cpu().numpy(), Ew.cpu().numpy()
                for (i, j) in idx:
                    Mi, Ei = exp2_weight_oracle(float(tv[i, j]))
                    Mc[i, j], Ec[i, j] = Mi, Ei
                if stats is not None:
                    stats.oracle_weights += len(idx)
                M, Ew = torch.from_numpy(Mc).to(dev), torch.from_numpy(Ec).to(dev)
            # --- tile partial ---
            top = torch.where(msk, Ew + 23, torch.full_like(Ew, NEG))
            TR = top.max(dim=1).values                                  # (Rb,)
            zt, zl = _floor_shift(M, Ew - (TR[:, None] - fz))
            TZ = torch.where(msk, zt, torch.zeros_like(zt)).sum(dim=1)
            TZe = (zl & msk).sum(dim=1)
            lv = lead_v[None, t0:t1, :]                                 # (1, T, dv)
            vmt, vet = vm[None, t0:t1, :], ve[None, t0:t1, :]
            valid = msk[:, :, None] & (vmt != 0)
            L = torch.where(valid, top[:, :, None] + 1 + lv, torch.full((Rb, t1 - t0, dv), NEG, dtype=I64, device=dev))
            TNR = L.max(dim=1).values                                   # (Rb, dv)
            nt, nl = _floor_shift(M[:, :, None] * vmt, Ew[:, :, None] + vet - (TNR[:, None, :] - fn))
            TN = torch.where(valid, nt, torch.zeros_like(nt)).sum(dim=1)
            TNe = (nl & valid).sum(dim=1)
            # --- merge into the running partial (any order) ---
            nR = torch.maximum(PR, TR)
            a1, e1 = _rescale(PZ, PZe, nR - PR)
            a2, e2 = _rescale(TZ, TZe, nR - TR)
            PR, PZ, PZe = nR, a1 + a2, e1 + e2
            nNR = torch.maximum(PNR, TNR)
            b1, f1 = _rescale(PN, PNe, nNR - PNR)
            b2, f2 = _rescale(TN, TNe, nNR - TNR)
            PNR, PN, PNe = nNR, b1 + b2, f1 + f2
            flag |= (torch.abs(PZ) + PZe) >= (1 << 62)
            flag |= ((torch.abs(PN) + PNe) >= (1 << 62)).any(dim=1)
        # --- certified final rounding ---
        has = PR > NEG // 2
        zunit = PR - fz
        Zlo, Zhi = PZ, PZ + PZe
        Nlo, Nhi = PN, PN + PNe
        chan = PNR > NEG // 2
        sh = (PNR - fn) - zunit[:, None]
        # Interval endpoints for N/Z.
        lo_num = Nlo
        lo_den = torch.where(Nlo >= 0, Zhi[:, None], Zlo[:, None])
        hi_num = Nhi
        hi_den = torch.where(Nhi <= 0, Zhi[:, None], Zlo[:, None])
        Zsafe = torch.clamp(Zlo, min=1)
        lo_den = torch.where(has[:, None], lo_den, Zsafe[:, None].expand_as(lo_den))
        hi_den = torch.where(has[:, None], hi_den, Zsafe[:, None].expand_as(hi_den))
        shc = torch.where(chan, sh, torch.zeros_like(sh))
        ql, qel = _ratio_round(torch.abs(lo_num), torch.clamp(lo_den, min=1), shc, p, qmin)
        qh, qeh = _ratio_round(torch.abs(hi_num), torch.clamp(hi_den, min=1), shc, p, qmin)
        sl = torch.where(lo_num < 0, -1, 1)
        sh_ = torch.where(hi_num < 0, -1, 1)
        same = ((ql == 0) & (qh == 0)) | ((ql == qh) & (qel == qeh) & (sl == sh_))
        same = same | ~chan
        flag |= has & ~same.all(dim=1)
        flag |= has & (Zlo <= 0)
        val = (ql.to(torch.float64) * torch.exp2(qel.to(torch.float64))) * sl.to(torch.float64)
        val = torch.where(chan & has[:, None] & (ql != 0), val, torch.zeros_like(val)).to(torch.float32)
        if out == "bf16":
            bits = (val.view(torch.int32) >> 16) & 0xFFFF
            res[r0:r1] = bits.cpu().numpy().astype(np.uint16)
        else:
            res[r0:r1] = val.cpu().numpy()
        fl_np = flag.cpu().numpy()
        bad_np = (bad | (mrows & (kbad.any(dim=1) | vbad.any(dim=1))[None, :]).any(dim=1)).cpu().numpy()
        flags[r0:r1] = np.where(bad_np, 2, np.where(fl_np, 1, 0)).astype(np.uint8)
    if stats is not None:
        stats.rows += R
        stats.flagged_rows += int((flags != 0).sum())
    if fallback:
        for r in np.flatnonzero(flags == 1):
            sel = np.flatnonzero(mask[r])
            res[r] = attention_row(q_bits[r], k_bits[sel], v_bits[sel], alpha, out)
    return res, flags
