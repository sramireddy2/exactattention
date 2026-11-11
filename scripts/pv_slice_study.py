"""P·V on tensor cores: how many int8 slices do weights and V need? Measured on real activations.

For sampled rows of the captured model, compute exact weights w_j = RN24(2^t_j), then
N_c = sum_j w_j v_jc through int8 slices:
  * V rows: block fixed point per key row (slices.slice_rows), with Sv digits
  * weights: fixed point relative to the row's largest term exponent, with Sw digits; one
    shared anchor per row (tensor-core friendly), unlike the CUDA-core path's per-channel anchor
Each output element gets a rigorous error interval, and we count how often bf16 RN(N_c / Z)
is certified. Z is exact (weights are cheap to sum on CUDA cores).

    python scripts/pv_slice_study.py --data <captures> --out results/pv_slice_study_qwen3-0.6b.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from exactattn.formats import BF16, decode_fp32, round_rational  # noqa: E402
from exactattn.reference import trace_row  # noqa: E402
from exactattn.slices import slice_rows  # noqa: E402
from validate_model import alpha_for  # noqa: E402

# (layout, Sw, Sv). Layouts:
#   row-chan  V exponent per key row; weight-truncation bound per channel, V-loss bound per key row (D9)
#   row-elem  same, but the V-loss bound charges only the elements that actually lost bits
#   col-elem  V exponent per (64-key tile, channel); weights anchored per tile; per-element bounds
CONFIGS = [("row-chan", 5, 3), ("row-elem", 4, 3), ("row-elem", 5, 3), ("col-elem", 3, 2), ("col-elem", 4, 2),
           ("col-elem", 3, 3), ("col-elem", 4, 3)]
TILE = 64


def rn(num, den):
    if num == 0:
        return (0, 0)
    m, e = round_rational(num, den, BF16)
    return (0, 0) if m == 0 else (m, e)


def _fix(m, sh):
    """Truncate the integer m * 2^sh toward zero. Returns (value, lost)."""
    if sh >= 0:
        return m << sh, False
    a = abs(m) >> -sh
    return (a if m >= 0 else -a), (a << -sh) != abs(m)


def certify(N, err, unit, tr):
    """N, err: per-channel exact ints in units 2^unit. Returns the number of certified channels."""
    Zm, Ze = tr.Z
    sh = unit - Ze
    ok = 0
    for n, e in zip(N, err):
        lo, hi = n - e, n + e
        r = (lambda x: rn(x << sh, Zm)) if sh >= 0 else (lambda x: rn(x, Zm << -sh))
        ok += r(lo) == r(hi)
    return ok


def study_row(tr, v, layout, Sw, Sv):
    M = [m for m, _ in tr.w]
    E = [e for _, e in tr.w]
    n, dv = v.shape
    if layout.startswith("row"):
        VX, vlsb, _, vlost = slice_rows(v, Sv)                         # vlost: (n, dv) per element
        Vfix = sum(VX[b].astype(object) * (1 << (7 * b)) for b in range(Sv))
        G = [e + int(l) for e, l in zip(E, vlsb)]
        base = max(g + 24 for g in G) - 7 * Sw
        W, lw = zip(*(_fix(m, g - base) for m, g in zip(M, G)))
        N = [sum(W[j] * int(Vfix[j, c]) for j in range(n)) for c in range(dv)]
        err = []
        for c in range(dv):
            e1 = sum(abs(int(Vfix[j, c])) for j in range(n) if lw[j])
            if layout == "row-chan":
                e2 = sum(W[j] + 1 for j in range(n) if vlost[j].any())
            else:
                e2 = sum(W[j] + 1 for j in range(n) if vlost[j, c])
            err.append(e1 + e2)
        return certify(N, err, base, tr), dv
    # col-elem: tile-local exponents per channel, exact combination across tiles
    from fractions import Fraction
    Nf = [Fraction(0)] * dv
    Ef = [Fraction(0)] * dv
    from exactattn.reference import decode_bf16_array
    vm, ve = decode_bf16_array(v)
    for t0 in range(0, n, TILE):
        js = range(t0, min(n, t0 + TILE))
        top = max(E[j] + 24 for j in js)
        wbase = top - 7 * Sw
        W, lw = zip(*(_fix(M[j], E[j] - wbase) for j in js))
        for c in range(dv):
            leads = [int(ve[j, c]) + abs(int(vm[j, c])).bit_length() - 1 for j in js if vm[j, c] != 0]
            if not leads:
                continue
            vbase = max(leads) + 1 - 7 * Sv
            Vc = [_fix(int(vm[j, c]), int(ve[j, c]) - vbase) for j in js]
            s = sum(w * vf for w, (vf, _) in zip(W, Vc))
            e = sum(abs(vf) for (vf, _), l in zip(Vc, lw) if l) + sum(w + 1 for w, (_, vl) in zip(W, Vc) if vl)
            scale = Fraction(2) ** (wbase + vbase)
            Nf[c] += s * scale
            Ef[c] += e * scale
    # Put everything over a common power-of-two denominator to reuse certify().
    den = max((x.denominator for x in Nf + Ef), default=1)
    unit = -(den.bit_length() - 1)
    N = [int(x * den) for x in Nf]
    err = [int(x * den) for x in Ef]
    return certify(N, err, unit, tr), dv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=24)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    L = meta["tokens"]
    rng = np.random.default_rng(3)
    picks = [(int(rng.integers(0, 28)), int(rng.integers(0, 16)), int(rng.integers(64, 600))) for _ in range(args.rows)]
    tot = {c: [0, 0] for c in CONFIGS}
    for li, h, pos in picks:
        lm = meta["layers"][str(li)]
        kv = h // lm["groups"]
        q = np.array(np.load(os.path.join(args.data, f"layer{li:02d}_q.npy"), mmap_mode="r")[h, pos])
        k = np.array(np.load(os.path.join(args.data, f"layer{li:02d}_k.npy"), mmap_mode="r")[kv, : pos + 1])
        v = np.array(np.load(os.path.join(args.data, f"layer{li:02d}_v.npy"), mmap_mode="r")[kv, : pos + 1])
        tr = trace_row(q, k, v, alpha_for(lm["scaling"]))
        for cfg in CONFIGS:
            ok, n = study_row(tr, v, *cfg)
            tot[cfg][0] += ok
            tot[cfg][1] += n
        print(f"row {li}/{h}/{pos} done", flush=True)
    lines = [f"# P·V on tensor cores: int8 slices for weights and V ({meta['model']})", "",
             f"{len(picks)} random rows (positions 64–600), every channel. One weight anchor per row,",
             "with exact Z. Generated by `scripts/pv_slice_study.py`.", "",
             "| layout | Sw | Sv | int8 MMA k-steps (value + bound) | outputs certified | uncertified rate |",
             "|---|---|---|---|---|---|"]
    for (layout, Sw, Sv), (ok, n) in tot.items():
        extra = Sv + Sw if layout.endswith("elem") else Sv
        lines.append(f"| {layout} | {Sw} | {Sv} | {Sw * Sv} + {extra} | {ok:,} / {n:,} | {1 - ok / n:.2e} |")
    lines += ["", "Bound MMAs: per-channel weight loss = lost-weight mask × |V| (Sv k-steps); per-element V loss =",
              "W × lost-V mask (Sw k-steps). Earlier results (D9): row-chan 4×3 1.2e-2, 5×3 4.2e-3."]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
