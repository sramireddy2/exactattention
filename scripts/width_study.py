"""Accumulator-width study on real attention activations (DESIGN.md Q1-Q3).

Reads captures from capture_qkv.py and measures, over sampled (layer, head, position) rows
attending causally to their prefix:

  A. spread of product exponents in each q.k dot product (int8 slices needed for exact scores)
  B. score cancellation: lead(max |q_i k_i|) - lead(|s|)
  C. logit spread per row, t_max - t_min, and how many keys sit within 24 / 48 bits of the max
  D. output cancellation per channel: lead(max |w_j v_jc|) - lead(|N_c|)
  E. exponent spread inside each K and V row (for block-exponent KV storage)
  F. actual certified-model fallback rates at several window widths, on a subsample of rows

Usage:
    python scripts/width_study.py --data <capture dir> --out results/width_study_qwen3-0.6b.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exactattn.certified import Config, Stats, split_attention_row  # noqa: E402
from exactattn.formats import FP32, decode_fp32, encode_fp32, round_dyadic  # noqa: E402
from exactattn.reference import attention_row, decode_bf16_array  # noqa: E402


def alpha_for(scaling: float) -> np.float32:
    from mpmath import log, mp, mpf

    with mp.workprec(200):
        man, exp = mp.mpf(mpf(scaling) / log(2)).man_exp
    return encode_fp32(*round_dyadic(int(man), int(exp), FP32))


def lead_of_int64(m: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Leading-bit exponent of |m| * 2**e (m != 0)."""
    a = np.abs(m).astype(np.float64)
    return e + np.floor(np.log2(np.maximum(a, 1))).astype(np.int64)


def row_stats(qb, Kb, Vb, alpha):
    qm, qe = decode_bf16_array(qb)
    km, ke = decode_bf16_array(Kb)
    vm, ve = decode_bf16_array(Vb)
    P = km * qm[None, :]
    E = ke + qe[None, :]
    nz = P != 0
    lead = np.where(nz, lead_of_int64(P, E), np.iinfo(np.int64).min // 4)
    lead_max = lead.max(axis=1)
    lead_min = np.where(nz, lead, np.iinfo(np.int64).max // 4).min(axis=1)
    has = nz.any(axis=1)
    spread = np.where(has, lead_max - lead_min, 0)
    # Exact-ish score: products are exact in float64, and fsum rounds the exact sum correctly.
    Pf = P.astype(np.float64) * np.exp2(E.astype(np.float64))
    s = np.array([math.fsum(row) for row in Pf])
    with np.errstate(divide="ignore"):
        lead_s = np.where(s != 0, np.floor(np.log2(np.abs(s))), -np.inf)
    cancel = np.where(has & (s != 0), lead_max - lead_s, np.nan)
    t = (s * float(alpha)).astype(np.float32).astype(np.float64)
    tmax = t.max()
    gap = tmax - t
    w = np.exp2(-gap)
    Vf = vm.astype(np.float64) * np.exp2(ve.astype(np.float64))
    terms = w[:, None] * Vf
    with np.errstate(divide="ignore"):
        max_term = np.abs(terms).max(axis=0)
        N = np.array([math.fsum(terms[:, c]) for c in range(terms.shape[1])])
        ncancel = np.where((N != 0) & (max_term > 0), np.log2(max_term) - np.log2(np.abs(N)), np.nan)

    def elem_spread(m, e):
        nzm = m != 0
        L = np.where(nzm, lead_of_int64(m, e), 0)
        hi = np.where(nzm, L, -10**9).max(axis=1)
        lo = np.where(nzm, L, 10**9).min(axis=1)
        return np.where(nzm.any(axis=1), hi - lo, 0)

    return {
        "prod_spread": spread,
        "score_cancel": cancel[np.isfinite(cancel)],
        "logit_spread": float(gap.max()),
        "within24": int((gap <= 24).sum()),
        "within48": int((gap <= 48).sum()),
        "n": int(len(t)),
        "out_cancel": ncancel[np.isfinite(ncancel)],
        "k_elem_spread": elem_spread(km, ke),
        "v_elem_spread": elem_spread(vm, ve),
        "q_elem_spread": int(elem_spread(qm[None, :], qe[None, :])[0]),
    }


def pct(x, ps=(50, 90, 99, 99.9, 100)):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return ["-"] * len(ps)
    return [f"{np.percentile(x, p):.1f}" for p in ps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default="2047,1500,1024,511,127,31")
    ap.add_argument("--cert-rows", type=int, default=48, help="rows to run through the certified model")
    ap.add_argument("--cert-widths", default="48,56,64")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    layers = sorted(int(k) for k in meta["layers"])
    positions = [int(p) for p in args.positions.split(",")]
    agg = {k: [] for k in ["prod_spread", "score_cancel", "out_cancel", "k_elem_spread", "v_elem_spread"]}
    rows_meta = []
    t0 = time.time()
    rng = np.random.default_rng(0)
    cert_pool = []
    for li in layers:
        lm = meta["layers"][str(li)]
        alpha = alpha_for(lm["scaling"])
        g = lm["groups"]
        Q = np.load(os.path.join(args.data, f"layer{li:02d}_q.npy"))
        K = np.load(os.path.join(args.data, f"layer{li:02d}_k.npy"))
        V = np.load(os.path.join(args.data, f"layer{li:02d}_v.npy"))
        H = Q.shape[0]
        for h in range(H):
            kv = h // g
            for pos in positions:
                if pos >= Q.shape[1]:
                    continue
                st = row_stats(Q[h, pos], K[kv, : pos + 1], V[kv, : pos + 1], alpha)
                for k in agg:
                    agg[k].append(np.atleast_1d(st[k]))
                rows_meta.append({"layer": li, "head": h, "pos": pos, "logit_spread": st["logit_spread"],
                                  "within24": st["within24"], "within48": st["within48"], "n": st["n"],
                                  "q_elem_spread": st["q_elem_spread"]})
                cert_pool.append((li, h, kv, pos))
        print(f"layer {li} done ({time.time() - t0:.0f}s)", flush=True)

    cat = {k: np.concatenate(v) for k, v in agg.items()}
    ls = [r["logit_spread"] for r in rows_meta]
    w24 = [r["within24"] / r["n"] for r in rows_meta]
    w48 = [r["within48"] / r["n"] for r in rows_meta]
    qs = [r["q_elem_spread"] for r in rows_meta]

    # F. certified model on a random subsample of rows
    widths = [int(x) for x in args.cert_widths.split(",")]
    pick = rng.choice(len(cert_pool), size=min(args.cert_rows, len(cert_pool)), replace=False)
    cert = {w: Stats() for w in widths}
    mismatches = 0
    for idx in pick:
        li, h, kv, pos = cert_pool[idx]
        lm = meta["layers"][str(li)]
        alpha = alpha_for(lm["scaling"])
        Q = np.load(os.path.join(args.data, f"layer{li:02d}_q.npy"), mmap_mode="r")
        K = np.load(os.path.join(args.data, f"layer{li:02d}_k.npy"), mmap_mode="r")
        V = np.load(os.path.join(args.data, f"layer{li:02d}_v.npy"), mmap_mode="r")
        q, k, v = np.array(Q[h, pos]), np.array(K[kv, : pos + 1]), np.array(V[kv, : pos + 1])
        n = k.shape[0]
        ref = None
        for w in widths:
            splits = np.array_split(np.arange(n), max(1, n // 512))
            cfg = Config(acc_bits=w, score_bits=w, head_dim_log2=7, key_log2=11)
            got = split_attention_row(q, k, v, alpha, splits, cfg, "bf16", None, cert[w], tile=64)
            if w == widths[-1]:
                ref = attention_row(q, k, v, alpha)
                mismatches += int(not np.array_equal(got, ref))
        print(f"cert row {li}/{h}/{pos} done ({time.time() - t0:.0f}s)", flush=True)

    hdr = "| quantity | p50 | p90 | p99 | p99.9 | max |\n|---|---|---|---|---|---|\n"
    lines = [
        f"# Accumulator-width study: {meta['model']}",
        "",
        f"{meta['tokens']} tokens of WikiText-2 (test), {len(layers)} layers, {len(rows_meta)} query rows",
        f"(every head × positions {positions}), causal prefix. Generated by `scripts/width_study.py`.",
        "",
        "## Distributions (bits)",
        "",
        hdr
        + f"| A. product exponent spread per q·k | {' | '.join(pct(cat['prod_spread']))} |\n"
        + f"| B. score cancellation | {' | '.join(pct(cat['score_cancel']))} |\n"
        + f"| C. logit spread per row (log2 units) | {' | '.join(pct(ls))} |\n"
        + f"| D. output cancellation per channel | {' | '.join(pct(cat['out_cancel']))} |\n"
        + f"| E1. exponent spread within a K row | {' | '.join(pct(cat['k_elem_spread']))} |\n"
        + f"| E2. exponent spread within a V row | {' | '.join(pct(cat['v_elem_spread']))} |\n"
        + f"| E3. exponent spread within a q row | {' | '.join(pct(qs))} |\n",
        "",
        f"Keys within 24 bits of the max logit (share of prefix): p50 {np.median(w24):.3f}, p10 {np.percentile(w24, 10):.3f}.",
        f"Keys within 48 bits: p50 {np.median(w48):.3f}, p10 {np.percentile(w48, 10):.3f}.",
        "",
        f"## Certified model on {len(pick)} sampled rows (bit-accurate, splits of 512, tiles of 64)",
        "",
        "| window bits | score fallbacks / scores | row fallbacks / rows |",
        "|---|---|---|",
    ]
    for w in widths:
        s = cert[w]
        lines.append(f"| {w} | {s.score_fallbacks} / {s.scores} | {s.row_fallbacks} / {s.rows} |")
    lines += ["", f"Mismatches against the exact reference at {widths[-1]} bits: {mismatches} (must be 0)."]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + "_rows.json", "w") as f:
        json.dump(rows_meta, f)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
