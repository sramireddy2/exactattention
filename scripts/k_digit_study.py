"""How many int8 digit planes do K (and V) rows actually use? Measured on real activations.

Tier 1 stores each K row as SK=4 int8 digit planes in fixed point below the row's largest
element (slices.slice_rows), and each V row as SV=3 planes. Plane 0 is the least significant.
A row whose low planes are all zero gives the same digits, and so the same bits, whether or
not those planes are read. A per-key flag could then skip their loads, the way D33 skips lost-V
rows.

For each row this counts the planes in use: S minus the number of all-zero planes from the
bottom (0 for an all-zero row). Rows that already lose bits at S planes are counted
separately. It also counts 64-key tiles in which every key leaves plane 0 unused, where a
per-tile skip could drop that plane's MMAs as well.

    python scripts/k_digit_study.py --data <captures> --out results/k_digit_study_qwen3-0.6b.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exactattn.slices import slice_rows  # noqa: E402

TILE = 64
ROW_BYTES = 128          # one digit plane of a d=128 row, int8
TIER1_BYTES = 925        # tier-1 bytes per key after D33 (DESIGN D33)


def planes_used(bits: np.ndarray, S: int):
    """bits (n, d) bf16 patterns -> (used (n,) int, lost (n,) bool)."""
    X, _, _, lost = slice_rows(bits.astype(np.uint16), S)
    nz = (X != 0).any(axis=2)                          # (S, n): plane s has a nonzero digit
    used = np.zeros(bits.shape[0], dtype=np.int64)
    for s in range(S):
        used = np.where(nz[s], s + 1, used)            # highest nonzero plane counts ...
    low = np.full(bits.shape[0], S, dtype=np.int64)    # ... down to the lowest nonzero one
    for s in reversed(range(S)):
        low = np.where(nz[s], s, low)
    span = np.where(used > 0, used - low, 0)
    return span, lost.any(axis=1)


def rows_of(a: np.ndarray) -> list[np.ndarray]:
    """(heads, tokens, d) -> one (tokens, d) block per head, so tiles are 64 positions of one head."""
    return [a[h] for h in range(a.shape[0])]


def summarize(blocks: list[np.ndarray], S: int) -> dict:
    spans, losts, tiles_free = [], [], []
    for b in blocks:
        span, lost = planes_used(b, S)
        spans.append(span)
        losts.append(lost)
        m = len(span) // TILE * TILE
        # plane 0 unused by every key of the tile, and no key already lossy
        tiles_free.append(((span[:m] < S) & ~lost[:m]).reshape(-1, TILE).all(axis=1))
    span = np.concatenate(spans)
    lost = np.concatenate(losts)
    tf = np.concatenate(tiles_free)
    out = {"keys": int(len(span)), "lost": float(lost.mean()), "tiles": int(len(tf)),
           "tile_skip": float(tf.mean()) if len(tf) else float("nan")}
    for k in range(S + 1):
        out[f"le{k}"] = float(((span <= k) & ~lost).mean())
    return out


def bytes_saved(k: dict, v: dict) -> float:
    """Bytes per key a per-key skip of unused low planes saves: K plane 0 and 1, V plane 0."""
    return ROW_BYTES * (k["le3"] + k["le2"] + v["le2"])


def fmt_row(label: str, k: dict, v: dict) -> str:
    sv = bytes_saved(k, v)
    return (f"| {label} | {k['keys']} | {k['le2']:.1%} | {k['le3']:.1%} | {1 - k['le3'] - k['lost']:.1%} | "
            f"{k['lost']:.2%} | {k['tile_skip']:.1%} | {v['le2']:.1%} | {sv:.0f} B ({sv / TIER1_BYTES:.0%}) |")


HEADER = ("| keys | count | K: ≤2 planes | K: ≤3 planes | K: all 4 | K: lossy at 4 | 64-key tiles with K plane 0 "
          "unused | V: ≤2 of 3 planes | per-key skip saves |\n|---|---|---|---|---|---|---|---|---|")


def bench_blocks(n: int, heads: int):
    import torch
    g = torch.Generator().manual_seed(0)
    k = torch.randn((heads, n, 128), generator=g).to(torch.bfloat16)
    v = torch.randn((heads, n, 128), generator=g).to(torch.bfloat16)
    as_bits = lambda t: t.contiguous().view(torch.int16).numpy().view(np.uint16)
    return rows_of(as_bits(k)), rows_of(as_bits(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="capture_qkv.py output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-keys", type=int, default=1 << 16, help="randn keys per head, 4 heads")
    args = ap.parse_args()
    json.load(open(os.path.join(args.data, "meta.json")))   # a capture_qkv.py directory
    nl = 1000                                                # every layer file present

    lines = ["# K and V digit planes in use (Qwen3-0.6B, WikiText-2, and the randn bench)", "",
             "Made by `scripts/k_digit_study.py`; see DESIGN D34. K has 4 int8 planes and V has 3,",
             "in fixed point below each row's largest element. \"≤3 planes\" means the row's lowest",
             "plane is all zero (and the row is not already lossy), so skipping that plane gives the",
             "same bits. Tiles are 64 consecutive positions of one KV head.", "", HEADER]

    kb, vb = bench_blocks(args.bench_keys, 4)
    k, v = summarize(kb, 4), summarize(vb, 3)
    lines.append(fmt_row("bench randn", k, v))
    print(lines[-1], flush=True)

    allk, allv, per_layer = [], [], []
    for li in range(nl):
        kp = os.path.join(args.data, f"layer{li:02d}_k.npy")
        if not os.path.exists(kp):
            break
        K = rows_of(np.load(kp))
        V = rows_of(np.load(os.path.join(args.data, f"layer{li:02d}_v.npy")))
        allk += K
        allv += V
        k, v = summarize(K, 4), summarize(V, 3)
        per_layer.append(fmt_row(f"layer {li}", k, v))
        print(per_layer[-1], flush=True)
    k, v = summarize(allk, 4), summarize(allv, 3)
    lines.append(fmt_row(f"Qwen3 all {len(per_layer)} layers", k, v))
    lines += ["", "Per layer (all KV heads):", "", HEADER] + per_layer
    lines += ["", f"Savings assume {TIER1_BYTES} tier-1 bytes per key and {ROW_BYTES} B per plane row (d=128)."]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
