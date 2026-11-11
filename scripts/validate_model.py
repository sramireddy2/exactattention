"""Full-model validation of the C++ certified kernel on captured activations.

For every layer and query head, runs every causal row, at every position, under two different
schedules:
  * schedule A: 1 split, 64-key tiles, 64-bit accumulators
  * schedule B: 8 splits, 37-key tiles, shuffled merge order, 128-bit accumulators
Checks:
  1. A and B are bit-identical on every row (invariance at scale)
  2. certification rate (rows needing a fallback)
  3. a random sample of rows against the exact Python reference
  4. accuracy against float64 attention, in bf16 ulps
Also reports throughput and writes a markdown report.

Usage (Linux/WSL):
    python scripts/validate_model.py --data <capture dir> --out results/validation_qwen3-0.6b.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exactattn import attention_row, bf16_bits_to_float32  # noqa: E402
from exactattn import cpu  # noqa: E402
from exactattn.formats import FP32, encode_fp32, round_dyadic  # noqa: E402


def alpha_for(scaling: float) -> np.float32:
    from mpmath import log, mp, mpf

    with mp.workprec(200):
        man, exp = mp.mpf(mpf(scaling) / log(2)).man_exp
    return encode_fp32(*round_dyadic(int(man), int(exp), FP32))


def causal_csr(L: int):
    lens = np.arange(1, L + 1, dtype=np.int64)
    row_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    key_idx = np.concatenate([np.arange(i + 1, dtype=np.int64) for i in range(L)])
    return row_ptr, key_idx


def float64_causal(q, k, v, alpha):
    qf = bf16_bits_to_float32(q).astype(np.float64)
    kf = bf16_bits_to_float32(k).astype(np.float64)
    vf = bf16_bits_to_float32(v).astype(np.float64)
    s = (qf @ kf.T) * float(alpha)
    L = s.shape[0]
    s[np.triu_indices(L, 1)] = -np.inf
    s -= s.max(axis=1, keepdims=True)
    p = np.exp2(s)
    return (p @ vf) / p.sum(axis=1, keepdims=True)


def fp32_causal(q, k, v, alpha):
    """Standard fp32 attention, FlashAttention-style: fp32 scores, fp32 exp2, fp32 sums."""
    qf = bf16_bits_to_float32(q)
    kf = bf16_bits_to_float32(k)
    vf = bf16_bits_to_float32(v)
    s = (qf @ kf.T).astype(np.float32) * np.float32(alpha)
    L = s.shape[0]
    s[np.triu_indices(L, 1)] = -np.inf
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp2(s).astype(np.float32)
    return ((p @ vf) / p.sum(axis=1, keepdims=True)).astype(np.float32)


def bf16_ordinal(bits):
    """Map bf16 bit patterns to integers where adjacent values differ by 1 (ulp distance)."""
    b = bits.astype(np.int32)
    return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-samples-per-layer", type=int, default=2)
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    layers = sorted(int(x) for x in meta["layers"]) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    L = meta["tokens"]
    row_ptr, key_idx = causal_csr(L)
    rng = np.random.default_rng(0)
    tot = {"rows": 0, "flag_a": 0, "flag_b": 0, "ab_mismatch": 0, "ref_checked": 0, "ref_mismatch": 0,
           "pairs": 0, "time_a": 0.0}
    ulp_hist = np.zeros(4, dtype=np.int64)  # |ulp| == 0, 1, 2, >2 vs float64 rounded to bf16
    ulp_hist32 = np.zeros(4, dtype=np.int64)  # same, for standard fp32 attention
    far_rel: list[float] = []
    for li in layers:
        lm = meta["layers"][str(li)]
        alpha = alpha_for(lm["scaling"])
        g = lm["groups"]
        Q = np.load(os.path.join(args.data, f"layer{li:02d}_q.npy"))
        K = np.load(os.path.join(args.data, f"layer{li:02d}_k.npy"))
        V = np.load(os.path.join(args.data, f"layer{li:02d}_v.npy"))
        for h in range(Q.shape[0]):
            kv = h // g
            q, k, v = Q[h], K[kv], V[kv]
            sa, sb = cpu.CpuStats(), cpu.CpuStats()
            t0 = time.perf_counter()
            a, fa = cpu.attention_cpu(q, k, v, alpha, row_ptr, key_idx, acc_bits=64, tile=64, splits=1,
                                      stats=sa, fallback=False)
            tot["time_a"] += time.perf_counter() - t0
            tot["pairs"] += int(row_ptr[-1])
            b, fb = cpu.attention_cpu(q, k, v, alpha, row_ptr, key_idx, acc_bits=128, tile=37, splits=8,
                                      order_seed=li * 100 + h, stats=sb, fallback=False)
            both = (fa == 0) & (fb == 0)
            tot["rows"] += L
            tot["flag_a"] += int((fa != 0).sum())
            tot["flag_b"] += int((fb != 0).sum())
            tot["ab_mismatch"] += int((~np.all(a == b, axis=1) & both).sum())
            # Accuracy vs float64 attention
            ref64 = float64_causal(q, k, v, alpha).astype(np.float32)
            from exactattn.formats import float_to_bf16_bits

            r16 = float_to_bf16_bits(ref64)  # float64 -> fp32 -> bf16; a reasonable truth proxy
            dist = np.abs(bf16_ordinal(a[both]) - bf16_ordinal(r16[both]))
            ulp_hist += np.array([(dist == 0).sum(), (dist == 1).sum(), (dist == 2).sum(), (dist > 2).sum()])
            s16 = float_to_bf16_bits(fp32_causal(q, k, v, alpha))
            dist32 = np.abs(bf16_ordinal(s16[both]) - bf16_ordinal(r16[both]))
            ulp_hist32 += np.array([(dist32 == 0).sum(), (dist32 == 1).sum(), (dist32 == 2).sum(), (dist32 > 2).sum()])
            # Where ExactAttention is > 2 ulp from float64: how small is the output relative to |v|?
            vmax = np.abs(bf16_bits_to_float32(v)).max(axis=0)  # per channel
            rel = np.abs(ref64[both]) / np.maximum(vmax[None, :], 1e-30)
            far_rel.extend(rel[dist > 2].tolist())
        # Exact reference spot checks
        for _ in range(args.ref_samples_per_layer):
            h = int(rng.integers(0, Q.shape[0]))
            pos = int(rng.integers(0, L))
            kv = h // g
            ref = attention_row(Q[h, pos], K[kv, : pos + 1], V[kv, : pos + 1], alpha)
            got, f = cpu.attention_cpu(Q[h, pos:pos + 1], K[kv, : pos + 1], V[kv, : pos + 1], alpha,
                                       splits=3, fallback=False)
            if f[0] == 0:
                tot["ref_checked"] += 1
                tot["ref_mismatch"] += int(not np.array_equal(got[0], ref))
        print(f"layer {li}: {tot}", flush=True)

    thr = tot["pairs"] / tot["time_a"]
    n_el = ulp_hist.sum()
    lines = [
        f"# Full-model validation: {meta['model']}",
        "",
        f"Every causal query row of every layer and head: {len(layers)} layers × "
        f"{meta['layers'][str(layers[0])]['q_shape'][0]} heads × {L} positions = {tot['rows']:,} rows, "
        f"{tot['pairs']:,} query-key pairs. Generated by `scripts/validate_model.py` (C++ kernel, CPU).",
        "",
        "| check | result |",
        "|---|---|",
        f"| rows not certified, schedule A (1 split, 64-key tiles, int64) | {tot['flag_a']} / {tot['rows']:,} |",
        f"| rows not certified, schedule B (8 splits, 37-key tiles, shuffled merges, int128) | {tot['flag_b']} / {tot['rows']:,} |",
        f"| rows where A ≠ B (both certified) | **{tot['ab_mismatch']}** |",
        f"| sampled rows ≠ exact Python reference | **{tot['ref_mismatch']}** / {tot['ref_checked']} |",
        "",
        "Accuracy of bf16 outputs against float64 attention (rounded to bf16), compared with standard",
        "fp32 attention computed FlashAttention-style (fp32 scores, fp32 exp2, fp32 sums):",
        "",
        "| distance from float64 | ExactAttention | share | standard fp32 | share |",
        "|---|---|---|---|---|",
    ]
    for lbl, c, c32 in zip(["0 ulp", "1 ulp", "2 ulp", "> 2 ulp"], ulp_hist, ulp_hist32):
        lines.append(f"| {lbl} | {int(c):,} | {c / n_el:.6f} | {int(c32):,} | {c32 / n_el:.6f} |")
    far = np.array(far_rel) if far_rel else np.zeros(0)
    lines += [
        "",
        "One-ulp differences are expected and are not errors in ExactAttention. The float64 proxy",
        "rounds twice (float64 → fp32 → bf16) and uses unrounded weights, so near-ties can land on",
        "either side.",
        "",
        f"The elements more than 2 ulp away ({far.size}) are outputs with heavy cancellation:",
        f"median |o| / max|v| = {np.median(far) if far.size else float('nan'):.2e}, max "
        f"{far.max() if far.size else float('nan'):.2e}. There the fp32 rounding of the logits",
        "(SPEC step 2) is amplified. Standard fp32 attention rounds scores to fp32 as well, and is",
        "shown in the column above for comparison.",
        "",
        f"Throughput, schedule A: {thr / 1e6:.1f} M query-key pairs/s on {os.cpu_count()} CPU threads "
        f"(d = 128, d_v = 128).",
    ]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
