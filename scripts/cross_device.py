"""Cross-hardware bit-identity on real activations.

Step 1 (Windows, PyTorch XPU env): compute every causal row of the selected heads on a torch
device and save the outputs.
    python scripts/cross_device.py run --device xpu --data <captures> --heads 0:0,13:5,27:11 --save xpu.npz

Step 2 (Linux/WSL): run the C++ CPU kernel on the same rows and compare bit for bit.
    python scripts/cross_device.py compare --data <captures> --load xpu.npz --out results/cross_device.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def alpha_for(scaling):
    from exactattn.formats import FP32, encode_fp32, round_dyadic
    from mpmath import log, mp, mpf

    with mp.workprec(200):
        man, exp = mp.mpf(mpf(scaling) / log(2)).man_exp
    return encode_fp32(*round_dyadic(int(man), int(exp), FP32))


def load_head(data, meta, li, h):
    lm = meta["layers"][str(li)]
    Q = np.load(os.path.join(data, f"layer{li:02d}_q.npy"), mmap_mode="r")
    K = np.load(os.path.join(data, f"layer{li:02d}_k.npy"), mmap_mode="r")
    V = np.load(os.path.join(data, f"layer{li:02d}_v.npy"), mmap_mode="r")
    kv = h // lm["groups"]
    return np.array(Q[h]), np.array(K[kv]), np.array(V[kv]), alpha_for(lm["scaling"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--heads", default="0:0,13:5,27:11")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--save")
    ap.add_argument("--load")
    ap.add_argument("--out")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    heads = [tuple(int(x) for x in s.split(":")) for s in args.heads.split(",")]
    L = meta["tokens"]
    mask = np.tril(np.ones((L, L), dtype=bool))

    if args.mode == "run":
        import torch
        from exactattn.torch_impl import TorchStats, attention_torch

        outs, flags, info = {}, {}, {}
        for li, h in heads:
            q, k, v, alpha = load_head(args.data, meta, li, h)
            st = TorchStats()
            t0 = time.perf_counter()
            o, f = attention_torch(q, k, v, alpha, mask, device=args.device, tile=128, row_chunk=128,
                                   stats=st, fallback=False)
            if args.device == "xpu":
                torch.xpu.synchronize()
            dt = time.perf_counter() - t0
            outs[f"{li}_{h}"] = o
            flags[f"{li}_{h}"] = f
            info[f"{li}_{h}"] = {"seconds": dt, "flagged": int((f != 0).sum()), "oracle_weights": st.oracle_weights}
            print(li, h, info[f"{li}_{h}"], flush=True)
        dev_name = args.device
        if args.device == "xpu":
            dev_name = torch.xpu.get_device_name(0)
        np.savez(args.save, **{f"out_{k}": v for k, v in outs.items()}, **{f"flag_{k}": v for k, v in flags.items()},
                 info=json.dumps({"device": dev_name, "torch": torch.__version__, "heads": info}))
        return

    from exactattn import cpu

    z = np.load(args.load)
    info = json.loads(str(z["info"]))
    row_ptr = np.concatenate([[0], np.cumsum(np.arange(1, L + 1))]).astype(np.int64)
    key_idx = np.concatenate([np.arange(i + 1) for i in range(L)]).astype(np.int64)
    lines = [
        "# Cross-hardware bit identity on real activations",
        "",
        f"Model {meta['model']}, {L} tokens of WikiText-2. Every causal row of each head, computed on",
        f"two different processors with two different implementations:",
        "",
        f"* **GPU:** {info['device']}, through the tensor-op implementation (`torch_impl.py`, "
        f"PyTorch {info['torch']}), with 128-key tiles and int64 accumulators that keep 61 bits",
        "  of headroom",
        f"* **CPU:** the x86 C++ kernel (`exact_attention_cpu.cpp`, {os.cpu_count()} threads), with 8 splits,",
        "  shuffled merges, 37-key tiles and int128 accumulators",
        "",
        "| layer:head | rows | GPU uncertified | CPU uncertified | rows with any differing bit |",
        "|---|---|---|---|---|",
    ]
    total_rows = total_diff = 0
    for li, h in heads:
        q, k, v, alpha = load_head(args.data, meta, li, h)
        c, fc = cpu.attention_cpu(q, k, v, alpha, row_ptr, key_idx, acc_bits=128, tile=37, splits=8,
                                  order_seed=li * 31 + h, fallback=False)
        g, fg = z[f"out_{li}_{h}"], z[f"flag_{li}_{h}"]
        both = (fc == 0) & (fg == 0)
        diff = int((~np.all(c == g, axis=1) & both).sum())
        total_rows += L
        total_diff += diff
        lines.append(f"| {li}:{h} | {L} | {int((fg != 0).sum())} | {int((fc != 0).sum())} | **{diff}** |")
    lines += ["", f"Total: {total_rows:,} rows × 128 channels; rows with any differing bit: **{total_diff}**.",
              "", "GPU timings (tensor-op form, unfused; not a performance number): " +
              ", ".join(f"{k}: {v['seconds']:.1f}s" for k, v in info["heads"].items())]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
