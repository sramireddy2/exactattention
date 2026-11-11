"""Portable bit-identity fixture: real attention heads plus the SHA-256 of their exact outputs.

    make   (once, from captures):  python scripts/fixture.py make --data <captures> --heads 0:0,27:11
    check  (any machine):          python scripts/fixture.py check --backend cpp|torch:cuda|torch:xpu|triton:cuda|...

`check` runs every causal row of each fixture head through the chosen backend, hashes the bf16
output bits, and compares the hash with the one recorded at `make` time. The recorded hash
comes from rows checked against the exact reference.

It also hashes torch's standard scaled_dot_product_attention on the same device. That hash
is device-dependent, which is the point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
FIX = os.path.join(ROOT, "fixtures", "qwen3_0.6b_heads.npz")
EXPECTED = os.path.join(ROOT, "fixtures", "expected_sha256.json")


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_backend(backend, q, k, v, alpha):
    L = q.shape[0]
    kv_len = np.arange(1, L + 1)
    if backend == "cpp":
        from exactattn import cpu

        row_ptr = np.concatenate([[0], np.cumsum(kv_len)]).astype(np.int64)
        key_idx = np.concatenate([np.arange(n) for n in kv_len]).astype(np.int64)
        return cpu.attention_cpu(q, k, v, alpha, row_ptr, key_idx, splits=8, tile=64)
    kind, _, dev = backend.partition(":")
    if kind == "torch":
        from exactattn.torch_impl import attention_torch

        mask = np.tril(np.ones((L, L), bool))
        return attention_torch(q, k, v, alpha, mask, device=dev or "cpu", tile=128, row_chunk=128)
    if kind in ("triton", "tritontc"):
        if not dev or dev == "cpu":
            os.environ["TRITON_INTERPRET"] = "1"
        from exactattn.triton_kernels import attention_triton

        return attention_triton(q, k, v, alpha, kv_len, splits=8, block_m=16, block_n=16, block_dv=32,
                                device=dev or "cpu", use_tc=(kind == "tritontc"), sq=3, sk=4)
    raise ValueError(backend)


def sdpa_hash(q, k, v, dev):
    import torch
    import torch.nn.functional as F

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)[None, None].to(dev)

    o = F.scaled_dot_product_attention(t(q), t(k), t(v), is_causal=True).to(torch.bfloat16)
    return sha(o.view(torch.int16).cpu().numpy().view(np.uint16)[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "check"])
    ap.add_argument("--data")
    ap.add_argument("--heads", default="0:0,27:11")
    ap.add_argument("--backend", default="cpp")
    ap.add_argument("--only", default=None, help="comma-separated head keys to check (default: all)")
    args = ap.parse_args()

    if args.mode == "make":
        from exactattn import attention_row, cpu
        from validate_model import alpha_for  # noqa: F401  (scripts/ on the path)

        meta = json.load(open(os.path.join(args.data, "meta.json")))
        arrays, expected = {}, {"model": meta["model"], "tokens": meta["tokens"], "heads": {}}
        rng = np.random.default_rng(0)
        for spec in args.heads.split(","):
            li, h = (int(x) for x in spec.split(":"))
            lm = meta["layers"][str(li)]
            kv = h // lm["groups"]
            q = np.load(os.path.join(args.data, f"layer{li:02d}_q.npy"))[h]
            k = np.load(os.path.join(args.data, f"layer{li:02d}_k.npy"))[kv]
            v = np.load(os.path.join(args.data, f"layer{li:02d}_v.npy"))[kv]
            alpha = alpha_for(lm["scaling"])
            out, flags = run_backend("cpp", q, k, v, alpha)
            for pos in rng.choice(q.shape[0], size=8, replace=False):  # spot-check against the reference
                assert np.array_equal(out[pos], attention_row(q[pos], k[: pos + 1], v[: pos + 1], alpha))
            key = f"{li}_{h}"
            arrays.update({f"q_{key}": q, f"k_{key}": k, f"v_{key}": v, f"alpha_{key}": np.float32(alpha)})
            expected["heads"][key] = {"sha256": sha(out), "rows": int(q.shape[0])}
            print(key, expected["heads"][key])
        os.makedirs(os.path.dirname(FIX), exist_ok=True)
        np.savez_compressed(FIX, **arrays)
        with open(EXPECTED, "w") as f:
            json.dump(expected, f, indent=1)
        return

    z = np.load(FIX)
    expected = json.load(open(EXPECTED))
    ok = True
    print(f"host {platform.node()} | {platform.processor() or platform.machine()} | backend {args.backend}")
    only = set(args.only.split(",")) if args.only else None   # e.g. --only 0_0 (one head per stage)
    for key, info in expected["heads"].items():
        if only is not None and key not in only:
            continue
        q, k, v, alpha = z[f"q_{key}"], z[f"k_{key}"], z[f"v_{key}"], np.float32(z[f"alpha_{key}"])
        t0 = time.perf_counter()
        out, flags = run_backend(args.backend, q, k, v, alpha)
        dt = time.perf_counter() - t0
        h = sha(out)
        match = h == info["sha256"]
        ok &= match
        line = f"head {key}: {'MATCH' if match else 'MISMATCH'} sha256 {h[:16]}…  ({dt:.1f}s, {int((flags != 0).sum())} fallback rows)"
        dev = args.backend.partition(":")[2]
        if dev:
            try:
                line += f" | standard SDPA on {dev}: {sdpa_hash(q, k, v, dev)[:16]}…"
            except Exception as e:  # noqa: BLE001
                line += f" | SDPA unavailable ({type(e).__name__})"
        print(line, flush=True)
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
