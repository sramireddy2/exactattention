"""Capture real post-RoPE q, k and v from every attention layer of a small LLM on CPU.

Usage:
    python scripts/capture_qkv.py --model Qwen/Qwen3-0.6B --tokens 2048 --out <dir>

Writes <out>/layer{i:02d}_{q,k,v}.npy as uint16 bf16 bit patterns, with shapes
(heads, tokens, head_dim), plus meta.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import sdpa_attention_forward

CAPTURED: dict[int, dict[str, np.ndarray]] = {}


def bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def capture_attention(module, query, key, value, attention_mask, **kwargs):
    li = module.layer_idx
    if li not in CAPTURED:
        CAPTURED[li] = {
            "q": bf16_bits(query[0]),  # (heads, tokens, d)
            "k": bf16_bits(key[0]),    # (kv_heads, tokens, d)
            "v": bf16_bits(value[0]),
            "scaling": float(kwargs.get("scaling") or module.scaling),
            "groups": int(getattr(module, "num_key_value_groups", 1)),
        }
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def load_text(min_chars: int) -> str:
    """WikiText-2 test split (raw), via the parquet file on the HF Hub."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    lines = pq.read_table(path).column("text").to_pylist()
    text = "".join(lines)
    return text[: max(min_chars, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    AttentionInterface.register("capture", capture_attention)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="capture")
    model.eval()
    text = load_text(args.tokens * 8)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens]
    with torch.no_grad():
        model(ids)
    meta = {"model": args.model, "tokens": int(ids.shape[1]), "layers": {}}
    for li, d in sorted(CAPTURED.items()):
        for name in ("q", "k", "v"):
            np.save(os.path.join(args.out, f"layer{li:02d}_{name}.npy"), d[name])
        meta["layers"][li] = {"scaling": d["scaling"], "groups": d["groups"],
                              "q_shape": list(d["q"].shape), "k_shape": list(d["k"].shape)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"captured {len(CAPTURED)} layers, {ids.shape[1]} tokens -> {args.out}")


if __name__ == "__main__":
    main()
