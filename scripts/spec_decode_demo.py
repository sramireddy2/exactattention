"""Bit-identical speculative decoding with a real model (Qwen3-0.6B, CPU).

Greedy decoding two ways, comparing every emitted token and every logit bit:
  sequential   one token per forward pass (decode, M = 1)
  speculative  n-gram drafts of k tokens, verified in one forward pass (M = k + 1), accepted
               greedily, rejected KV cropped from the cache

Two numerics modes:
  standard     PyTorch SDPA plus batched matmuls. Verify and decode run different kernels
               and shapes, so the logits differ in their low bits.
  exact        ExactAttention C++ kernel for attention. Every linear and RMSNorm is computed
               one token row at a time, standing in for a batch-invariant GEMM, which is solved
               elsewhere (Thinking Machines, FlashInfer). Elementwise ops are row-independent
               already.

The claim checked: in exact mode, speculative decoding returns the same tokens **and the same
logit bits** as sequential decoding.

Usage (Linux/WSL, after `make -C csrc`):
    python scripts/spec_decode_demo.py --new-tokens 48 --draft 4 --out results/spec_decode_qwen3-0.6b.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402
from transformers.models.qwen3 import modeling_qwen3  # noqa: E402

from exactattn import cpu  # noqa: E402

_ALPHA_CACHE: dict[float, np.float32] = {}


def _alpha(scaling: float) -> np.float32:
    if scaling not in _ALPHA_CACHE:
        from validate_model import alpha_for

        _ALPHA_CACHE[scaling] = alpha_for(scaling)
    return _ALPHA_CACHE[scaling]


def _bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def exact_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """ExactAttention for one sequence: query (1, H, ql, D), key/value (1, Hkv, kl, D), causal with
    the queries at the end of the key range."""
    _, H, ql, D = query.shape
    Hkv, kl = key.shape[1], key.shape[2]
    g = H // Hkv
    q = _bits(query[0]).reshape(H * ql, D)
    k = _bits(key[0]).reshape(Hkv * kl, D)
    v = _bits(value[0]).reshape(Hkv * kl, D)
    lens = np.tile(np.arange(kl - ql + 1, kl + 1, dtype=np.int64), H)    # per row
    heads = np.repeat(np.arange(H) // g, ql).astype(np.int64)
    row_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    key_idx = np.concatenate([hk * kl + np.arange(n) for hk, n in zip(heads, lens)]).astype(np.int64)
    out, _ = cpu.attention_cpu(q, k, v, _alpha(float(scaling or module.scaling)), row_ptr, key_idx,
                               splits=4, tile=64)
    o = torch.from_numpy(out.view(np.int16)).view(torch.bfloat16).reshape(H, ql, D)
    return o.permute(1, 0, 2).unsqueeze(0).contiguous(), None


def install_rowwise():
    """Make linear and RMSNorm row-independent: each token row is computed alone, so the result
    for a row cannot depend on how many rows share the call."""
    lin_fwd = torch.nn.Linear.forward
    norm_fwd = modeling_qwen3.Qwen3RMSNorm.forward

    def linear_rows(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.cat([lin_fwd(self, x2[i:i + 1]) for i in range(x2.shape[0])])
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def norm_rows(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.cat([norm_fwd(self, x2[i:i + 1]) for i in range(x2.shape[0])])
        return out.reshape(x.shape)

    torch.nn.Linear.forward = linear_rows
    modeling_qwen3.Qwen3RMSNorm.forward = norm_rows


def ngram_draft(seq: list[int], k: int, n: int = 2) -> list[int]:
    """Prompt-lookup drafting: find the previous occurrence of the trailing n-gram and propose the k
    tokens that followed it."""
    tail = seq[-n:]
    for start in range(len(seq) - n - 1, -1, -1):
        if seq[start:start + n] == tail:
            cont = seq[start + n:start + n + k]
            if cont:
                return (cont + [cont[-1]] * k)[:k]
    return [seq[-1]] * k


@torch.no_grad()
def forward(model, ids, cache):
    out = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True)
    return out.logits[0]  # (len(ids), vocab)


@torch.no_grad()
def sequential(model, prompt, n_new):
    cache = DynamicCache()
    logits = forward(model, prompt, cache)
    toks, rows = [], []
    last = logits[-1]
    for _ in range(n_new):
        t = int(torch.argmax(last))
        toks.append(t)
        rows.append(_bits(last))
        last = forward(model, [t], cache)[-1]
    return toks, rows, 1 + n_new


@torch.no_grad()
def speculative(model, prompt, n_new, k):
    cache = DynamicCache()
    logits = forward(model, prompt, cache)
    t0 = int(torch.argmax(logits[-1]))
    toks, rows = [t0], [_bits(logits[-1])]
    seq = list(prompt) + [t0]
    passes, accepted, drafted = 1, 0, 0
    while len(toks) < n_new:
        draft = ngram_draft(seq, k)
        base = cache.get_seq_length()
        lg = forward(model, [seq[-1]] + draft, cache)   # verify: k + 1 rows in one pass
        passes += 1
        drafted += k
        n_acc = 0
        for j in range(k + 1):
            pred = int(torch.argmax(lg[j]))
            toks.append(pred)
            rows.append(_bits(lg[j]))
            seq.append(pred)
            if len(toks) >= n_new:
                break
            if j < k and pred == draft[j]:
                n_acc += 1
                continue
            break
        accepted += n_acc
        # Keep the KV for the pending token and the accepted drafts; drop the rejected ones.
        remove = cache.get_seq_length() - (base + 1 + n_acc)
        if remove > 0:
            cache.crop(-remove)
    return toks[:n_new], rows[:n_new], passes, accepted, drafted


def run_mode(mode, args, prompt_ids):
    if mode == "exact":
        AttentionInterface.register("exact", exact_attention)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="exact" if mode == "exact" else "sdpa")
    model.eval()
    t0 = time.perf_counter()
    st, sr, sp = sequential(model, prompt_ids, args.new_tokens)
    t1 = time.perf_counter()
    pt, pr, pp, acc, dr = speculative(model, prompt_ids, args.new_tokens, args.draft)
    t2 = time.perf_counter()
    diff_rows = sum(int(not np.array_equal(a, b)) for a, b in zip(sr, pr))
    first_tok_diff = next((i for i, (a, b) in enumerate(zip(st, pt)) if a != b), None)
    return {
        "tokens_equal": st == pt, "first_token_diff": first_tok_diff, "logit_rows_differing": diff_rows,
        "rows": len(sr), "seq_passes": sp, "spec_passes": pp, "accepted": acc, "drafted": dr,
        "seq_s": t1 - t0, "spec_s": t2 - t1, "text": None, "tokens": pt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--draft", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    tok = AutoTokenizer.from_pretrained(args.model)
    from capture_qkv import load_text

    text = load_text(4000)
    prompt_ids = tok(text).input_ids[200:200 + args.prompt_tokens]
    res = {"standard": run_mode("standard", args, prompt_ids)}
    install_rowwise()
    res["rows_only"] = run_mode("rows_only", args, prompt_ids)  # row-invariant linears, SDPA attention
    res["exact"] = run_mode("exact", args, prompt_ids)
    for m in res:
        res[m]["text"] = tok.decode(res[m]["tokens"])
    lines = [
        f"# Bit-identical speculative decoding: {args.model}",
        "",
        f"Greedy decoding of {args.new_tokens} tokens after a {len(prompt_ids)}-token WikiText-2 prompt,",
        f"on CPU. The speculative run uses prompt-lookup drafts of {args.draft} tokens, verified in one",
        "forward pass, with rejected KV cropped from the cache. Generated by `scripts/spec_decode_demo.py`.",
        "",
        "| | standard (SDPA, batched matmuls) | row-invariant linears + SDPA | exact (ExactAttention + row-invariant linears) |",
        "|---|---|---|---|",
    ]
    S, Ro, E = res["standard"], res["rows_only"], res["exact"]
    cols = (S, Ro, E)
    lines += [
        "| tokens identical to sequential decoding | " + " | ".join(str(m["tokens_equal"]) for m in cols) + " |",
        "| first differing token position | " + " | ".join("—" if m["first_token_diff"] is None else str(m["first_token_diff"]) for m in cols) + " |",
        "| logit rows with any differing bit | " + " | ".join(f"**{m['logit_rows_differing']} / {m['rows']}**" for m in cols) + " |",
        "| forward passes: sequential → speculative | " + " | ".join(f"{m['seq_passes']} → {m['spec_passes']}" for m in cols) + " |",
        "| draft tokens accepted | " + " | ".join(f"{m['accepted']} / {m['drafted']}" for m in cols) + " |",
        "| wall time: sequential / speculative (s) | " + " | ".join(f"{m['seq_s']:.1f} / {m['spec_s']:.1f}" for m in cols) + " |",
        "",
        ("The middle column isolates attention. With the linears made row-invariant and standard "
         "SDPA attention, verification still diverges from decoding, so the attention kernel "
         "itself depends on how many query rows are processed together."
         if Ro["logit_rows_differing"] > 0 else
         "The middle column isolates attention. With the linears made row-invariant, SDPA "
         "verification already matched decoding on this run, so the divergence in the standard "
         "column comes from the batched linears. ExactAttention's contribution is the guarantee: "
         "it holds by construction for any split, tile, draft tree or device, rather than "
         "happening to hold for one kernel configuration."),
        "",
        "In the exact column, each logit row the speculative run uses to pick a token is bit-identical",
        "to the row sequential decoding computes at that position. So the two runs agree on every",
        "token, not only in distribution: verifying k + 1 rows at once and decoding one row at a",
        "time are the same computation.",
        "",
        "Exact-mode wall time is not a performance number: the row-wise linears are a CPU",
        "stand-in for a batch-invariant GEMM.",
        "",
        f"Sample output (exact): {E['text']!r}",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
