"""ExactAttention vs FlashInfer / FlashAttention at serving shapes, on one NVIDIA GPU.

Workloads (bf16, d = d_v = 128, GQA like Qwen3-0.6B: 16 query heads over 8 KV heads):

  decode   B sequences, 1 new token each, n cached keys per sequence
  verify   B sequences, T draft tokens each (chain), causal among the drafts, n keys in total

Backends, each in its natural layout, all from the same random data:

  flashinfer   BatchDecodeWithPagedKVCacheWrapper (decode) / BatchPrefillWithPagedKVCacheWrapper
               (verify, causal), page size 16, NHD
  fa2-sdpa     torch SDPA forced to the FLASH_ATTENTION backend (FlashAttention-2 inside PyTorch);
               decode only (SDPA's is_causal is top-left aligned, wrong for verify)
  flash_attn   flash_attn_with_kvcache, if the flash-attn package is installed
  exact-*      ExactAttention Triton kernels, one segment per (sequence, KV head), the GQA
               group's query heads (x T drafts) as rows. int64 = windowed CUDA-core path;
               tc = int8 tensor-core scores; tcpv = tensor-core scores and P·V. q is sliced
               inside the kernel; K/V slicing is done once, as at cache write. Each exact mode
               is timed over several tile configs (channel block x warps); the table shows the
               best, its config and the split kernel's registers/spills. The spec makes the
               output independent of the tiling, so every config must give the same bits, and
               the benchmark checks that. Uncertified rows are counted, not recomputed (they
               would go to the exact path).

Timing: each backend's step is captured in a CUDA graph when possible (as serving engines run
decode) and replayed; otherwise eager. Mean of up to --iters runs after warmup (fewer for slow
steps, about half a second's worth, at least 3), CUDA events, warm L2.
`GB/s` counts only the bf16 K and V bytes every backend must read.

    python scripts/bench_vs_flashinfer.py --out results/bench_vs_flashinfer_<gpu>.md
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exactattn import default_alpha  # noqa: E402
from exactattn.triton_kernels import _LAST_DEBUG, launch, tc_slices_device, tc_v_slices_device  # noqa: E402

HQ, HKV, D = 16, 8, 128
DEV = os.environ.get("EA_BENCH_DEVICE", "cuda")   # "cpu" + TRITON_INTERPRET=1 for a smoke test
PAGE = 16
# (block_dv, num_warps) per exact mode. block_dv = 128 computes each score once per row block
# instead of once per channel block, at the price of larger P·V accumulators.
CONFIGS = {"int64": [(32, 4)], "tc": [(32, 4), (64, 8)],
           "tcpv": [(32, 4), (64, 4), (64, 8), (128, 8)]}


def time_step(fn, iters: int, warmup: int = 3):
    """Mean ms per call; tries a CUDA graph first. Returns (ms, 'graph' | 'eager')."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    mode = "eager"
    run = fn
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        run, mode = g.replay, "graph"
    except Exception:  # noqa: BLE001 - any capture failure means eager timing
        torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    run()
    en.record()
    torch.cuda.synchronize()
    iters = int(max(3, min(iters, 500.0 / max(st.elapsed_time(en), 1e-3))))
    st.record()
    for _ in range(iters):
        run()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters, mode


class Case:
    def __init__(self, B: int, n: int, T: int, seed: int = 0):
        self.B, self.n, self.T = B, n, T
        g = torch.Generator(device=DEV).manual_seed(seed)
        dev = DEV
        # Canonical data: K/V (B, HKV, n, D), new-token queries (B, T, HQ, D).
        self.k = torch.randn((B, HKV, n, D), generator=g, device=dev).to(torch.bfloat16)
        self.v = torch.randn((B, HKV, n, D), generator=g, device=dev).to(torch.bfloat16)
        self.q = torch.randn((B, T, HQ, D), generator=g, device=dev).to(torch.bfloat16)
        self.kv_bytes = 2 * B * HKV * n * D * 2

    # ---------------- FlashInfer ----------------
    def flashinfer(self):
        import flashinfer

        B, n, T = self.B, self.n, self.T
        npg = -(-n // PAGE)
        # NHD paged cache: (pages, 2, PAGE, HKV, D), each sequence's pages contiguous.
        kv = torch.zeros((B * npg, 2, PAGE, HKV, D), dtype=torch.bfloat16, device=DEV)
        kk = torch.zeros((B, npg * PAGE, HKV, D), dtype=torch.bfloat16, device=DEV)
        vv = torch.zeros_like(kk)
        kk[:, :n] = self.k.transpose(1, 2)
        vv[:, :n] = self.v.transpose(1, 2)
        kv[:, 0] = kk.reshape(B * npg, PAGE, HKV, D)
        kv[:, 1] = vv.reshape(B * npg, PAGE, HKV, D)
        i32 = dict(dtype=torch.int32, device=DEV)
        indptr = torch.arange(0, B + 1, **i32) * npg
        indices = torch.arange(0, B * npg, **i32)
        last = torch.full((B,), n - (npg - 1) * PAGE, **i32)
        ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEV)
        if T == 1:
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
            w.plan(indptr, indices, last, HQ, HKV, D, PAGE, pos_encoding_mode="NONE",
                   q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            q = self.q[:, 0].contiguous()                                  # (B, HQ, D)
        else:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
            qo = torch.arange(0, B + 1, **i32) * T
            w.plan(qo, indptr, indices, last, HQ, HKV, D, PAGE, causal=True,
                   q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            q = self.q.reshape(B * T, HQ, D).contiguous()
        out = w.run(q, kv)
        return (lambda: w.run(q, kv, out=out)), out

    # ---------------- FlashAttention-2 via SDPA ----------------
    def fa2_sdpa(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if self.T != 1:
            raise RuntimeError("decode only")
        q = self.q.permute(0, 2, 1, 3).contiguous()                         # (B, HQ, 1, D)
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, self.k, self.v, enable_gqa=True)
            k, v, note = self.k, self.v, ""
        except Exception:  # noqa: BLE001 - older flash backends lack GQA
            k = self.k.repeat_interleave(HQ // HKV, dim=1)
            v = self.v.repeat_interleave(HQ // HKV, dim=1)
            note = " (KV heads expanded)"

        def fn():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=(k is self.k))
        return fn, note

    # ---------------- flash-attn package ----------------
    def flash_attn(self):
        from flash_attn import flash_attn_with_kvcache

        kc = self.k.transpose(1, 2).contiguous()                            # (B, n, HKV, D)
        vc = self.v.transpose(1, 2).contiguous()
        q = self.q.contiguous()                                             # (B, T, HQ, D)
        seqlens = torch.full((self.B,), self.n, dtype=torch.int32, device=DEV)
        return lambda: flash_attn_with_kvcache(q, kc, vc, cache_seqlens=seqlens, causal=True)

    # ---------------- fp32 reference, for the correctness gate ----------------
    def reference(self):
        """fp32 attention in ExactAttention's (B, HKV, T*grp, D) row layout."""
        if getattr(self, "_ref", None) is None:
            B, n, T, grp = self.B, self.n, self.T, HQ // HKV
            q = self.q.float().view(B, T, HKV, grp, D).permute(0, 2, 1, 3, 4).reshape(B, HKV, T * grp, D)
            # Weights are 2^(alpha*s) with alpha = log2(e)/sqrt(d): softmax of alpha*s*ln 2.
            s = (q @ self.k.float().transpose(-1, -2)) * (float(default_alpha(D)) * 0.6931471805599453)
            t = torch.arange(T, device=DEV).repeat_interleave(grp)
            allowed = torch.arange(n, device=DEV)[None, :] < (n - T + 1 + t)[:, None]
            s = s.masked_fill(~allowed, float("-inf"))
            self._ref = torch.softmax(s, dim=-1) @ self.v.float()
        return self._ref

    # ---------------- ExactAttention ----------------
    def exact(self, mode: str, sms: int, block_dv: int = 32, num_warps: int = 4):
        B, n, T = self.B, self.n, self.T
        grp = HQ // HKV
        G, r_seg = B * HKV, T * grp
        # Rows of segment (b, h): draft t, query head h*grp + j  ->  row t*grp + j.
        qv = self.q.view(torch.int16).view(B, T, HKV, grp, D)
        Qbuf = torch.empty((B, HKV, T, grp, D), dtype=torch.int16, device=DEV)
        Qt = Qbuf.view(G, r_seg, D)
        Kt = self.k.view(torch.int16).reshape(G, n, D)
        Vt = self.v.view(torch.int16).reshape(G, n, D)
        t = torch.arange(T, device=DEV).repeat_interleave(grp)
        KL = (n - T + 1 + t).to(torch.int32).expand(G, r_seg).contiguous()
        alpha = default_alpha(D)
        block_m = 16
        ncb = D // block_dv
        progs = G * -(-r_seg // block_m) * ncb
        splits = int(max(1, min(n // 256, -(-4 * sms // progs))))
        kw = dict(splits=splits, block_m=block_m, block_n=32 if mode == "tcpv" else 16, block_dv=block_dv,
                  num_warps=num_warps)
        if mode != "int64":
            if getattr(self, "_k_tc", None) is None:
                self._k_tc = tc_slices_device(Kt.reshape(G * n, D), 4)      # once, at cache write
            kw["k_tc"], kw["sq"] = self._k_tc, 3                            # q is sliced in the kernel
        if mode == "tcpv":
            if getattr(self, "_v_tc", None) is None:
                self._v_tc = tc_v_slices_device(Vt.reshape(G * n, D), 3)
            kw["v_tc"] = self._v_tc

        def fn():
            Qbuf.copy_(qv.permute(0, 2, 1, 3, 4))                          # layout change, timed
            return launch(Qt, Kt, Vt, KL, n, alpha, **kw)

        OUT, FL, OF = fn()
        regs = (_LAST_DEBUG.get("n_regs"), _LAST_DEBUG.get("n_spills"))
        if DEV == "cuda":
            torch.cuda.synchronize()
        flagged = (FL.amax(dim=(0, 2)) > 0) | (OF.amax(dim=1) > 0)
        unc = int(flagged.sum())
        # Correctness gate: certified rows are correctly rounded bf16, so they lie within half a
        # bf16 ulp of the true value; fp32 reference noise is far below 2^-8 relative.
        got = (OUT.to(torch.int32) << 16).view(torch.float32).view(B, HKV, r_seg, D)
        ref = self.reference()
        off = (got - ref).abs() > (ref.abs() * 2.0 ** -8 + 1e-6)
        off = off.reshape(G * r_seg, D).any(dim=1) & ~flagged
        return fn, unc, int(off.sum()), OUT.clone(), flagged, regs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", default="1,8,32")
    ap.add_argument("--keys", default="1024,4096,16384")
    ap.add_argument("--drafts", default="1,8", help="T: 1 = decode, >1 = verify T draft tokens")
    ap.add_argument("--modes", default="int64,tc,tcpv")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    name = torch.cuda.get_device_name(0)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    vers = [f"torch {torch.__version__}"]
    for mod in ("triton", "flashinfer", "flash_attn"):
        try:
            vers.append(f"{mod} {__import__(mod).__version__}")
        except Exception:  # noqa: BLE001
            vers.append(f"{mod} n/a")
    modes = args.modes.split(",")
    cols = ["flashinfer", "fa2-sdpa", "flash_attn"] + [f"exact-{m}" for m in modes]
    lines = [f"# ExactAttention vs FlashInfer / FlashAttention on {name}", "",
             f"{', '.join(vers)}. {HQ} query heads, {HKV} KV heads, d = 128, bf16. ms per step "
             f"(mean of {args.iters}); `g` = CUDA graph, `e` = eager. Generated by `scripts/bench_vs_flashinfer.py`.", "",
             "| T | B | n | " + " | ".join(cols) + " | best exact / flashinfer | flashinfer GB/s | exact uncertified rows (of B*16*T) |",
             "|---|---|---|" + "---|" * len(cols) + "---|---|---|"]
    errors = []
    for T in [int(x) for x in args.drafts.split(",")]:
        for B in [int(x) for x in args.batch.split(",")]:
            for n in [int(x) for x in args.keys.split(",")]:
                case = Case(B, n, T)
                res, unc = {}, []

                def record(col, thunk):
                    try:
                        fn, note = thunk()
                        ms, mode = time_step(fn, args.iters)
                        res[col] = (ms, f"{ms:.3f} {mode[0]}{note}")
                    except Exception as ex:  # noqa: BLE001 - a missing backend is a table cell
                        res[col] = (None, "n/a")
                        errors.append((col, f"{type(ex).__name__}: {ex} (T={T} B={B} n={n})"))
                        if os.environ.get("EA_BENCH_TRACE"):
                            traceback.print_exc()

                record("flashinfer", lambda: (case.flashinfer()[0], ""))
                record("fa2-sdpa", case.fa2_sdpa)
                record("flash_attn", lambda: (case.flash_attn(), ""))
                for m in modes:
                    best, first, note_u = None, None, ""
                    for bdv, nw in CONFIGS[m]:
                        try:
                            fn, u, bad, out, flg, (nr, ns) = case.exact(m, sms, bdv, nw)
                            ms, gmode = time_step(fn, args.iters)
                        except Exception as ex:  # noqa: BLE001
                            errors.append((f"exact-{m}", f"{type(ex).__name__}: {ex} (T={T} B={B} n={n} cfg={bdv}/{nw})"))
                            if os.environ.get("EA_BENCH_TRACE"):
                                traceback.print_exc()
                            continue
                        if first is None:
                            first = (out, flg)
                            note_u = f"{m}:{u}" + (f" **{bad} rows WRONG**" if bad else "")
                        else:
                            both = ~(first[1] | flg)                     # rows certified by both
                            o1 = first[0].reshape(-1, D)[both]
                            if not torch.equal(o1, out.reshape(-1, D)[both]):
                                note_u += f" **{m} config {bdv}/{nw} changed bits**"
                        if best is None or ms < best[0]:
                            sp = f" r{nr}" + (f" spill{ns}" if ns else "") if nr is not None else ""
                            best = (ms, f"{ms:.3f} {gmode[0]} [{bdv}/{nw}{sp}]")
                        del fn, out, flg
                    res[f"exact-{m}"] = best if best else (None, "n/a")
                    if note_u:
                        unc.append(note_u)
                fi = res["flashinfer"][0]
                ex_best = min((res[f"exact-{m}"][0] for m in modes if res[f"exact-{m}"][0]), default=None)
                ratio = f"{ex_best / fi:.1f}x" if fi and ex_best else "n/a"
                gbs = f"{case.kv_bytes / (fi * 1e-3) / 1e9:.0f}" if fi else "n/a"
                lines.append(f"| {T} | {B} | {n} | " + " | ".join(res[c][1] for c in cols)
                             + f" | {ratio} | {gbs} | {' '.join(unc)} |")
                print(lines[-1], flush=True)
                del case
                torch.cuda.empty_cache()
    lines.insert(4, "Exact cells: `ms mode [block_dv/num_warps rREGS spillBYTES]` for the fastest tile config.")
    lines.insert(5, "")
    if errors:
        lines += ["", "Backend errors (first per backend and error type):", ""]
        seen = set()
        for col, e in errors:
            key = (col, e.split(":")[0])
            if key not in seen:
                seen.add(key)
                lines.append(f"- {col}: {e[:300]}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
