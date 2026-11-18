"""Where an exact step's GPU time goes: device time per kernel, and how much work tier 2 reruns
(DESIGN D18). Same data and layout as scripts/bench_vs_flashinfer.py.

Tier 2 recomputes, with int64 P·V, each (row, 32-channel block) pair the tensor-core P·V could
not certify, over every KV split, from a work list the tier-1 combine appends to (DESIGN D23).
So its cost follows the number of such pairs.

For tcpv cases it also runs the tier-1 split kernel's diagnostic phase clocks (TIMERS, DESIGN
D26): SM cycles per program in the prologue, in each phase of a key tile (scores, weights, tile
partial, merge) and elsewhere in the loop, plus each program's start, end and SM. From those:
cycles per tile by phase, how busy the SMs are, and how many programs share an SM.

    python scripts/profile_step.py --out results/profile_step.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(__file__))
import bench_vs_flashinfer as bench  # noqa: E402
from exactattn.triton_kernels import _LAST_DEBUG  # noqa: E402

# (T, B, n, mode, block_dv, num_warps, block_m, block_n, sq); the n32 and q3 rows are A/Bs.
CASES = [(1, 1, 16384, "tcpv", 128, 8, None, 64, 4), (8, 1, 16384, "tcpv", 128, 8, None, 64, 4),
         (8, 1, 16384, "tcpv", 128, 8, None, 64, 3), (8, 1, 16384, "tcpv", 128, 8, None, 32, 4),
         (8, 8, 4096, "tcpv", 128, 8, None, 64, 4), (8, 1, 16384, "tc", 64, 8, None, 16, 4),
         (1, 8, 16384, "tcpv", 128, 8, None, 64, 4)]


def kernel_times(fn, iters):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    # Tier 1 and tier 2 launch the same kernels, so label each launch by its place in the step.
    evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    evs.sort(key=lambda e: e.time_range.start)
    per = max(1, len(evs) // iters)
    out = {}
    for i, e in enumerate(evs[:per * iters]):
        key = f"{i % per}: {e.name}"
        ms, _ = out.get(key, (0.0, 1))
        out[key] = (ms + e.time_range.elapsed_us() / 1000.0 / iters, 1)
    return out


def timer_report(fn, sms):
    """Markdown lines from the split kernel's phase clocks (columns: see triton_kernels TIMERS)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    TM = _LAST_DEBUG["timers"].cpu().numpy().astype(np.float64)
    pro, s1, s2, s3, s4, nt, loop, tot, g0, g1, sm, wf, ef, mx, pm, ps, uc, ul, ue, uv, ux, cf, ct = (
        TM[:, i] for i in range(23))
    dur = g1 - g0                                             # ns per program
    ghz = tot.sum() / dur.sum()
    span = g1.max() - g0.min()
    conc = 0                                                  # most programs at once on one SM
    for m in np.unique(sm):
        ev = sorted([(a, 1) for a in g0[sm == m]] + [(b, -1) for b in g1[sm == m]])
        c = 0
        for _, d in ev:
            c += d
            conc = max(conc, c)
    tiles = max(nt.sum(), 1.0)
    ph = {"scores (_tc_logits, windowed fallback)": s1.sum(), "weights (exp2)": s2.sum(),
          "tile partial (Z, _tc_pv)": s3.sum(), "merge": s4.sum()}
    ph["rest of the loop (load waits, loop control)"] = loop.sum() - sum(ph.values())
    lines = [f"Phase clocks (TIMERS build): {len(TM)} programs on {len(np.unique(sm))} SMs "
             f"(at most {conc} at once per SM), {nt.mean():.1f} tiles each; clock {ghz:.2f} GHz; "
             f"launch span {span / 1e3:.1f} us; SMs busy {100 * dur.sum() / (span * sms):.0f}% of the span; "
             f"program {dur.mean() / 1e3:.1f} us mean ({dur.min() / 1e3:.1f}-{dur.max() / 1e3:.1f}).", "",
             "| part | cycles per tile | us per tile | share of program time |", "|---|---|---|---|"]
    for k, v in ph.items():
        lines.append(f"| {k} | {v / tiles:.0f} | {v / tiles / ghz / 1e3:.2f} | {100 * v / tot.sum():.1f}% |")
    lines.append(f"| prologue (per program) | {pro.mean():.0f} | {pro.mean() / ghz / 1e3:.2f} | "
                 f"{100 * pro.sum() / tot.sum():.1f}% |")
    epi = tot - pro - loop
    lines.append(f"| epilogue (per program) | {epi.mean():.0f} | {epi.mean() / ghz / 1e3:.2f} | "
                 f"{100 * epi.sum() / tot.sum():.1f}% |")
    # Fallbacks and stragglers (DESIGN D27).
    us = dur / 1e3
    anyf = (wf + ef) > 0

    def mean(x):
        return f"{x.mean():.1f}" if len(x) else "n/a"

    lines += ["", f"Fallbacks: windowed scores in {int(wf.sum())} of {int(tiles)} tiles "
              f"({int((wf > 0).sum())} programs; {int(ct.sum())} of them by key column, {int(cf.sum())} "
              f"columns, the other {int(wf.sum() - ct.sum())} whole tile), float64 exp2 in {int(ef.sum())} tiles "
              f"({int((ef > 0).sum())} programs). Program us mean: {mean(us[anyf])} with a fallback "
              f"({int(anyf.sum())}), {mean(us[~anyf])} without ({int((~anyf).sum())}). "
              f"Longest tile: {mx.max() / ghz / 1e3:.1f} us (median program's longest "
              f"{np.median(mx) / ghz / 1e3:.1f}).", "",
              # Where the uncertified scores sit (DESIGN D29): key columns per tile that has any.
              f"Uncertified scores: {int(ue.sum())} in {int(uc.sum())} key columns of {int(wf.sum())} tiles "
              f"({uc.sum() / max(1, wf.sum()):.2f} columns per such tile, at most {int(ux.max())}); "
              f"{int(ul.sum())} of the columns are keys that lost bits when sliced; the columns' "
              f"visible rows are {100 * ue.sum() / max(1, uv.sum()):.0f}% uncertified.", "",
              "Slowest programs (us; phases summed over the program's tiles):", "",
              "| pid_m | pid_s | SM | start | program | tiles | windowed | f64 exp2 | longest tile | "
              "prologue | scores | weights | partial | merge | rest |", "|" + "---|" * 15]

    def u(c):
        return f"{c / ghz / 1e3:.1f}"

    for i in np.argsort(-dur)[:8]:
        rest = loop[i] - s1[i] - s2[i] - s3[i] - s4[i]
        lines.append(f"| {int(pm[i])} | {int(ps[i])} | {int(sm[i])} | {(g0[i] - g0.min()) / 1e3:.1f} | "
                     f"{us[i]:.1f} | {int(nt[i])} | {int(wf[i])} | {int(ef[i])} | {u(mx[i])} | {u(pro[i])} | "
                     f"{u(s1[i])} | {u(s2[i])} | {u(s3[i])} | {u(s4[i])} | {u(rest)} |")
    return lines + [""]


def tier2_share():
    NEED = _LAST_DEBUG.get("tier2_need")
    if NEED is None:
        return "no tier 2"
    cnt = int(_LAST_DEBUG["tier2_cnt"][0])                  # work-list items: one pair each
    return (f"NEED pairs {int((NEED != 0).sum())}/{NEED.numel()}; tier 2 reran {cnt} work items "
            f"(row x 32-channel block, each over every KV split)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    lines = [f"# Exact step profile on {torch.cuda.get_device_name(0)}", "",
             "Device ms per step by kernel (torch.profiler, eager, mean of "
             f"{a.iters}), and tier 2's rerun share. Generated by `scripts/profile_step.py`.", ""]
    for T, B, n, mode, bdv, nw, bm, bn, sq in CASES:
        case = bench.Case(B, n, T)
        try:
            fn, unc, bad, out0, _, regs = case.exact(mode, sms, bdv, nw, bm, "int", bn, sq, None)
            share = tier2_share()
            kt = kernel_times(fn, a.iters)
        except Exception as ex:  # noqa: BLE001
            lines += [f"## T={T} B={B} n={n} {mode} {bdv}/{nw}/n{bn}/q{sq}: {type(ex).__name__}: {ex}", ""]
            print(lines[-2], flush=True)
            continue
        tot = sum(v[0] for v in kt.values())
        lines += [f"## T={T} B={B} n={n} {mode} {bdv}/{nw}/m{regs[2]}/n{bn}/q{sq}: {tot:.3f} ms, "
                  f"{unc} uncertified, {bad} wrong, regs {regs[0]} spill {regs[1]}", "", f"{share}", "",
                  "| launch (order: kernel) | ms |", "|---|---|"]
        for k, (ms, _) in sorted(kt.items(), key=lambda kv: int(kv[0].split(":")[0])):
            lines.append(f"| {k[:70]} | {ms:.3f} |")
        lines.append("")
        if mode == "tcpv":
            try:
                fn_t, unc_t, bad_t, out_t, _, _ = case.exact(mode, sms, bdv, nw, bm, "int", bn, sq, None, timers=True)
                same = bool(torch.equal(out_t, out0))
                kl = case._k_tc[3] & 1                                   # KLOST bit 0, per key
                lines += [f"Timers build: {unc_t} uncertified, {bad_t} wrong, output bits "
                          f"{'equal' if same else 'DIFFERENT'} to the plain build. Keys that lost bits "
                          f"when sliced: {int(kl.sum())} of {kl.numel()} in the cache.", ""]
                lines += timer_report(fn_t, sms)
                del fn_t
            except Exception as ex:  # noqa: BLE001
                lines += [f"Timers build failed: {type(ex).__name__}: {ex}", ""]
        print("\n".join(lines[-40:]), flush=True)
        del case, fn
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
