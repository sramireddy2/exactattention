"""What the exact step's split kernel waits on: Nsight Compute on one step (DESIGN D31).

D30 left verify B=1 16K's tier 1 at about 24 us per 64-key tile, 14.5 of them in the key loop
outside the four timed phases: load waits and loop control. The phase clocks cannot say which
loads, or whether DRAM, L2 or latency is the limit. Nsight Compute can. For each case this
profiles the step's two split-kernel launches (tier 1 and tier 2) and reports:
- duration, DRAM / L2 / L1 throughput, bytes read from DRAM, L1 and L2 hit rates;
- achieved occupancy, registers, warp cycles per issued instruction;
- the warp-stall breakdown (sampled, per issued instruction);
- per launch, stall samples by SASS opcode, the float64 share, and the hottest lines with
  their main stall reasons.

The cache is not flushed between replay passes (--cache-control none), as in a decode loop where
the previous step leaves data in L2. Kernel replay saves and restores device memory, so L2
contents can still differ from a real step.

    python scripts/ncu_step.py --out results/ncu_step.md [--ncu /path/to/ncu]

It runs itself under ncu once per case (--target), with profiling limited to one step.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, T, B, n, mode, block_dv, num_warps, block_m, block_n, sq), as in profile_step.
CASES = [("decode_b1_16k", 1, 1, 16384, "tcpv", 128, 8, None, 64, 4),
         ("verify_b1_16k", 8, 1, 16384, "tcpv", 128, 8, None, 64, 4),
         ("verify_b8_4k", 8, 8, 4096, "tcpv", 128, 8, None, 64, 4),
         ("decode_b8_16k", 1, 8, 16384, "tcpv", 128, 8, None, 64, 4)]

SECTIONS = ["SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "LaunchStats", "WarpStateStats",
            "SchedulerStats", "SourceCounters"]
EXTRA = ["dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_sector_hit_rate.pct",
         "l1tex__t_sector_hit_rate.pct", "sm__warps_active.avg.pct_of_peak_sustained_active",
         "smsp__inst_executed.sum", "gpu__time_duration.sum"]
STALL = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
# Speed-of-light and memory rows worth a table line, by the details page's metric names.
KEEP = ["Duration", "DRAM Throughput", "Memory Throughput", "L1/TEX Cache Throughput", "L2 Cache Throughput",
        "Compute (SM) Throughput", "Mem Busy", "Max Bandwidth", "L1/TEX Hit Rate", "L2 Hit Rate",
        "Achieved Occupancy", "Theoretical Occupancy", "Registers Per Thread", "Grid Size", "Waves Per SM",
        "Warp Cycles Per Issued Instruction", "Issue Slots Busy", "Issued Warp Per Scheduler",
        "No Eligible", "Active Warps Per Scheduler", "Eligible Warps Per Scheduler",
        "Dynamic Shared Memory Per Block", "Static Shared Memory Per Block"]


def target(name):
    """Build the case, warm up, then run one step inside a cudaProfilerStart/Stop range."""
    import torch
    sys.path.insert(0, HERE)
    import bench_vs_flashinfer as bench
    case_ = next(c for c in CASES if c[0] == name)
    _, T, B, n, mode, bdv, nw, bm, bn, sq = case_
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    case = bench.Case(B, n, T)
    fn, unc, bad, _, _, _ = case.exact(mode, sms, bdv, nw, bm, "int", bn, sq, None)
    print(f"{name}: {unc} uncertified, {bad} wrong", flush=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def csv_rows(text):
    """Rows of ncu's CSV output, skipping its ==PROF== and other non-CSV lines."""
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    return list(csv.reader(io.StringIO("\n".join(lines))))


def run(cmd, timeout, env=None):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr, time.time() - t0
    except subprocess.TimeoutExpired as ex:
        return -1, ex.stdout or "", f"timeout after {timeout}s", time.time() - t0


def fnum(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return None


def details_table(text):
    """Per launch: [(section, metric, unit, value)] for the KEEP metrics, from --page details."""
    rows = csv_rows(text)
    if not rows:
        return {}
    h = rows[0]
    ix = {k: h.index(k) for k in ("ID", "Kernel Name", "Section Name", "Metric Name", "Metric Unit", "Metric Value")
          if k in h}
    out = {}
    for r in rows[1:]:
        if len(r) < len(h) or r[ix["Metric Name"]] not in KEEP:
            continue
        out.setdefault(r[ix["ID"]], []).append((r[ix["Section Name"]], r[ix["Metric Name"]],
                                                r[ix["Metric Unit"]], r[ix["Metric Value"]]))
    return out


def raw_metrics(text):
    """Per launch: {metric: value} from --page raw (a header row, a units row, then one row per launch)."""
    rows = csv_rows(text)
    if len(rows) < 3:
        return {}
    h = rows[0]
    out = {}
    for r in rows[2:]:
        d = dict(zip(h, r))
        out[d.get("ID", str(len(out)))] = d
    return out


REASONS = ["barrier", "branch_resolving", "dispatch", "drain", "lg", "long_sb", "math", "membar", "mio", "misc",
           "no_inst", "not_selected", "selected", "short_sb", "sleep", "tex", "wait"]
F64 = re.compile(r"\b(DFMA|DADD|DMUL|DSETP|DMNMX)\b|\.F64\b")


def source_report(text, top=20):
    """Markdown from --page source --print-source sass (one CSV block per launch, each opened by a
    "Kernel Name" line): per launch, stall samples by opcode, the float64 share, and the hottest
    lines with their main stall reasons. Launch 1 is tier 1."""
    blocks, cur = [], None
    for r in csv_rows(text):
        if r and r[0] == "Kernel Name":
            cur = {"h": None, "rows": []}
            blocks.append(cur)
        elif cur is not None and cur["h"] is None:
            cur["h"] = r
        elif cur is not None:
            cur["rows"].append(r)
    out = []
    for k, b in enumerate(blocks):
        h = b["h"] or []
        if "Warp Stall Sampling (All Samples)" not in h:
            continue
        si, ie = h.index("Warp Stall Sampling (All Samples)"), h.index("Instructions Executed")
        ri = {x: h.index("stall_" + x) for x in REASONS if "stall_" + x in h}
        rows = [r for r in b["rows"] if len(r) == len(h)]

        def val(r, i):
            return fnum(r[i]) or 0.0
        tot = sum(val(r, si) for r in rows)
        totx = sum(val(r, ie) for r in rows) or 1.0
        if tot < 1:
            continue
        ops, opx = {}, {}
        for r in rows:
            m = re.match(r"\s*(@!?U?P\w+\s+)?([A-Z0-9_]+)", r[1])
            o = m.group(2) if m else "?"
            ops[o] = ops.get(o, 0.0) + val(r, si)
            opx[o] = opx.get(o, 0.0) + val(r, ie)
        f64 = sum(val(r, si) for r in rows if F64.search(r[1]))
        out += [f"Launch {k + 1} ({'tier 1' if k == 0 else 'tier 2'}) SASS: {len(rows)} lines, {tot:.0f} stall "
                f"samples; float64 instructions hold {100 * f64 / tot:.1f}% of them.", "",
                "| opcode | stall samples | instructions |", "|---|---|---|"]
        for o, v in sorted(ops.items(), key=lambda x: -x[1])[:15]:
            out.append(f"| {o} | {100 * v / tot:.1f}% | {100 * opx[o] / totx:.1f}% |")
        out += ["", "| samples | executed | SASS | main stalls |", "|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -val(r, si))[:top]:
            rs = sorted(((x, val(r, i)) for x, i in ri.items()), key=lambda x: -x[1])[:2]
            why = ", ".join(f"{x} {100 * v / max(val(r, si), 1):.0f}%" for x, v in rs if v > 0)
            out.append(f"| {100 * val(r, si) / tot:.1f}% | {int(val(r, ie))} | `{r[1].strip()[:60]}` | {why} |")
        out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--target")
    ap.add_argument("--ncu", default=os.environ.get("EA_NCU", "ncu"))
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    if a.target:
        return target(a.target)
    outdir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(outdir, exist_ok=True)
    rc, vout, verr, _ = run([a.ncu, "--version"], 60)
    lines = ["# Nsight Compute on one exact step (DESIGN D31)", "",
             f"`{a.ncu}`: {(vout.strip().splitlines() or [verr.strip()])[-1]}. Per case, the step's two "
             "`_ea_split_kernel` launches (1 = tier 1, 2 = tier 2), --cache-control none, kernel replay. "
             "Generated by `scripts/ncu_step.py`.", ""]
    for name in a.cases.split(","):
        rep = os.path.join(outdir, f"ncu_{name}")
        cmd = [a.ncu, "--profile-from-start", "off", "--kernel-name", "regex:_ea_split_kernel", "--launch-count", "2",
               "--cache-control", "none", "--replay-mode", "kernel", "--force-overwrite", "--export", rep,
               *sum((["--section", s] for s in SECTIONS), []),
               "--metrics", ",".join(EXTRA + [f"regex:{STALL}.*{STALL_SUFFIX.replace('.', chr(92) + '.')}"]),
               sys.executable, os.path.abspath(__file__), "--target", name]
        rc, so, se, sec = run(cmd, a.timeout)
        lines += [f"## {name}", "", f"ncu rc={rc}, {sec:.0f} s. "
                  + " ".join(ln for ln in so.splitlines() if "uncertified" in ln), ""]
        if rc != 0 or not os.path.exists(rep + ".ncu-rep"):
            err = "\n".join((so + "\n" + se).strip().splitlines()[-25:])
            lines += ["ncu failed:", "", "```", err, "```", ""]
            print("\n".join(lines[-30:]), flush=True)
            open(a.out, "w").write("\n".join(lines) + "\n")
            continue
        # Imports in a clean environment: under the job's venv and module paths, ncu's embedded
        # Python (its rules, the source page) fails with an ImportError (run 1129).
        imp = [a.ncu, "--import", rep + ".ncu-rep", "--csv"]
        env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin"}
        _, det, _, _ = run(imp + ["--page", "details"], 300, env)
        _, raw, _, _ = run(imp + ["--page", "raw"], 300, env)
        _, src, srcerr, _ = run(imp + ["--page", "source", "--print-source", "sass"], 600, env)
        open(rep + "_details.csv", "w").write(det)
        open(rep + "_raw.csv", "w").write(raw)
        open(rep + "_source.csv", "w").write(src)
        dt, rm = details_table(det), raw_metrics(raw)
        ids = sorted(set(dt) | set(rm), key=lambda s: fnum(s) or 0)
        for k, lid in enumerate(ids):
            lines += [f"### launch {k + 1} ({'tier 1' if k == 0 else 'tier 2'})", "",
                      "| section | metric | value |", "|---|---|---|"]
            for sname, m, unit, v in dt.get(lid, []):
                lines.append(f"| {sname} | {m} | {v} {unit} |")
            d = rm.get(lid, {})
            for m in EXTRA:
                if m in d:
                    lines.append(f"| raw | {m} | {d[m]} |")
            st = sorted(((m[len(STALL):-len(STALL_SUFFIX)], fnum(v)) for m, v in d.items()
                         if m.startswith(STALL) and m.endswith(STALL_SUFFIX) and fnum(v)), key=lambda x: -x[1])
            tot = sum(v for _, v in st) or 1.0
            lines += ["", "Warp stalls (cycles per issued instruction, sampled):", "",
                      "| reason | cycles | share |", "|---|---|---|"]
            lines += [f"| {r} | {v:.2f} | {100 * v / tot:.0f}% |" for r, v in st if v >= 0.005]
            lines.append("")
        sr = source_report(src)
        lines += sr or [f"No source-page samples ({(srcerr.strip().splitlines() or ['no output'])[-1][:200]}).", ""]
        print("\n".join(lines[-60:]), flush=True)
        open(a.out, "w").write("\n".join(lines) + "\n")      # after every case: a timeout keeps the rest
    open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
