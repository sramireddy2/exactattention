"""Static SASS profile of the split kernel's key loop on sm_120, with no GPU (DESIGN D18).

Compiles one config (scripts/sm120_compile_check.py names, plus KEY=VALUE overrides of the
kernel's constexprs or num_warps) with line info, disassembles it with Triton's bundled nvdisasm,
finds the key loop (the largest backward branch), and counts its SASS instructions by the kernel
line they come from. Tiles that need the windowed score recompute or float64 exp2 are rare, so
those branches are counted apart. One loop iteration is one key tile, so the hot count per warp,
times the warps, is the tile's instruction cost: the metric this script reports.

    python scripts/sm120_sass_profile.py "tcpv m16 128/8"
    python scripts/sm120_sass_profile.py "tcpv m2 128/8" BLOCK_N=32 --top 20
"""
import argparse
import collections
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import sm120_compile_check as cc  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "exactattn", "triton_kernels.py")


def rare_lines(src):
    """Source lines of the rare branches: the windowed recompute and float64 exp2."""
    out = set()
    for name in ("_windowed_logits", "_qk_chunk", "_exp2_weight_f64"):
        a = next(i for i, ln in enumerate(src, 1) if ln.startswith(f"def {name}("))
        b = next(i for i, ln in enumerate(src[a:], a + 1) if ln.startswith(("@triton.jit", "def ")))
        out |= set(range(a, b))
    out |= {i for i, ln in enumerate(src, 1) if "= _windowed_logits(" in ln or "= _exp2_weight_f64(" in ln}
    return out


def parse(sass):
    """[(line chain, opcode, operands)] and label positions. nvdisasm -gi gives each instruction's
    line and, for inlined code, the caller's line; the last entry is the kernel-level line."""
    ins, labels, chain, more = [], {}, (), False
    for s in sass.splitlines():
        if "//##" in s:
            # Deeper inlining prints one comment per level ("a inlined at b", then "b inlined at
            # c"); consecutive comments are one chain, innermost line first.
            new = tuple(int(x) for x in re.findall(r"line (\d+)", s))
            chain = chain + tuple(x for x in new if not chain or x != chain[-1]) if more else new
            more = True
            continue
        more = False
        m = re.match(r"^(\.L_x_\d+):", s.strip())
        if m:
            labels[m.group(1)] = len(ins)
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)(.*)", s)
        if m:
            ins.append((chain, m.group(2).split(".")[0], m.group(3)))
    return ins, labels


def body_lines(src):
    """Source lines of _split_program, the per-program body both tiers inline (DESIGN D23), the
    kernel lines that call it, and the lines of tier 2's work-list loop. (Only those: ptxas may move
    other kernel-level instructions, such as a program_id compare, into the key loop.)"""
    a = next(i for i, ln in enumerate(src, 1) if ln.startswith("def _split_program("))
    b = next(i for i, ln in enumerate(src[a:], a + 1) if ln.startswith(("@triton.jit", "def ")))
    calls = {i for i, ln in enumerate(src, 1) if "_split_program(" in ln and not ln.startswith("def ")}
    k = next(i for i, ln in enumerate(src, 1) if ln.startswith("def _ea_split_kernel("))
    e = next(i for i, ln in enumerate(src[k:], k + 1) if ln.startswith(("@triton.jit", "def ")))
    wl = {i for i in range(k, e) if "for it in range(" in src[i - 1] or "tl.load(WL + it)" in src[i - 1]}
    return set(range(a, b)), calls, wl


def body_line(chain, body):
    """The kernel-level line of an instruction, read inside _split_program: inlined helpers are
    charged to the body line that calls them, not to the kernel's call of the body."""
    inside = [ln for ln in chain if ln in body]
    return inside[-1] if inside else (chain[-1] if chain else -1)


def key_loop(ins, labels, kern=()):
    """The largest backward branch, i.e. the key loop. A loop that holds tier 2's work-list loop
    code (kern) does not count."""
    best = None
    for i, (_, op, rest) in enumerate(ins):
        m = re.search(r"`\((\.L_x_\d+)\)", rest) if op == "BRA" else None
        if m and m.group(1) in labels and labels[m.group(1)] <= i:
            lo = labels[m.group(1)]
            if any(c and c[-1] in kern for c, _, _ in ins[lo:i + 1]):
                continue
            if best is None or i - lo > best[1] - best[0]:
                best = (lo, i)
    return best


def profile(name, over):
    cfg, nw, _ = cc.kernel_config(over)
    with tempfile.TemporaryDirectory() as td:
        cubin = os.path.join(td, "k.cubin")
        ptx, smem = cc.build_ptx(over)
        info = cc.ptxas(ptx, cubin, "-lineinfo")
        sass = subprocess.run([cc.tool("nvdisasm"), "-gi", "-c", cubin], capture_output=True, text=True).stdout
    regs = re.findall(r"Used (\d+) registers", info)
    spill = re.findall(r"(\d+) bytes spill stores", info)
    stack = re.findall(r"(\d+) bytes stack frame", info)
    src = open(SRC).read().splitlines()
    rare = rare_lines(src)
    body, _, kern = body_lines(src)
    ins, labels = parse(sass)
    lo, hi = key_loop(ins, labels, kern)
    by_line, ops, n_rare = collections.Counter(), collections.Counter(), 0
    for chain, op, _ in ins[lo:hi + 1]:
        if any(ln in rare for ln in chain):
            n_rare += 1
            continue
        by_line[body_line(chain, body)] += 1
        ops[op] += 1
    hot = sum(by_line.values())
    return dict(name=name, warps=nw, block_n=cfg["BLOCK_N"], regs=int(regs[0]) if regs else None,
                spill=int(spill[0]) if spill else 0, smem=smem, stack=int(stack[0]) if stack else 0, hot=hot, rare=n_rare,
                local=ops["LDL"] + ops["STL"], by_line=by_line, ops=ops, src=src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="a name from sm120_compile_check.CONFIGS")
    ap.add_argument("overrides", nargs="*", help="KEY=VALUE (constexprs or num_warps)")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    over = dict(cc.CONFIGS[a.config])
    for kv in a.overrides:
        k, v = kv.split("=", 1)
        over[k] = eval(v)  # noqa: S307 (numbers and booleans from the command line)
    p = profile(a.config, over)
    per32 = p["hot"] * p["warps"] * 32 / p["block_n"]
    print(f"{a.config} {' '.join(a.overrides)}: regs {p['regs']}, spill {p['spill']} B, stack {p['stack']} B, "
          f"shared {p['smem']} B; "
          f"key loop {p['hot']} hot SASS per warp per tile (+{p['rare']} in rare branches), "
          f"{p['local']} of them local-memory spill loads/stores; {per32:.0f} warp-instructions and "
          f"{p['local'] * p['warps'] * 32 / p['block_n']:.0f} warp spill ops per 32 keys")
    print("ops:", ", ".join(f"{k} {v}" for k, v in p["ops"].most_common(16)))
    for ln, c in p["by_line"].most_common(a.top):
        txt = p["src"][ln - 1].strip()[:90] if 0 < ln <= len(p["src"]) else "?"
        print(f"{c:6d}  L{ln}: {txt}")


if __name__ == "__main__":
    main()
