"""Compile the split kernel for sm_120 (RTX PRO 6000 / RTX 50) without a GPU, and report compile
time, registers and spills per config (DESIGN D17).

The CPU interpreter runs the kernels but never compiles them, so it cannot see compile-time
blow-ups or register spills. This needs only Triton's bundled ptxas-blackwell. Run it before a
PACE job:

    python scripts/sm120_compile_check.py            # the configs the GPU stages build
    python scripts/sm120_compile_check.py --max-seconds 120 --max-spill 4096

Exit status 1 if a config exceeds either limit.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BASE = dict(D=128, DV=128, BLOCK_M=16, BLOCK_N=64, BLOCK_DV=32, EXP2_MARGIN=2.0 ** -26, USE_TC=True,
            SQ=4, SK=4, TC_FALLBACK=True, USE_TC_PV=True, SW=5, SV=3, RERUN=False, PACK=1,
            EXP2_INT=True, num_warps=4)
INT64 = dict(USE_TC=False, USE_TC_PV=False, BLOCK_N=16)
TC = dict(USE_TC_PV=False, BLOCK_N=16)
DECODE = dict(BLOCK_M=2, PACK=8)
CONFIGS = {
    "int64 m2": {**INT64, **DECODE},
    "int64 m16": INT64,
    "tc m2": {**TC, **DECODE},
    "tc m16": TC,
    "tcpv m2 32/4": DECODE,
    "tcpv m2 64/8": {**DECODE, "BLOCK_DV": 64, "num_warps": 8},
    "tcpv m2 128/8": {**DECODE, "BLOCK_DV": 128, "num_warps": 8},
    "tcpv m2 128/8 n32": {**DECODE, "BLOCK_DV": 128, "num_warps": 8, "BLOCK_N": 32},
    "tcpv m16 32/4": {},
    "tcpv m16 128/8": {"BLOCK_DV": 128, "num_warps": 8},     # verify steps (T = 8: 16 rows)
    "tcpv m16 128/8 q3": {"BLOCK_DV": 128, "num_warps": 8, "SQ": 3},
    "tcpv tier2 m2": {**DECODE, "USE_TC_PV": False, "BLOCK_N": 16, "RERUN": True},
}
PTR = {"Q": "*i16", "K": "*i16", "V": "*i16", "KVLEN": "*i32", "PR": "*i64", "PZ": "*i64", "PZE": "*i64",
       "PNR": "*i64", "PN": "*i64", "PNE": "*i64", "FLAG": "*i32", "KX": "*i8", "KLSB": "*i64",
       "KL1S": "*fp64", "KLOST": "*i32", "VX": "*i8", "VLSB": "*i64", "VLOST": "*i8", "NEED": "*i32",
       "EXP2T": "*i64"}


def kernel_config(over):
    """The split kernel's constexprs and num_warps for a config, as launch computes them."""
    from exactattn.triton_kernels import SCORE_FS

    cfg = {**BASE, **over, "FS": SCORE_FS}
    nw = cfg.pop("num_warps")
    ns = cfg.pop("num_stages", None)
    if ns is None and cfg["USE_TC_PV"] and cfg["BLOCK_N"] >= 64:
        ns = 2                                                  # as launch picks it
    if not cfg["USE_TC_PV"]:
        cfg.update(SW=1, SV=1)
    cfg["ANCH"] = 7 * cfg["SW"] + 7 * cfg["SV"] + (cfg["BLOCK_N"] - 1).bit_length()
    cfg["WDC"] = max(1, min(cfg["D"], (512 * nw) // (cfg["BLOCK_M"] * cfg["BLOCK_N"])))
    cfg["FL_DIV"] = 1
    return cfg, nw, ns


def build_ptx(over, ttgir_out=None):
    """Lower the split kernel to sm_120 PTX with Triton's own pipeline (no GPU needed).
    Returns (ptx, shared memory bytes per program)."""
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    from triton._C.libtriton import ir

    from exactattn.triton_kernels import _ea_split_kernel

    cfg, nw, ns = kernel_config(over)
    sig = {a: ("constexpr" if a in cfg else PTR.get(a, "i32")) for a in _ea_split_kernel.arg_names}
    # The GPU's JIT specializes arguments divisible by 16 (all pointers, and these sizes at serving
    # shapes); without the same hints, loads are not vectorized or pipelined through shared memory
    # and the kernel differs from the one the GPU runs.
    div16 = set(PTR) | {"ROWS", "SPLIT_LEN", "NKEYS", "NK_SEG"}
    attrs = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(_ea_split_kernel.arg_names)
             if a in div16 and a not in cfg}
    src = ASTSource(_ea_split_kernel, sig, {a: cfg[a] for a in _ea_split_kernel.arg_names if a in cfg}, attrs)
    target = GPUTarget("cuda", 120, 32)
    be = CUDABackend(target)
    opts = be.parse_options({"num_warps": nw, **({} if ns is None else {"num_stages": ns})})
    ctx = ir.context()
    ir.load_dialects(ctx)
    be.load_dialects(ctx)
    stages = {}
    be.add_stages(stages, opts, src.language)
    cur = src.make_ir(target, opts, be.get_codegen_implementation(opts), be.get_module_map(), ctx)
    meta = {}
    for st, fn in stages.items():
        if st == "cubin":
            break
        cur = fn(cur, meta)
        if st == "ttgir" and ttgir_out:
            open(ttgir_out, "w").write(str(cur))
    return cur, meta.get("shared", 0)


def tool(name):
    import triton
    return os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", name)


def ptxas(ptx, cubin, *flags):
    """Assemble with Triton's bundled ptxas-blackwell. Returns its -v output."""
    pf = cubin + ".ptx"
    open(pf, "w").write(ptx)
    r = subprocess.run([tool("ptxas-blackwell"), "-v", *flags, "--gpu-name=sm_120a", "-O3", pf, "-o", cubin],
                       capture_output=True, text=True)
    return r.stderr + r.stdout


def compile_one(item):
    name, over = item
    t0 = time.time()
    ptx, smem = build_ptx(over)
    with tempfile.TemporaryDirectory() as td:
        out = ptxas(ptx, os.path.join(td, "k.cubin"))
    regs = re.findall(r"Used (\d+) registers", out)
    spill = re.findall(r"(\d+) bytes spill stores", out)
    return (name, time.time() - t0, int(regs[0]) if regs else None, int(spill[0]) if spill else None, smem,
            "" if regs else out[-400:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--max-spill", type=int, default=8192)
    ap.add_argument("--max-shared", type=int, default=101376, help="bytes per block (sm_120: 99 KB)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    bad = False
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, sec, regs, spill, smem, err in ex.map(compile_one, CONFIGS.items()):
            over = regs is None or sec > a.max_seconds or spill > a.max_spill or smem > a.max_shared
            bad |= over
            print(f"{'FAIL' if over else 'ok  '} {name:18s} {sec:6.1f} s  regs {regs}  spill {spill} B  "
                  f"shared {smem} B {err}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
