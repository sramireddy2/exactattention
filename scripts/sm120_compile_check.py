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

BASE = dict(D=128, DV=128, BLOCK_M=16, BLOCK_N=32, BLOCK_DV=32, EXP2_MARGIN=2.0 ** -26, USE_TC=True,
            SQ=3, SK=4, TC_FALLBACK=True, USE_TC_PV=True, SW=5, SV=3, ANCH=61, RERUN=False, PACK=1,
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
    "tcpv m16 32/4": {},
    "tcpv tier2 m2": {**DECODE, "USE_TC_PV": False, "BLOCK_N": 16, "RERUN": True},
}
PTR = {"Q": "*i16", "K": "*i16", "V": "*i16", "KVLEN": "*i32", "PR": "*i64", "PZ": "*i64", "PZE": "*i64",
       "PNR": "*i64", "PN": "*i64", "PNE": "*i64", "FLAG": "*i32", "KX": "*i8", "KLSB": "*i64",
       "KL1S": "*fp64", "KLOST": "*i32", "VX": "*i8", "VLSB": "*i64", "VLOST": "*i8", "NEED": "*i32",
       "EXP2T": "*i64"}


def compile_one(item):
    name, over = item
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.compiler import ASTSource
    from triton._C.libtriton import ir

    from exactattn.triton_kernels import SCORE_FS, _ea_split_kernel

    cfg = {**BASE, **over, "FS": SCORE_FS}
    nw = cfg.pop("num_warps")
    if not cfg["USE_TC_PV"]:
        cfg.update(SW=1, SV=1, ANCH=14 + (cfg["BLOCK_N"] - 1).bit_length())   # as launch computes it
    cfg["WDC"] = max(1, min(cfg["D"], (512 * nw) // (cfg["BLOCK_M"] * cfg["BLOCK_N"])))
    sig = {a: ("constexpr" if a in cfg else PTR.get(a, "i32")) for a in _ea_split_kernel.arg_names}
    src = ASTSource(_ea_split_kernel, sig, {a: cfg[a] for a in _ea_split_kernel.arg_names if a in cfg})
    target = GPUTarget("cuda", 120, 32)
    be = CUDABackend(target)
    opts = be.parse_options({"num_warps": nw})
    ctx = ir.context()
    ir.load_dialects(ctx)
    be.load_dialects(ctx)
    stages = {}
    be.add_stages(stages, opts, src.language)
    t0 = time.time()
    cur = src.make_ir(target, opts, be.get_codegen_implementation(opts), be.get_module_map(), ctx)
    meta = {}
    for st, fn in stages.items():
        if st == "cubin":
            break
        cur = fn(cur, meta)
    with tempfile.TemporaryDirectory() as td:
        pf = os.path.join(td, "k.ptx")
        open(pf, "w").write(cur)
        ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas-blackwell")
        r = subprocess.run([ptxas, "-v", "--gpu-name=sm_120a", "-O3", pf, "-o", os.path.join(td, "k.cubin")],
                           capture_output=True, text=True)
    out = r.stderr + r.stdout
    regs = re.findall(r"Used (\d+) registers", out)
    spill = re.findall(r"(\d+) bytes spill stores", out)
    return name, time.time() - t0, int(regs[0]) if regs else None, int(spill[0]) if spill else None, \
        "" if regs else out[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=180.0)
    ap.add_argument("--max-spill", type=int, default=8192)
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    bad = False
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, sec, regs, spill, err in ex.map(compile_one, CONFIGS.items()):
            over = regs is None or sec > a.max_seconds or spill > a.max_spill
            bad |= over
            print(f"{'FAIL' if over else 'ok  '} {name:16s} {sec:6.1f} s  regs {regs}  spill {spill} B {err}",
                  flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
