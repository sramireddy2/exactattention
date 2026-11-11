"""ctypes binding for the C++ certified kernel (csrc/exact_attention_cpu.cpp).

`attention_cpu` returns exactly the spec's bits. Rows the kernel certifies come straight from
C++. Flagged rows are recomputed by the exact reference, which is hardware-independent pure
integer arithmetic.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass

import numpy as np

from .formats import ExactAttentionDomainError
from .reference import attention_row

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_PATH = os.path.join(_ROOT, "build", "libexactattn_cpu.so")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    src = os.path.join(_ROOT, "csrc", "exact_attention_cpu.cpp")
    if not os.path.exists(_LIB_PATH) or os.path.getmtime(_LIB_PATH) < os.path.getmtime(src):
        subprocess.run(["make", "-C", os.path.join(_ROOT, "csrc")], check=True,
                       stdout=subprocess.DEVNULL)
    lib = ctypes.CDLL(_LIB_PATH)
    P = ctypes.c_void_p
    lib.ea_attention_cpu.argtypes = [P, ctypes.c_int64, ctypes.c_int, P, P, ctypes.c_int64, ctypes.c_int,
                                     ctypes.c_float, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_uint64, ctypes.c_int, P, P, P]
    lib.ea_attention_cpu.restype = ctypes.c_int
    lib.ea_round_ratio.argtypes = [ctypes.c_uint64] * 4 + [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_int)]
    lib.ea_round_ratio.restype = ctypes.c_float
    _lib = lib
    return lib


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


@dataclass
class CpuStats:
    score_fallbacks: int = 0
    weight_fallbacks: int = 0
    final_fallbacks: int = 0
    rows: int = 0
    flagged_rows: int = 0


def csr_from_mask(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    counts = mask.sum(axis=1)
    row_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    key_idx = np.concatenate([np.flatnonzero(m) for m in mask]).astype(np.int64) if mask.size else np.zeros(0, np.int64)
    return row_ptr, key_idx


def attention_cpu(q, k, v, alpha, row_ptr=None, key_idx=None, out="bf16", acc_bits=64, tile=64,
                  splits=1, order_seed=0, threads=None, stats: CpuStats | None = None,
                  fallback=True) -> np.ndarray:
    """q: (rows, d) bf16 bits; k: (n, d); v: (n, dv). The mask is in CSR form (row_ptr,
    key_idx); None means every row attends every key."""
    lib = _load()
    q = np.ascontiguousarray(q, dtype=np.uint16)
    k = np.ascontiguousarray(k, dtype=np.uint16)
    v = np.ascontiguousarray(v, dtype=np.uint16)
    rows, d = q.shape
    n, dv = v.shape
    if row_ptr is None:
        row_ptr = (np.arange(rows + 1, dtype=np.int64) * n)
        key_idx = np.tile(np.arange(n, dtype=np.int64), rows)
    row_ptr = np.ascontiguousarray(row_ptr, dtype=np.int64)
    key_idx = np.ascontiguousarray(key_idx, dtype=np.int64)
    fp32 = out == "fp32"
    res = np.zeros((rows, dv), dtype=np.float32 if fp32 else np.uint16)
    flags = np.zeros(rows, dtype=np.uint8)
    st = np.zeros(3, dtype=np.int64)
    threads = threads or os.cpu_count() or 1
    rc = lib.ea_attention_cpu(q.ctypes.data, rows, d, k.ctypes.data, v.ctypes.data, n, dv,
                              ctypes.c_float(float(np.float32(alpha))), row_ptr.ctypes.data,
                              key_idx.ctypes.data, int(fp32), acc_bits, tile, splits, order_seed,
                              threads, res.ctypes.data, flags.ctypes.data, st.ctypes.data)
    if rc != 0:
        raise ValueError(f"ea_attention_cpu: invalid arguments (code {rc})")
    if stats is not None:
        stats.score_fallbacks += int(st[0])
        stats.weight_fallbacks += int(st[1])
        stats.final_fallbacks += int(st[2])
        stats.rows += rows
        stats.flagged_rows += int((flags != 0).sum())
    if np.any(flags == 2):
        raise ExactAttentionDomainError("non-finite input")
    if fallback:
        for r in np.flatnonzero(flags):
            sel = key_idx[row_ptr[r]:row_ptr[r + 1]]
            res[r] = attention_row(q[r], k[sel], v[sel], alpha, out)
    return res, flags


def round_ratio(a: int, b: int, E: int, neg: bool, fp32: bool) -> tuple[float, bool]:
    lib = _load()
    ov = ctypes.c_int(0)
    f = lib.ea_round_ratio(a >> 64, a & ((1 << 64) - 1), b >> 64, b & ((1 << 64) - 1), E, int(neg), int(fp32),
                           ctypes.byref(ov))
    return float(f), bool(ov.value)
