# ExactAttention

Attention defined so that reduction order cannot change the answer.

Today's inference engines make attention reproducible by **fixing the reduction order**: fixed
split sizes, fixed merge trees, one kernel per path. That has to be re-engineered for every
kernel shape, split heuristic and GPU vendor, and it breaks as soon as a draft tree, a
different page size or a new GPU appears.

ExactAttention **defines** attention with exact sums and a single final rounding
([docs/SPEC.md](docs/SPEC.md)). The result then depends only on the query and the *set* of
keys and values it attends to. One theorem (SPEC §5.3) gives all of the following:

- invariance to batch composition, KV split count and page layout;
- prefill producing the same bits as decode;
- speculative and tree verification producing the same bits as sequential decoding, for any
  draft tree;
- bit-identical results across CPUs and GPUs from any vendor.

The fast kernels don't do big-integer arithmetic. They accumulate in narrow fixed-point
windows with rigorous error bounds, and **certify** that the bounds decide the final rounding
(Ziv's strategy). A certified result equals the spec's by definition. A row that cannot be
certified falls back to the exact path.

## Results so far (Qwen3-0.6B, WikiText-2, laptop-only)

| | result |
|---|---|
| Full model, every causal row: 917,504 rows, 940M query-key pairs | **0** rows need a fallback with int64 accumulators |
| Same rows, two schedules (1 split, 64-key tiles, int64 vs 8 splits, 37-key tiles, shuffled merges, int128) | **0** rows differ |
| Intel Arc 140V GPU vs x86 C++ kernel, 6,144 real rows | **0** rows differ ([details](results/cross_device_qwen3-0.6b.md)) |
| Standard `scaled_dot_product_attention`, same rows, CPU vs Intel GPU | **98.7%** of rows differ in bf16, 9.7% in fp32 |
| Elements off float64 attention by at least 1 ulp, vs standard fp32 attention | **3× fewer** |
| Narrowest window with zero fallbacks on sampled real rows | 44 bits (cliff at 40: 8/150 rows) |
| **Speculative decoding** (Qwen3-0.6B, 48 greedy tokens, 4-token drafts) vs sequential decoding | standard: tokens diverge at position 4, 47/48 logit rows differ. Row-invariant linears with SDPA: still 47/48. **ExactAttention: identical tokens, 0/48 rows differ** ([details](results/spec_decode_qwen3-0.6b.md)) |
| Portable fixture: 2 real heads, 4,096 rows | same SHA-256 from the C++ kernel, torch on CPU and torch on the Intel GPU; standard SDPA hashes differ per device |
| Same fixture on an **NVIDIA RTX PRO 6000 Blackwell** (PACE free tier) | **same SHA-256** from torch:cuda, the fused Triton kernel, and Triton with int8 tensor-core scores, 0 fallback rows |

**Check it on any machine:** `python scripts/fixture.py check --backend cpp` (or `torch:cuda`,
`torch:xpu`, `triton:cuda`). It prints `ALL MATCH` when the output bits equal the hash recorded in
`fixtures/expected_sha256.json`. For NVIDIA runs on Georgia Tech's free PACE tier, see
[scripts/pace/README.md](scripts/pace/README.md).

Reports: [validation](results/validation_qwen3-0.6b.md), [window widths](results/width_study_qwen3-0.6b.md),
[cross-device](results/cross_device_qwen3-0.6b.md). Design decisions and measurements are in
[docs/DESIGN.md](docs/DESIGN.md).

## Layout

| path | what |
|---|---|
| `docs/SPEC.md` | the definition, a no-ties proof for exp2, and the invariance theorem |
| `src/exactattn/reference.py` | exact big-integer reference (ground truth) |
| `src/exactattn/exp2.py` | rigorous interval oracle for `RN24(2^t)`, plus a certified fast path |
| `src/exactattn/certified.py` | bit-accurate model of the windowed split-KV algorithm |
| `csrc/exact_attention_cpu.cpp` | C++ certified kernel: int64/int128, threads, CSR masks |
| `src/exactattn/torch_impl.py` | the same algorithm as int64 tensor ops (Intel XPU / CUDA / CPU) |
| `src/exactattn/triton_kernels.py` | fused split-KV Triton kernels: int64 path, int8 tensor-core scores (3×4 slices) and int8 tensor-core P·V (5×3 slices), all certified |
| `fixtures/` | two real Qwen3-0.6B heads and the SHA-256 of their exact outputs |
| `scripts/` | activation capture, width study, full-model validation, cross-device check |

## Running

Python reference and model (any OS):

```bash
python -m venv .venv && .venv/bin/pip install -e .[test] && .venv/bin/pytest
```

C++ kernel (Linux/WSL, gcc): `make -C csrc`, then the same `pytest` picks up
`tests/test_cpu_kernel.py`.

Intel GPU: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/xpu`.
Newer XPU wheels need a newer Intel GPU driver than this laptop has. Then `pytest
tests/test_torch_impl.py` runs on both `cpu` and `xpu`.

```python
import numpy as np
from exactattn import attention, default_alpha, float_to_bf16_bits

rng = np.random.default_rng(0)
q = float_to_bf16_bits(rng.standard_normal((2, 64)))
k = float_to_bf16_bits(rng.standard_normal((100, 64)))
v = float_to_bf16_bits(rng.standard_normal((100, 64)))
out_bits = attention(q, k, v, default_alpha(64))   # bf16 bit patterns, shape (2, 64)
```

## Speed goals

The kernels are measured on an NVIDIA RTX PRO 6000 Blackwell (sm_120) against FlashInfer
0.7.4, with `scripts/bench_vs_flashinfer.py`. The workload is bf16, d = 128, and 16 query heads
over 8 KV heads. A shape counts only if every row is certified. The ratio is our best exact step
time divided by FlashInfer's.

1. **Decode within 2× of FlashInfer on DRAM-bound shapes.** These are one-token decode steps
   whose KV cache comes from DRAM: 268 MB or more, where FlashInfer reads at 1.44–1.53 TB/s.
   - Smaller caches (134 MB or less) stay in L2 when the bench repeats one layer, and
     FlashInfer reads them at 2–4 TB/s. A model's other layers would evict them.
   - Those shapes are reported but are not part of this goal.
2. **Then: verify within 5× of FlashInfer.** Verify scores 8 draft tokens per sequence. The
   goal covers every verify shape with n ≥ 4K, L2-resident ones included. Verify is the step
   speculative decoding adds, and its gap is largest on the small shapes.

Status as of run 1133 (DESIGN D35). Verify ratios are the bench's, from unrounded times:

| goal | shape | FlashInfer (ms) | exact (ms) | ratio | met |
|---|---|---|---|---|---|
| 1 | decode B=1, 64K | 0.187 | 0.393 | 2.10× | no |
| 1 | decode B=8, 16K | 0.359 | 0.734 | 2.04× | no |
| 1 | decode B=32, 4K | 0.359 | 0.869 | 2.42× | no |
| 1 | decode B=32, 16K | 1.406 | 3.173 | 2.26× | no |
| 2 | verify B=1, 4K | 0.012 | 0.086 | 6.9× | no |
| 2 | verify B=1, 16K | 0.021 | 0.326 | 15.6× | no |
| 2 | verify B=8, 4K | 0.040 | 0.540 | 13.6× | no |
| 2 | verify B=8, 16K | 0.363 | 1.609 | 4.4× | yes |

## Roadmap

1. ~~Run the fused Triton kernel on NVIDIA~~: bit-identical on Blackwell (DESIGN D11). Intel iGPU
   Triton still needs a newer driver for `triton-xpu`.
2. ~~Exact scores and P·V on tensor cores (int8 slices)~~: implemented (DESIGN D8, D10) and
   bit-identical on NVIDIA. Pairs that tensor-core P·V cannot certify are recomputed on the
   device with int64 P·V (tier 2, DESIGN D12). In progress: the speed goals above.
3. Draft-tree verification end to end: bit-identical speculative decoding.
4. Spec v1: optionally a higher-precision logit, which removes the remaining amplification on
   heavily cancelling outputs (DESIGN D5).

## License

Apache License 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

`fixtures/qwen3_0.6b_heads.npz` holds activations from two heads of
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0), computed on text from
WikiText-2 (CC BY-SA).
