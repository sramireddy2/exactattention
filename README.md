# ExactAttention

Attention defined so that reduction order cannot change the answer.

Today's inference engines make attention reproducible by **fixing the reduction order**: fixed
split sizes, fixed merge trees, one kernel per path. That has to be re-engineered for every
kernel shape, split heuristic and GPU vendor, and it breaks the moment a draft tree, a
different page size or a new GPU enters the picture.

ExactAttention takes the other route. It **defines** attention with exact sums and a single
final rounding ([docs/SPEC.md](docs/SPEC.md)), so the result depends only on the query and the
*set* of keys and values it attends to. These all follow from one theorem (SPEC §5.3):

- invariance to batch composition, KV split count and page layout;
- prefill producing the same bits as decode;
- speculative and tree verification producing the same bits as sequential decoding, for any
  draft tree;
- bit-identical results across CPU, Intel, AMD and NVIDIA.

The fast kernels don't compute exact big integers. They accumulate in narrow fixed-point
windows with rigorous error bounds, and **certify** that the bounds decide the final rounding
(Ziv's strategy). A certified result equals the spec's by definition. The rare uncertified
row falls back to a wider path.

## Status

| step | state |
|---|---|
| Formal spec v0 | done: [docs/SPEC.md](docs/SPEC.md) |
| Exact big-integer reference (Python) | done: `src/exactattn/reference.py` |
| Correctly rounded exp2 oracle and fast path | done: `src/exactattn/exp2.py` |
| Windowed, certified split-KV algorithm model | in progress |
| Accumulator-width study on real model activations | planned |
| CPU kernel (C++) bit-matched to reference | planned |
| Intel GPU kernel (Arc 140V, SYCL/Triton) | planned |
| NVIDIA kernel (CUDA, int8 tensor-core scores) | planned |

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[test]   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest
```

```python
import numpy as np
from exactattn import attention, default_alpha, float_to_bf16_bits

rng = np.random.default_rng(0)
q = float_to_bf16_bits(rng.standard_normal((2, 64)))
k = float_to_bf16_bits(rng.standard_normal((100, 64)))
v = float_to_bf16_bits(rng.standard_normal((100, 64)))
out_bits = attention(q, k, v, default_alpha(64))   # bf16 bit patterns, shape (2, 64)
```
