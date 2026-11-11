# Cross-hardware bit identity on real activations

Model Qwen/Qwen3-0.6B, 2048 tokens of WikiText-2. Every causal row of each head, computed on
two different processors with two different implementations:

* **GPU:** Intel(R) Arc(TM) 140V GPU (16GB), through the tensor-op implementation (`torch_impl.py`, PyTorch 2.6.0+xpu), with 128-key tiles and int64 accumulators that keep 61 bits
  of headroom
* **CPU:** the x86 C++ kernel (`exact_attention_cpu.cpp`, 8 threads), with 8 splits,
  shuffled merges, 37-key tiles and int128 accumulators

| layer:head | rows | GPU uncertified | CPU uncertified | rows with any differing bit |
|---|---|---|---|---|
| 0:0 | 2048 | 0 | 0 | **0** |
| 13:5 | 2048 | 0 | 0 | **0** |
| 27:11 | 2048 | 0 | 0 | **0** |

Total: 6,144 rows × 128 channels; rows with any differing bit: **0**.

GPU timings (tensor-op form, unfused; not a performance number): 0_0: 7.2s, 13_5: 6.3s, 27_11: 6.5s

## Contrast: standard attention on the same inputs

`torch.nn.functional.scaled_dot_product_attention` (causal), PyTorch 2.6.0, run on the same
6,144 rows on both devices, with outputs compared after conversion to bf16:

| implementation | rows differing between CPU and Intel GPU |
|---|---|
| SDPA, bf16 inputs | **6,062 / 6,144** (98.7%) |
| SDPA, inputs upcast to fp32 | **594 / 6,144** (9.7%) |
| ExactAttention | **0 / 6,144** |