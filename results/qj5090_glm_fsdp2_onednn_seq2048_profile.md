# GLM-4.5-Air FSDP2 + oneDNN, seq 2048 profiler comparison

Both runs used qj5090, eight GPUs, global batch 8, BF16 compute, FP32 master
weights, GPU-only activation checkpointing, one optimizer step, no warmup, and
the same data. They are diagnostics, not eight-step stable throughput points.
PyTorch profiler traced rank 0 with CPU, CUDA, and memory events.

| CPU expert weight access | Forward (s) | Backward (s) | Optimizer (s) | Step (s) | TPS | Loss | Peak GPU (GiB) | Peak process-tree RSS (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `weight[local_id]` | 30.993 | 186.459 | 27.791 | 245.214 | 66.815 | 3.335126 | 29.625 | 1183.4 |
| `weight.unbind(0)[local_id]` | 32.837 | 81.233 | 27.334 | 141.307 | 115.946 | 3.335126 | 29.625 | 1189.1 |

The old code produced 1,440 `SelectBackward0` events: 45 MoE layers times 16
owned experts times two weight groups. Each contained a full-parent-gradient
`aten::fill_`; together those 1,440 fills occupied 41.8 seconds on rank 0.
The expert weights total 1.03125 GiB per rank per layer, so repeated parent
zeroing represents roughly 742.5 GiB across 45 layers. The new code creates
the expert views in one `unbind(0)` per group. Its trace has zero
`SelectBackward0` and 90 `UnbindBackward0` events. A small BF16 synthetic check
gave bitwise equal weight gradients; in the full run the loss matched to six
decimals and the global gradient norm changed from 102.171 to 102.124.

The marked old trace shows 30.913 seconds in forward, 186.432 seconds in
backward, and 27.645 seconds in optimizer on rank 0. During its 247.402-second
profiler window, rank 0 GPU kernel or copy activity covered 6.084 seconds;
the rest of the wall time had no GPU work recorded on that device. Forward
CPU expert spans totaled 29.2 seconds; the backward slowdown was on the CPU
autograd path. The optimized trace covers 143.380 seconds and records 5.323
seconds of rank 0 GPU activity. These are rank 0 trace measurements, not a
hardware-wide GPU utilization counter.

- [Before: run configuration, timings, logs, trace](../runs/pytorch-fsdp2-onednn-qj-20261010/diagnostics/glm45_air_fsdp2_onednn_seq2048_8gpu_20261010_153545_3515498/)
- [After: run configuration, timings, logs, trace](../runs/pytorch-fsdp2-onednn-qj-20261010/diagnostics/glm45_air_fsdp2_onednn_seq2048_8gpu_20261010_154549_3518448/)

The `.json.gz` traces in each `case/` directory can be loaded locally in
Chrome Perfetto. Phase labels are `step 1/forward`, `step 1/backward`, and
`step 1/optimizer`. Nested labels mark GPU attention and MLP, recomputation,
GPU-to-CPU routing, Gloo dispatch/return, CPU oneDNN forward, CPU-to-GPU
return, gradient clipping, and each optimizer.
