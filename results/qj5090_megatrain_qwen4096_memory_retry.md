# qj5090 MegaTrain Qwen seq4096 host-memory retry

Both runs used the same eight-GPU BF16 full fine-tuning case, checkpoint
interval 2, global batch 8, four steps with two warmup steps, alias layer
slots, and direct shared initialization. The only runtime setting changed was
the number of pinned CPU gradient slabs per worker, from four to three.

| Slabs / worker | Outcome | Lowest sampled MemAvailable | Step 3–4 | Evidence |
|---:|---|---:|---|---|
| 4 | Host-memory guard stopped the first step at 16 GiB available; no throughput result | 26.98 GB (2 s samples missed the lower guard reading) | — | [`run_config.json`](../runs/megatrain-qwen4096-20261011/qwen35_122b_megatrain_seq4096_8gpu_20261011_141452_41259_benchmark/run_config.json), [`host_memory_guard.txt`](../runs/megatrain-qwen4096-20261011/qwen35_122b_megatrain_seq4096_8gpu_20261011_141452_41259_benchmark/host_memory_guard.txt), [`resources.json`](../runs/megatrain-qwen4096-20261011/qwen35_122b_megatrain_seq4096_8gpu_20261011_141452_41259_benchmark/resources.json) |
| 3 | Completed all four steps, exit 0; loss 0.580 → 0.339 → 0.199 → 0.109 | 37.41 GB | 61.458 s/step; 533.18 TPS | [`run_config.json`](../runs/megatrain-qwen4096-20261011/qwen35_122b_megatrain_seq4096_8gpu_20261011_142731_58036_benchmark/run_config.json), [`step_timing.json`](../runs/megatrain-qwen4096-20261011/qwen35_122b_megatrain_seq4096_8gpu_20261011_142731_58036_benchmark/case/step_timing.json), [`resources.json`](../runs/megatrain-qwen4096-20261011/qwen35_122b_megatrain_seq4096_8gpu_20261011_142731_58036_benchmark/resources.json) |

`MemAvailable` is an actual host headroom estimate. The performance table's
CPU RSS field sums process-tree RSS and counts shared CPU pages repeatedly;
it is not the host's physical allocation. The retry kept the 20 GiB guard
enabled and saved no checkpoint.
