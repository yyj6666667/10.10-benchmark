# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 5.942

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.010715 | 27.425453 | 64.666721 | 101.142779 | 2.531 |
| 2 | 1 | 7.399072 | 38.233211 | 12.213869 | 57.904522 | 4.421 |
| 3 | 1 | 7.668040 | 27.981219 | 7.023829 | 42.713797 | 5.993 |
| 4 | 1 | 7.275494 | 28.531092 | 7.603298 | 43.447317 | 5.892 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.471767 | 28.256156 | 7.313563 | 43.080557 |
