# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.758

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.370020 | 24.331854 | 7.973224 | 38.708115 | 6.614 |
| 2 | 1 | 6.047395 | 23.849308 | 3.793948 | 33.702061 | 7.596 |
| 3 | 1 | 6.020382 | 23.794695 | 4.493690 | 34.319543 | 7.459 |
| 4 | 1 | 6.021794 | 30.154684 | 5.255846 | 41.445280 | 6.177 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.021088 | 26.974689 | 4.874768 | 37.882411 |
