# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 743.263

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.945083 | 26.760084 | 12.550499 | 46.294134 | 707.822 |
| 2 | 1 | 6.550596 | 28.083979 | 5.917728 | 40.571260 | 807.665 |
| 3 | 1 | 6.568139 | 29.330052 | 11.443624 | 47.357555 | 691.928 |
| 4 | 1 | 6.545226 | 26.847441 | 7.410280 | 40.815803 | 802.826 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.556682 | 28.088747 | 9.426952 | 44.086679 |
