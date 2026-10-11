# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 194.438

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.540340 | 20.653363 | 11.368343 | 38.598795 | 212.235 |
| 2 | 1 | 6.191432 | 26.027803 | 2.942392 | 35.174714 | 232.895 |
| 3 | 1 | 6.192450 | 24.386792 | 11.158495 | 41.752707 | 196.203 |
| 4 | 1 | 6.192498 | 29.772888 | 6.534643 | 42.510559 | 192.705 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.192474 | 27.079840 | 8.846569 | 42.131633 |
