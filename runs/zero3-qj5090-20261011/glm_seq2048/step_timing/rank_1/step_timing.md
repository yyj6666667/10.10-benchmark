# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 428.965

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.812332 | 22.125112 | 12.856955 | 41.831295 | 391.668 |
| 2 | 1 | 6.338697 | 25.208189 | 12.068107 | 43.629545 | 375.525 |
| 3 | 1 | 6.304080 | 22.341753 | 8.263901 | 36.921265 | 443.755 |
| 4 | 1 | 6.333356 | 25.714627 | 7.407393 | 39.467285 | 415.129 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.318718 | 24.028190 | 7.835647 | 38.194275 |
