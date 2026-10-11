# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 209.731

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.542803 | 20.636302 | 7.564566 | 34.785258 | 235.502 |
| 2 | 1 | 6.190494 | 26.042453 | 4.709274 | 36.959300 | 221.649 |
| 3 | 1 | 6.196295 | 24.400590 | 5.934841 | 36.543017 | 224.174 |
| 4 | 1 | 6.191004 | 29.761674 | 5.610734 | 41.575925 | 197.037 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.193650 | 27.081132 | 5.772788 | 39.059471 |
