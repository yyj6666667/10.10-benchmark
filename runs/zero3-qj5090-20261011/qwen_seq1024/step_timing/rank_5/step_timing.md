# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 149.552

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.979531 | 34.397622 | 27.608476 | 70.060081 | 116.928 |
| 2 | 1 | 8.139656 | 38.939243 | 16.482550 | 63.641792 | 128.720 |
| 3 | 1 | 7.469860 | 35.647350 | 16.125333 | 59.303577 | 138.137 |
| 4 | 1 | 7.993350 | 31.016588 | 11.182136 | 50.250066 | 163.025 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.731605 | 33.331969 | 13.653734 | 54.776822 |
