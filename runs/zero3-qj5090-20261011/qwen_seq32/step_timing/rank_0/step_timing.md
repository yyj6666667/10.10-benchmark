# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.535

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.510054 | 35.605863 | 47.071440 | 90.223514 | 2.837 |
| 2 | 1 | 7.181284 | 25.895297 | 6.839786 | 39.945959 | 6.409 |
| 3 | 1 | 7.684215 | 26.953770 | 6.157809 | 40.828553 | 6.270 |
| 4 | 1 | 7.106702 | 25.864581 | 4.499304 | 37.518235 | 6.823 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.395458 | 26.409176 | 5.328556 | 39.173394 |
