# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 30.176

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.483393 | 20.231736 | 10.063637 | 36.820570 | 27.811 |
| 2 | 1 | 6.075679 | 24.055697 | 5.252586 | 35.393164 | 28.932 |
| 3 | 1 | 6.068645 | 23.298486 | 4.004654 | 33.391113 | 30.667 |
| 4 | 1 | 6.072876 | 23.168225 | 5.225522 | 34.476820 | 29.701 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.070761 | 23.233356 | 4.615088 | 33.933966 |
