# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 30.016

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.492074 | 20.245977 | 6.567081 | 33.336502 | 30.717 |
| 2 | 1 | 6.073673 | 24.056756 | 3.220600 | 33.362364 | 30.693 |
| 3 | 1 | 6.067659 | 23.297547 | 4.371785 | 33.748807 | 30.342 |
| 4 | 1 | 6.073003 | 23.167112 | 5.231712 | 34.481721 | 29.697 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.070331 | 23.232330 | 4.801748 | 34.115264 |
