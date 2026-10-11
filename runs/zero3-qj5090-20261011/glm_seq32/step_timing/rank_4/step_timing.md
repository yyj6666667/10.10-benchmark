# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.248

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.370516 | 24.333677 | 11.304345 | 42.039618 | 6.089 |
| 2 | 1 | 6.047989 | 23.851509 | 4.694516 | 34.603686 | 7.398 |
| 3 | 1 | 6.019877 | 23.783906 | 9.186068 | 38.999981 | 6.564 |
| 4 | 1 | 6.022961 | 30.170398 | 6.744657 | 42.947467 | 5.961 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.021419 | 26.977152 | 7.965362 | 40.973724 |
