# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 14.303

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.460106 | 20.303236 | 13.945363 | 40.738722 | 12.568 |
| 2 | 1 | 6.060253 | 23.590893 | 4.505606 | 34.166561 | 14.985 |
| 3 | 1 | 6.069608 | 20.865336 | 8.869199 | 35.813417 | 14.296 |
| 4 | 1 | 6.042276 | 24.346608 | 5.383690 | 35.782033 | 14.309 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.055942 | 22.605972 | 7.126444 | 35.797725 |
