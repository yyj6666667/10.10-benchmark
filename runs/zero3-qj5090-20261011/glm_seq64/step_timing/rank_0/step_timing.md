# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 14.827

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.458640 | 20.284244 | 8.704615 | 35.479329 | 14.431 |
| 2 | 1 | 6.056395 | 23.602077 | 4.830713 | 34.502417 | 14.840 |
| 3 | 1 | 6.066967 | 20.861621 | 5.086819 | 32.025761 | 15.987 |
| 4 | 1 | 6.040735 | 24.342667 | 6.643028 | 37.035843 | 13.824 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.053851 | 22.602144 | 5.864924 | 34.530802 |
