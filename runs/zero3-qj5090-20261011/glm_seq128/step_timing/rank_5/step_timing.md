# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 30.596

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.491693 | 20.230433 | 6.458873 | 33.214640 | 30.830 |
| 2 | 1 | 6.073289 | 24.059079 | 4.796620 | 34.938678 | 29.308 |
| 3 | 1 | 6.067821 | 23.297705 | 4.995338 | 34.370783 | 29.793 |
| 4 | 1 | 6.072896 | 23.183629 | 3.297582 | 32.566525 | 31.443 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.070359 | 23.240667 | 4.146460 | 33.468654 |
