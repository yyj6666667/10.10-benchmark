# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.299

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.369837 | 24.338886 | 10.956894 | 41.699610 | 6.139 |
| 2 | 1 | 6.048074 | 23.835066 | 11.346712 | 41.239134 | 6.208 |
| 3 | 1 | 6.021023 | 23.814309 | 6.912259 | 36.761045 | 6.964 |
| 4 | 1 | 6.024516 | 30.285737 | 8.198148 | 44.521401 | 5.750 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.022770 | 27.050023 | 7.555203 | 40.641223 |
