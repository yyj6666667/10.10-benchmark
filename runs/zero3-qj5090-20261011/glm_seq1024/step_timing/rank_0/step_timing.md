# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 209.938

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.538966 | 20.638070 | 11.824990 | 39.041604 | 209.827 |
| 2 | 1 | 6.189977 | 26.029295 | 7.682620 | 39.911922 | 205.252 |
| 3 | 1 | 6.195676 | 24.388745 | 3.964479 | 34.562214 | 237.022 |
| 4 | 1 | 6.190862 | 29.758861 | 7.519408 | 43.479859 | 188.409 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.193269 | 27.073803 | 5.741943 | 39.021037 |
