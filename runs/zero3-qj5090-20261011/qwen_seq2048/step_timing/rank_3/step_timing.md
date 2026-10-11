# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 355.406

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.613238 | 28.443044 | 47.817352 | 84.973919 | 192.812 |
| 2 | 1 | 7.616414 | 30.112264 | 11.000085 | 48.811491 | 335.659 |
| 3 | 1 | 8.005005 | 31.022782 | 7.334373 | 46.457281 | 352.668 |
| 4 | 1 | 7.978285 | 31.265737 | 6.397646 | 45.741424 | 358.187 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.991645 | 31.144260 | 6.866009 | 46.099353 |
