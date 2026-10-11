# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 11.968

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.659122 | 33.475784 | 45.831817 | 87.016331 | 5.884 |
| 2 | 1 | 7.341835 | 28.655134 | 8.762936 | 44.811029 | 11.426 |
| 3 | 1 | 7.866714 | 27.117653 | 7.485656 | 42.509570 | 12.044 |
| 4 | 1 | 7.253520 | 28.114478 | 7.647150 | 43.052194 | 11.893 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.560117 | 27.616066 | 7.566403 | 42.780882 |
