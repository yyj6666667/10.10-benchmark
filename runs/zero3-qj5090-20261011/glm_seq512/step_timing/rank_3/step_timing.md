# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 100.637

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.475000 | 25.280157 | 9.639161 | 41.427054 | 98.873 |
| 2 | 1 | 6.113955 | 25.672326 | 7.604783 | 39.401657 | 103.955 |
| 3 | 1 | 6.079963 | 24.802000 | 8.680957 | 39.574083 | 103.502 |
| 4 | 1 | 6.088665 | 28.185185 | 7.541845 | 41.827005 | 97.927 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.084314 | 26.493593 | 8.111401 | 40.700544 |
