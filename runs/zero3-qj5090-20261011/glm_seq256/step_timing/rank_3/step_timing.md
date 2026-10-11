# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 54.443

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.525644 | 23.922147 | 8.724934 | 39.204507 | 52.239 |
| 2 | 1 | 6.177092 | 24.936815 | 5.933038 | 37.056865 | 55.266 |
| 3 | 1 | 6.191800 | 22.713978 | 6.692384 | 35.610491 | 57.511 |
| 4 | 1 | 6.159755 | 25.590781 | 7.862750 | 39.623528 | 51.686 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.175777 | 24.152380 | 7.277567 | 37.617010 |
