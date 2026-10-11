# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 23.966

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.710396 | 26.110048 | 18.171370 | 52.039382 | 19.677 |
| 2 | 1 | 7.411914 | 28.864960 | 9.232654 | 45.543496 | 22.484 |
| 3 | 1 | 7.266999 | 28.005377 | 8.239956 | 43.552799 | 23.512 |
| 4 | 1 | 7.272693 | 28.074964 | 6.508948 | 41.899840 | 24.439 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.269846 | 28.040171 | 7.374452 | 42.726320 |
