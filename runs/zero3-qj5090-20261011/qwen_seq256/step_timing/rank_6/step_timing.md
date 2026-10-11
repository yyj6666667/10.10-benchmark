# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 51.493

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.751974 | 28.086557 | 42.836763 | 78.716165 | 26.018 |
| 2 | 1 | 7.877695 | 28.271225 | 5.864200 | 42.076550 | 48.673 |
| 3 | 1 | 7.190961 | 27.255501 | 5.222140 | 39.727583 | 51.551 |
| 4 | 1 | 7.192176 | 26.842948 | 5.738614 | 39.816592 | 51.436 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.191569 | 27.049225 | 5.480377 | 39.772087 |
