# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 198.374

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.546586 | 20.652453 | 11.621900 | 38.853455 | 210.844 |
| 2 | 1 | 6.193325 | 26.028277 | 12.545860 | 44.776515 | 182.953 |
| 3 | 1 | 6.197187 | 24.387516 | 7.844240 | 38.440688 | 213.108 |
| 4 | 1 | 6.192044 | 29.773564 | 8.174102 | 44.150733 | 185.546 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.194616 | 27.080540 | 8.009171 | 41.295711 |
