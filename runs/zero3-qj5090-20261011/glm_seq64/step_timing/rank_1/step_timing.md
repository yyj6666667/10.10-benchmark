# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 15.539

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.460016 | 20.284937 | 7.594577 | 34.371577 | 14.896 |
| 2 | 1 | 6.059572 | 23.605066 | 3.586478 | 33.262662 | 15.393 |
| 3 | 1 | 6.069449 | 20.864544 | 4.673362 | 31.616788 | 16.194 |
| 4 | 1 | 6.041740 | 24.346394 | 3.883290 | 34.282113 | 14.935 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.055595 | 22.605469 | 4.278326 | 32.949451 |
