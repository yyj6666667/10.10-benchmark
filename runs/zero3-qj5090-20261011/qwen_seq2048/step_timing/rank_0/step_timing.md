# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 345.977

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.625952 | 28.447913 | 46.610285 | 83.776097 | 195.569 |
| 2 | 1 | 7.617507 | 30.128704 | 9.586862 | 47.403581 | 345.628 |
| 3 | 1 | 8.006262 | 31.028378 | 8.843239 | 47.954058 | 341.660 |
| 4 | 1 | 7.978526 | 31.269480 | 7.442548 | 46.757456 | 350.404 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.992394 | 31.148929 | 8.142894 | 47.355757 |
