# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 140.839

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.993623 | 34.260437 | 29.402320 | 71.709540 | 114.239 |
| 2 | 1 | 8.142497 | 38.938968 | 20.295237 | 67.449000 | 121.455 |
| 3 | 1 | 7.468919 | 35.648873 | 18.063923 | 61.242009 | 133.764 |
| 4 | 1 | 7.994804 | 31.005982 | 16.043312 | 55.089125 | 148.704 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.731862 | 33.327427 | 17.053617 | 58.165567 |
