# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 596.526

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.565600 | 33.318213 | 43.485483 | 86.478852 | 378.913 |
| 2 | 1 | 9.139746 | 35.883498 | 8.218622 | 53.358655 | 614.108 |
| 3 | 1 | 9.117650 | 37.847358 | 8.404614 | 55.481001 | 590.617 |
| 4 | 1 | 8.756278 | 37.086455 | 8.427020 | 54.381754 | 602.555 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.936964 | 37.466907 | 8.415817 | 54.931377 |
