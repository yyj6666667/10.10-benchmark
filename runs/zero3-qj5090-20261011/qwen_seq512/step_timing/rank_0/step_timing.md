# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 85.319

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.963642 | 25.910174 | 19.691472 | 53.612042 | 76.401 |
| 2 | 1 | 8.044371 | 31.901826 | 11.229211 | 51.225632 | 79.960 |
| 3 | 1 | 7.499681 | 35.834382 | 7.561990 | 50.947581 | 80.396 |
| 4 | 1 | 7.500149 | 30.454699 | 7.043268 | 45.067995 | 90.885 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.499915 | 33.144541 | 7.302629 | 48.007788 |
