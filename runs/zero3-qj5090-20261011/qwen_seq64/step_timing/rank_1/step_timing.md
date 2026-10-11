# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 11.595

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.657559 | 33.482874 | 45.864193 | 87.063182 | 5.881 |
| 2 | 1 | 7.342678 | 28.662904 | 9.131601 | 45.190094 | 11.330 |
| 3 | 1 | 7.869221 | 27.139271 | 8.765293 | 43.808654 | 11.687 |
| 4 | 1 | 7.259784 | 28.100022 | 9.110120 | 44.501914 | 11.505 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.564502 | 27.619647 | 8.937707 | 44.155284 |
