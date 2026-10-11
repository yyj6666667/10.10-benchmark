# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.250

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.011753 | 27.442102 | 65.036449 | 101.529491 | 2.521 |
| 2 | 1 | 7.397081 | 38.229966 | 7.005788 | 52.685873 | 4.859 |
| 3 | 1 | 7.663514 | 27.979568 | 4.863762 | 40.554705 | 6.312 |
| 4 | 1 | 7.272614 | 28.528666 | 5.529930 | 41.363536 | 6.189 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.468064 | 28.254117 | 5.196846 | 40.959120 |
