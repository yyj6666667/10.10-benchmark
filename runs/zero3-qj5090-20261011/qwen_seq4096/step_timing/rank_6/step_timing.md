# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 589.758

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.564132 | 33.315863 | 47.972340 | 90.971435 | 360.201 |
| 2 | 1 | 9.142099 | 35.884202 | 8.933200 | 54.115642 | 605.518 |
| 3 | 1 | 9.114614 | 37.835202 | 9.924109 | 57.039258 | 574.482 |
| 4 | 1 | 8.749104 | 37.097242 | 8.119390 | 54.084293 | 605.869 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.931859 | 37.466222 | 9.021749 | 55.561775 |
