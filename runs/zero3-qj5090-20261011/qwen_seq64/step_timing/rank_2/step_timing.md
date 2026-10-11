# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 12.350

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.657747 | 33.480384 | 48.551680 | 89.729523 | 5.706 |
| 2 | 1 | 7.343773 | 28.670359 | 7.292914 | 43.367758 | 11.806 |
| 3 | 1 | 7.866163 | 27.113866 | 6.841087 | 41.882306 | 12.225 |
| 4 | 1 | 7.256584 | 28.110158 | 5.613618 | 41.033191 | 12.478 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.561374 | 27.612012 | 6.227352 | 41.457749 |
