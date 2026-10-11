# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.217

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.511006 | 35.604328 | 49.516164 | 92.663203 | 2.763 |
| 2 | 1 | 7.180611 | 25.894731 | 8.863133 | 41.967921 | 6.100 |
| 3 | 1 | 7.684138 | 26.954761 | 7.262882 | 41.929026 | 6.106 |
| 4 | 1 | 7.106122 | 25.867368 | 7.425139 | 40.424167 | 6.333 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.395130 | 26.411065 | 7.344010 | 41.176596 |
