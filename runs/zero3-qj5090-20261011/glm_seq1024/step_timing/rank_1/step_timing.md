# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 182.335

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.547064 | 20.652932 | 11.190068 | 38.422667 | 213.207 |
| 2 | 1 | 6.191571 | 26.028137 | 2.684318 | 34.921537 | 234.583 |
| 3 | 1 | 6.197801 | 24.387123 | 14.297071 | 44.892235 | 182.481 |
| 4 | 1 | 6.193413 | 29.757035 | 9.003731 | 44.964515 | 182.188 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.195607 | 27.072079 | 11.650401 | 44.928375 |
