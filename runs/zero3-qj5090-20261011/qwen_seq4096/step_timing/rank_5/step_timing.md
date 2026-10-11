# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 599.299

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.565949 | 33.314749 | 45.800261 | 88.782919 | 369.080 |
| 2 | 1 | 9.139945 | 35.862973 | 7.059783 | 52.180166 | 627.978 |
| 3 | 1 | 9.118996 | 37.845433 | 8.199588 | 55.265658 | 592.918 |
| 4 | 1 | 8.756385 | 37.107504 | 8.126273 | 54.088814 | 605.818 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.937691 | 37.476468 | 8.162931 | 54.677236 |
