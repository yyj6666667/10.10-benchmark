# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 84.243

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.964932 | 25.890700 | 15.507908 | 49.412638 | 82.894 |
| 2 | 1 | 8.041822 | 31.924271 | 9.828436 | 49.846931 | 82.172 |
| 3 | 1 | 7.497191 | 35.839851 | 8.320034 | 51.708067 | 79.214 |
| 4 | 1 | 7.498704 | 30.439986 | 7.556765 | 45.534829 | 89.953 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.497947 | 33.139918 | 7.938399 | 48.621448 |
