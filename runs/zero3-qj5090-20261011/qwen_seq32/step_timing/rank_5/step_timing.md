# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.242

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.510878 | 35.581559 | 48.145586 | 91.277579 | 2.805 |
| 2 | 1 | 7.181281 | 25.892453 | 7.238827 | 40.348645 | 6.345 |
| 3 | 1 | 7.683251 | 26.950505 | 7.141764 | 41.802909 | 6.124 |
| 4 | 1 | 7.107366 | 25.844702 | 7.245000 | 40.223558 | 6.364 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.395309 | 26.397603 | 7.193382 | 41.013233 |
