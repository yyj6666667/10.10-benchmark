# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 335.819

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.626255 | 28.447434 | 49.545191 | 86.701925 | 188.969 |
| 2 | 1 | 7.618687 | 30.120903 | 8.901291 | 46.719952 | 350.685 |
| 3 | 1 | 8.005768 | 31.028141 | 10.145872 | 49.247875 | 332.684 |
| 4 | 1 | 7.981070 | 31.266194 | 9.015502 | 48.328568 | 339.013 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.993419 | 31.147167 | 9.580687 | 48.788221 |
