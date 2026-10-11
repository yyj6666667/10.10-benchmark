# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.510

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.369457 | 24.321311 | 11.279394 | 42.004660 | 6.095 |
| 2 | 1 | 6.047811 | 23.869545 | 8.970565 | 38.896997 | 6.581 |
| 3 | 1 | 6.022450 | 23.783284 | 6.480956 | 36.296207 | 7.053 |
| 4 | 1 | 6.023148 | 30.155414 | 6.159395 | 42.349247 | 6.045 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.022799 | 26.969349 | 6.320175 | 39.322727 |
