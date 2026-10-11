# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 772.426

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.945362 | 26.761857 | 12.584997 | 46.330261 | 707.270 |
| 2 | 1 | 6.547671 | 28.085966 | 13.615563 | 48.266063 | 678.904 |
| 3 | 1 | 6.564344 | 29.330770 | 9.698480 | 45.610937 | 718.424 |
| 4 | 1 | 6.542952 | 26.865611 | 5.805240 | 39.233484 | 835.205 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.553648 | 28.098190 | 7.751860 | 42.422211 |
