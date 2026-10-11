# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 99.908

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.477409 | 25.280137 | 10.717277 | 42.511201 | 96.351 |
| 2 | 1 | 6.114978 | 25.699208 | 15.103371 | 46.927102 | 87.284 |
| 3 | 1 | 6.081396 | 24.799280 | 8.813857 | 39.703940 | 103.164 |
| 4 | 1 | 6.088190 | 28.199459 | 7.992269 | 42.291215 | 96.852 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.084793 | 26.499370 | 8.403063 | 40.997578 |
