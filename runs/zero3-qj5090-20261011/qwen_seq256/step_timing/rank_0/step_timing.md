# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 52.327

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.748510 | 28.066953 | 30.238359 | 66.094259 | 30.986 |
| 2 | 1 | 7.877301 | 28.291493 | 5.665578 | 41.905003 | 48.872 |
| 3 | 1 | 7.190376 | 27.257286 | 4.498776 | 38.987517 | 52.530 |
| 4 | 1 | 7.189044 | 26.842138 | 5.219445 | 39.289422 | 52.126 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.189710 | 27.049712 | 4.859110 | 39.138469 |
