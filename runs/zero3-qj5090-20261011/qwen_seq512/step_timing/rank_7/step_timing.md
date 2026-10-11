# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 83.551

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.965280 | 25.890418 | 18.001522 | 51.907754 | 78.909 |
| 2 | 1 | 8.042073 | 31.924003 | 10.369362 | 50.376405 | 81.308 |
| 3 | 1 | 7.497573 | 35.838101 | 8.954702 | 52.333211 | 78.268 |
| 4 | 1 | 7.498931 | 30.456986 | 7.722074 | 45.714541 | 89.599 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.498252 | 33.147543 | 8.338388 | 49.023876 |
