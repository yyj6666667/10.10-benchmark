# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 78.025

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.965508 | 25.911409 | 22.070729 | 55.991406 | 73.154 |
| 2 | 1 | 8.044423 | 31.904492 | 3.670828 | 43.665815 | 93.803 |
| 3 | 1 | 7.497399 | 35.819448 | 13.102537 | 56.459870 | 72.547 |
| 4 | 1 | 7.504064 | 30.456634 | 10.534679 | 48.532351 | 84.397 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.500731 | 33.138041 | 11.818608 | 52.496111 |
