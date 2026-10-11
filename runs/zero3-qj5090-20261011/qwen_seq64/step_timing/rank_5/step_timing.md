# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 12.420

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.658565 | 33.469931 | 28.294549 | 69.454505 | 7.372 |
| 2 | 1 | 7.340580 | 28.653198 | 9.121876 | 45.163451 | 11.337 |
| 3 | 1 | 7.867349 | 27.111281 | 6.477484 | 41.487266 | 12.341 |
| 4 | 1 | 7.255142 | 28.111552 | 5.555015 | 40.963302 | 12.499 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.561245 | 27.611416 | 6.016250 | 41.225284 |
