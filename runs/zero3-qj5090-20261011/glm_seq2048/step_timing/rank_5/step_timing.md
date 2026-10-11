# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 416.769

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.777068 | 22.139232 | 12.796889 | 41.775727 | 392.189 |
| 2 | 1 | 6.336653 | 25.179227 | 3.349473 | 34.883286 | 469.681 |
| 3 | 1 | 6.301189 | 22.341664 | 10.155822 | 38.810606 | 422.153 |
| 4 | 1 | 6.332580 | 25.703205 | 7.767389 | 39.813253 | 411.521 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.316884 | 24.022435 | 8.961605 | 39.311929 |
