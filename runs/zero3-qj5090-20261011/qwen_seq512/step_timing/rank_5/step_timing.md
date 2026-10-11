# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 88.642

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.964443 | 25.910619 | 17.623359 | 51.548336 | 79.459 |
| 2 | 1 | 8.042385 | 31.924130 | 10.417400 | 50.427702 | 81.225 |
| 3 | 1 | 7.497726 | 35.817373 | 5.679686 | 49.042274 | 83.520 |
| 4 | 1 | 7.498898 | 30.458266 | 5.371033 | 43.374836 | 94.433 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.498312 | 33.137820 | 5.525360 | 46.208555 |
