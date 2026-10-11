# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 349.400

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.626759 | 28.447843 | 48.066824 | 85.224250 | 192.246 |
| 2 | 1 | 7.618036 | 30.119628 | 8.918559 | 46.737616 | 350.553 |
| 3 | 1 | 8.003989 | 31.005986 | 7.973406 | 47.064729 | 348.116 |
| 4 | 1 | 7.978196 | 31.289946 | 7.364151 | 46.718874 | 350.693 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.991093 | 31.147966 | 7.668778 | 46.891802 |
