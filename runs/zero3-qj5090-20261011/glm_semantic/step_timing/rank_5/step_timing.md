# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.712

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.372553 | 32.274468 | 11.696272 | 51.375722 | 4.983 |
| 2 | 1 | 6.029800 | 31.408594 | 5.516844 | 42.966927 | 5.958 |
| 3 | 1 | 6.109186 | 24.977191 | 8.399698 | 39.496849 | 6.482 |
| 4 | 1 | 5.998175 | 23.944901 | 6.831273 | 36.787009 | 6.959 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.053680 | 24.461046 | 7.615485 | 38.141929 |
