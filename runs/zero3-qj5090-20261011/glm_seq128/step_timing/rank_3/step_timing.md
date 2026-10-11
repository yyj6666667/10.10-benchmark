# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 29.488

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.491683 | 20.245771 | 9.582837 | 36.353549 | 28.168 |
| 2 | 1 | 6.075531 | 24.056125 | 6.555747 | 36.696468 | 27.905 |
| 3 | 1 | 6.068899 | 23.295492 | 5.387514 | 34.761954 | 29.457 |
| 4 | 1 | 6.074494 | 23.181152 | 5.424852 | 34.689394 | 29.519 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.071696 | 23.238322 | 5.406183 | 34.725674 |
