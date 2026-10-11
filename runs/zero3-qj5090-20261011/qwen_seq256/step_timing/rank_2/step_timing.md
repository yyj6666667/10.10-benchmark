# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 52.587

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.751817 | 28.083780 | 14.467473 | 50.347120 | 40.678 |
| 2 | 1 | 7.876354 | 28.292283 | 6.604424 | 42.819087 | 47.829 |
| 3 | 1 | 7.191584 | 27.238315 | 5.420880 | 39.884546 | 51.348 |
| 4 | 1 | 7.192246 | 26.861688 | 3.900723 | 38.004808 | 53.888 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.191915 | 27.050001 | 4.660801 | 38.944677 |
