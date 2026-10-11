# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.291

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.510627 | 35.604963 | 50.640984 | 93.795186 | 2.729 |
| 2 | 1 | 7.180479 | 25.896691 | 8.847302 | 41.958745 | 6.101 |
| 3 | 1 | 7.682711 | 26.956389 | 6.789319 | 41.460137 | 6.175 |
| 4 | 1 | 7.107307 | 25.866944 | 6.918601 | 39.924077 | 6.412 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.395009 | 26.411667 | 6.853960 | 40.692107 |
