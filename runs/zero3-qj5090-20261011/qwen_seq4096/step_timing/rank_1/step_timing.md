# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 618.349

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.565897 | 33.314769 | 42.540720 | 85.523903 | 383.144 |
| 2 | 1 | 9.140117 | 35.883314 | 7.491126 | 52.642256 | 622.466 |
| 3 | 1 | 9.118855 | 37.847388 | 6.461511 | 53.541348 | 612.013 |
| 4 | 1 | 8.756190 | 37.106027 | 6.467475 | 52.444186 | 624.817 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.937523 | 37.476707 | 6.464493 | 52.992767 |
