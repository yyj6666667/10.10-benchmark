# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 81.309

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.964071 | 25.927445 | 22.135003 | 56.067800 | 73.054 |
| 2 | 1 | 8.042673 | 31.905969 | 10.907976 | 50.898005 | 80.475 |
| 3 | 1 | 7.497162 | 35.836230 | 9.837980 | 53.216049 | 76.969 |
| 4 | 1 | 7.498511 | 30.461600 | 9.535132 | 47.535040 | 86.168 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.497836 | 33.148915 | 9.686556 | 50.375544 |
