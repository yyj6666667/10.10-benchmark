# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 56.940

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.529059 | 23.909087 | 10.317022 | 40.792219 | 50.206 |
| 2 | 1 | 6.179104 | 24.937932 | 10.309338 | 41.438064 | 49.423 |
| 3 | 1 | 6.194549 | 22.695185 | 6.585639 | 35.485646 | 57.713 |
| 4 | 1 | 6.161881 | 25.608221 | 4.667649 | 36.449658 | 56.187 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.178215 | 24.151703 | 5.626644 | 35.967652 |
