# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.636

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.372392 | 32.292590 | 11.511848 | 51.207589 | 4.999 |
| 2 | 1 | 6.030225 | 31.410236 | 8.829119 | 46.279181 | 5.532 |
| 3 | 1 | 6.109630 | 24.960452 | 6.844889 | 37.927417 | 6.750 |
| 4 | 1 | 5.997606 | 23.945141 | 9.270506 | 39.226776 | 6.526 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.053618 | 24.452797 | 8.057697 | 38.577097 |
