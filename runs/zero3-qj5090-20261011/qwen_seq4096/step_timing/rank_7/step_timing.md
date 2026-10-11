# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 619.130

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.566019 | 33.293470 | 22.295292 | 65.270615 | 502.033 |
| 2 | 1 | 9.137294 | 35.882707 | 6.337572 | 51.470608 | 636.635 |
| 3 | 1 | 9.119816 | 37.844804 | 6.505614 | 53.569586 | 611.690 |
| 4 | 1 | 8.756898 | 37.103659 | 6.314241 | 52.282220 | 626.752 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.938357 | 37.474231 | 6.409928 | 52.925903 |
