# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 158.134

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.993955 | 34.259926 | 26.965273 | 69.271155 | 118.260 |
| 2 | 1 | 8.139401 | 38.940649 | 16.161290 | 63.319869 | 129.375 |
| 3 | 1 | 7.471855 | 35.649320 | 8.981549 | 52.150486 | 157.084 |
| 4 | 1 | 7.992795 | 30.996809 | 12.419597 | 51.457641 | 159.199 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.732325 | 33.323065 | 10.700573 | 51.804064 |
