# 10.10 benchmark

Local Git repository for the sap4 single-node, full-parameter fine-tuning
benchmark. The intended GitHub destination is the private repository
`yyj6666667/10.10-benchmark`, once that account's authentication is restored.
No GitHub remote is configured yet.

## Contents

- `references/`: copies of the two user-provided historical test analyses.
  Their tables are **historical results**, not results of the current sweep.
- `results/historical_throughput.csv`: machine-readable step time and TPS
  extracted from those analyses by `tools/extract_historical.py`. It keeps
  `pytorch_torch` distinct from the figure's `pytorch_onednn` backend.
- `results/current_throughput.csv`: completed new throughput measurements.
- `results/semantic_checks.csv`: fixed-batch loss checks; these are not
  throughput measurements.
- `runs/kt-2026-10-10/`: small raw KT evidence: run configurations, software
  versions, timing, logs, and the isolated Qwen text-path patch. No model
  weights or optimizer checkpoints are included.

## Measurement convention

For eight-GPU server throughput, global batch size is 8. Each case has 15
optimizer steps, with the first 5 excluded. TPS is `8 * sequence_length /
mean_step_seconds` over the remaining 10 steps. A semantic check repeats a
fixed four-example batch on four GPUs and must not be plotted as throughput.

The new GLM KT 4096 case ran while a separate service occupied about 10.5
GiB on GPU 0. Its result is retained with that condition in the run evidence;
it is not a clean, same-host comparison against the historical framework
results. Future cases should each include their configuration, environment
versions, step timings, resource measurements, and final status.

Model checkpoints, adapter weights, optimizer state, and regenerable training
checkpoints are excluded from this data repository.
