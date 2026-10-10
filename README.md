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
- `tools/fsdp2_onednn_full_ft.py`: PyTorch FSDP2 + oneDNN CPU expert baseline.
- `runs/pytorch-fsdp2-onednn-2026-10-10/`: run configuration, per-step timings,
  environment versions, resource monitor, and loss evidence for this baseline.

## Measurement convention

For eight-GPU server throughput, global batch size is 8. Each case has 15
optimizer steps, with the first 5 excluded. TPS is `8 * sequence_length /
mean_step_seconds` over the remaining 10 steps. KT semantic checks repeat a
fixed four-example batch on four GPUs; the FSDP2 check repeats a fixed
eight-example batch on eight GPUs. Semantic checks must not be plotted as
throughput.

The FSDP2 + oneDNN baseline has data parallel size 8, pipeline size 1, and
tensor parallel size 1. FSDP2 owns the GPU trunk while each rank owns a
disjoint set of CPU experts. Expert assignments use differentiable Gloo
all-to-all; groups of four experts use CPU oneDNN AMX BF16 batched matmul.
GPU and CPU master parameters are FP32, compute is BF16, CPU AdamW moments
are BF16, GPU AdamW moments are FP32, and the global gradient norm is clipped
to 1.0. The learning rate is a constant 1e-5. Its forward, backward, and
optimizer host-wall timers synchronize CUDA at each boundary, so phase times
have a different boundary from the historical coarse-host-wall KT runs.
Whole-decoder gradient checkpointing is disabled because recomputing a decoder
also reentered its Gloo expert collectives in different orders across ranks
in a smoke test. Any later checkpointing must cover GPU-only submodules.

The new GLM KT 4096 case ran while a separate service occupied about 10.5
GiB on GPU 0. Its result is retained with that condition in the run evidence;
it is not a clean, same-host comparison against the historical framework
results. Future cases should each include their configuration, environment
versions, step timings, resource measurements, and final status.

Model checkpoints, adapter weights, optimizer state, and regenerable training
checkpoints are excluded from this data repository.

## qj5090 KT rerun

The qj5090 rerun uses `/mnt/hurry-bench` on the remote host. The base KT
training source is the clean `bfe3e57c` snapshot with the exact BF16 gradient
norm patch from KT `fa04bb2` applied. Updated Accelerate `9a0d243` and
Transformers `e04dcff` are loaded through an isolated `PYTHONPATH` overlay.
The built KT wheel is also installed only in that overlay. Source revisions
and the patch are in `tools/kt_qj_source_revisions.md` and
`tools/kt_grad_norm.patch`.

Run one GLM case on qj5090 with `tools/run_qj_kt_case.sh SEQ 15`. Each case
uses eight GPUs, BF16 full fine-tuning, global batch 8, gradient clipping at
1.0, and 15 steps with the first five excluded. Timing follows the earlier
KT host-wall phase timer, with no forced CUDA synchronization. The GPU peak is
the largest `nvidia-smi` used-memory sample over the eight cards. CPU RSS is
the peak sum of the benchmark process tree's RSS, so shared pages can be
counted more than once. Both resource measurements are sampled every two
seconds. No model or optimizer checkpoint is retained.

The official ModelScope `Qwen/Qwen3.5-122B-A10B` checkpoint downloads to
`/mnt/data/models/Qwen3.5-122B-A10B` with
`tools/download_qwen_bf16.py`. Its Qwen text-decoder path correction is
isolated in `/mnt/hurry-bench/py-overlay-qwen` and recorded as
`tools/qwen_text_path_contract.patch`. A Qwen case uses
`tools/run_qj_kt_case.sh SEQ 15 qwen`; the no-training preflight is under
`runs/kt-qj5090-20261010/preflight/`. The queued GLM sweep waits for both
concurrent model downloads to finish so download I/O does not affect timing.

qj5090 has an archived MegaTrain source and a prior 397B single-GPU routed
LoRA run, but the current benchmark venv has no installed MegaTrain module or
verified eight-GPU BF16 full-finetuning setup for the figure's 122B and GLM
models.
