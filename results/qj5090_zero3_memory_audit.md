# qj5090 ZeRO-3 CPU offload memory audit (2026-10-11)

This is the running audit for GLM-4.5-Air and Qwen3.5-122B-A10B. The eight-GPU
GLM diagnostic uses the actual ZeRO-3 CPU parameter and optimizer offload
configuration, BF16 master weights/gradients and BF16 optimizer states. No
checkpoint is saved. The pre-fix diagnostic was stopped deliberately after
the memory cause was established, while 248.6 GiB remained available.

| Phase | Actual host memory | Increment | Allocation / copy | Evidence |
|---|---:|---:|---|---|
| Idle, before ranks | MemAvailable 1464.9 GiB | — | Baseline | [pre-fix samples](../runs/zero3-qj5090-20261011/diagnostic_before/memory_diagnostic.jsonl) |
| Model constructed under `zero.Init` | Task cgroup 210.7 GiB; eight-rank PSS 210.0 GiB | +~210 GiB | ZeRO model construction and parameter partitions; no optimizer yet | [pre-fix phases](../runs/zero3-qj5090-20261011/diagnostic_before/launcher.log) |
| 47 checkpoint shards opened, before weight conversion finishes | Task cgroup 226.6 GiB; rank PSS 215.1 GiB | +15.9 GiB cgroup | Checkpoint tensors and mappings; no optimizer yet | [pre-fix samples](../runs/zero3-qj5090-20261011/diagnostic_before/memory_diagnostic.jsonl) |
| Original all-shard ZeRO load, deliberately stopped | Task cgroup 1425.1 GiB; rank PSS 1377.8 GiB, including 1246.2 GiB anonymous PSS; MemAvailable 248.6 GiB | +1198.5 GiB cgroup from shard-open point | Source merges all checkpoint shards in every rank, then applies MoE weight conversion before loading; exact split of the private allocation between conversion output and loader temporaries is still unmeasured | [pre-fix samples](../runs/zero3-qj5090-20261011/diagnostic_before/memory_diagnostic.jsonl) |
| Bounded shard loading, before optimizer | Peak task cgroup 247.9 GiB; rank PSS 245.9 GiB; checkpoint load and model finalize completed | At least 1177 GiB below the observed pre-fix cgroup peak | One complete GLM MoE layer is contained in each safetensors shard, so conversion and loading can release one shard before the next | [streaming phases](../runs/zero3-qj5090-20261011/diagnostic_streaming_load/launcher.log), [samples](../runs/zero3-qj5090-20261011/diagnostic_streaming_load/memory_diagnostic.jsonl) |
| ZeRO optimizer initialization | Closest sample 2.9 s before completion: task cgroup 876.5 GiB; rank PSS 820.1 GiB | +~665 GiB from optimizer start | BF16 parameter partition and BF16 master partition have different storage pointers; gradients are allocated. Adam moments are lazy and were empty at this boundary | [first-step log](../runs/zero3-qj5090-20261011/diagnostic_first_step/launcher.log), [samples](../runs/zero3-qj5090-20261011/diagnostic_first_step/memory_diagnostic.jsonl) |
| First forward/backward/step | Peak task cgroup 1269.7 GiB; rank PSS 1148.3 GiB; MemAvailable 441.5 GiB; GPU about 15.2 GiB/card | +~391 GiB from optimizer-initialized state | The first Adam step allocates moments lazily; the exact moment tensor dtypes will be logged in the next semantic run | [first-step log](../runs/zero3-qj5090-20261011/diagnostic_first_step/launcher.log), [samples](../runs/zero3-qj5090-20261011/diagnostic_first_step/memory_diagnostic.jsonl) |

The task cgroup had `memory.max=max` and `memory.events` showed no new OOM in
the diagnostics. Rank PSS was collected from `/proc/<pid>/smaps_rollup`; summed
RSS is not used as a physical-memory estimate. Locked memory was 0 KiB in all
rank samples. GPU usage stayed near 6.1 GiB per card during the pre-fix load.
The fixed one-step run passed with finite loss 9.416 and saved no checkpoint.
The earlier streaming trial stopped after the parameter contract because the
original DeepSpeed `weight_decay=0.01` differed from inherited TrainingArguments
`0.0`; the runner now aligns the training argument with 0.01.

**Confirmed:** The failure occurred during checkpoint loading, before
optimizer initialization, forward, backward, or step. The source branch
merged all 47 shards in each rank and performed MoE expert conversion on the
merged state dict. Private rank memory and the task cgroup rose together.
Streaming one shard at a time completed loading at a far lower peak. All 46
GLM expert layers have their source tensors within one shard each.

**Separate confirmed Qwen failure:** A meta-device model has 1164 expected
keys, all present in the 1949-key checkpoint. The nested text-model renaming
maps `model.language_model.*` to nonexistent `model.*`, so the original run
reported valid weights as missing and exhausted GPU memory while gathering
parameters for reinitialization. The source patch skips simple renaming when
every expected key already exists in the checkpoint. This path still needs a
full trial.

The reproducible source change is
[`patch_transformers_zero3_streaming.py`](../tools/patch_transformers_zero3_streaming.py).
The original `modeling_utils.py` SHA-256 is
`48c64b69a7aa1f2de111acbbe697aae3ce30db59e861ff6be552e08f67843006`;
the patched SHA-256 is
`a28fd6bbb07528db7372ebc06995be3aba4c8033a7a3f4f220a2d117bc574541`.
The original file is backed up on qj5090 at
`/mnt/hurry-bench/diagnostics/zero3_20261011/modeling_utils_before.py`.
