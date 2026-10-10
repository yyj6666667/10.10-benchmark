#!/usr/bin/env python3
"""Eight-GPU BF16 full fine-tuning through the official MegaTrain CPU master.

The harness supplies the same text-only checkpoint and tokenized examples as
the KT and FSDP2 cases. MegaTrain owns layer streaming, recomputation,
gradient accumulation, and GPU worker processes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from fsdp2_onednn_full_ft import MODELS, encode_samples, load_records, text_config


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=MODELS, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--repeat-batch", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.steps <= args.warmup_steps or args.sequence_length < 2:
        parser.error("steps must exceed warmup; sequence length must be at least 2")
    return args


def checkpoint_keys(root: Path) -> dict[str, str]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def source_name(name: str, model_kind: str) -> str:
    if model_kind == "qwen35_122b" and name.startswith("model."):
        return "model.language_model." + name.removeprefix("model.")
    return name


def make_text_model(model_kind: str, *, load_weights: bool):
    spec = MODELS[model_kind]
    config = text_config(
        AutoConfig.from_pretrained(spec.checkpoint, trust_remote_code=True,
                                   local_files_only=True),
        model_kind,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    if type(model).__name__ != spec.model_class:
        raise RuntimeError(f"wrong model class: {type(model).__name__}")
    count = sum(p.numel() for p in model.parameters())
    if count != spec.parameter_count:
        raise RuntimeError(f"parameter contract {count} != {spec.parameter_count}")
    weight_map = checkpoint_keys(spec.checkpoint)
    assignments: dict[str, list[tuple[str, torch.nn.Parameter, tuple]]] = {}
    names = set()
    for name, parameter in model.named_parameters():
        if model_kind == "glm45_air" and name.endswith(".experts.gate_up_proj"):
            parent = name.rsplit(".", 1)[0]
            half = parameter.shape[1] // 2
            for expert in range(parameter.shape[0]):
                for projection, selector in (
                    ("gate_proj", (expert, slice(0, half))),
                    ("up_proj", (expert, slice(half, None))),
                ):
                    source = f"{parent}.{expert}.{projection}.weight"
                    names.add(source)
                    if source in weight_map:
                        assignments.setdefault(weight_map[source], []).append(
                            (source, parameter, selector)
                        )
        elif model_kind == "glm45_air" and name.endswith(".experts.down_proj"):
            parent = name.rsplit(".", 1)[0]
            for expert in range(parameter.shape[0]):
                source = f"{parent}.{expert}.down_proj.weight"
                names.add(source)
                if source in weight_map:
                    assignments.setdefault(weight_map[source], []).append(
                        (source, parameter, (expert,))
                    )
        else:
            source = source_name(name, model_kind)
            names.add(source)
            if source in weight_map:
                assignments.setdefault(weight_map[source], []).append(
                    (source, parameter, ())
                )
    missing = sorted(names - weight_map.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoint tensors missing: {missing[:4]}")
    if not load_weights:
        return None, {"parameters": count, "checkpoint_tensors": len(names)}

    # MegaTrain's shared-memory flats and gradient buffers use this dtype.
    model.to(dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config, device="cpu")
    with torch.no_grad():
        for shard_no, (shard, entries) in enumerate(sorted(assignments.items()), 1):
            with safe_open(str(spec.checkpoint / shard), framework="pt", device="cpu") as reader:
                for source, parameter, selector in entries:
                    tensor = reader.get_tensor(source)
                    target = parameter[selector] if selector else parameter
                    if tensor.shape != target.shape:
                        raise RuntimeError(f"checkpoint shape differs: {source}")
                    target.copy_(tensor)
            print(f"[load] shard {shard_no}/{len(assignments)}", flush=True)
    model.train()
    return model, {"parameters": count, "checkpoint_tensors": len(names)}


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, contract = make_text_model(args.model_kind, load_weights=not args.check_only)
    print("[contract] " + json.dumps(contract), flush=True)
    if args.check_only:
        return

    from infinity import CPUMasterModel
    from infinity.config import CPUMasterConfig

    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[args.model_kind].checkpoint, trust_remote_code=True,
        local_files_only=True,
    )
    records = load_records(args.dataset_dir, "fft_real_100")
    examples = encode_samples(tokenizer, MODELS[args.model_kind], records,
                              args.sequence_length)
    config = CPUMasterConfig(
        model_name=str(MODELS[args.model_kind].checkpoint),
        devices=list(range(8)), batch_size=8, max_seq_len=args.sequence_length,
        num_steps=args.steps, learning_rate=1e-5, max_grad_norm=1.0,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        dataset_name="fft_real_100", dataset_dir=str(args.dataset_dir),
        checkpoint_interval=1, num_grad_slabs=2, seed=42,
    )
    engine = CPUMasterModel(model, config)
    del model
    torch.set_num_threads(64)
    parameters = engine.get_parameters()
    if sum(p.numel() for p in parameters) != contract["parameters"]:
        raise RuntimeError("MegaTrain optimizer does not own the full text model")
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    # MegaTrain's normal fast CPU optimizer. BF16 moments keep the 100B+
    # model within this host's memory budget.
    optimizer = DeepSpeedCPUAdam(parameters, lr=1e-5, weight_decay=0.01,
                                 adamw_mode=True, fp32_optimizer_states=False)
    rows = []
    try:
        for step in range(args.steps):
            indices = ([i % 8 for i in range(8)] if args.repeat_batch else
                       [(step * 8 + i) % len(examples) for i in range(8)])
            batch = {key: torch.cat([examples[i][key] for i in indices])
                     for key in ("input_ids", "attention_mask", "labels")}
            started = time.perf_counter()
            loss, tokens, timing = engine.forward_and_backward(**batch)
            optimizer_started = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            engine._sync_params_to_gpu()
            engine.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            completed = time.perf_counter()
            if not math.isfinite(loss) or not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"non-finite loss or gradient at step {step + 1}")
            row = {
                "step": step + 1, "loss": float(loss), "grad_norm": float(grad_norm),
                "tokens": tokens, "step_sec": completed - started,
                "forward_sec": timing["forward"],
                "backward_sec": timing["backward"],
                "optimizer_sec": completed - optimizer_started,
            }
            rows.append(row)
            print("[step] " + json.dumps(row), flush=True)
    finally:
        engine.cleanup()

    stable = rows[args.warmup_steps:]
    mean = lambda field: statistics.fmean(row[field] for row in stable)
    result = {
        "backend": "megatrain", "model_kind": args.model_kind,
        "sequence_length": args.sequence_length, "global_batch": 8,
        "steps_requested": args.steps, "steps_completed": len(rows),
        "warmup_steps": args.warmup_steps,
        "contract": contract, "repeat_batch": args.repeat_batch,
        "optimizer": "DeepSpeedCPUAdam(BF16 states)",
        "timing_method": "MegaTrain CUDA event forward/backward; host optimizer/step",
        "rows": rows,
        "stable": {field: mean(field) for field in
                   ("step_sec", "forward_sec", "backward_sec", "optimizer_sec")},
    }
    result["stable"]["tps"] = 8 * args.sequence_length / result["stable"]["step_sec"]
    (args.output_dir / "step_timing.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
