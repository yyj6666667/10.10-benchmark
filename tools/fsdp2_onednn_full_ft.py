#!/usr/bin/env python3
"""Full fine-tuning with FSDP2 GPU trunk and sharded oneDNN CPU experts.

The routed experts are owned by rank, rather than replicated. CPU all-to-all
dispatch has an autograd backward, so both expert and router gradients flow.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F


MODEL_DEFAULTS = {
    "qwen35_122b": Path("/mnt/data2/models/Qwen3.5-122B-A10B"),
    "glm45_air": Path("/mnt/data2/models/GLM-4.5-Air"),
}


class ShardedCPUExperts(nn.Module):
    """Own a disjoint expert-ID range and route token assignments over Gloo."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        world_size: int,
        cpu_group: Any,
        activation: Any = F.silu,
    ) -> None:
        super().__init__()
        if num_experts < world_size or rank not in range(world_size):
            raise ValueError("expert count must cover all ranks")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank
        self.world_size = world_size
        self.cpu_group = cpu_group
        self.activation = activation
        self.layer_index = -1
        self._route_call = 0
        counts = [num_experts // world_size + (r < num_experts % world_size) for r in range(world_size)]
        self.expert_start = sum(counts[:rank])
        self.expert_end = self.expert_start + counts[rank]
        # FSDP2 moves module buffers to CUDA even when expert parameters are
        # ignored. Keep this routing metadata as a plain CPU tensor.
        self.owner_boundaries = torch.tensor(
            [sum(counts[:r]) for r in range(1, world_size)], dtype=torch.long
        )
        self.gate_up_proj = nn.Parameter(torch.empty(
            counts[rank], 2 * intermediate_size, hidden_size,
            device="cpu", dtype=torch.float32,
        ))
        self.down_proj = nn.Parameter(torch.empty(
            counts[rank], hidden_size, intermediate_size,
            device="cpu", dtype=torch.float32,
        ))

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        self._route_call += 1
        route_call = self._route_call
        def trace(stage: str) -> None:
            if os.getenv("FSDP2_DEBUG_ROUTING") == "1":
                print(
                    f"[route] rank={self.rank} layer={self.layer_index} "
                    f"call={route_call} stage={stage}", flush=True,
                )
        trace("enter")
        if hidden_states.device.type != "cpu" or selected_experts.device.type != "cpu":
            raise RuntimeError("CPU experts require CPU input and routing indices")
        tokens, hidden_size = hidden_states.shape
        top_k = selected_experts.shape[1]
        if selected_experts.shape[0] != tokens or hidden_size != self.hidden_size:
            raise ValueError("invalid routed expert input shape")
        if routing_weights.shape != selected_experts.shape:
            raise ValueError("routing weight shape mismatch")

        expert_ids = selected_experts.reshape(-1).long()
        owners = torch.bucketize(expert_ids, self.owner_boundaries, right=True)
        order = torch.argsort(owners, stable=True)
        send_counts = torch.bincount(owners, minlength=self.world_size).to(torch.long)
        send_splits = send_counts.tolist()
        send_hidden = hidden_states.repeat_interleave(top_k, dim=0).index_select(0, order).contiguous()
        send_ids = expert_ids.index_select(0, order).contiguous()

        if self.world_size > 1:
            recv_counts = torch.empty_like(send_counts)
            trace("counts")
            dist.all_to_all_single(recv_counts, send_counts, group=self.cpu_group)
            recv_splits = recv_counts.tolist()
            recv_total = sum(recv_splits)
            recv_hidden = dist_nn.all_to_all_single(
                torch.empty((recv_total, hidden_size), dtype=send_hidden.dtype),
                send_hidden,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.cpu_group,
            )
            trace("hidden")
            recv_ids = torch.empty(recv_total, dtype=torch.long)
            dist.all_to_all_single(
                recv_ids, send_ids,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.cpu_group,
            )
            trace("ids")
        else:
            recv_hidden, recv_ids = send_hidden, send_ids
            recv_splits = send_splits

        # Four contiguous experts per oneDNN batched matmul. The small padding
        # within each group avoids separate tiny GEMM launches for every expert.
        # Keep a zero autograd path if a rank happens to receive no assignments.
        local_output = recv_hidden * 0.0
        for group_start in range(0, self.expert_end - self.expert_start, 4):
            group_end = min(group_start + 4, self.expert_end - self.expert_start)
            indices = [
                torch.nonzero(recv_ids == self.expert_start + local_index, as_tuple=True)[0]
                for local_index in range(group_start, group_end)
            ]
            padded_count = max(index.numel() for index in indices)
            if padded_count == 0:
                continue
            packed = torch.stack([
                F.pad(recv_hidden.index_select(0, index),
                      (0, 0, 0, padded_count - index.numel()))
                for index in indices
            ])
            gate, up = torch.bmm(
                packed, self.gate_up_proj[group_start:group_end].transpose(1, 2)
            ).chunk(2, dim=-1)
            gated = self.activation(gate) * up
            outputs = torch.bmm(
                gated, self.down_proj[group_start:group_end].transpose(1, 2)
            )
            for slot, index in enumerate(indices):
                if index.numel():
                    local_output.index_copy_(0, index, outputs[slot, :index.numel()])

        if self.world_size > 1:
            trace("return")
            sent_back = dist_nn.all_to_all_single(
                torch.empty_like(send_hidden),
                local_output.contiguous(),
                output_split_sizes=send_splits,
                input_split_sizes=recv_splits,
                group=self.cpu_group,
            )
        else:
            sent_back = local_output
        trace("done")
        restored = sent_back.index_select(0, torch.argsort(order))
        return (restored.reshape(tokens, top_k, hidden_size) * routing_weights.unsqueeze(-1)).sum(dim=1)


class HybridMoE(nn.Module):
    """Run the router and shared path on CUDA and routed experts on CPU."""

    def __init__(self, block: nn.Module, model_kind: str, rank: int, world_size: int, cpu_group: Any) -> None:
        super().__init__()
        self.model_kind = model_kind
        self.gate = block.gate
        original_experts = block.experts
        self.experts = ShardedCPUExperts(
            num_experts=original_experts.num_experts,
            hidden_size=original_experts.hidden_dim,
            intermediate_size=original_experts.intermediate_dim,
            rank=rank,
            world_size=world_size,
            cpu_group=cpu_group,
            activation=original_experts.act_fn,
        )
        if model_kind == "qwen35_122b":
            self.shared_expert = block.shared_expert
            self.shared_expert_gate = block.shared_expert_gate
        else:
            self.shared_experts = block.shared_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if self.model_kind == "qwen35_122b":
            shared = self.shared_expert(flat)
            _, routing_weights, selected_experts = self.gate(flat)
        else:
            router_logits = self.gate(flat)
            selected_experts, routing_weights = self._glm_route(router_logits)
            shared = self.shared_experts(flat)
            shared = shared.reshape(-1, shared.shape[-1])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            routed = self.experts(
                flat.to(device="cpu"),
                selected_experts.to(device="cpu"),
                routing_weights.to(device="cpu"),
            )
        routed = routed.to(device=flat.device, dtype=flat.dtype)
        if self.model_kind == "qwen35_122b":
            shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
        return (routed + shared).reshape(original_shape)

    def _glm_route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = router_logits.sigmoid()
        correction = self.gate.e_score_correction_bias
        group_scores = (
            (scores + correction)
            .view(-1, self.gate.n_group, self.gate.n_routed_experts // self.gate.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.gate.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(
            -1, self.gate.n_group, self.gate.n_routed_experts // self.gate.n_group
        ).reshape(-1, self.gate.n_routed_experts)
        masked = (scores + correction).masked_fill(~score_mask.bool(), float("-inf"))
        indices = torch.topk(masked, k=self.gate.top_k, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        if self.gate.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return indices, weights * self.gate.routed_scaling_factor


class BF16StateAdamW:
    """AdamW for FP32 CPU masters with compact BF16 first/second moments."""

    def __init__(self, parameters: list[nn.Parameter], lr: float, weight_decay: float = 0.0) -> None:
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: dict[nn.Parameter, dict[str, Any]] = {}

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2 = 0.9, 0.999
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
                raise RuntimeError("BF16StateAdamW requires CPU FP32 master parameters")
            state = self.state.get(parameter)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(parameter, dtype=torch.bfloat16),
                    "exp_avg_sq": torch.zeros_like(parameter, dtype=torch.bfloat16),
                }
                self.state[parameter] = state
            state["step"] += 1
            grad_bf16 = gradient.to(torch.bfloat16)
            state["exp_avg"].mul_(beta1).add_(grad_bf16, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad_bf16, grad_bf16, value=1.0 - beta2)
            moment = state["exp_avg"].float()
            denom = state["exp_avg_sq"].float().sqrt_().div_(
                (1.0 - beta2 ** state["step"]) ** 0.5
            ).add_(1e-8)
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            parameter.addcdiv_(moment, denom, value=-self.lr / (1.0 - beta1 ** state["step"]))

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None


def _load_records(dataset_dir: Path, dataset_name: str) -> list[dict[str, Any]]:
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    data_path = dataset_dir / info[dataset_name]["file_name"]
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"empty dataset: {data_path}")
    return records


def _text_config(source_config: Any, model_kind: str) -> Any:
    if model_kind == "glm45_air":
        if source_config.model_type != "glm4_moe":
            raise RuntimeError(f"unexpected GLM checkpoint type: {source_config.model_type}")
        return source_config
    if source_config.model_type != "qwen3_5_moe":
        raise RuntimeError(f"unexpected Qwen checkpoint type: {source_config.model_type}")
    config = copy.deepcopy(source_config.text_config)
    if config.model_type != "qwen3_5_moe_text" or config.num_hidden_layers != 48 or config.num_experts != 256:
        raise RuntimeError("unexpected Qwen text checkpoint contract")
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    config.tie_word_embeddings = bool(getattr(source_config, "tie_word_embeddings", False))
    return config


def _materialize_meta(module: nn.Module, device: torch.device) -> None:
    """Allocate meta parameters on CUDA; pre-created expert shards stay on CPU."""
    for child in module.modules():
        for name, parameter in list(child._parameters.items()):
            if parameter is not None and parameter.is_meta:
                child._parameters[name] = nn.Parameter(
                    torch.empty(parameter.shape, device=device, dtype=parameter.dtype),
                    requires_grad=parameter.requires_grad,
                )
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and buffer.is_meta:
                child._buffers[name] = torch.empty(buffer.shape, device=device, dtype=buffer.dtype)


def _checkpoint_candidate(name: str, model_kind: str, weight_map: dict[str, str]) -> str | None:
    candidates = [name]
    if model_kind == "qwen35_122b" and name.startswith("model."):
        candidates.append("model.language_model." + name[len("model."):])
    return next((candidate for candidate in candidates if candidate in weight_map), None)


def _load_weights(model: nn.Module, model_path: Path, model_kind: str) -> dict[str, int]:
    """Load dense weights on every rank and only owned CPU expert slices."""
    from safetensors import safe_open

    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    tensors: dict[str, torch.Tensor] = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    requests: dict[str, list[tuple[str, torch.Tensor, str, int | None, int | None]]] = {}
    missing: list[str] = []
    for name, tensor in tensors.items():
        if tensor.is_meta:
            raise RuntimeError(f"parameter was not materialized: {name}")
        source = _checkpoint_candidate(name, model_kind, weight_map)
        if source is not None:
            kind = "expert_slice" if ".experts." in name and model_kind == "qwen35_122b" else "direct"
            if kind == "expert_slice":
                expert_module = model.get_submodule(name.rsplit(".", 1)[0])
                span = (expert_module.expert_start, expert_module.expert_end)
            else:
                span = (None, None)
            requests.setdefault(weight_map[source], []).append((source, tensor, kind, *span))
            continue
        if model_kind == "glm45_air" and name.endswith("mlp.experts.gate_up_proj"):
            expert_module = model.get_submodule(name.rsplit(".", 1)[0])
            half = tensor.shape[1] // 2
            for local_index, expert_id in enumerate(range(expert_module.expert_start, expert_module.expert_end)):
                for part, offset in (("gate_proj.weight", 0), ("up_proj.weight", half)):
                    source = f"{name.rsplit('.', 1)[0]}.{expert_id}.{part}"
                    if source not in weight_map:
                        missing.append(source)
                        continue
                    requests.setdefault(weight_map[source], []).append((
                        source, tensor, "glm_gate_up", local_index, offset
                    ))
            continue
        if model_kind == "glm45_air" and name.endswith("mlp.experts.down_proj"):
            expert_module = model.get_submodule(name.rsplit(".", 1)[0])
            for local_index, expert_id in enumerate(range(expert_module.expert_start, expert_module.expert_end)):
                source = f"{name.rsplit('.', 1)[0]}.{expert_id}.down_proj.weight"
                if source not in weight_map:
                    missing.append(source)
                    continue
                requests.setdefault(weight_map[source], []).append((
                    source, tensor, "glm_down", local_index, None
                ))
            continue
        if "rotary_emb" not in name:
            missing.append(name)
    if missing:
        raise RuntimeError(f"checkpoint is missing required tensors: {missing[:8]}")

    loaded = 0
    expert_tensors = 0
    with torch.no_grad():
        for shard_name, shard_requests in requests.items():
            with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as handle:
                for source, target, kind, local_index, offset in shard_requests:
                    if kind == "expert_slice":
                        value = handle.get_slice(source)[local_index:offset]
                        expert_tensors += 1
                    else:
                        value = handle.get_tensor(source)
                    if kind in ("direct", "expert_slice"):
                        target.copy_(value.to(device=target.device, dtype=target.dtype))
                    elif kind == "glm_gate_up":
                        target[local_index, offset:offset + value.shape[0]].copy_(value.to(target.dtype))
                    else:
                        target[local_index].copy_(value.to(target.dtype))
                    loaded += 1
                    del value
    return {"loaded_checkpoint_tensors": loaded, "sliced_expert_tensors": expert_tensors}


def _set_local_cpu_affinity(local_rank: int, threads: int) -> list[int]:
    """Keep each GPU's CPU work on its socket's physical cores on sap4."""
    if local_rank < 4:
        first_core = local_rank * 12
    else:
        first_core = 48 + (local_rank - 4) * 12
    desired = set(range(first_core, first_core + 12))
    allowed = os.sched_getaffinity(0)
    selected = desired & allowed
    if selected:
        os.sched_setaffinity(0, selected)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return sorted(os.sched_getaffinity(0))


def _build_model(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group: Any,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    model_path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    source_config = AutoConfig.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    config = _text_config(source_config, args.model_kind)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    expected_class = "Glm4MoeForCausalLM" if args.model_kind == "glm45_air" else "Qwen3_5MoeForCausalLM"
    if type(model).__name__ != expected_class:
        raise RuntimeError(f"unexpected model class: {type(model).__name__}")
    # BF16-only optimizer weights can round away small AdamW updates.
    # Keep FP32 master parameters and moments; cast to BF16 for compute.
    model.to(dtype=torch.float32)
    # The GLM router correction bias is stored and used in FP32.
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and "e_score_correction_bias" in name:
                module._buffers[name] = torch.empty(buffer.shape, dtype=torch.float32, device="meta")
    model.config.use_cache = False

    replaced = 0
    for layer_index, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            layer.mlp = HybridMoE(layer.mlp, args.model_kind, rank, world_size, cpu_group)
            layer.mlp.experts.layer_index = layer_index
            replaced += 1
    expected_moe = 45 if args.model_kind == "glm45_air" else 48
    if replaced != expected_moe:
        raise RuntimeError(f"expected {expected_moe} MoE layers, got {replaced}")

    _materialize_meta(model, device)
    wrong_device = [name for name, parameter in model.named_parameters()
                    if parameter.device.type == "cuda" and parameter.device != device]
    if wrong_device:
        raise RuntimeError(f"GPU parameters are on the wrong device: {wrong_device[:4]}")
    residency = {
        "model_layers": len(model.model.layers),
        "moe_layers": replaced,
        "cpu_expert_parameters_local": sum(
            p.numel() for p in model.parameters() if p.device.type == "cpu"
        ),
        "gpu_trunk_parameters_full_before_fsdp": sum(
            p.numel() for p in model.parameters() if p.device.type == "cuda"
        ),
    }
    residency["logical_parameter_count"] = (
        residency["gpu_trunk_parameters_full_before_fsdp"]
        + world_size * residency["cpu_expert_parameters_local"]
    )
    expected_total = 106852245504 if args.model_kind == "glm45_air" else 122111526912
    if residency["logical_parameter_count"] != expected_total:
        raise RuntimeError(f"full-parameter contract mismatch: {residency['logical_parameter_count']} != {expected_total}")
    loaded = _load_weights(model, model_path, args.model_kind)
    residency.update(loaded)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    mesh = init_device_mesh("cuda", (world_size,))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        ignored = {p for p in layer.parameters() if p.device.type == "cpu"}
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, ignored_params=ignored,
                    reshard_after_forward=True)
    ignored = {p for p in model.parameters() if p.device.type == "cpu"}
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, ignored_params=ignored,
                reshard_after_forward=True)
    if any(p.device.type != "cpu" for p in ignored):
        raise RuntimeError("FSDP2 moved an ignored expert parameter off CPU")
    if any(not hasattr(p, "to_local") for p in model.parameters() if p.device.type == "cuda"):
        raise RuntimeError("GPU trunk contains an unsharded parameter")
    residency["gpu_trunk_parameters_shard_local"] = sum(
        p.to_local().numel() for p in model.parameters() if p.device.type == "cuda"
    )
    residency["gpu_allocated_gib_after_shard"] = torch.cuda.memory_allocated(device) / 2**30
    residency["gpu_reserved_gib_before_cache_release"] = torch.cuda.memory_reserved(device) / 2**30
    torch.cuda.empty_cache()
    residency["gpu_reserved_gib_after_cache_release"] = torch.cuda.memory_reserved(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    return model, tokenizer, residency


def _prepare_samples(tokenizer: Any, records: list[dict[str, Any]], sequence_length: int) -> list[dict[str, torch.Tensor]]:
    samples = []
    for record in records:
        text = str(record["instruction"])
        if record.get("input"):
            text += "\n" + str(record["input"])
        text += "\n" + str(record["output"])
        encoded = tokenizer(
            text, max_length=sequence_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        samples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        })
    return samples


def _clip_sharded_grad_norm(
    gpu_params: list[nn.Parameter],
    cpu_params: list[nn.Parameter],
    world_size: int,
    device: torch.device,
    max_norm: float,
) -> float:
    """Clip one global norm over FSDP GPU shards and unique CPU expert shards."""
    cpu_sumsq = torch.zeros((), dtype=torch.float64)
    gpu_sumsq = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in cpu_params:
        if parameter.grad is None:
            continue
        # CPU gradients are unique-owned, while FSDP already averages GPU
        # gradients across data-parallel ranks.
        parameter.grad.div_(world_size)
        norm = torch.linalg.vector_norm(parameter.grad.detach()).to(torch.float64)
        cpu_sumsq += norm * norm
    for parameter in gpu_params:
        if parameter.grad is None:
            continue
        norm = torch.linalg.vector_norm(parameter.grad.detach().to_local()).to(torch.float64)
        gpu_sumsq += norm * norm
    global_sumsq = gpu_sumsq + cpu_sumsq.to(device)
    dist.all_reduce(global_sumsq, op=dist.ReduceOp.SUM)
    global_norm = float(global_sumsq.sqrt().item())
    if not math.isfinite(global_norm):
        raise RuntimeError(f"non-finite global gradient norm: {global_norm}")
    coefficient = min(1.0, max_norm / (global_norm + 1e-6))
    if coefficient < 1.0:
        for parameter in cpu_params:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
        for parameter in gpu_params:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return global_norm


def _save_results(args: argparse.Namespace, rows: list[dict[str, Any]], residency: dict[str, Any]) -> None:
    stable = rows[args.warmup_steps:]
    mean_step = statistics.fmean(row["step_sec"] for row in stable) if stable else None
    summary = {
        "backend": "pytorch_fsdp2_onednn_cpu",
        "model_kind": args.model_kind,
        "sequence_length": args.sequence_length,
        "global_batch_size": 8,
        "dp": 8,
        "pp": 1,
        "tp": 1,
        "steps_requested": args.steps,
        "steps_completed": len(rows),
        "warmup_steps": args.warmup_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "repeat_batch": args.repeat_batch,
        "parameter_dtype": "float32",
        "compute_dtype": "bfloat16",
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": 0.0,
        "cpu_optimizer_state_dtype": "bfloat16",
        "gpu_optimizer_state_dtype": "float32",
        "residency": residency,
        "stable": {
            "step_sec": mean_step,
            "tps": 8 * args.sequence_length / mean_step if mean_step else None,
            "forward_sec": statistics.fmean(row["forward_sec"] for row in stable) if stable else None,
            "backward_sec": statistics.fmean(row["backward_sec"] for row in stable) if stable else None,
            "optimizer_sec": statistics.fmean(row["optimizer_sec"] for row in stable) if stable else None,
        },
        "rows": rows,
    }
    (args.output_dir / "step_timing.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "step_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="fft_real_100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--repeat-batch", action="store_true",
                        help="Repeat the same eight samples to check training loss direction")
    parser.add_argument("--threads-per-rank", type=int, default=12)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.steps <= args.warmup_steps or args.sequence_length <= 0
            or args.threads_per_rank <= 0 or args.max_grad_norm <= 0):
        raise ValueError("invalid steps, warmup, sequence length, or thread count")
    args.model_path = args.model_path or MODEL_DEFAULTS[args.model_kind]
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"this benchmark requires 8 ranks, got {world_size}")
    affinity = _set_local_cpu_affinity(local_rank, args.threads_per_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    if not torch.backends.mkldnn.is_available():
        raise RuntimeError("this PyTorch installation has no oneDNN support")
    torch.backends.mkldnn.enabled = True
    torch.manual_seed(1234)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"dp": 8, "pp": 1, "tp": 1, "expert_ownership": "contiguous_expert_ids",
                       "cpu_affinity_rank0": affinity, "torch": torch.__version__,
                       "onednn_available": torch.backends.mkldnn.is_available()})
        (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    dist.barrier(device_ids=[local_rank])

    model, tokenizer, residency = _build_model(args, rank, world_size, device, cpu_group)
    if rank == 0:
        print(f"[fsdp2_onednn] loaded model and sharded GPU trunk: {residency}", flush=True)
    records = _load_records(args.dataset_dir, args.dataset_name)
    samples = _prepare_samples(tokenizer, records, args.sequence_length)
    gpu_params = [p for p in model.parameters() if p.device.type == "cuda"]
    cpu_params = [p for p in model.parameters() if p.device.type == "cpu"]
    if not gpu_params or not cpu_params:
        raise RuntimeError("expected both FSDP2 GPU and sharded CPU parameters")
    gpu_optimizer = torch.optim.AdamW(
        gpu_params, lr=args.learning_rate, weight_decay=0.0, fused=True
    )
    cpu_optimizer = BF16StateAdamW(cpu_params, lr=args.learning_rate)

    rows: list[dict[str, Any]] = []
    try:
        for step in range(1, args.steps + 1):
            sample_index = rank if args.repeat_batch else (step - 1) * world_size + rank
            sample = samples[sample_index % len(samples)]
            batch = {key: value.to(device) for key, value in sample.items()}
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            output = model(**batch, use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step} rank {rank}: {loss}")
            torch.cuda.synchronize(device)
            forward_sec = time.perf_counter() - step_start

            backward_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            backward_sec = time.perf_counter() - backward_start

            optimizer_start = time.perf_counter()
            grad_norm = _clip_sharded_grad_norm(
                gpu_params, cpu_params, world_size, device, args.max_grad_norm
            )
            gpu_optimizer.step()
            cpu_optimizer.step()
            gpu_optimizer.zero_grad(set_to_none=True)
            cpu_optimizer.zero_grad()
            torch.cuda.synchronize(device)
            # Different routed batches retain different allocator blocks. The
            # active GPU state is ~11 GiB here, but stale cache reached 45 GiB.
            # Account for cache release in optimizer/step time.
            torch.cuda.empty_cache()
            optimizer_sec = time.perf_counter() - optimizer_start
            step_sec = time.perf_counter() - step_start

            loss_box = torch.tensor([float(loss.detach())], device=device, dtype=torch.float64)
            dist.all_reduce(loss_box, op=dist.ReduceOp.SUM)
            maxima = torch.tensor([
                forward_sec, backward_sec, optimizer_sec, step_sec,
                torch.cuda.max_memory_reserved(device) / 2**30,
                torch.cuda.memory_allocated(device) / 2**30,
            ], device=device, dtype=torch.float64)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            forward_max, backward_max, optimizer_max, step_max, gpu_max, gpu_live = maxima.tolist()

            row = {
                "step": step,
                "loss": float(loss_box.item()) / world_size,
                "grad_norm": grad_norm,
                "forward_sec": forward_max,
                "backward_sec": backward_max,
                "optimizer_sec": optimizer_max,
                "step_sec": step_max,
                "tps": 8 * args.sequence_length / step_max,
                "gpu_max_reserved_gib": gpu_max,
                "gpu_allocated_gib_after_step": gpu_live,
            }
            if rank == 0:
                rows.append(row)
                print(
                    f"[fsdp2_onednn] model={args.model_kind} seq={args.sequence_length} "
                    f"step={step}/{args.steps} loss={row['loss']:.6f} "
                    f"step_sec={row['step_sec']:.3f} tps={row['tps']:.3f}",
                    flush=True,
                )
                _save_results(args, rows, residency)
        if rank == 0:
            (args.output_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    finally:
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[fsdp2_onednn] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
