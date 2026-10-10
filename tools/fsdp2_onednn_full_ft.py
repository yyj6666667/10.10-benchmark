#!/usr/bin/env python3
"""Full fine-tuning with FSDP2 on GPU and oneDNN routed experts on CPU.

One process uses one GPU and owns a disjoint slice of CPU experts. FSDP2
shards the dense GPU trunk; differentiable all-to-all connects the CPU
experts across data-parallel ranks. The original Transformers MoE block
keeps its router and shared expert, so both model variants retain their
native routing semantics.
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
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    checkpoint: Path
    model_class: str
    template: str
    moe_layers: int
    parameter_count: int


MODELS = {
    "glm45_air": ModelSpec(
        Path("/mnt/data/models/GLM-4.5-Air"),
        "Glm4MoeForCausalLM",
        "glm4_moe",
        45,
        106_852_245_504,
    ),
    "qwen35_122b": ModelSpec(
        Path("/mnt/data/models/Qwen3.5-122B-A10B"),
        "Qwen3_5MoeForCausalLM",
        "qwen3",
        48,
        122_111_526_912,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=MODELS, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="fft_real_100")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--threads-per-rank", type=int, default=8)
    parser.add_argument("--repeat-batch", action="store_true")
    parser.add_argument("--release-cache-each-step", action="store_true")
    # Whole-layer checkpointing would replay CPU all-to-all during backward.
    # It is intentionally unsupported until a GPU-only policy is validated.
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if (
        args.sequence_length < 2
        or args.steps <= args.warmup_steps
        or args.warmup_steps < 0
    ):
        parser.error("sequence length must be >= 2 and steps must exceed warmup steps")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.threads_per_rank <= 0:
        parser.error("learning rate, gradient norm, and CPU threads must be positive")
    if args.gradient_checkpointing:
        parser.error("whole-layer checkpointing is unsafe with CPU expert all-to-all")
    args.model_path = args.model_path or MODELS[args.model_kind].checkpoint
    return args


def pin_cpu_threads(local_rank: int, world_size: int, thread_count: int) -> list[int]:
    """Partition physical cores by socket so CPU experts use local memory."""
    cores_by_socket: dict[int, dict[int, int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        socket = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        cores_by_socket.setdefault(socket, {}).setdefault(core, cpu)

    sockets = sorted(cores_by_socket)
    if not sockets or world_size % len(sockets):
        raise RuntimeError("GPU ranks cannot be split evenly across CPU sockets")
    ranks_per_socket = world_size // len(sockets)
    socket = sockets[local_rank // ranks_per_socket]
    cores = list(cores_by_socket[socket].values())
    cores_per_rank = len(cores) // ranks_per_socket
    if thread_count > cores_per_rank:
        raise RuntimeError(
            f"rank {local_rank} requested {thread_count} threads but has "
            f"only {cores_per_rank} physical cores on socket {socket}"
        )
    slot = local_rank % ranks_per_socket
    selected = cores[slot * cores_per_rank : slot * cores_per_rank + thread_count]
    os.sched_setaffinity(0, selected)
    torch.set_num_threads(thread_count)
    torch.set_num_interop_threads(1)
    return selected


def load_records(dataset_dir: Path, dataset_name: str) -> list[dict[str, str]]:
    manifest = json.loads(
        (dataset_dir / "dataset_info.json").read_text(encoding="utf-8")
    )
    data_file = dataset_dir / manifest[dataset_name]["file_name"]
    records = json.loads(data_file.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"dataset is empty: {data_file}")
    for index, record in enumerate(records):
        if not all(key in record for key in ("instruction", "input", "output")):
            raise ValueError(f"record {index} is not an Alpaca instruction example")
    return records


def encode_samples(
    tokenizer: Any,
    spec: ModelSpec,
    records: list[dict[str, str]],
    sequence_length: int,
) -> list[dict[str, torch.Tensor]]:
    """Use the same LLaMA Factory template and label mask as the KT run."""
    from llamafactory.data.processor.processor_utils import infer_seqlen
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments

    data_args = DataArguments(template=spec.template, cutoff_len=sequence_length)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    samples = []
    for record in records:
        user_text = str(record["instruction"])
        if record["input"]:
            user_text += "\n" + str(record["input"])
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": str(record["output"])},
        ]
        source, target = template.encode_oneturn(
            tokenizer, messages, system="", tools=""
        )
        budget = sequence_length - int(template.efficient_eos)
        source_length, target_length = infer_seqlen(len(source), len(target), budget)
        input_ids = source[:source_length] + target[:target_length]
        labels = [-100] * source_length + target[:target_length]
        if template.efficient_eos:
            input_ids.append(tokenizer.eos_token_id)
            labels.append(tokenizer.eos_token_id)
        if not any(label != -100 for label in labels):
            raise ValueError("sequence cutoff left no supervised token")
        valid_length = len(input_ids)
        padding = sequence_length - valid_length
        input_ids.extend([pad_id] * padding)
        labels.extend([-100] * padding)
        samples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.tensor(
                    [1] * valid_length + [0] * padding, dtype=torch.long
                ).unsqueeze(0),
                "labels": torch.tensor(labels, dtype=torch.long).unsqueeze(0),
            }
        )
    return samples


def named_tensors(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    yield from module.named_parameters()
    yield from module.named_buffers()


class ShardedCPUExperts(nn.Module):
    """Replace only the original experts module, leaving routing untouched."""

    def __init__(
        self, original: nn.Module, rank: int, world_size: int, cpu_group: Any
    ) -> None:
        super().__init__()
        if original.num_experts < world_size:
            raise ValueError("expert count must be at least the rank count")
        self.num_experts = original.num_experts
        self.hidden_size = original.hidden_dim
        self.intermediate_size = original.intermediate_dim
        self.activation = original.act_fn
        self.cpu_group = cpu_group
        self.world_size = world_size

        counts = [
            self.num_experts // world_size + int(r < self.num_experts % world_size)
            for r in range(world_size)
        ]
        self.first_expert = sum(counts[:rank])
        self.last_expert = self.first_expert + counts[rank]
        # A plain tensor stays on CPU when FSDP2 processes module buffers.
        self.owner_boundaries = torch.tensor(
            [sum(counts[:r]) for r in range(1, world_size)],
            dtype=torch.long,
            device="cpu",
        )
        owned = counts[rank]
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                owned,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=torch.float32,
                device="cpu",
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                owned,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float32,
                device="cpu",
            )
        )

    def _compute_owned(
        self, inputs: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Execute ordinary oneDNN-backed linear layers, one expert at a time."""
        output = inputs * 0
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for expert_id in range(self.first_expert, self.last_expert):
                positions = torch.nonzero(expert_ids == expert_id, as_tuple=True)[0]
                if positions.numel() == 0:
                    continue
                x = inputs.index_select(0, positions)
                local_id = expert_id - self.first_expert
                gate, up = F.linear(x, self.gate_up_proj[local_id]).chunk(2, dim=-1)
                y = F.linear(self.activation(gate) * up, self.down_proj[local_id])
                output.index_copy_(0, positions, y.to(output.dtype))
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = hidden_states.device, hidden_states.dtype
        tokens, width = hidden_states.shape
        top_k = selected_experts.shape[1]
        if width != self.hidden_size or selected_experts.shape != routing_weights.shape:
            raise ValueError("routed expert input shape mismatch")
        if selected_experts.shape[0] != tokens:
            raise ValueError("expert indices do not match token count")

        # GPU -> CPU: hidden states and routing decisions enter the CPU experts.
        cpu_hidden = hidden_states.to("cpu")
        expert_ids = selected_experts.reshape(-1).to("cpu", dtype=torch.long)
        owners = torch.bucketize(expert_ids, self.owner_boundaries, right=True)
        order = torch.argsort(owners, stable=True)
        send_counts = torch.bincount(owners, minlength=self.world_size)
        send_splits = send_counts.tolist()
        send_hidden = cpu_hidden.repeat_interleave(top_k, dim=0).index_select(0, order)
        send_ids = expert_ids.index_select(0, order)

        if self.world_size > 1:
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=self.cpu_group)
            recv_splits = recv_counts.tolist()
            recv_total = sum(recv_splits)
            recv_hidden = dist_nn.all_to_all_single(
                torch.empty((recv_total, width), dtype=send_hidden.dtype),
                send_hidden.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.cpu_group,
            )
            recv_ids = torch.empty(recv_total, dtype=torch.long)
            dist.all_to_all_single(
                recv_ids,
                send_ids.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.cpu_group,
            )
        else:
            recv_hidden, recv_ids = send_hidden, send_ids
            recv_splits = send_splits

        owned_output = self._compute_owned(recv_hidden, recv_ids)

        if self.world_size > 1:
            sent_back = dist_nn.all_to_all_single(
                torch.empty_like(send_hidden),
                owned_output.contiguous(),
                output_split_sizes=send_splits,
                input_split_sizes=recv_splits,
                group=self.cpu_group,
            )
        else:
            sent_back = owned_output

        restored = sent_back.index_select(0, torch.argsort(order))
        weighted = (
            restored.reshape(tokens, top_k, width)
            * routing_weights.to("cpu").unsqueeze(-1)
        ).sum(dim=1)
        # CPU -> GPU: autograd propagates gradients back across device copies.
        return weighted.to(device=device, dtype=dtype)


class CheckpointReader:
    """Read one tensor at a time without assembling the 100B+ checkpoint."""

    def __init__(self, model_path: Path, model_kind: str) -> None:
        self.model_path = model_path.resolve()
        self.model_kind = model_kind
        index = self.model_path / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))[
            "weight_map"
        ]
        self.stack = ExitStack()
        self.handles: dict[str, Any] = {}
        self.loaded_tensors = 0

    def __enter__(self) -> CheckpointReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def _source_name(self, name: str) -> str:
        if name in self.weight_map:
            return name
        if self.model_kind == "qwen35_122b" and name.startswith("model."):
            source = "model.language_model." + name.removeprefix("model.")
            if source in self.weight_map:
                return source
        raise KeyError(f"checkpoint has no tensor for {name}")

    def read(self, name: str, expert_slice: slice | None = None) -> torch.Tensor:
        from safetensors import safe_open

        source = self._source_name(name)
        shard = self.weight_map[source]
        if shard not in self.handles:
            self.handles[shard] = self.stack.enter_context(
                safe_open(str(self.model_path / shard), framework="pt", device="cpu")
            )
        handle = self.handles[shard]
        self.loaded_tensors += 1
        if expert_slice is None:
            return handle.get_tensor(source)
        return handle.get_slice(source)[expert_slice]

    @torch.no_grad()
    def load_tensors(
        self, model: nn.Module, tensors: Iterator[tuple[str, torch.Tensor]]
    ) -> None:
        for name, target in tensors:
            if target.is_meta:
                raise RuntimeError(f"unmaterialized tensor: {name}")
            if name.startswith("model.rotary_emb."):
                # RoPE frequencies are computed from config, not checkpointed.
                continue
            if ".experts." in name:
                experts = self._load_expert_tensor(model, name, target)
                if experts:
                    continue
            target.copy_(self.read(name))

    @torch.no_grad()
    def _load_expert_tensor(
        self, model: nn.Module, name: str, target: torch.Tensor
    ) -> bool:
        if not name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            return False
        parent = name.rsplit(".", 1)[0]
        owner = model.get_submodule(parent)
        if not isinstance(owner, ShardedCPUExperts):
            raise RuntimeError(f"expert ownership missing for {name}")
        first, last = owner.first_expert, owner.last_expert
        if self.model_kind == "qwen35_122b":
            target.copy_(self.read(name, slice(first, last)))
            return True

        for local_id, global_id in enumerate(range(first, last)):
            if name.endswith(".gate_up_proj"):
                half = target.shape[1] // 2
                for source, offset in (
                    ("gate_proj.weight", 0),
                    ("up_proj.weight", half),
                ):
                    value = self.read(f"{parent}.{global_id}.{source}")
                    target[local_id, offset : offset + half].copy_(value)
            else:
                target[local_id].copy_(
                    self.read(f"{parent}.{global_id}.down_proj.weight")
                )
        return True


def text_config(source_config: Any, model_kind: str) -> Any:
    if model_kind == "glm45_air":
        if source_config.model_type != "glm4_moe":
            raise RuntimeError("GLM checkpoint has unexpected model type")
        return source_config
    if source_config.model_type != "qwen3_5_moe":
        raise RuntimeError("Qwen checkpoint has unexpected model type")
    config = copy.deepcopy(source_config.text_config)
    if config.model_type != "qwen3_5_moe_text":
        raise RuntimeError("Qwen text config has unexpected model type")
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    config.tie_word_embeddings = bool(
        getattr(source_config, "tie_word_embeddings", False)
    )
    return config


def materialize_meta(
    module: nn.Module,
    device: torch.device,
    skip: set[nn.Module] | None = None,
) -> None:
    """Allocate meta tensors only; the owned expert tensors stay on CPU."""
    skipped = skip or set()
    stack = [module]
    while stack:
        child = stack.pop()
        if child in skipped:
            continue
        stack.extend(child.children())
        for name, parameter in list(child._parameters.items()):
            if parameter is not None and parameter.is_meta:
                child._parameters[name] = nn.Parameter(
                    torch.empty(parameter.shape, dtype=parameter.dtype, device=device),
                    requires_grad=parameter.requires_grad,
                )
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and buffer.is_meta:
                child._buffers[name] = torch.empty(
                    buffer.shape, dtype=buffer.dtype, device=device
                )


def make_model(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group: Any,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load and shard one GPU decoder layer at a time to bound peak VRAM."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[args.model_kind]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    source_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    config = text_config(source_config, args.model_kind)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    if type(model).__name__ != spec.model_class:
        raise RuntimeError(f"unexpected Transformers class: {type(model).__name__}")
    model.to(dtype=torch.float32)
    model.config.use_cache = False

    replaced = 0
    for layer in model.model.layers:
        if hasattr(layer.mlp, "experts"):
            layer.mlp.experts = ShardedCPUExperts(
                layer.mlp.experts, rank, world_size, cpu_group
            )
            replaced += 1
    if replaced != spec.moe_layers:
        raise RuntimeError(f"expected {spec.moe_layers} MoE layers, found {replaced}")

    cpu_params = [p for p in model.parameters() if p.device.type == "cpu"]
    full_dense = sum(p.numel() for p in model.parameters() if p.is_meta)
    logical_parameters = full_dense + world_size * sum(p.numel() for p in cpu_params)
    if logical_parameters != spec.parameter_count:
        raise RuntimeError(
            f"parameter contract changed: {logical_parameters} != {spec.parameter_count}"
        )
    stats = {
        "model_layers": len(model.model.layers),
        "moe_layers": replaced,
        "logical_parameters": logical_parameters,
        "cpu_expert_parameters_per_rank": sum(p.numel() for p in cpu_params),
        "gpu_parameters_before_sharding": full_dense,
    }

    mesh = init_device_mesh("cuda", (world_size,))
    precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    model.train()
    with CheckpointReader(args.model_path, args.model_kind) as checkpoint:
        for layer_id, layer in enumerate(model.model.layers):
            materialize_meta(layer, device)
            prefix = f"model.layers.{layer_id}."
            checkpoint.load_tensors(
                model,
                ((prefix + name, tensor) for name, tensor in named_tensors(layer)),
            )
            ignored = {p for p in layer.parameters() if p.device.type == "cpu"}
            fully_shard(
                layer,
                mesh=mesh,
                mp_policy=precision,
                ignored_params=ignored,
                reshard_after_forward=True,
            )
            torch.cuda.empty_cache()
            if rank == 0 and (layer_id + 1) % 10 == 0:
                print(
                    f"[load] sharded {layer_id + 1}/{len(model.model.layers)} layers",
                    flush=True,
                )

        # RoPE inverse frequencies are nonpersistent buffers. Materializing
        # their meta storage with torch.empty would leave undefined values.
        rotary = model.model.rotary_emb
        model.model.rotary_emb = type(rotary)(config, device=device)
        materialize_meta(model, device, skip={model.model.layers})
        checkpoint.load_tensors(
            model,
            (
                (name, tensor)
                for name, tensor in named_tensors(model)
                if not name.startswith("model.layers.")
            ),
        )
        stats["loaded_checkpoint_tensors"] = checkpoint.loaded_tensors

    ignored = {p for p in model.parameters() if p.device.type == "cpu"}
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=precision,
        ignored_params=ignored,
        reshard_after_forward=True,
    )
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("model still has a meta parameter")
    if any(p.device.type != "cpu" for p in ignored):
        raise RuntimeError("FSDP moved an ignored CPU expert parameter")
    gpu_params = [p for p in model.parameters() if p.device.type == "cuda"]
    if any(not hasattr(p, "to_local") for p in gpu_params):
        raise RuntimeError("a GPU parameter was not sharded by FSDP2")
    stats["gpu_parameters_per_rank"] = sum(p.to_local().numel() for p in gpu_params)
    stats["gpu_allocated_gib_after_load"] = torch.cuda.memory_allocated(device) / 2**30
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return model, tokenizer, stats


class CompactCPUAdamW:
    """FP32 expert weights with BF16 resident moments to fit host memory."""

    def __init__(self, params: list[nn.Parameter], learning_rate: float) -> None:
        self.params = params
        self.learning_rate = learning_rate
        self.state: dict[nn.Parameter, tuple[int, torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def step(self) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            if param.device.type != "cpu" or param.dtype != torch.float32:
                raise RuntimeError("CPU expert optimizer expects FP32 CPU weights")
            if param not in self.state:
                self.state[param] = (
                    0,
                    torch.zeros_like(param, dtype=torch.bfloat16),
                    torch.zeros_like(param, dtype=torch.bfloat16),
                )
            step, first_bf16, second_bf16 = self.state[param]
            step += 1
            # Perform the arithmetic in FP32; only the persistent state is
            # compact. This also avoids rounding beta2=0.999 to 1 in BF16.
            first = first_bf16.float()
            second = second_bf16.float()
            first.mul_(beta1).add_(grad, alpha=1 - beta1)
            second.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            first_bf16.copy_(first)
            second_bf16.copy_(second)
            denominator = second.sqrt_().div_(math.sqrt(1 - beta2**step)).add_(eps)
            param.addcdiv_(
                first,
                denominator,
                value=-self.learning_rate / (1 - beta1**step),
            )
            self.state[param] = (step, first_bf16, second_bf16)

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None


def clip_global_grad_norm(
    gpu_params: list[nn.Parameter],
    cpu_params: list[nn.Parameter],
    world_size: int,
    device: torch.device,
    max_norm: float,
) -> float:
    """Clip one norm over unique FSDP GPU shards and owned CPU expert shards."""
    cpu_sumsq = torch.zeros((), dtype=torch.float64)
    gpu_sumsq = torch.zeros((), dtype=torch.float64, device=device)
    for param in cpu_params:
        if param.grad is not None:
            # Gloo all-to-all accumulates gradients from every rank's local
            # loss. FSDP2 has already averaged its GPU gradients.
            param.grad.div_(world_size)
            norm = torch.linalg.vector_norm(param.grad.detach()).to(torch.float64)
            cpu_sumsq += norm.square()
    for param in gpu_params:
        if param.grad is not None:
            norm = torch.linalg.vector_norm(param.grad.detach().to_local()).to(
                torch.float64
            )
            gpu_sumsq += norm.square()
    total_sumsq = gpu_sumsq + cpu_sumsq.to(device)
    dist.all_reduce(total_sumsq, op=dist.ReduceOp.SUM)
    global_norm = float(total_sumsq.sqrt().item())
    if not math.isfinite(global_norm):
        raise RuntimeError(f"nonfinite global gradient norm: {global_norm}")
    scale = min(1.0, max_norm / (global_norm + 1e-6))
    if scale < 1:
        for param in (*gpu_params, *cpu_params):
            if param.grad is not None:
                param.grad.mul_(scale)
    return global_norm


def learning_rate_at_step(base_rate: float, step: int, steps: int) -> float:
    """Cosine schedule with zero warmup, matching the KT benchmark config."""
    return base_rate * (1 + math.cos(math.pi * (step - 1) / steps)) / 2


def write_results(
    args: argparse.Namespace,
    rows: list[dict[str, float | int]],
    model_stats: dict[str, Any],
    world_size: int,
) -> None:
    stable = rows[args.warmup_steps :]

    def average(key: str) -> float | None:
        return statistics.fmean(float(row[key]) for row in stable) if stable else None

    step_seconds = average("step_sec")
    summary = {
        "backend": "pytorch_fsdp2_onednn_cpu",
        "model_kind": args.model_kind,
        "sequence_length": args.sequence_length,
        "global_batch_size": world_size,
        "dp": world_size,
        "tp": 1,
        "steps_requested": args.steps,
        "steps_completed": len(rows),
        "warmup_steps": args.warmup_steps,
        "repeat_batch": args.repeat_batch,
        "template": MODELS[args.model_kind].template,
        "compute_dtype": "bfloat16",
        "gpu_master_dtype": "float32",
        "cpu_master_dtype": "float32",
        "cpu_optimizer_state_dtype": "bfloat16",
        "gpu_optimizer_state_dtype": "float32",
        "gradient_checkpointing": False,
        "release_cache_each_step": args.release_cache_each_step,
        "model_stats": model_stats,
        "stable": {
            "step_sec": step_seconds,
            "tps": world_size * args.sequence_length / step_seconds
            if step_seconds
            else None,
            "forward_sec": average("forward_sec"),
            "backward_sec": average("backward_sec"),
            "optimizer_sec": average("optimizer_sec"),
        },
        "rows": rows,
    }
    (args.output_dir / "step_timing.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "step_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["step"])
        writer.writeheader()
        writer.writerows(rows)


def run_training(
    args: argparse.Namespace,
    model: nn.Module,
    samples: list[dict[str, torch.Tensor]],
    model_stats: dict[str, Any],
    rank: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    gpu_params = [p for p in model.parameters() if p.device.type == "cuda"]
    cpu_params = [p for p in model.parameters() if p.device.type == "cpu"]
    if not gpu_params or not cpu_params:
        raise RuntimeError("expected both sharded GPU and owned CPU parameters")
    gpu_optimizer = torch.optim.AdamW(
        gpu_params, lr=args.learning_rate, weight_decay=0, fused=True
    )
    cpu_optimizer = CompactCPUAdamW(cpu_params, args.learning_rate)
    rows: list[dict[str, float | int]] = []

    for step in range(1, args.steps + 1):
        sample_id = rank if args.repeat_batch else ((step - 1) * world_size + rank)
        batch = {
            name: value.to(device)
            for name, value in samples[sample_id % len(samples)].items()
        }
        torch.cuda.synchronize(device)
        start = time.perf_counter()

        loss = model(**batch, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss on rank {rank}, step {step}")
        torch.cuda.synchronize(device)
        forward_sec = time.perf_counter() - start

        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_sec = time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        rate = learning_rate_at_step(args.learning_rate, step, args.steps)
        gpu_optimizer.param_groups[0]["lr"] = rate
        cpu_optimizer.learning_rate = rate
        norm = clip_global_grad_norm(
            gpu_params, cpu_params, world_size, device, args.max_grad_norm
        )
        gpu_optimizer.step()
        cpu_optimizer.step()
        gpu_optimizer.zero_grad(set_to_none=True)
        cpu_optimizer.zero_grad()
        torch.cuda.synchronize(device)
        if args.release_cache_each_step:
            torch.cuda.empty_cache()
        optimizer_sec = time.perf_counter() - optimizer_start
        step_sec = time.perf_counter() - start

        mean_loss = loss.detach().to(torch.float64)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        maxima = torch.tensor(
            [
                forward_sec,
                backward_sec,
                optimizer_sec,
                step_sec,
                torch.cuda.max_memory_reserved(device) / 2**30,
                torch.cuda.memory_allocated(device) / 2**30,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        if rank == 0:
            row = {
                "step": step,
                "loss": mean_loss.item() / world_size,
                "grad_norm": norm,
                "learning_rate": rate,
                "forward_sec": maxima[0].item(),
                "backward_sec": maxima[1].item(),
                "optimizer_sec": maxima[2].item(),
                "step_sec": maxima[3].item(),
                "tps": world_size * args.sequence_length / maxima[3].item(),
                "gpu_max_reserved_gib": maxima[4].item(),
                "gpu_allocated_gib_after_step": maxima[5].item(),
            }
            rows.append(row)
            print(
                f"[train] model={args.model_kind} seq={args.sequence_length} "
                f"step={step}/{args.steps} loss={row['loss']:.6f} "
                f"step_sec={row['step_sec']:.3f} tps={row['tps']:.3f}",
                flush=True,
            )
            write_results(args, rows, model_stats, world_size)
        # Rank 0 file I/O must not become the next step's forward latency.
        dist.barrier(device_ids=[local_rank])
    if rank == 0:
        (args.output_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"this benchmark requires DP=8, got {world_size}")
    affinity = pin_cpu_threads(local_rank, world_size, args.threads_per_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    try:
        if not torch.backends.mkldnn.is_available():
            raise RuntimeError("this PyTorch build has no oneDNN CPU support")
        torch.backends.mkldnn.enabled = True
        torch.manual_seed(42)
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            config.update(
                {
                    "dp": world_size,
                    "tp": 1,
                    "cpu_affinity_rank0": affinity,
                    "torch": torch.__version__,
                    "oneDNN_available": torch.backends.mkldnn.is_available(),
                }
            )
            (args.output_dir / "run_config.json").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
        dist.barrier(device_ids=[local_rank])
        model, tokenizer, model_stats = make_model(
            args, rank, world_size, device, cpu_group
        )
        records = load_records(args.dataset_dir, args.dataset_name)
        samples = encode_samples(
            tokenizer, MODELS[args.model_kind], records, args.sequence_length
        )
        if rank == 0:
            print(f"[load] {model_stats}", flush=True)
            print(
                f"[data] encoded {len(samples)} examples with "
                f"{MODELS[args.model_kind].template} template",
                flush=True,
            )
        dist.barrier(device_ids=[local_rank])
        run_training(
            args,
            model,
            samples,
            model_stats,
            rank,
            local_rank,
            world_size,
            device,
        )
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
