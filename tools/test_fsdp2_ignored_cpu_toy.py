#!/usr/bin/env python3
"""Check that FSDP2 GPU sharding and CPU expert routing backpropagate together."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from fsdp2_onednn_full_ft import BF16StateAdamW, ShardedCPUExperts, _clip_sharded_grad_norm


class ToyModel(nn.Module):
    def __init__(self, rank, world_size, cpu_group, device):
        super().__init__()
        self.pre = nn.Linear(4, 4, dtype=torch.float32, device=device)
        self.experts = ShardedCPUExperts(4, 4, 5, rank, world_size, cpu_group)
        self.post = nn.Linear(4, 1, dtype=torch.float32, device=device)

    def forward(self, x, expert_ids, weights):
        hidden = self.pre(x)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            expert = self.experts(hidden.cpu(), expert_ids.cpu(), weights.cpu()).to(x.device)
        return self.post(hidden + expert)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = ToyModel(rank, world_size, cpu_group, device)
    with torch.no_grad():
        for parameter in model.experts.parameters():
            nn.init.normal_(parameter, std=0.2)
    mesh = init_device_mesh("cuda", (world_size,))
    ignored = set(model.experts.parameters())
    fully_shard(model, mesh=mesh,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                               reduce_dtype=torch.float32),
                ignored_params=ignored, reshard_after_forward=True)
    gpu_params = [p for p in model.parameters() if p.device.type == "cuda"]
    cpu_params = [p for p in model.parameters() if p.device.type == "cpu"]
    assert len(gpu_params) == 4 and len(cpu_params) == 2
    assert all(hasattr(p, "to_local") for p in gpu_params)
    assert all(not hasattr(p, "to_local") for p in cpu_params)
    before_gpu = [p.to_local().detach().clone() for p in gpu_params]
    before_cpu = [p.detach().clone() for p in cpu_params]
    gpu_optimizer = torch.optim.AdamW(gpu_params, lr=1e-2, fused=True)
    cpu_optimizer = BF16StateAdamW(cpu_params, lr=1e-1)
    x = torch.randn(3, 4, dtype=torch.bfloat16, device=device)
    expert_ids = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.long)
    weights = torch.ones(3, 2, dtype=torch.bfloat16) / 2
    loss = model(x, expert_ids, weights).float().square().sum() * 1000
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad.to_local()).all() for p in gpu_params)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in cpu_params)
    norm = _clip_sharded_grad_norm(gpu_params, cpu_params, world_size, device, 1.0)
    assert norm > 1.0
    gpu_optimizer.step()
    cpu_optimizer.step()
    torch.cuda.synchronize()
    assert any(not torch.equal(p.to_local(), old) for p, old in zip(gpu_params, before_gpu))
    assert any(not torch.equal(p, old) for p, old in zip(cpu_params, before_cpu))
    print(f"rank={rank} FSDP2 GPU and CPU expert weights updated; loss={loss.item():.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
