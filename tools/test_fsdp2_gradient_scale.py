#!/usr/bin/env python3
"""Verify the installed FSDP2 reduces rank-local loss gradients by DP size."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(17)
    model = nn.Linear(4, 4, bias=False, device=device, dtype=torch.float32)
    original_weight = model.weight.detach().cpu().clone()
    fully_shard(model, mesh=init_device_mesh("cuda", (world_size,)))
    x = torch.arange(1 + rank * 4, 5 + rank * 4, dtype=torch.float32, device=device)[None]
    model(x).square().sum().backward()
    actual = model.weight.grad.to_local().cpu()

    reference_weight = original_weight.detach().clone().requires_grad_()
    reference_loss = 0
    for r in range(world_size):
        reference_input = torch.arange(1 + r * 4, 5 + r * 4, dtype=torch.float32)[None]
        reference_loss = reference_loss + nn.functional.linear(reference_input, reference_weight).square().sum()
    (reference_loss / world_size).backward()
    expected = reference_weight.grad[rank * 2:(rank + 1) * 2]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(f"rank={rank} FSDP2 gradient equals global mean", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
