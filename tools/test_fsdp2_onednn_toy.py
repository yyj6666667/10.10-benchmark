#!/usr/bin/env python3
"""Compare two-rank CPU expert routing and gradients with a dense reference."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fsdp2_onednn_full_ft import ShardedCPUExperts


def reference(x, indices, weights, gate_up, down):
    outputs = []
    for token, ids, token_weights in zip(x, indices, weights):
        selected = []
        for expert_id, routing_weight in zip(ids, token_weights):
            gate, up = F.linear(token[None], gate_up[expert_id]).chunk(2, dim=-1)
            value = F.linear(F.silu(gate) * up, down[expert_id])
            selected.append(value.squeeze(0) * routing_weight)
        outputs.append(torch.stack(selected).sum(dim=0))
    return torch.stack(outputs)


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2
    torch.set_num_threads(2)
    torch.manual_seed(7)
    full_gate_up = torch.randn(4, 10, 4, dtype=torch.float32) * 0.2
    full_down = torch.randn(4, 4, 5, dtype=torch.float32) * 0.2
    module = ShardedCPUExperts(4, 4, 5, rank, world_size, dist.group.WORLD)
    with torch.no_grad():
        module.gate_up_proj.copy_(full_gate_up[module.expert_start:module.expert_end])
        module.down_proj.copy_(full_down[module.expert_start:module.expert_end])

    xs, indices, ws = [], [], []
    assignments = [
        [[0, 2], [1, 3], [2, 0]],
        [[3, 1], [0, 3], [2, 1]],
    ]
    for r in range(world_size):
        torch.manual_seed(100 + r)
        xs.append((torch.randn(3, 4, dtype=torch.float32) * 0.1).requires_grad_())
        indices.append(torch.tensor(assignments[r], dtype=torch.long))
        ws.append(torch.randn(3, 2, dtype=torch.float32).requires_grad_())

    output = module(xs[rank], indices[rank], ws[rank])
    output.float().square().sum().backward()

    ref_x = [x.detach().clone().requires_grad_() for x in xs]
    ref_w = [w.detach().clone().requires_grad_() for w in ws]
    ref_gate_up = full_gate_up.detach().clone().requires_grad_()
    ref_down = full_down.detach().clone().requires_grad_()
    ref_outputs = [reference(ref_x[r], indices[r], ref_w[r], ref_gate_up, ref_down) for r in range(world_size)]
    sum(y.float().square().sum() for y in ref_outputs).backward()

    torch.testing.assert_close(output, ref_outputs[rank], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(xs[rank].grad, ref_x[rank].grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ws[rank].grad, ref_w[rank].grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(module.gate_up_proj.grad, ref_gate_up.grad[module.expert_start:module.expert_end], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(module.down_proj.grad, ref_down.grad[module.expert_start:module.expert_end], atol=1e-5, rtol=1e-5)
    print(f"rank={rank} route and gradients match dense reference", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
