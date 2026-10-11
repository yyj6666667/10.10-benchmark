#!/usr/bin/env python3
"""Keep two prefetched layer slots while sharing flat and template storage."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")
MARKER = "MEGATRAIN_ALIAS_LAYER_SLOTS"


def replace(source: str, old: str, new: str, expected: int = 1) -> str:
    count = source.count(old)
    if count != expected:
        raise RuntimeError(f"expected {expected} patch sites, found {count}: {old[:75]!r}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().path
    source = path.read_text()
    if MARKER in source:
        print(f"already patched: {path}")
        return
    if "MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD" not in source:
        raise RuntimeError("install the output-head residency patch first")
    if "MEGATRAIN_BACKWARD_CACHE_EVICTION" in source:
        raise RuntimeError("restore the pre-cache-eviction worker before aliasing slots")

    source = replace(
        source,
        "def _create_worker_gpu_context(rank, device_id, shared_state):\n",
        '''ALIAS_LAYER_SLOTS = os.environ.get('MEGATRAIN_ALIAS_LAYER_SLOTS') == '1'


def _alias_template_weights(template, flat):
    """Make a template's Parameter storages views of its slot's flat buffer."""
    offset = 0
    for parameter in template.parameters():
        numel = parameter.numel()
        parameter.data = flat[offset:offset + numel].view(parameter.shape)
        offset += numel


def _create_worker_gpu_context(rank, device_id, shared_state):
''',
    )
    source = replace(
        source,
        "        for _ in range(2):\n"
        "            template = copy.deepcopy(shared_state.cpu_layers[representative_idx])\n"
        "            _preserve_attn_implementation(template, shared_state._model_config)\n"
        "            template = template.to(device)\n"
        "            for p in template.parameters():\n"
        "                p.requires_grad_(False)\n"
        "            templates.append(template)\n",
        "        for slot in range(2):\n"
        "            template = copy.deepcopy(shared_state.cpu_layers[representative_idx])\n"
        "            _preserve_attn_implementation(template, shared_state._model_config)\n"
        "            template = template.to(device)\n"
        "            for p in template.parameters():\n"
        "                p.requires_grad_(False)\n"
        "            if ctx.alias_layer_slots:\n"
        "                _alias_template_weights(template, ctx.gpu_flat_buffers[slot])\n"
        "            templates.append(template)\n",
        expected=2,
    )
    source = replace(
        source,
        "    ctx.param_sync_event.record(current_stream)\n"
        "    current_stream.synchronize()\n\n"
        "    return ctx\n",
        "    ctx.param_sync_event.record(current_stream)\n"
        "    current_stream.synchronize()\n"
        "    if ctx.alias_layer_slots:\n"
        "        torch.cuda.empty_cache()\n\n"
        "    return ctx\n",
    )
    source = replace(
        source,
        "        for parameter in template.parameters():\n"
        "            parameter.requires_grad_(False)\n"
        "        ctx.gpu_layer_templates[group_id][0] = template\n",
        "        for parameter in template.parameters():\n"
        "            parameter.requires_grad_(False)\n"
        "        if ctx.alias_layer_slots:\n"
        "            _alias_template_weights(template, ctx.gpu_flat_buffers[0])\n"
        "        ctx.gpu_layer_templates[group_id][0] = template\n"
        "    if ctx.alias_layer_slots:\n"
        "        torch.cuda.empty_cache()\n",
    )
    source = replace(
        source,
        "    checkpoints[num_layers] = hidden.detach()\n\n"
        "    if ctx.norm_gpu:\n",
        "    checkpoints[num_layers] = hidden.detach()\n"
        "    del gpu_layer, out\n\n"
        "    if ctx.norm_gpu:\n",
    )
    source = replace(
        source,
        "                hidden = out[0] if isinstance(out, tuple) else out\n"
        "                _bench_finite_probe(rank, f'layer {i}', hidden)\n",
        "                hidden = out[0] if isinstance(out, tuple) else out\n"
        "                _bench_finite_probe(rank, f'layer {i}', hidden)\n"
        "                if ctx.alias_layer_slots:\n"
        "                    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)\n",
    )
    source = replace(
        source,
        "                    hidden_recompute = out[0] if isinstance(out, tuple) else out\n\n"
        "                recompute_cache[j] = hidden_recompute.detach()\n",
        "                    hidden_recompute = out[0] if isinstance(out, tuple) else out\n"
        "                    if ctx.alias_layer_slots:\n"
        "                        ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)\n\n"
        "                recompute_cache[j] = hidden_recompute.detach()\n",
    )
    source = replace(
        source,
        "                hidden = out[0] if isinstance(out, tuple) else out\n\n"
        "        # Norm + lm_head\n",
        "                hidden = out[0] if isinstance(out, tuple) else out\n"
        "                if ctx.alias_layer_slots:\n"
        "                    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)\n\n"
        "        # Norm + lm_head\n",
    )
    source = replace(
        source,
        "    for p in gpu_layer.parameters():\n"
        "        numel = p.numel()\n"
        "        p.data.copy_(flat[offset:offset + numel].view(p.shape))\n"
        "        offset += numel\n",
        "    if not ctx.alias_layer_slots:\n"
        "        for p in gpu_layer.parameters():\n"
        "            numel = p.numel()\n"
        "            p.data.copy_(flat[offset:offset + numel].view(p.shape))\n"
        "            offset += numel\n",
    )
    source = replace(
        source,
        "    ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)\n\n\n"
        "def _worker_collect_layer_grads_async",
        "    if not ctx.alias_layer_slots:\n"
        "        ctx.buffer_free_events[buffer_idx].record(ctx.compute_stream)\n\n\n"
        "def _worker_collect_layer_grads_async",
    )
    source = replace(
        source,
        "        ctx.layer_slab_events[slab_idx].record(ctx.grad_stream)\n"
        "        ctx.template_free_events[buffer_idx].record(ctx.grad_stream)\n",
        "        ctx.layer_slab_events[slab_idx].record(ctx.grad_stream)\n"
        "        ctx.template_free_events[buffer_idx].record(ctx.grad_stream)\n"
        "        if ctx.alias_layer_slots:\n"
        "            ctx.buffer_free_events[buffer_idx].record(ctx.grad_stream)\n",
    )
    source = source.replace("ctx.alias_layer_slots", "ALIAS_LAYER_SLOTS")
    path.write_text(source)
    print(f"installed double-buffered aliased layer slots: {path}")


if __name__ == "__main__":
    main()
