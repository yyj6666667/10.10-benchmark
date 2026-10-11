#!/usr/bin/env python3
"""Temporarily free one streamed layer slot while the output head runs."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")
MARKER = "MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD"


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one worker patch site, found {count}: {old[:70]!r}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT)
    path = parser.parse_args().path
    source = path.read_text()
    if MARKER in source:
        print(f"already patched: {path}")
        return

    source = replace_once(
        source,
        "def _run_forward_backward(rank, ctx, shared_state, grad_queue, cmd):\n",
        '''def _release_layer_slot_for_head(ctx):
    """The head and streamed layers never compute at the same time."""
    ctx.compute_stream.synchronize()
    ctx.weight_stream.synchronize()
    ctx.gpu_flat_buffers[0] = None
    for templates in ctx.gpu_layer_templates.values():
        templates[0] = None
    torch.cuda.empty_cache()


def _restore_layer_slot_after_head(ctx, shared_state):
    """Recreate the slot before backward recomputation loads any layer."""
    torch.cuda.empty_cache()
    ctx.gpu_flat_buffers[0] = torch.empty(
        shared_state.max_layer_numel,
        dtype=shared_state.config.dtype,
        device=ctx.device,
    )
    for group_id, group in shared_state.layer_groups.items():
        representative = group['indices'][0]
        template = copy.deepcopy(shared_state.cpu_layers[representative])
        _preserve_attn_implementation(template, shared_state._model_config)
        template = template.to(ctx.device)
        for parameter in template.parameters():
            parameter.requires_grad_(False)
        ctx.gpu_layer_templates[group_id][0] = template


def _run_forward_backward(rank, ctx, shared_state, grad_queue, cmd):
''',
    )
    source = replace_once(
        source,
        "    B, T = input_ids.shape\n\n"
        "    offload_emb_head = os.environ.get('MEGATRAIN_OFFLOAD_EMB_HEAD') == '1'\n"
        "    if offload_emb_head and shared_state.tied_lm_head:\n",
        "    B, T = input_ids.shape\n\n"
        "    offload_emb_head = os.environ.get('MEGATRAIN_OFFLOAD_EMB_HEAD') == '1'\n"
        "    release_layer_slot = os.environ.get('MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD') == '1'\n"
        "    if release_layer_slot and not offload_emb_head:\n"
        "        raise RuntimeError('head layer slot release requires embedding/head offload')\n"
        "    if offload_emb_head and shared_state.tied_lm_head:\n",
    )
    source = replace_once(
        source,
        "    # === LOSS + BACKWARD ===\n"
        "    if offload_emb_head:\n"
        "        ctx.lm_head_gpu = copy.deepcopy(shared_state.lm_head).to(ctx.device)\n",
        "    # === LOSS + BACKWARD ===\n"
        "    if offload_emb_head:\n"
        "        if release_layer_slot:\n"
        "            _release_layer_slot_for_head(ctx)\n"
        "        ctx.lm_head_gpu = copy.deepcopy(shared_state.lm_head).to(ctx.device)\n",
    )
    source = replace_once(
        source,
        "    if offload_emb_head:\n"
        "        del loss\n"
        "        ctx.head_slab_event.synchronize()\n"
        "        ctx.lm_head_gpu = None\n"
        "\n"
        "    # Backward through layers\n",
        "    if offload_emb_head:\n"
        "        del loss\n"
        "        ctx.head_slab_event.synchronize()\n"
        "        ctx.lm_head_gpu = None\n"
        "        if release_layer_slot:\n"
        "            _restore_layer_slot_after_head(ctx, shared_state)\n"
        "\n"
        "    # Backward through layers\n",
    )
    path.write_text(source)
    print(f"installed optional head/layer slot residency: {path}")


if __name__ == "__main__":
    main()
