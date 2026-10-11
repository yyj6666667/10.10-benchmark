#!/usr/bin/env python3
"""Keep MegaTrain embedding and output head on GPU only when used."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path("/mnt/hurry-bench/src/megatrain/infinity/model/mp_worker.py")


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
    if "MEGATRAIN_OFFLOAD_EMB_HEAD" in source:
        print(f"already patched: {path}")
        return

    source = replace_once(
        source,
        """    for p_gpu, p_cpu in zip(ctx.emb_gpu.parameters(), shared_state.embedding.parameters()):
        p_gpu.data.copy_(p_cpu.data, non_blocking=True)
""",
        """    if ctx.emb_gpu is not None:
        for p_gpu, p_cpu in zip(ctx.emb_gpu.parameters(), shared_state.embedding.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)
""",
    )
    source = replace_once(
        source,
        """    if not shared_state.tied_lm_head:
        for p_gpu, p_cpu in zip(ctx.lm_head_gpu.parameters(), shared_state.lm_head.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)
""",
        """    if not shared_state.tied_lm_head and ctx.lm_head_gpu is not None:
        for p_gpu, p_cpu in zip(ctx.lm_head_gpu.parameters(), shared_state.lm_head.parameters()):
            p_gpu.data.copy_(p_cpu.data, non_blocking=True)
""",
    )
    source = replace_once(
        source,
        """    B, T = input_ids.shape

    ctx.compute_stream.wait_event(ctx.param_sync_event)

    start = torch.cuda.Event(enable_timing=True)
""",
        """    B, T = input_ids.shape

    offload_emb_head = os.environ.get('MEGATRAIN_OFFLOAD_EMB_HEAD') == '1'
    if offload_emb_head and shared_state.tied_lm_head:
        raise RuntimeError('embedding/head offload requires untied weights')
    ctx.compute_stream.wait_event(ctx.param_sync_event)
    if offload_emb_head:
        ctx.param_sync_event.synchronize()
        # The output head is unused until all transformer layers finish.
        ctx.lm_head_gpu = None

    start = torch.cuda.Event(enable_timing=True)
""",
    )
    source = replace_once(
        source,
        """    start.record(ctx.compute_stream)

    # === FORWARD ===
""",
        """    start.record(ctx.compute_stream)
    if offload_emb_head and ctx.emb_gpu is None:
        ctx.emb_gpu = copy.deepcopy(shared_state.embedding).to(ctx.device)

    # === FORWARD ===
""",
    )
    source = replace_once(
        source,
        """    input_ids_gpu = input_ids.to(ctx.device)
    hidden = ctx.emb_gpu(input_ids_gpu)
    _bench_finite_probe(rank, 'embedding', hidden)
    del input_ids_gpu
""",
        """    input_ids_gpu = input_ids.to(ctx.device)
    if offload_emb_head:
        # The embedding gradient is recomputed after layer backward.
        with torch.no_grad():
            hidden = ctx.emb_gpu(input_ids_gpu)
    else:
        hidden = ctx.emb_gpu(input_ids_gpu)
    _bench_finite_probe(rank, 'embedding', hidden)
    del input_ids_gpu
    if offload_emb_head:
        ctx.emb_gpu = None
""",
    )
    source = replace_once(
        source,
        """    # === LOSS + BACKWARD ===
    labels_gpu = labels.to(ctx.device)
""",
        """    # === LOSS + BACKWARD ===
    if offload_emb_head:
        ctx.lm_head_gpu = copy.deepcopy(shared_state.lm_head).to(ctx.device)
    labels_gpu = labels.to(ctx.device)
""",
    )
    source = replace_once(
        source,
        """        total_loss = total_loss + loss_chunk
        del logits, loss_chunk
""",
        """        total_loss = total_loss + loss_chunk
        del logits, loss_chunk, flat_logits, flat_y, h, y
        if ctx.ce_loss is not None:
            del per_tok, valid
""",
    )
    source = replace_once(
        source,
        """    del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss

    # Backward through layers
""",
        """    del labels_gpu, hidden_after_norm, hidden_before_norm, total_loss
    if offload_emb_head:
        del loss
        ctx.head_slab_event.synchronize()
        ctx.lm_head_gpu = None

    # Backward through layers
""",
    )
    source = replace_once(
        source,
        """    # === BACKWARD THROUGH EMBEDDING ===
    input_ids_gpu = input_ids.to(ctx.device)
""",
        """    # === BACKWARD THROUGH EMBEDDING ===
    if offload_emb_head:
        ctx.emb_gpu = copy.deepcopy(shared_state.embedding).to(ctx.device)
    input_ids_gpu = input_ids.to(ctx.device)
""",
    )
    source = replace_once(
        source,
        """    del input_ids_gpu, emb_out
    del mask, cache_position, position_ids, position_embeddings, grad_hidden
""",
        """    del input_ids_gpu, emb_out
    if offload_emb_head:
        ctx.embed_slab_event.synchronize()
        ctx.emb_gpu = None
    del mask, cache_position, position_ids, position_embeddings, grad_hidden
""",
    )
    source = replace_once(
        source,
        """    ctx.compute_stream.wait_event(ctx.param_sync_event)

    with torch.no_grad():
        # Embedding
""",
        """    ctx.compute_stream.wait_event(ctx.param_sync_event)
    if ctx.emb_gpu is None:
        ctx.emb_gpu = copy.deepcopy(shared_state.embedding).to(ctx.device)
    if ctx.lm_head_gpu is None:
        ctx.lm_head_gpu = copy.deepcopy(shared_state.lm_head).to(ctx.device)

    with torch.no_grad():
        # Embedding
""",
    )
    path.write_text(source)
    print(f"installed optional GPU embedding/head lifecycle: {path}")


if __name__ == "__main__":
    main()
