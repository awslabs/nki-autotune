"""``LoadTranspose`` rewrite — fuse ``NKITranspose`` into its feeding ``NKILoad``.

Pattern::

    [i] NKILoad(data=param)  → sbuf_param         (p=dP, f=dF)
    [j] NKITranspose(data=sbuf_param) → sbuf_param_T  (p=dF, f=dP)
    ... (any consumer uses sbuf_param_T, not sbuf_param)

Rewrite::

    [i] NKILoad(data=param, attrs={"transpose": True})  → sbuf_param_T

The ``transpose=True`` flag tells the renderer to emit
``load_block(..., transpose=True)``, which dispatches to
``nisa.dma_transpose`` and performs the HBM→SBUF transposition in one
DMA instead of a round-trip through a separate Tensor-Engine transpose.

Guardrails:

* Only fires when the transpose's input sbuf is produced by a single
  load **and** has no other consumers — otherwise removing the
  intermediate ``sbuf_param`` would break some other op.
* Preserves every other IR knob: scope / num_buffers / emission_depth
  entries keyed under the original ``sbuf_param`` are re-keyed under
  the fused destination ``sbuf_param_T``.
"""

from dataclasses import replace

from nkigym.kernel_ir.ir import KernelIR, Op


class LoadTranspose:
    """Rewrite: fuse each ``NKITranspose`` into its sole-producer ``NKILoad``."""

    def __call__(self, ir: KernelIR) -> KernelIR:
        """Apply the fuse-transpose-into-load rewrite to ``ir``.

        Returns a new ``KernelIR`` with the fusion applied to every
        matching ``(NKILoad, NKITranspose)`` pair. Non-matching ops
        pass through unchanged.
        """
        ops = list(ir.ops)
        physical_buffers = dict(ir.physical_buffers)
        buffer_scopes = dict(ir.buffer_scopes)
        num_buffers = dict(ir.num_buffers)
        emission_depth = dict(ir.emission_depth)

        while True:
            match = _find_load_transpose_pair(ops)
            if match is None:
                break
            load_idx, transpose_idx = match
            load_op = ops[load_idx]
            transpose_op = ops[transpose_idx]
            old_sbuf = load_op.outputs[0]
            new_sbuf = transpose_op.outputs[0]

            """Update the load: dst becomes the transpose's output buffer,
            and set the transpose flag so the renderer fires dma_transpose."""
            new_attrs = dict(load_op.attrs)
            new_attrs["transpose"] = True
            ops[load_idx] = replace(load_op, outputs=[new_sbuf], attrs=new_attrs)
            """Drop the NKITranspose op entirely."""
            ops.pop(transpose_idx)

            """Retire the now-unused intermediate buffer and re-key any
            knob entries onto the fused destination buffer."""
            physical_buffers.pop(old_sbuf, None)
            _migrate_knob(buffer_scopes, old_sbuf, new_sbuf)
            _migrate_knob(num_buffers, old_sbuf, new_sbuf)
            _migrate_knob(emission_depth, old_sbuf, new_sbuf)

        edges = _derive_edges(ops)

        return KernelIR(
            func_name=ir.func_name,
            param_names=ir.param_names,
            return_name=ir.return_name,
            dimensions=ir.dimensions,
            logical_tensors=ir.logical_tensors,
            physical_buffers=physical_buffers,
            ops=ops,
            edges=edges,
            dim_order=ir.dim_order,
            ltiles_per_block=ir.ltiles_per_block,
            buffer_scopes=buffer_scopes,
            num_buffers=num_buffers,
            emission_depth=emission_depth,
        )


def _find_load_transpose_pair(ops: list[Op]) -> tuple[int, int] | None:
    """Return ``(load_idx, transpose_idx)`` for the first fusable pair, or ``None``.

    Fusable means: a ``NKITranspose`` whose input sbuf is the sole
    output of a ``NKILoad`` op, and that sbuf has no other consumers
    anywhere in ``ops``.
    """
    for j, op in enumerate(ops):
        if op.kind != "NKITranspose":
            continue
        src_buffer = op.inputs.get("data")
        if src_buffer is None:
            continue
        producer_idx = _sole_producer(ops, src_buffer)
        if producer_idx is None or ops[producer_idx].kind != "NKILoad":
            continue
        if _has_other_consumers(ops, src_buffer, transpose_idx=j):
            continue
        return producer_idx, j
    return None


def _sole_producer(ops: list[Op], tensor: str) -> int | None:
    """Return the index of the op that outputs ``tensor`` if exactly one op does."""
    indices = [i for i, op in enumerate(ops) if tensor in op.outputs]
    if len(indices) != 1:
        return None
    return indices[0]


def _has_other_consumers(ops: list[Op], tensor: str, transpose_idx: int) -> bool:
    """True iff any op other than ``ops[transpose_idx]`` reads ``tensor``."""
    for i, op in enumerate(ops):
        if i == transpose_idx:
            continue
        if tensor in op.inputs.values():
            return True
    return False


def _migrate_knob(knob: dict, old_key: str, new_key: str) -> None:
    """Move ``knob[old_key]`` to ``knob[new_key]``; skip when old is absent."""
    if old_key in knob:
        value = knob.pop(old_key)
        """Only install the migrated value when there isn't already an
        explicit entry for the fused destination buffer."""
        knob.setdefault(new_key, value)


def _derive_edges(ops: list[Op]) -> list[tuple[int, int]]:
    """Return ``[(producer_idx, consumer_idx), ...]`` for the new ops list."""
    producer_of: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        for tname in op.inputs.values():
            src = producer_of.get(tname)
            if src is not None and src != i:
                edges.append((src, i))
        for out in op.outputs:
            producer_of[out] = i
    return edges
