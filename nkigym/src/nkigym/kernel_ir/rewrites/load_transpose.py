"""``LoadTranspose`` rewrite — collapse ``NKILoad → transpose`` into one DMA transpose.

Pattern (either ``NKITranspose`` or ``NKIDMATranspose`` qualifies —
both describe the same math; the fused form subsumes either gadget)::

    [i] NKILoad(data=param)               → sbuf_param    (p=dP, f=dF)
    [j] NKI[DMA]Transpose(data=sbuf_param) → sbuf_param_T (p=dF, f=dP)

Rewrite::

    [i] NKIDMATranspose(data=param) → sbuf_param_T

The fused op reads directly from the HBM parameter and writes the
transposed result into ``sbuf_param_T``. The renderer lowers it to
``dma_transpose_block(cur_sbuf_param_T, param[...])`` — exactly one
HBM→SBUF DMA transpose, same perf as the previous
``load_block(transpose=True)`` path but with no attribute side-channel
or dedicated load-transpose codepath.

Two-stage interface (see :class:`GraphRewrite`):

* ``analyze(ir)`` returns a list of :class:`LoadTransposeMatch` — one
  per fusable ``(load, transpose)`` pair, in IR order. Pure inspection.
* ``apply(ir, matches)`` rewrites exactly the pairs in ``matches``
  (the caller can prune the list). Returns a new :class:`KernelIR`.

Guardrails (enforced during ``analyze``):

* Only fires when the transpose's input sbuf is produced by a single
  load **and** has no other consumers — otherwise removing the
  intermediate ``sbuf_param`` would break some other op.
* Preserves every other IR knob: scope / num_buffers / emission_depth
  entries keyed under the original ``sbuf_param`` are re-keyed under
  the fused destination ``sbuf_param_T``.
"""

from dataclasses import dataclass, replace

from nkigym.kernel_ir.ir import KernelIR, Op
from nkigym.kernel_ir.rewrites.base import GraphRewrite

_TRANSPOSE_KINDS = frozenset({"NKITranspose", "NKIDMATranspose"})


@dataclass(frozen=True)
class LoadTransposeMatch:
    """One fusable ``(NKILoad, transpose)`` site.

    Attributes:
        load_idx: Index of the ``NKILoad`` op in ``ir.ops``.
        transpose_idx: Index of the transpose op in ``ir.ops``.
        hbm_param: HBM tensor the load reads from.
        old_sbuf: Intermediate SBUF buffer name to be retired.
        new_sbuf: Transpose output SBUF buffer name — fused destination.
    """

    load_idx: int
    transpose_idx: int
    hbm_param: str
    old_sbuf: str
    new_sbuf: str


class LoadTranspose(GraphRewrite[LoadTransposeMatch]):
    """Rewrite: replace ``(NKILoad, transpose)`` pairs with a single ``NKIDMATranspose``."""

    def analyze(self, ir: KernelIR) -> list[LoadTransposeMatch]:
        """Return every fusable ``(load, transpose)`` pair in ``ir`` without mutating it."""
        matches: list[LoadTransposeMatch] = []
        used_sbufs: set[str] = set()
        for j, op in enumerate(ir.ops):
            if op.kind not in _TRANSPOSE_KINDS:
                continue
            src_buffer = op.inputs.get("data")
            if src_buffer is None or src_buffer in used_sbufs:
                continue
            producer_idx = _sole_producer(ir.ops, src_buffer)
            if producer_idx is None or ir.ops[producer_idx].kind != "NKILoad":
                continue
            if _has_other_consumers(ir.ops, src_buffer, transpose_idx=j):
                continue
            load_op = ir.ops[producer_idx]
            matches.append(
                LoadTransposeMatch(
                    load_idx=producer_idx,
                    transpose_idx=j,
                    hbm_param=load_op.inputs["data"],
                    old_sbuf=src_buffer,
                    new_sbuf=op.outputs[0],
                )
            )
            used_sbufs.add(src_buffer)
        return matches

    def apply(self, ir: KernelIR, matches: list[LoadTransposeMatch]) -> KernelIR:
        """Fuse exactly the pairs in ``matches`` and return a new IR."""
        if not matches:
            return ir

        ops = list(ir.ops)
        physical_buffers = dict(ir.physical_buffers)
        buffer_scopes = dict(ir.buffer_scopes)
        num_buffers = dict(ir.num_buffers)
        emission_depth = dict(ir.emission_depth)

        transpose_indices_to_drop: list[int] = []
        for match in matches:
            load_op = ops[match.load_idx]
            transpose_op = ops[match.transpose_idx]
            ops[match.load_idx] = replace(
                load_op,
                kind="NKIDMATranspose",
                inputs={"data": match.hbm_param},
                outputs=[match.new_sbuf],
                axis_map=dict(transpose_op.axis_map),
                blocking_dims=set(transpose_op.blocking_dims),
                attrs=dict(transpose_op.attrs),
            )
            transpose_indices_to_drop.append(match.transpose_idx)

            physical_buffers.pop(match.old_sbuf, None)
            _migrate_knob(buffer_scopes, match.old_sbuf, match.new_sbuf)
            _migrate_knob(num_buffers, match.old_sbuf, match.new_sbuf)
            _migrate_knob(emission_depth, match.old_sbuf, match.new_sbuf)

        """Drop the retired transpose ops in descending order so earlier
        indices remain valid during removal."""
        for idx in sorted(transpose_indices_to_drop, reverse=True):
            ops.pop(idx)

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
