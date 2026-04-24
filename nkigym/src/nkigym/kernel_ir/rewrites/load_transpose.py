"""``LoadTranspose``: fuse ``NKILoad`` + ``NKITranspose`` into a
single ``NKILoad`` with ``attrs["transpose"] = True``.

The fused op reads HBM in (free, partition) order and DMA-transposes
on the fly — codegen dispatches to ``nisa.dma_transpose`` per leaf
(see :func:`nkigym.codegen.gadgets.load_block`).

Match conditions (all must hold):

1. An ``NKILoad`` ``L`` whose sole output buffer ``B_L`` is consumed
   by exactly one op, an ``NKITranspose`` ``T`` via role ``"data"``.
2. ``T.attrs["mode"]`` is ``"dma_transpose"`` (or absent — DMA is
   the default). ``nc_transpose`` cannot be load-fused because it
   reads from SBUF, not HBM.
3. ``L.attrs.get("transpose", False)`` is False — the load is not
   already fused.
4. ``T``'s output is not the kernel return tensor (storing a fused
   load directly to HBM is a different rewrite).

Apply (produces a new IR):

* Drop ops ``L`` and ``T``; splice in a new ``NKILoad`` with inputs
  ``{"data": L.inputs["data"]}``, outputs ``[T.outputs[0]]``,
  ``attrs={"transpose": True}``.
* Drop ``B_L`` from ``physical_buffers`` / ``buffer_scopes`` /
  ``num_buffers`` / ``emission_depth`` — it no longer exists as a
  distinct buffer; the fused load writes directly into ``T``'s
  output buffer.
* Rebuild ``edges`` to skip ``L``/``T``: the fused op inherits
  every inbound edge of ``L`` (on the producer side — there aren't
  any for a Load on a kernel param) and every outbound edge of
  ``T``. Indices in the edges list are re-numbered against the new
  ops list.
"""

from dataclasses import dataclass, replace

from nkigym.kernel_ir.ir import KernelIR, Op


@dataclass(frozen=True)
class LoadTransposeMatch:
    """One ``Load → Transpose`` chain to fold.

    Attributes:
        load_index: Index of the ``NKILoad`` op in ``ir.ops``.
        transpose_index: Index of the ``NKITranspose`` op in ``ir.ops``.
    """

    load_index: int
    transpose_index: int


class LoadTranspose:
    """Collapse adjacent ``NKILoad`` + ``NKITranspose`` into one fused load."""

    name = "load_transpose"

    def match(self, ir: KernelIR) -> list[LoadTransposeMatch]:
        """Enumerate every load→transpose pair that satisfies the fusion predicate."""
        matches: list[LoadTransposeMatch] = []
        for li, load in enumerate(ir.ops):
            if load.kind != "NKILoad":
                continue
            if load.attrs.get("transpose", False):
                continue
            if len(load.outputs) != 1:
                continue
            load_out = load.outputs[0]
            consumers = _consumers_of(ir, load_out)
            if len(consumers) != 1:
                continue
            ti, role = consumers[0]
            if role != "data":
                continue
            transpose = ir.ops[ti]
            if transpose.kind != "NKITranspose":
                continue
            mode = transpose.attrs.get("mode", "dma_transpose")
            if mode != "dma_transpose":
                continue
            if len(transpose.outputs) != 1:
                continue
            if transpose.outputs[0] == ir.return_name:
                continue
            matches.append(LoadTransposeMatch(load_index=li, transpose_index=ti))
        return matches

    def apply(self, ir: KernelIR, instance: LoadTransposeMatch) -> KernelIR:
        """Return a new IR with the load+transpose pair fused."""
        load = ir.ops[instance.load_index]
        transpose = ir.ops[instance.transpose_index]
        if load.kind != "NKILoad" or transpose.kind != "NKITranspose":
            raise ValueError(
                f"LoadTranspose.apply: expected NKILoad+NKITranspose at "
                f"({instance.load_index}, {instance.transpose_index}); got "
                f"{load.kind!r}+{transpose.kind!r}"
            )
        dead_buf = load.outputs[0]
        fused_out = transpose.outputs[0]
        fused_attrs = dict(load.attrs)
        fused_attrs["transpose"] = True
        fused = Op(
            kind="NKILoad",
            inputs=dict(load.inputs),
            outputs=[fused_out],
            kwargs=dict(load.kwargs),
            attrs=fused_attrs,
            axis_map=dict(transpose.axis_map),
            tile_sizes=dict(transpose.tile_sizes),
            blocking_dims=set(transpose.blocking_dims),
        )

        """Build the new op list with the fused op at the load's position,
        and the transpose removed."""
        new_ops: list[Op] = []
        index_map: dict[int, int] = {}
        for i, op in enumerate(ir.ops):
            if i == instance.load_index:
                index_map[i] = len(new_ops)
                new_ops.append(fused)
            elif i == instance.transpose_index:
                """Both the load and transpose fold into the fused op —
                route transpose-index readers to the fused op too."""
                index_map[i] = index_map[instance.load_index]
            else:
                index_map[i] = len(new_ops)
                new_ops.append(op)

        """Rebuild edges: drop every edge into/out-of the transpose's old
        slot that originated on the dead buffer (load → transpose), and
        renumber the rest."""
        new_edges: list[tuple[int, int, str, str]] = []
        for src, dst, tensor, role in ir.edges:
            if tensor == dead_buf and src == instance.load_index and dst == instance.transpose_index:
                continue
            new_src = index_map[src]
            new_dst = index_map[dst]
            if new_src == new_dst:
                continue
            new_edges.append((new_src, new_dst, tensor, role))

        """Drop the dead SBUF buffer and every knob referencing it."""
        new_physical = {n: b for n, b in ir.physical_buffers.items() if n != dead_buf}
        new_scopes = {n: s for n, s in ir.buffer_scopes.items() if n != dead_buf}
        new_num_buffers = {n: nb for n, nb in ir.num_buffers.items() if n != dead_buf}
        new_emission = {n: d for n, d in ir.emission_depth.items() if n != dead_buf}

        return replace(
            ir,
            ops=new_ops,
            edges=new_edges,
            physical_buffers=new_physical,
            buffer_scopes=new_scopes,
            num_buffers=new_num_buffers,
            emission_depth=new_emission,
        )


def _consumers_of(ir: KernelIR, tensor_name: str) -> list[tuple[int, str]]:
    """All ``(op_index, role)`` pairs that read ``tensor_name`` as input."""
    consumers: list[tuple[int, str]] = []
    for i, op in enumerate(ir.ops):
        for role, name in op.inputs.items():
            if name == tensor_name:
                consumers.append((i, role))
    return consumers
