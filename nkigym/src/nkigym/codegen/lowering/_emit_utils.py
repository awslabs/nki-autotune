"""Shared utilities for the lowering pipeline.

Holds the indentation-tracking :class:`_Writer`, the per-emit
:class:`EmitCtx`, and the affine-slice emission helper
:func:`emit_slice`. Lifted into a shared module so
:mod:`nkigym.codegen.lowering.emit_ops` and
:mod:`nkigym.codegen.lowering.emit_source` can depend on these helpers
without forming an import cycle with :mod:`emit_source`.

Slice emission is shape-aware. The :class:`BufferAccess.pattern` carries
one :class:`AccessRange` per **logical** tensor dim (matching
``tensor.dim_ids``), but :func:`place_buffers` promotes SBUF/PSUM tensor
**physical** shapes to 3+ dims: ``(P_tile, num_P_slots, *middle, F_tile
* num_F_tiles)``. The renderer must emit slices matching the physical
shape:

- HBM tensors (2D): all dims emit tile-size-scaled affine starts —
  ``(iv) * tile : (iv) * tile + tile``.
- SBUF/PSUM tensors (3+D): partition axis emits ``0:P_tile``; P-slot and
  any middle slot axes emit the iter-var slot index (no scaling); F-axis
  emits a tile-size-scaled slice.

Fused iter vars: after :class:`Fuse` retires an outer/inner pair and
allocates ``v_fused``, ``module.fused_iter_var_map`` records each retired
id → ``(fused_id, inner_extent, is_outer)``. :func:`_resolve_iv_name`
decomposes retired references into ``(fused // inner_extent)`` for the
outer component and ``(fused % inner_extent)`` for the inner component.
"""

from dataclasses import dataclass

from nkigym.codegen.ir import AccessRange, BufferAccess, KernelModule, Tensor


class _Writer:
    """Line-based writer with indentation tracking."""

    def __init__(self) -> None:
        """Initialize an empty writer at indent depth 0."""
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        """Open a nested block — subsequent ``line`` calls indent one level deeper."""
        self._depth += 1

    def dedent(self) -> None:
        """Close a nested block."""
        self._depth -= 1

    def line(self, text: str = "") -> None:
        """Append a source line at the current indent."""
        self._lines.append(("    " * self._depth + text) if text else "")

    def getvalue(self) -> str:
        """Return the accumulated source with a trailing newline."""
        return "\n".join(self._lines) + "\n"


@dataclass
class EmitCtx:
    """Per-emit context carried through the forest walker.

    Attributes:
        iter_var_to_name: Live mapping from IterVar.var_id to the
            rendered loop variable name. Extended by the walker on
            ForNode entry, restored on exit.
        tensors: Reference to ``module.tensors`` for shape/dtype/location
            lookups.
        module: Reference to the enclosing :class:`KernelModule`. The
            slice-emission helpers consult ``module.fused_iter_var_map``
            to decompose retired outer/inner iter-var references into
            ``(fused // inner_extent)`` / ``(fused % inner_extent)``.
    """

    iter_var_to_name: dict[int, str]
    tensors: dict[str, Tensor]
    module: KernelModule


def _resolve_iv_name(iv_id: int, ctx: EmitCtx) -> str:
    """Return the source name to use for ``iv_id``.

    Resolution order:

    1. Live binding — the iter var is currently bound by an enclosing
       ``ForNode`` (normal case).
    2. Retired via ``Fuse`` — ``module.fused_iter_var_map`` records the
       decomposition. Emits ``(fused_name // inner_extent)`` for the
       outer component and ``(fused_name % inner_extent)`` for the inner
       component. Recursively resolves the fused id so chains of fuses
       resolve correctly.

    Raises:
        KeyError: ``iv_id`` is neither live nor retired.
    """
    if iv_id in ctx.iter_var_to_name:
        return ctx.iter_var_to_name[iv_id]
    entry = ctx.module.fused_iter_var_map.get(iv_id)
    if entry is None:
        raise KeyError(f"iter var {iv_id} is neither live nor recorded in fused_iter_var_map")
    fused_id, inner_extent, is_outer = entry
    fused_name = _resolve_iv_name(fused_id, ctx)
    if is_outer:
        return f"({fused_name} // {inner_extent})"
    return f"({fused_name} % {inner_extent})"


def emit_slice(tensor: Tensor, access: BufferAccess, ctx: EmitCtx) -> str:
    """Emit ``<tensor.name>[per-dim slice expressions]`` for ``access``.

    Shape-aware dispatch: SBUF/PSUM tensors (3+D after ``place_buffers``)
    use the ``[0:P_tile, slot_index, F_slice]`` convention; HBM tensors
    emit a flat 2D slice with tile-size-scaled starts on every dim.

    Args:
        tensor: Fully-lowered :class:`Tensor` (shape set by ``place_buffers``).
        access: BufferAccess with one :class:`AccessRange` per logical
            tensor dim.
        ctx: :class:`EmitCtx` carrying the live iter-var naming and the
            enclosing :class:`KernelModule` (used to decompose retired
            iter-var references via :attr:`KernelModule.fused_iter_var_map`).

    Returns:
        Source fragment ``"name[slice_0, slice_1, ...]"``.
    """
    if tensor.location in {"sbuf", "psum"}:
        return _emit_slice_sbuf_psum(tensor, access, ctx)
    return _emit_slice_hbm(tensor, access, ctx)


def _emit_slice_hbm(tensor: Tensor, access: BufferAccess, ctx: EmitCtx) -> str:
    """Flat HBM slice: every dim emits a tile-size-scaled affine range.

    HBM tensors keep their declared logical shape, so the number of
    slice dims matches ``len(access.pattern)``. Each dim's iter var is
    a tile index; the byte/element start = ``iv * tile_size``.
    """
    slice_parts: list[str] = []
    for ar in access.pattern:
        start = _emit_scaled_affine_start(ar, ctx, ar.extent)
        slice_parts.append(f"{start} : {start} + {ar.extent}")
    return f"{tensor.name}[{', '.join(slice_parts)}]"


def _emit_slice_sbuf_psum(tensor: Tensor, access: BufferAccess, ctx: EmitCtx) -> str:
    """3+D SBUF/PSUM slice: ``[0:P_tile, slot_index(s), F_slice]``.

    Physical shape is ``(P_tile, num_P_slots, *middle_slots, F_tile *
    num_F_tiles)`` per :func:`place_buffers`. Conventions:

    - Dim 0 (partition): always full — ``0:P_tile``.
    - Dim 1 (P-slot): integer slot index from the P-dim iter var(s).
      No tile-size scaling (this is an index into a slot array).
    - Middle dims (N-D > 2): integer slot indices — same rule as P-slot.
    - Last dim (F): tile-size-scaled slice ``(iv) * F_tile : (iv) * F_tile
      + F_tile``.

    1D logical tensors (e.g. ``reduce_res``, ``operand0``) have physical
    shape ``(P_tile, num_P_slots, 1)`` per :func:`place_buffers`. The
    access carries only one P-dim range; this helper emits the 3D slice
    ``[0:P_tile, p_slot, 0:1]`` so the resulting tile matches NKI's
    ``(P, 1)`` contract for ``nisa.activation_reduce.reduce_res`` and
    ``nisa.tensor_scalar.operand0``.
    """
    p_tile = tensor.shape[0]
    partition_slice = f"0:{p_tile}"

    p_ar = access.pattern[0]
    p_slot = _emit_index_expr(p_ar, ctx)

    if len(tensor.dim_ids) == 1:
        return f"{tensor.name}[{partition_slice}, {p_slot}, 0:1]"

    middle_slots: list[str] = []
    for ar in access.pattern[1:-1]:
        middle_slots.append(_emit_index_expr(ar, ctx))

    f_ar = access.pattern[-1]
    f_start = _emit_scaled_affine_start(f_ar, ctx, f_ar.extent)
    f_slice = f"{f_start} : {f_start} + {f_ar.extent}"

    parts = [partition_slice, p_slot, *middle_slots, f_slice]
    return f"{tensor.name}[{', '.join(parts)}]"


def _emit_index_expr(ar: AccessRange, ctx: EmitCtx) -> str:
    """Emit a bare integer index expression (no extent scaling, no slice).

    Used for SBUF/PSUM slot axes. For a simple 1:1 access (one iter var,
    coeff 1, offset 0) the result is just the iter var name.

    When the access has no iter-var coefficients and ``extent > 1``, the
    block-level access spans the full slot dimension — emit an explicit
    slice ``0:extent``. This lets a close-reduce block (e.g. the RFactor
    closing :class:`NKITensorReduce`) express a full-range read along a
    middle slot dim whose iter var is not bound at the consumer scope.
    """
    terms: list[str] = []
    for iv_id, coeff in ar.iter_var_coeffs:
        name = _resolve_iv_name(iv_id, ctx)
        if coeff == 1:
            terms.append(name)
        else:
            terms.append(f"{name} * {coeff}")
    if ar.const_offset != 0:
        terms.append(str(ar.const_offset))
    if not terms:
        return f"0:{ar.extent}" if ar.extent > 1 else "0"
    return " + ".join(terms)


def _emit_scaled_affine_start(ar: AccessRange, ctx: EmitCtx, scale: int) -> str:
    """Emit a tile-size-scaled affine start expression.

    Each iter var's coefficient is multiplied by ``scale`` (the tile
    size along the dim). Parenthesisation matches v1 output:
    coeff-1 single-iter-var emits as ``(name) * scale``; coeff != 1
    emits as ``(name * coeff) * scale`` (via grouped product expressed
    as the flat ``name * total_coeff`` form).
    """
    terms: list[str] = []
    for iv_id, coeff in ar.iter_var_coeffs:
        name = _resolve_iv_name(iv_id, ctx)
        total_coeff = coeff * scale
        if coeff == 1:
            terms.append(f"({name}) * {total_coeff}")
        else:
            terms.append(f"({name} * {coeff}) * {scale}")
    if ar.const_offset != 0:
        terms.append(str(ar.const_offset))
    if not terms:
        return "0"
    return " + ".join(terms)
