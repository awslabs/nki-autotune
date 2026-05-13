"""Shared utilities for the lowering pipeline.

Holds the indentation-tracking :class:`_Writer`, the per-emit
:class:`EmitCtx`, and the affine-slice emission helper
:func:`emit_slice`. Lifted into a shared module so
:mod:`nkigym.codegen.emit_ops` and
:mod:`nkigym.codegen.emit_source` can depend on these helpers
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

Fused iter vars: :class:`Fuse` eagerly rewrites affected
:class:`BufferAccess` patterns when it retires an outer/inner pair
(outer dropped from coeffs — extent-1 contributes 0; inner renamed to
the fused var_id). No side-table is consulted at emission.
"""

from dataclasses import dataclass, field

from nkigym.ir.ir import AccessRange, BufferAccess, KernelIR, Tensor


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
        module: Reference to the enclosing :class:`KernelIR`. Used
            for tensor/axis lookups during slice emission.
        innermost_tile_ids: Set of :class:`IterVar` ``var_id``\\ s that
            are the innermost tile loop for their respective axis at a
            descendant :class:`SBlock`. When the walker enters a ForNode
            whose iter-var is in this set, it elides the ``for`` header
            and binds the iter-var's name to ``"0"`` (the semantically
            correct symbolic value). The iter-var's ``extent`` is
            consumed as the slice width on the ISA call's buffer-access
            pattern. :func:`_emit_affine_start` and
            :func:`_emit_index_expr` detect elision by membership in
            this set (not by string-matching the bound name) so the
            ``"0"`` binding remains a robust symbolic value rather than
            a sentinel.
    """

    iter_var_to_name: dict[int, str]
    tensors: dict[str, Tensor]
    module: KernelIR
    innermost_tile_ids: set[int] = field(default_factory=set)


def _resolve_iv_name(iv_id: int, ctx: EmitCtx) -> str:
    """Return the source name to use for ``iv_id``.

    The iter-var must be bound by an enclosing ForNode at render time;
    otherwise the access pattern references a ghost id and the module is
    ill-formed.

    Raises:
        KeyError: ``iv_id`` is not bound by any enclosing ForNode.
    """
    if iv_id not in ctx.iter_var_to_name:
        raise KeyError(f"iter var {iv_id} has no live binding")
    return ctx.iter_var_to_name[iv_id]


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
            enclosing :class:`KernelIR` for tensor/axis lookups.

    Returns:
        Source fragment ``"name[slice_0, slice_1, ...]"``.
    """
    if tensor.location in {"sbuf", "psum"}:
        return _emit_slice_sbuf_psum(tensor, access, ctx)
    return _emit_slice_hbm(tensor, access, ctx)


def _emit_slice_hbm(tensor: Tensor, access: BufferAccess, ctx: EmitCtx) -> str:
    """Flat HBM slice: every dim emits the physical affine range.

    HBM tensors keep their declared logical shape, so the number of
    slice dims matches ``len(access.pattern)``. Each :class:`AccessRange`
    carries per-iter-var coefficients already in physical-address units
    (Task 4: outer coeff = tile, inner coeff = 1), so the affine start is
    just ``sum(coeff_i * iv_i) + const_offset``. The slice width is
    ``ar.extent`` (the innermost tile width; the inner iter-var is
    elided to ``0`` by :func:`_emit_node`, so it contributes nothing to
    the start).
    """
    slice_parts: list[str] = []
    for ar in access.pattern:
        start = _emit_affine_start(ar, ctx)
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
    f_start = _emit_affine_start(f_ar, ctx)
    f_slice = f"{f_start} : {f_start} + {f_ar.extent}"

    parts = [partition_slice, p_slot, *middle_slots, f_slice]
    return f"{tensor.name}[{', '.join(parts)}]"


def _emit_index_expr(ar: AccessRange, ctx: EmitCtx) -> str:
    """Emit a bare integer slot-index expression (no extent scaling, no slice).

    Used for SBUF/PSUM slot axes (dim 1 and middle dims), where the
    physical index into the buffer is the *slot number*, not the
    physical byte offset. Under Task 4's IR the :class:`AccessRange`
    along a slot axis carries coefficients in physical-address units
    (``outer_coeff = tile``, ``inner_coeff = 1``), so to recover the
    slot index we divide each iter-var coefficient by ``ar.extent``
    (the tile width). The outer iter-var's coefficient divides cleanly
    to ``1``, yielding just the iter-var name; the inner iter-var is
    elided (its ``var_id`` is recorded in :attr:`EmitCtx.innermost_tile_ids`)
    and contributes nothing to the slot index.

    When the access has no iter-var coefficients and ``extent > 1``, the
    block-level access spans the full slot dimension — emit an explicit
    slice ``0:extent``. This lets a close-reduce block (e.g. the RFactor
    closing :class:`NKITensorReduce`) express a full-range read along a
    middle slot dim whose iter var is not bound at the consumer scope.
    """
    terms: list[str] = []
    tile = ar.extent
    for iv_id, coeff in ar.iter_var_coeffs:
        if iv_id in ctx.innermost_tile_ids:
            continue
        name = _resolve_iv_name(iv_id, ctx)
        slot_coeff = coeff // tile if tile > 0 else coeff
        if slot_coeff == 0:
            continue
        if slot_coeff == 1:
            terms.append(name)
        else:
            terms.append(f"{name} * {slot_coeff}")
    if ar.const_offset != 0:
        terms.append(str(ar.const_offset))
    if not terms:
        return f"0:{ar.extent}" if ar.extent > 1 else "0"
    return " + ".join(terms)


def _emit_affine_start(ar: AccessRange, ctx: EmitCtx) -> str:
    """Emit the physical affine start expression for a slice.

    Each iter-var coefficient is already in physical-address units
    (Task 4: outer coeff = tile, inner coeff = 1), so the expression
    is simply ``sum(coeff_i * iv_i) + const_offset``. Coeff-1 terms
    emit bare (``name``); other coeff values emit as ``(name) * coeff``
    to match the renderer's v1 parenthesisation. When the renderer has
    elided the inner tile loop (the iter-var's ``var_id`` is recorded
    in :attr:`EmitCtx.innermost_tile_ids`) the term is dropped as a
    statically-zero contribution, so the emitted start is just the
    outer component.
    """
    terms: list[str] = []
    for iv_id, coeff in ar.iter_var_coeffs:
        if iv_id in ctx.innermost_tile_ids:
            continue
        name = _resolve_iv_name(iv_id, ctx)
        if coeff == 1:
            terms.append(name)
        else:
            terms.append(f"({name}) * {coeff}")
    if ar.const_offset != 0:
        terms.append(str(ar.const_offset))
    if not terms:
        return "0"
    return " + ".join(terms)
