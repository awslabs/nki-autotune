"""Iter-var-identity IR for the nkigym tune stage.

TVM-style dataclasses (IterVar, ForNode, SBlock, NKIOpCall, BufferAccess,
AccessRange) replace the earlier path-based LoopNode/BodyLeaf IR. See
``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md``.

Roles:

- :class:`KernelModule` - envelope. Holds signature + tensor/dim declarations +
  tree body + dep cache.
- :class:`TreeIR` / :class:`ForNode` / :class:`SBlock` - schedule tree with
  TVM-style iter-var identity.
- :class:`DepCache` - per-scope dependency cache (defined in
  :mod:`nkigym.codegen.dep_cache`).
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from nkigym.ops.base import AxisRole

"""``DepCache`` import deferred until after ``ForNode``/``SBlock`` are
defined — ``dep_cache`` imports those names at module top. The deferred
import executes before ``KernelModule`` is defined so the annotation +
default factory resolve normally."""

TensorOrigin = Literal["param", "intermediate", "return"]


@dataclass
class Tensor:
    """Named tensor appearing in the kernel body.

    Attributes:
        name: Source-level variable name.
        dim_ids: Concrete dim ids in operand order.
        shape: Element sizes aligned with ``dim_ids``.
        dtype: Element dtype (e.g. ``"bfloat16"``).
        origin: ``"param"`` (HBM input), ``"intermediate"`` (SBUF handoff),
            ``"return"`` (final output).
        location: ``"hbm"`` / ``"sbuf"`` / ``"psum"`` — which memory
            the allocation targets. For params, always ``"hbm"``.
        buffer_degree: Multi-buffer degree per dim; defaults to 1.
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin
    location: Literal["hbm", "sbuf", "psum"] = "sbuf"
    buffer_degree: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for d in self.dim_ids:
            self.buffer_degree.setdefault(d, 1)


@dataclass
class Axis:
    """Logical axis — opaque integer identity plus display name.

    Identity is the ``axis_id`` int; two IterVars are on the same logical
    axis iff their ``axis_id``s are equal. The ``name`` is a cosmetic
    property consumed by the renderer to spell ``i_<name>_<ordinal>``;
    changing it never affects logic.

    For axes created by ``Fuse`` across distinct axes, ``source_axes``
    records the component axis_ids in fuse order (outer-first). Original
    axes have ``source_axes = None``.

    Attributes:
        axis_id: Monotonic unique id per module (allocated via
            :meth:`KernelModule.allocate_axis`).
        name: Display name, e.g. ``"d0"``, ``"row"``. Purely cosmetic.
        total_size: Axis extent in elements.
        source_axes: Tuple of component axis_ids if this axis was created
            by cross-axis Fuse; ``None`` for original axes.
    """

    axis_id: int
    name: str
    total_size: int
    source_axes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class IterVar:
    """Stable identity for a loop iteration variable.

    Created by the canonical builder and every Split / Fuse. Never
    mutated — atoms retire iter vars and emit fresh ones.

    Attributes:
        var_id: Monotonic unique id per module.
        axis_id: Logical axis this iter var traverses (integer, stable).
        extent: Trip count (# tiles).
        role: PARALLEL / SEQUENTIAL / ACCUMULATION.
    """

    var_id: int
    axis_id: int
    extent: int
    role: AxisRole


@dataclass(frozen=True)
class AccessRange:
    """Affine access for one buffer dim: sum(coeff * iv) + const_offset.

    ``iter_var_coeffs`` stored as a sorted tuple of (id, coeff) pairs for
    hashability. Construct from a dict via :meth:`make`; read via the
    :attr:`coeffs` property.

    Attributes:
        iter_var_coeffs: Sorted tuple of ``(iter_var_id, coefficient)`` pairs.
        const_offset: Constant offset added to the affine form.
        extent: Per-iteration extent (the tile size along this dim).
    """

    iter_var_coeffs: tuple[tuple[int, int], ...]
    const_offset: int
    extent: int

    @classmethod
    def make(cls, coeffs: dict[int, int], const_offset: int, extent: int) -> "AccessRange":
        """Construct from a coefficient dict; normalizes ordering."""
        return cls(iter_var_coeffs=tuple(sorted(coeffs.items())), const_offset=const_offset, extent=extent)

    @property
    def coeffs(self) -> dict[int, int]:
        """Return coefficients as a dict."""
        return dict(self.iter_var_coeffs)


@dataclass(frozen=True)
class BufferAccess:
    """Which region of a tensor a block reads or writes.

    TVM BufferRegion analog. The renderer consumes ``pattern`` to emit
    slice expressions; ``cache_read`` / ``cache_write`` atoms (future)
    consume it to infer staging buffer shapes.

    Attributes:
        tensor_name: Name of the tensor in ``module.tensors``.
        iter_var_ids: Tuple of iter_var ids that index this buffer.
        pattern: One AccessRange per tensor dim, in tensor-order.
    """

    tensor_name: str
    iter_var_ids: tuple[int, ...]
    pattern: tuple[AccessRange, ...]


@dataclass(frozen=True)
class NKIOpCall:
    """One ISA call inside an SBlock.body.

    Attributes:
        op_cls: The NKIOp subclass (e.g. ``NKIMatmul``).
        kwargs: Op-level kwargs (e.g. ``value=0.0`` for memset, ``op="square"``
            for activation_reduce).
        axis_map: Abstract axis → axis_id.
        dim_role: axis_id → AxisRole. Op-local; same axis can have
            different roles across ops in the module.
    """

    op_cls: type
    kwargs: dict[str, Any]
    axis_map: dict[str, int]
    dim_role: dict[int, AxisRole]


@dataclass
class SBlock:
    """Atomic (or fused) compute block.

    Multi-leaf blocks are supported in the data model; canonical builder
    always emits single-leaf. Fusion atoms (future) produce len > 1.

    Attributes:
        iter_vars: Block-local iter vars, canonical order (output-axis
            dims first, then reduction dims).
        reads: slot_name → BufferAccess (read-only operands).
        writes: slot_name → BufferAccess (write-only operands).
        reads_writes: slot_name → BufferAccess (RMW operands).
        body: Ordered list of NKIOpCalls.
        annotations: Keyed annotations consumed by lowering passes
            (e.g. ``buffer_degree`` on alloc SBlocks).
    """

    iter_vars: list[IterVar]
    reads: dict[str, BufferAccess]
    writes: dict[str, BufferAccess]
    reads_writes: dict[str, BufferAccess]
    body: list[NKIOpCall]
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForNode:
    """A for-loop in the schedule tree. Binds one IterVar by reference.

    ``annotations`` holds keyed annotations (e.g.
    ``software_pipeline_depth``) consumed by lowering passes. Replaces
    v1's scalar ``pipeline_depth`` field.

    Attributes:
        iter_var: The IterVar bound to this loop.
        children: Nested ForNodes and SBlocks.
        name: Canonical rendered name ``i_<dim>_<ordinal>``; assigned by
            the canonicalize pass (``None`` until then).
        annotations: Keyed annotations consumed by lowering passes.
    """

    iter_var: IterVar
    children: "list[ForNode | SBlock]"
    name: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


"""TreeIR is the top-level forest type — list of root ForNodes/SBlocks."""
TreeIR = list["ForNode | SBlock"]


"""Deferred import: ``dep_cache`` imports ``ForNode``/``SBlock`` from this
module at its own top; placing this line after those classes are defined
breaks the cycle. Executed before ``KernelModule`` so the annotation +
default factory resolve normally."""
from nkigym.codegen.dep_cache import DepCache  # noqa: E402


@dataclass
class KernelModule:
    """Envelope IR — signature + tensor/axis declarations + schedule tree.

    Analog of TVM's PrimFunc + buffer_map.

    Attributes:
        func_name: Emitted kernel name.
        param_names: Signature order.
        return_name: Tensor name of the return.
        tensors: All named tensors, keyed by name.
        axes: All logical axes, keyed by axis_id.
        iter_var_counter: Monotonic counter for allocating IterVar.var_id.
        axis_counter: Monotonic counter for allocating Axis.axis_id.
        body: Schedule tree — list of ForNode / SBlock roots.
        dep: Per-scope dependency cache (lazy).
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    axes: dict[int, Axis]
    iter_var_counter: int = 0
    axis_counter: int = 0
    body: TreeIR = field(default_factory=list)
    dep: DepCache = field(default_factory=lambda: DepCache(scopes={}))

    def allocate_axis(self, name: str, total_size: int, source_axes: tuple[int, ...] | None = None) -> Axis:
        """Allocate a fresh :class:`Axis` with a monotonic unique ``axis_id``.

        Args:
            name: Display name for the axis (cosmetic only).
            total_size: Axis extent in elements.
            source_axes: Tuple of component axis_ids if this axis is
                the result of a cross-axis fuse; ``None`` for original axes.

        Returns:
            The freshly-allocated :class:`Axis`.
        """
        axis = Axis(axis_id=self.axis_counter, name=name, total_size=total_size, source_axes=source_axes)
        self.axes[axis.axis_id] = axis
        self.axis_counter += 1
        return axis

    def allocate_iter_var(self, axis_id: int, extent: int, role: AxisRole) -> IterVar:
        """Allocate a fresh IterVar with a monotonic unique var_id.

        Never reuses retired var_ids.

        Args:
            axis_id: The logical axis this iter var traverses.
            extent: The trip count (# tiles).
            role: PARALLEL / SEQUENTIAL / ACCUMULATION.

        Returns:
            A freshly-allocated IterVar.
        """
        iv = IterVar(var_id=self.iter_var_counter, axis_id=axis_id, extent=extent, role=role)
        self.iter_var_counter += 1
        return iv

    def axis_id_by_name(self, name: str) -> int:
        """Return the axis_id for the axis with the given display name.

        Raises:
            KeyError: No axis has this name, or multiple axes share it.
        """
        matches = [a.axis_id for a in self.axes.values() if a.name == name]
        if not matches:
            raise KeyError(f"no axis with name {name!r}")
        if len(matches) > 1:
            raise KeyError(f"multiple axes with name {name!r}: {matches}")
        return matches[0]

    def pprint(self) -> str:
        """Human-readable multi-line repr of the whole module.

        Sections: signature, axes, tensors, schedule tree. The tree is
        indented; each ForNode prints as ``for i_<axis.name>_<var_id> in
        range(extent) [role]``; each SBlock prints as ``SBlock[op_cls]``
        followed by indented operand access lines.
        """
        lines: list[str] = []

        """Signature."""
        lines.append(f"KernelModule {self.func_name}({', '.join(self.param_names)}) -> {self.return_name}")

        """Axes."""
        lines.append("axes:")
        for axis in sorted(self.axes.values(), key=lambda a: a.axis_id):
            src = f" <- {axis.source_axes}" if axis.source_axes is not None else ""
            lines.append(f"  axis_id={axis.axis_id} name={axis.name!r} total_size={axis.total_size}{src}")

        """Tensors."""
        lines.append("tensors:")
        for t in self.tensors.values():
            degree = dict(t.buffer_degree) if any(v != 1 for v in t.buffer_degree.values()) else None
            degree_suffix = f" buffer_degree={degree}" if degree else ""
            lines.append(
                f"  {t.name}: {t.location} {t.dtype} shape={t.shape} dim_ids={t.dim_ids} origin={t.origin}{degree_suffix}"
            )

        """Body."""
        lines.append("body:")
        for root in self.body:
            _pprint_node(root, self, indent=1, lines=lines)

        return "\n".join(lines)


def _pprint_node(node: "ForNode | SBlock", module: "KernelModule", indent: int, lines: list[str]) -> None:
    """Recursively render one tree node into ``lines`` with ``indent`` level."""
    prefix = "  " * indent
    if isinstance(node, ForNode):
        iv = node.iter_var
        axis_name = module.axes[iv.axis_id].name if iv.axis_id in module.axes else f"axis{iv.axis_id}"
        lines.append(f"{prefix}for i_{axis_name}_{iv.var_id} in range({iv.extent}) [{iv.role.name}]")
        for c in node.children:
            _pprint_node(c, module, indent + 1, lines)
    else:
        op_names = ",".join(call.op_cls.__name__ for call in node.body) if node.body else "empty"
        iv_summary = ",".join(
            f"{module.axes[iv.axis_id].name if iv.axis_id in module.axes else '?'}:{iv.var_id}(ext={iv.extent})"
            for iv in node.iter_vars
        )
        lines.append(f"{prefix}SBlock[{op_names}] iter_vars=[{iv_summary}]")
        for slot, access in node.reads.items():
            lines.append(f"{prefix}  read {slot}: {_fmt_access(access)}")
        for slot, access in node.writes.items():
            lines.append(f"{prefix}  write {slot}: {_fmt_access(access)}")
        for slot, access in node.reads_writes.items():
            lines.append(f"{prefix}  rmw {slot}: {_fmt_access(access)}")


def _fmt_access(access: "BufferAccess") -> str:
    """One-line repr of a :class:`BufferAccess`."""
    pattern_parts = []
    for ar in access.pattern:
        terms = []
        for iv_id, coeff in ar.iter_var_coeffs:
            terms.append(f"{coeff}*iv{iv_id}")
        if ar.const_offset:
            terms.append(str(ar.const_offset))
        start_expr = " + ".join(terms) if terms else "0"
        pattern_parts.append(f"{start_expr}:+{ar.extent}")
    return f"{access.tensor_name}[{', '.join(pattern_parts)}]"


def resolve_node(forest: TreeIR, path: tuple[int, ...]) -> "ForNode | SBlock | None":
    """Walk ``path`` from the forest root; return node or None on invalid path.

    Returns None for: empty path; index out of range; attempting to descend
    through an SBlock.

    Args:
        forest: Top-level list of trees.
        path: Tuple of child indices from forest root down to target.

    Returns:
        Node at ``path``, or ``None`` if invalid.
    """
    result: ForNode | SBlock | None = None
    if path:
        siblings: list[ForNode | SBlock] = list(forest)
        node: ForNode | SBlock | None = None
        valid = True
        for idx in path:
            if idx < 0 or idx >= len(siblings):
                valid = False
                break
            node = siblings[idx]
            if isinstance(node, SBlock):
                siblings = []
            else:
                siblings = node.children
        if valid:
            result = node
    return result


def replace_at_path(forest: TreeIR, path: tuple[int, ...], replacement: "ForNode | SBlock") -> TreeIR:
    """Return a new forest with ``replacement`` placed at ``path``.

    Ancestors along ``path`` are rebuilt; untouched subtrees pass by reference.

    Args:
        forest: Top-level list of trees.
        path: Non-empty tuple of child indices.
        replacement: New node to place at ``path``.

    Returns:
        A new forest.

    Raises:
        ValueError: ``path`` is empty or traverses a non-ForNode ancestor.
    """
    if not path:
        raise ValueError("replace_at_path: path must be non-empty")
    result: TreeIR
    if len(path) == 1:
        idx = path[0]
        result = [*forest[:idx], replacement, *forest[idx + 1 :]]
    else:
        idx, rest = path[0], path[1:]
        parent = forest[idx]
        if not isinstance(parent, ForNode):
            raise ValueError("replace_at_path: non-ForNode ancestor")
        new_children = replace_at_path(parent.children, rest, replacement)
        new_parent = ForNode(
            iter_var=parent.iter_var, children=new_children, name=parent.name, annotations=dict(parent.annotations)
        )
        result = [*forest[:idx], new_parent, *forest[idx + 1 :]]
    return result


def blocks_under(node: "ForNode | SBlock") -> Iterator[SBlock]:
    """Yield every SBlock in ``node``'s subtree.

    Includes ``node`` itself when ``node`` is an SBlock.
    """
    if isinstance(node, SBlock):
        yield node
        return
    for child in node.children:
        yield from blocks_under(child)


def validate_dataflow_ordering(module: KernelModule) -> bool:
    """Enforce 5 dataflow legality rules in pre-order DFS of the forest.

    Rules:
    1. Alloc precedes use — a tensor name cannot appear in any block's
       reads/writes/reads_writes before its NKIAlloc block.
    2. Non-alloc blocks' operand names must be allocated earlier (or be params).
    3. Reads after real writes — every read name must be written by some
       prior non-alloc block (or be a param). An ``NKIAlloc`` block declares
       storage but does NOT count as a value-producing write for this rule,
       so a consumer placed above its value-producing writer is rejected.
    4. RMW finalization — for tensor T with any RMW writer, every non-RMW
       reader of T must come after the LAST RMW write.
    5. Return produced — every tensor with origin == "return" must be in
       the written set by end of walk (allocs alone do not satisfy this).

    Returns:
        True if every rule holds; False on any violation.
    """
    allocated: set[str] = set(module.param_names)
    written: set[str] = set(module.param_names)
    rmw_total = _count_rmw_writers(module.body)
    rmw_seen: dict[str, int] = {}

    walker_valid = True
    for root in module.body:
        if not _walk_node(root, allocated, written, rmw_total, rmw_seen):
            walker_valid = False
            break

    result = walker_valid
    if result:
        for tname, tensor in module.tensors.items():
            if tensor.origin == "return" and tname not in written:
                result = False
                break
    return result


def _count_rmw_writers(forest: TreeIR) -> dict[str, int]:
    """Return ``{tensor_name: count_of_RMW_write_blocks}`` across the tree."""
    counts: dict[str, int] = {}

    def walk(node: ForNode | SBlock) -> None:
        if isinstance(node, SBlock):
            for access in node.reads_writes.values():
                counts[access.tensor_name] = counts.get(access.tensor_name, 0) + 1
            return
        for child in node.children:
            walk(child)

    for root in forest:
        walk(root)
    return counts


def _walk_node(
    node: ForNode | SBlock, allocated: set[str], written: set[str], rmw_total: dict[str, int], rmw_seen: dict[str, int]
) -> bool:
    """Recursively validate ``node``; mutates ``allocated`` / ``written`` /
    ``rmw_seen`` as it goes. Returns False on any rule violation."""
    result = True
    if isinstance(node, SBlock):
        result = _check_block(node, allocated, written, rmw_total, rmw_seen)
    else:
        for child in node.children:
            if not _walk_node(child, allocated, written, rmw_total, rmw_seen):
                result = False
                break
    return result


def _check_block(
    block: SBlock, allocated: set[str], written: set[str], rmw_total: dict[str, int], rmw_seen: dict[str, int]
) -> bool:
    """Validate one SBlock against rules 1-4. Mutates the running sets."""
    is_alloc = len(block.body) == 1 and block.body[0].op_cls.__name__ == "NKIAlloc"
    if is_alloc:
        tname = block.body[0].kwargs["tensor_name"]
        allocated.add(tname)
        return True

    touched = (
        {a.tensor_name for a in block.reads.values()}
        | {a.tensor_name for a in block.writes.values()}
        | {a.tensor_name for a in block.reads_writes.values()}
    )
    for tname in touched:
        if tname not in allocated:
            return False

    read_set = {a.tensor_name for a in block.reads.values()} | {a.tensor_name for a in block.reads_writes.values()}
    for tname in read_set:
        if tname not in written:
            return False

    for access in block.reads.values():
        tname = access.tensor_name
        if rmw_total.get(tname, 0) > 0 and rmw_seen.get(tname, 0) < rmw_total[tname]:
            return False

    for access in block.writes.values():
        written.add(access.tensor_name)
    for access in block.reads_writes.values():
        written.add(access.tensor_name)
        rmw_seen[access.tensor_name] = rmw_seen.get(access.tensor_name, 0) + 1
    return True
