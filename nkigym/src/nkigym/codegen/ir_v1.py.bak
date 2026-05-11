"""Core IR types for nkigym scheduling.

The IR has three roles:

- :class:`KernelModule` - envelope. Holds signature + tensor/dim declarations +
  tree body + dep cache.
- :class:`TreeIR` / :class:`LoopNode` / :class:`BodyLeaf` - schedule tree. Leaves
  self-describe all op metadata (no back-reference to a sidecar).
- :class:`DepCache` - per-scope dependency cache (defined in
  :mod:`nkigym.codegen.dep_cache`).

Analogous to TVM's PrimFunc + buffer_map + SBlockNode. Leaves mirror TVM's
self-describing blocks.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from nkigym.codegen.dep_cache import DepCache
from nkigym.ops.base import AxisRole

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
class DimInfo:
    """Concrete dimension metadata."""

    dim_id: str
    total_size: int
    tile_size: int
    num_tiles: int


@dataclass
class LoopNode:
    """One loop in the schedule tree."""

    dim_id: str
    trip_count: int
    role: AxisRole
    children: "list[LoopNode | BodyLeaf]" = field(default_factory=list)
    reduce_op: str | None = None
    name: str | None = None
    pipeline_depth: int = 1


@dataclass
class BodyLeaf:
    """Self-describing leaf: one op + the metadata needed to render it.

    Every non-parameter tensor referenced in ``reads`` / ``writes`` /
    ``reads_writes`` is declared by an ``NKIAlloc`` leaf earlier in
    pre-order DFS.
    """

    op_cls: type
    reads: dict[str, str] = field(default_factory=dict)
    writes: tuple[str, ...] = ()
    reads_writes: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    dim_role: dict[str, AxisRole] = field(default_factory=dict)


TreeIR = list[LoopNode | BodyLeaf]


@dataclass
class KernelModule:
    """Envelope IR - signature + declarations + body + dep cache.

    Analog of TVM's PrimFunc + buffer_map.

    Attributes:
        func_name: Emitted kernel name.
        param_names: Signature order.
        return_name: Tensor name of the return value.
        tensors: All named tensors, keyed by name.
        dims: All dims, keyed by dim id.
        body: The schedule tree.
        dep: Per-scope dependency cache.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    body: TreeIR = field(default_factory=list)
    dep: DepCache = field(default_factory=lambda: DepCache(scopes={}))


def resolve_node(forest: TreeIR, path: tuple[int, ...]) -> "LoopNode | BodyLeaf | None":
    """Walk ``path`` from the forest root; return the node or ``None`` on invalid path.

    Args:
        forest: Top-level list of trees.
        path: Tuple of child indices from forest root down to target.

    Returns:
        The node at ``path``; ``None`` if the path is empty, indexes out of
        range, or attempts to descend through a ``BodyLeaf``.
    """
    if not path:
        return None
    siblings: list[LoopNode | BodyLeaf] = list(forest)
    node: LoopNode | BodyLeaf | None = None
    for idx in path:
        if idx < 0 or idx >= len(siblings):
            return None
        node = siblings[idx]
        if isinstance(node, BodyLeaf):
            siblings = []
        else:
            siblings = node.children
    return node


def replace_at_path(forest: TreeIR, path: tuple[int, ...], replacement: "LoopNode | BodyLeaf") -> TreeIR:
    """Return a new forest with ``replacement`` placed at ``path``.

    Ancestors along ``path`` are rebuilt; untouched subtrees pass by reference.

    Raises:
        ValueError: ``path`` is empty or traverses a non-loop ancestor.
    """
    if not path:
        raise ValueError("replace_at_path: path must be non-empty")
    if len(path) == 1:
        idx = path[0]
        return [*forest[:idx], replacement, *forest[idx + 1 :]]
    idx, rest = path[0], path[1:]
    parent = forest[idx]
    if not isinstance(parent, LoopNode):
        raise ValueError("replace_at_path: non-loop ancestor")
    new_children = replace_at_path(parent.children, rest, replacement)
    new_parent = LoopNode(
        dim_id=parent.dim_id,
        trip_count=parent.trip_count,
        role=parent.role,
        children=new_children,
        reduce_op=parent.reduce_op,
        name=parent.name,
        pipeline_depth=parent.pipeline_depth,
    )
    return [*forest[:idx], new_parent, *forest[idx + 1 :]]


def leaves_under(node: "LoopNode | BodyLeaf") -> Iterator[BodyLeaf]:
    """Yield every ``BodyLeaf`` reachable from ``node`` (pre-order)."""
    if isinstance(node, BodyLeaf):
        yield node
        return
    for child in node.children:
        yield from leaves_under(child)


def emission_order_leaves(body: TreeIR) -> Iterator[BodyLeaf]:
    """Yield every ``BodyLeaf`` in emission order (pre-order DFS over the forest)."""
    for root in body:
        yield from leaves_under(root)


def validate_dataflow_ordering(module: KernelModule) -> bool:
    """Return True iff the forest's emission order preserves dataflow.

    Rules:
    - **Reads-after-writes:** every name in ``leaf.reads ∪ leaf.reads_writes``
      must be a parameter or already-written by a prior leaf in pre-order
      DFS.
    - **Writes update the written set:** names in ``leaf.writes ∪ leaf.reads_writes``
      are added after the leaf fires.
    - **Return tensor produced:** every tensor with ``origin == "return"``
      (legacy — being phased out) must be written by some leaf.
    - **RMW finalization:** for a tensor T with any RMW writer, every
      non-RMW read of T (a ``reads``-slot consumer) must appear after the
      LAST RMW write. Otherwise the consumer reads a partial-accumulation
      value. Enforced by a pre-pass that records the last RMW-writer
      position per tensor and rejects consumers that appear earlier.

    RMW operands encode "init must precede update" structurally —
    `NKIMatmul.reads_writes = ('psum_acc',)` means the matmul leaf
    requires a prior writer of ``psum_acc`` (the memset). The RMW
    finalization rule additionally ensures drain/consumer ops (e.g.
    NKITensorCopy) don't read psum_acc before all accumulating
    nc_matmul iterations complete.
    """
    written: set[str] = set()
    params = {t.name for t in module.tensors.values() if t.origin == "param"}
    returns = {t.name for t in module.tensors.values() if t.origin == "return"}

    """Allocated tensor names — NKIAlloc reserves storage but does NOT
    initialize content. Downstream readers (including RMW operands like
    matmul's psum dst) need a content-writer (NKIMemset, NKILoad, another
    compute op) to appear after the alloc but before the read."""
    allocated: set[str] = set()

    """Pre-pass: find the emission index of the last RMW write per tensor."""
    leaves_list = list(emission_order_leaves(module.body))
    last_rmw_write: dict[str, int] = {}
    for idx, leaf in enumerate(leaves_list):
        for tname in leaf.reads_writes:
            last_rmw_write[tname] = idx

    for idx, leaf in enumerate(leaves_list):
        op_name = leaf.op_cls.__name__
        if op_name == "NKIAlloc":
            """Alloc leaf: the tensor must appear for the FIRST time (no prior
            non-alloc read or write). If another leaf already touched this
            name, we'd be using the tensor before its storage exists —
            reject the state."""
            for tname in leaf.writes:
                if tname in written or tname in allocated:
                    return False
                allocated.add(tname)
            continue
        """Non-alloc leaves: all reads/writes/reads_writes names must have
        been allocated first (storage reserved) — params are exempt."""
        touched = set(leaf.reads.values()) | set(leaf.writes) | set(leaf.reads_writes)
        for name in touched:
            if name in params:
                continue
            if name not in allocated:
                return False
        read_set = set(leaf.reads.values()) | set(leaf.reads_writes)
        for name in read_set:
            if name in params:
                continue
            if name not in written:
                return False
        """RMW finalization: a non-RMW read (leaf.reads, not reads_writes) of
        a tensor with any RMW writer must come after the last RMW writer.
        A leaf that itself RMW's the tensor is exempt (same-tensor RMW chains
        are the accumulation iterations themselves)."""
        for name in leaf.reads.values():
            if name in last_rmw_write and idx < last_rmw_write[name]:
                return False
        write_set = set(leaf.writes) | set(leaf.reads_writes)
        written |= write_set

    for ret_name in returns:
        if ret_name not in written:
            return False
    return True
