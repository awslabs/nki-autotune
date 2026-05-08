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
        buffer_degree: Multi-buffer degree per dim; defaults to 1.
    """

    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin
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
class OpLocalBuffer:
    """Resolved op-local buffer ready for renderer emission."""

    logical_name: str
    emitted_name: str
    location: Literal["sbuf", "psum"]
    dtype: str
    axis_ids: tuple[str, ...]
    shape: tuple[int, ...]


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
    """Self-describing leaf: an op (or op phase) + the metadata needed to render it.

    Every metadata field that used to live on ``ParsedOp`` now lives here, so
    legality checks and rendering can work from the leaf alone without
    consulting a sidecar op graph.

    Attributes:
        op_cls: The NKIOp subclass.
        phase: ``"main"`` for single-phase ops; one of the op class's phases
            otherwise (e.g. ``"psum_init"``, ``"compute"``, ``"drain"`` for
            matmul; ``"reduce_step"``, ``"reduce_close"`` for activation_reduce).
        reads: Maps operand slot name to referenced tensor name.
        writes: Tuple of tensor names this leaf writes.
        kwargs: Merged literal kwargs from the NKIOp call.
        axis_map: Abstract axis label (``"K"`` etc.) to concrete dim id.
        dim_role: Concrete dim id to :class:`AxisRole` (op-local).
        op_local_buffers: Op-local buffers keyed by logical name.
    """

    op_cls: type
    phase: str = "main"
    reads: dict[str, str] = field(default_factory=dict)
    writes: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    dim_role: dict[str, AxisRole] = field(default_factory=dict)
    op_local_buffers: dict[str, OpLocalBuffer] = field(default_factory=dict)


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

    Three conditions must hold:

    * **Reads-after-writes** — every ``leaf.reads`` refers to a tensor that
      is either a parameter (``origin == "param"``) or has already been
      written by an earlier leaf in pre-order DFS over the forest.
    * **Return tensor must be produced** — every tensor whose
      ``origin == "return"`` must be written by at least one leaf in the
      body. Rewrites that structurally clone a subtree while keeping only
      one leaf (e.g. ``DecomposeReduction``) can drop the store leaf
      downstream of the reducer, leaving ``hbm_out`` unwritten — caught
      here before the sampler pool accepts the state.
    * **Matmul phase order** — for each matmul op (identified by shared
      ``axis_map`` among ``NKIMatmul`` leaves), ``psum_init`` must precede
      ``compute``, and ``compute`` must precede ``drain`` in emission
      order. The inline ``psum_tile = nl.ndarray(...)`` binding emitted by
      ``psum_init`` is a Python-level local; if ``compute`` or ``drain``
      appears first, the reference is a use-before-def at runtime.

    Tensor-level only for the first two checks — op-local buffers in
    ``leaf.op_local_buffers`` are emitted at the kernel top and are not
    subject to emission-order constraints at leaf granularity.

    Returns:
        ``True`` when the forest linearization is consistent; ``False``
        otherwise.
    """
    written: set[str] = set()
    params = {t.name for t in module.tensors.values() if t.origin == "param"}
    returns = {t.name for t in module.tensors.values() if t.origin == "return"}
    out: bool = True
    matmul_phase_order = {"psum_init": 0, "compute": 1, "drain": 2}
    matmul_seen: dict[tuple, int] = {}
    for leaf in emission_order_leaves(module.body):
        for tensor_name in leaf.reads.values():
            if tensor_name in params:
                continue
            if tensor_name in written:
                continue
            out = False
            break
        if not out:
            break
        if leaf.op_cls.__name__ == "NKIMatmul" and leaf.phase in matmul_phase_order:
            """Same-op phase leaves share ``reads`` + ``writes`` identically
            (set by ``_make_leaf`` from the parent ``_ParsedOp``), so the
            (op_cls, sorted-reads, writes) triple uniquely identifies this
            matmul instance even across rewrite-induced rebuilds."""
            key = (leaf.op_cls.__name__, tuple(sorted(leaf.reads.items())), leaf.writes)
            required = matmul_phase_order[leaf.phase]
            prior = matmul_seen.get(key, -1)
            if required < prior:
                out = False
                break
            matmul_seen[key] = max(prior, required)
        for t in leaf.writes:
            written.add(t)
    if out:
        for ret_name in returns:
            if ret_name not in written:
                out = False
                break
    return out
