"""Per-scope dependency cache keyed on SBlock paths.

``_classify_edge`` unions ``reads_writes`` into both the reads side and the
writes side of the intersection so RMW operands contribute correctly to
RAW / WAR / WAW edges.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` §7.
"""

from dataclasses import dataclass, field
from enum import Enum

from nkigym.ir.ir import ForNode, SBlock


class DepKind(Enum):
    """Dependency edge classification. Same as v1."""

    RAW = 0
    WAR = 1
    WAW = 2
    OPAQUE = 3


@dataclass(frozen=True)
class LeafId:
    """Structural identifier for an SBlock — path from forest root."""

    path: tuple[int, ...]


@dataclass(frozen=True)
class ScopeId:
    """Structural identifier for a scope root.

    Empty tuple = forest root is the scope.
    """

    path: tuple[int, ...]


@dataclass(frozen=True)
class Dependency:
    """One directed dep edge between two blocks."""

    src: LeafId
    dst: LeafId
    kind: DepKind


@dataclass
class SBlockScope:
    """Per-scope dep graph.

    Attributes:
        src2deps: source LeafId → outgoing dependencies.
        dst2deps: destination LeafId → incoming dependencies.
        buffer_writers: tensor name → writer LeafIds (in DFS order).
        signature: Structural hash of the scope's subtree — used to
            detect staleness on next ``for_scope`` call.
    """

    src2deps: dict[LeafId, list[Dependency]]
    dst2deps: dict[LeafId, list[Dependency]]
    buffer_writers: dict[str, list[LeafId]]
    signature: int = 0


@dataclass
class DepCache:
    """Per-scope dep cache. Lazy-rebuilds on signature mismatch.

    Attributes:
        scopes: ScopeId → SBlockScope.
    """

    scopes: dict[ScopeId, SBlockScope] = field(default_factory=dict)

    def for_scope(self, scope_id: ScopeId, children: "list[ForNode | SBlock]") -> SBlockScope:
        """Return the scope's dep graph; rebuild lazily on signature mismatch.

        Args:
            scope_id: Scope identifier.
            children: Current top-level children of the scope.

        Returns:
            Cached SBlockScope if signature matches, freshly-built otherwise.
        """
        current_sig = hash(tuple(subtree_signature(c) for c in children))
        cached = self.scopes.get(scope_id)
        result: SBlockScope
        if cached is not None and cached.signature == current_sig:
            result = cached
        else:
            fresh = rebuild_scope(children)
            self.scopes[scope_id] = fresh
            result = fresh
        return result


def subtree_signature(node: "ForNode | SBlock") -> int:
    """Structural hash of a subtree.

    Folds iter-var ids, buffer access patterns, block bodies, and ForNode
    children so any atom-driven change triggers a cache rebuild.
    """
    result: int
    if isinstance(node, SBlock):
        reads_key = tuple(sorted((k, v.tensor_name, v.iter_var_ids, v.pattern) for k, v in node.reads.items()))
        writes_key = tuple(sorted((k, v.tensor_name, v.iter_var_ids, v.pattern) for k, v in node.writes.items()))
        rmw_key = tuple(sorted((k, v.tensor_name, v.iter_var_ids, v.pattern) for k, v in node.reads_writes.items()))
        body_key = tuple(c.op_cls.__name__ for c in node.body)
        iv_key = tuple((iv.var_id, iv.axis_id, iv.extent, iv.role.value) for iv in node.iter_vars)
        result = hash(("block", iv_key, reads_key, writes_key, rmw_key, body_key))
    else:
        iv = node.iter_var
        result = hash(
            ("for", iv.var_id, iv.axis_id, iv.extent, iv.role.value, tuple(subtree_signature(c) for c in node.children))
        )
    return result


def rebuild_scope(children: "list[ForNode | SBlock]") -> SBlockScope:
    """Build an :class:`SBlockScope` for the given scope's top-level children.

    Walks every descendant SBlock, classifies pair-wise edges by
    ``_classify_edge``, returns the resulting graph.

    Args:
        children: The scope's top-level children.

    Returns:
        Fresh SBlockScope.
    """
    blocks: list[tuple[LeafId, SBlock]] = []

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        if isinstance(node, SBlock):
            blocks.append((LeafId(path), node))
            return
        for i, c in enumerate(node.children):
            walk(c, path + (i,))

    for i, c in enumerate(children):
        walk(c, (i,))

    src2deps: dict[LeafId, list[Dependency]] = {}
    dst2deps: dict[LeafId, list[Dependency]] = {}
    buffer_writers: dict[str, list[LeafId]] = {}

    for i, (src_id, src_block) in enumerate(blocks):
        for dst_id, dst_block in blocks[i + 1 :]:
            kind = _classify_edge(src_block, dst_block)
            if kind is not None:
                dep = Dependency(src=src_id, dst=dst_id, kind=kind)
                src2deps.setdefault(src_id, []).append(dep)
                dst2deps.setdefault(dst_id, []).append(dep)
        for access in src_block.writes.values():
            buffer_writers.setdefault(access.tensor_name, []).append(src_id)
        for access in src_block.reads_writes.values():
            buffer_writers.setdefault(access.tensor_name, []).append(src_id)

    sig = hash(tuple(subtree_signature(c) for c in children))
    return SBlockScope(src2deps=src2deps, dst2deps=dst2deps, buffer_writers=buffer_writers, signature=sig)


def _classify_edge(src: SBlock, dst: SBlock) -> DepKind | None:
    """Classify the strongest edge from ``src`` to ``dst``.

    Unions ``reads_writes`` into BOTH the reads and writes sides —
    fixes v1's RMW-blind spot. Precedence: RAW > WAW > WAR.

    Returns:
        DepKind.RAW / WAR / WAW, or None if no shared tensor.
    """
    src_reads = {a.tensor_name for a in src.reads.values()} | {a.tensor_name for a in src.reads_writes.values()}
    src_writes = {a.tensor_name for a in src.writes.values()} | {a.tensor_name for a in src.reads_writes.values()}
    dst_reads = {a.tensor_name for a in dst.reads.values()} | {a.tensor_name for a in dst.reads_writes.values()}
    dst_writes = {a.tensor_name for a in dst.writes.values()} | {a.tensor_name for a in dst.reads_writes.values()}
    result: DepKind | None = None
    if src_writes & dst_reads:
        result = DepKind.RAW
    elif src_writes & dst_writes:
        result = DepKind.WAW
    elif src_reads & dst_writes:
        result = DepKind.WAR
    return result
