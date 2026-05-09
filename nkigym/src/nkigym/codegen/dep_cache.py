"""Per-scope dependency cache. Analog of TVM's SBlockScope.

Dep information is *not* mutated alongside tree edits — instead, each scope
entry stores a structural signature of its subtree, and ``for_scope`` rebuilds
lazily when the signature has changed.

This keeps transform implementations simple (no explicit invalidate calls)
at the cost of one signature hash per read.
"""

from dataclasses import dataclass, field
from enum import Enum


class DepKind(Enum):
    """Dependency edge classification.

    Mirrors TVM's DepKind enum: RAW (read-after-write), WAR (write-after-read),
    WAW (write-after-write), OPAQUE (unclassified).
    """

    RAW = 0
    WAR = 1
    WAW = 2
    OPAQUE = 3


@dataclass(frozen=True)
class LeafId:
    """Structural identifier for a BodyLeaf — path from forest root.

    Stable across tree edits only if the leaf's ancestors' child lists are
    unchanged. Callers recompute LeafIds after each rewrite.
    """

    path: tuple[int, ...]


@dataclass(frozen=True)
class ScopeId:
    """Structural identifier for a scope root.

    ``path`` is a root-to-node tuple of child indices (empty tuple = the forest
    itself is the scope).
    """

    path: tuple[int, ...]


@dataclass(frozen=True)
class Dependency:
    """One directed dep edge between two leaves."""

    src: LeafId
    dst: LeafId
    kind: DepKind


@dataclass
class SBlockScope:
    """Per-scope dep graph.

    Attributes:
        src2deps: Maps a source leaf to its outgoing edges.
        dst2deps: Maps a destination leaf to its incoming edges.
        buffer_writers: Maps tensor name to writer leaves (in source order).
        signature: Structural hash of the scope's subtree when this entry was
            built. Used by ``DepCache.for_scope`` to detect staleness.
    """

    src2deps: dict[LeafId, list[Dependency]]
    dst2deps: dict[LeafId, list[Dependency]]
    buffer_writers: dict[str, list[LeafId]]
    signature: int = 0


@dataclass
class DepCache:
    """Per-scope dep cache.

    Stores one :class:`SBlockScope` per scope id; lazy-rebuilds on signature
    mismatch so transforms do not need explicit invalidate calls.

    Attributes:
        scopes: Maps ``ScopeId`` to its cached :class:`SBlockScope`.
    """

    scopes: dict[ScopeId, SBlockScope] = field(default_factory=dict)

    def for_scope(self, scope_id: ScopeId, children: "list[LoopNode | BodyLeaf]") -> SBlockScope:
        """Return the scope's dep graph, rebuilding lazily on signature mismatch.

        Args:
            scope_id: Scope identifier.
            children: The current top-level children of this scope (as resolved
                from the live tree).

        Returns:
            Fresh :class:`SBlockScope` if the cache was stale or missing;
            cached entry otherwise.
        """
        current_sig = hash(tuple(subtree_signature(c) for c in children))
        cached = self.scopes.get(scope_id)
        if cached is not None and cached.signature == current_sig:
            return cached
        fresh = rebuild_scope(children)
        self.scopes[scope_id] = fresh
        return fresh


def subtree_signature(node: "LoopNode | BodyLeaf") -> int:
    """Return a deterministic structural hash of ``node``'s subtree.

    Two subtrees produce the same signature iff they have the same tree
    structure and every leaf's read/write sets are equal.
    """
    from nkigym.codegen.ir import BodyLeaf

    if isinstance(node, BodyLeaf):
        return hash(("leaf", id(node.op_cls), tuple(sorted(node.reads.items())), node.writes, node.reads_writes))
    return hash(
        (
            "node",
            node.dim_id,
            node.trip_count,
            node.role.value,
            node.reduce_op,
            node.pipeline_depth,
            tuple(subtree_signature(c) for c in node.children),
        )
    )


def rebuild_scope(children: "list[LoopNode | BodyLeaf]") -> SBlockScope:
    """Build an :class:`SBlockScope` for the given scope's top-level children.

    Walks every descendant leaf, classifies pair-wise edges by buffer name,
    returns the resulting dep graph.
    """
    from nkigym.codegen.ir import BodyLeaf

    leaves: list[tuple[LeafId, BodyLeaf]] = []

    def walk(node: "LoopNode | BodyLeaf", path: tuple[int, ...]) -> None:
        if isinstance(node, BodyLeaf):
            leaves.append((LeafId(path), node))
            return
        for i, child in enumerate(node.children):
            walk(child, path + (i,))

    for i, child in enumerate(children):
        walk(child, (i,))

    src2deps: dict[LeafId, list[Dependency]] = {}
    dst2deps: dict[LeafId, list[Dependency]] = {}
    buffer_writers: dict[str, list[LeafId]] = {}

    for i, (src_id, src_leaf) in enumerate(leaves):
        for dst_id, dst_leaf in leaves[i + 1 :]:
            kind = _classify_edge(src_leaf, dst_leaf)
            if kind is not None:
                dep = Dependency(src=src_id, dst=dst_id, kind=kind)
                src2deps.setdefault(src_id, []).append(dep)
                dst2deps.setdefault(dst_id, []).append(dep)
        for t in src_leaf.writes:
            buffer_writers.setdefault(t, []).append(src_id)

    sig = hash(tuple(subtree_signature(c) for c in children))
    return SBlockScope(src2deps=src2deps, dst2deps=dst2deps, buffer_writers=buffer_writers, signature=sig)


def _classify_edge(src: "BodyLeaf", dst: "BodyLeaf") -> DepKind | None:
    """Classify the strongest data dependency from ``src`` to ``dst``.

    Returns ``None`` when ``src`` and ``dst`` have no shared buffers.
    Precedence: RAW > WAW > WAR (RAW is the strongest ordering constraint
    and takes priority when multiple edge types could apply).
    """
    src_reads = set(src.reads.values())
    src_writes = set(src.writes)
    dst_reads = set(dst.reads.values())
    dst_writes = set(dst.writes)
    if src_writes & dst_reads:
        return DepKind.RAW
    if src_writes & dst_writes:
        return DepKind.WAW
    if src_reads & dst_writes:
        return DepKind.WAR
    return None
