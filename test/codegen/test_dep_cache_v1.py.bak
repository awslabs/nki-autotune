"""Tests for dep_cache module."""

from nkigym.codegen.dep_cache import (
    DepCache,
    Dependency,
    DepKind,
    LeafId,
    SBlockScope,
    ScopeId,
    rebuild_scope,
    subtree_signature,
)
from nkigym.codegen.ir import BodyLeaf, LoopNode
from nkigym.ops.base import AxisRole


def test_depkind_enum_values():
    assert DepKind.RAW.value == 0
    assert DepKind.WAR.value == 1
    assert DepKind.WAW.value == 2
    assert DepKind.OPAQUE.value == 3


def test_dependency_construction():
    d = Dependency(src=LeafId((0,)), dst=LeafId((1,)), kind=DepKind.RAW)
    assert d.src == LeafId((0,))
    assert d.kind == DepKind.RAW


def test_sblock_scope_empty():
    s = SBlockScope(src2deps={}, dst2deps={}, buffer_writers={})
    assert s.src2deps == {}


def _make_leaf(op_cls=object, reads=None, writes=()):
    return BodyLeaf(op_cls=op_cls, reads=reads or {}, writes=writes)


def test_subtree_signature_stable():
    a = _make_leaf(writes=("t1",))
    b = _make_leaf(reads={"x": "t1"}, writes=("t2",))
    node1 = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.PARALLEL, children=[a, b])
    node2 = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.PARALLEL, children=[a, b])
    assert subtree_signature(node1) == subtree_signature(node2)


def test_subtree_signature_detects_order_change():
    a = _make_leaf(writes=("t1",))
    b = _make_leaf(reads={"x": "t1"}, writes=("t2",))
    sig1 = subtree_signature(LoopNode("d0", 4, AxisRole.PARALLEL, [a, b]))
    sig2 = subtree_signature(LoopNode("d0", 4, AxisRole.PARALLEL, [b, a]))
    assert sig1 != sig2


def test_rebuild_scope_produces_raw_edge():
    producer = _make_leaf(writes=("t1",))
    consumer = _make_leaf(reads={"x": "t1"}, writes=("t2",))
    scope_root_children = [producer, consumer]
    scope = rebuild_scope(scope_root_children)
    assert len(scope.src2deps) == 1
    edges = next(iter(scope.src2deps.values()))
    assert edges[0].kind == DepKind.RAW


def test_dep_cache_for_scope_rebuilds_on_miss():
    producer = _make_leaf(writes=("t1",))
    consumer = _make_leaf(reads={"x": "t1"}, writes=("t2",))
    cache = DepCache(scopes={})
    scope_id = ScopeId(())
    scope = cache.for_scope(scope_id, [producer, consumer])
    assert scope_id in cache.scopes
    assert len(scope.src2deps) == 1


def test_dep_cache_for_scope_rebuilds_on_stale_signature():
    producer = _make_leaf(writes=("t1",))
    consumer = _make_leaf(reads={"x": "t1"}, writes=("t2",))
    children = [producer, consumer]
    scope_id = ScopeId(())
    stub = SBlockScope(src2deps={}, dst2deps={}, buffer_writers={}, signature=0)
    cache = DepCache(scopes={scope_id: stub})
    scope = cache.for_scope(scope_id, children)
    assert scope is not stub
    assert len(scope.src2deps) == 1
    assert cache.scopes[scope_id] is scope
