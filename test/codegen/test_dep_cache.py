"""Unit tests for v2 DepCache — classifier folds reads_writes correctly."""

from nkigym.codegen.dep_cache import DepCache, DepKind, ScopeId, _classify_edge, rebuild_scope, subtree_signature
from nkigym.codegen.ir import BufferAccess, NKIOpCall, SBlock
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy


def _mk_access(name: str) -> BufferAccess:
    """Zero-iter-var access (sufficient for name-based dep classification)."""
    return BufferAccess(tensor_name=name, iter_var_ids=(), pattern=())


def test_classify_edge_raw_rmw_into_reads() -> None:
    """A block that RMWs T → another block that reads T is RAW (strongest).

    Fixes the v1 blind spot where src.reads_writes intersected with
    dst.reads was not captured as RAW.
    """
    matmul = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={"dst": _mk_access("psum_acc")},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    tcopy = SBlock(
        iter_vars=[],
        reads={"src": _mk_access("psum_acc")},
        writes={},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKITensorCopy, kwargs={}, axis_map={}, dim_role={})],
    )
    assert _classify_edge(matmul, tcopy) == DepKind.RAW


def test_classify_edge_memset_then_matmul_rmw_is_raw() -> None:
    """memset writes psum_acc; matmul RMWs psum_acc.

    The matmul's RMW counts as both reads AND writes. src.writes ∩
    dst.reads (through reads_writes fold) → RAW.
    """
    memset = SBlock(
        iter_vars=[],
        reads={},
        writes={"dst": _mk_access("psum_acc")},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKIMemset, kwargs={"value": 0.0}, axis_map={}, dim_role={})],
    )
    matmul = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={"dst": _mk_access("psum_acc")},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    assert _classify_edge(memset, matmul) == DepKind.RAW


def test_classify_edge_disjoint_returns_none() -> None:
    """Blocks touching disjoint tensors have no edge."""
    a = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x")}, reads_writes={}, body=[])
    b = SBlock(iter_vars=[], reads={"src": _mk_access("y")}, writes={}, reads_writes={}, body=[])
    assert _classify_edge(a, b) is None


def test_classify_edge_war() -> None:
    """src reads T, dst writes T → WAR (without any earlier write)."""
    reader = SBlock(iter_vars=[], reads={"src": _mk_access("t")}, writes={}, reads_writes={}, body=[])
    writer = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("t")}, reads_writes={}, body=[])
    assert _classify_edge(reader, writer) == DepKind.WAR


def test_classify_edge_waw() -> None:
    """src writes T, dst writes T (both pure writes) → WAW."""
    writer_a = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("t")}, reads_writes={}, body=[])
    writer_b = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("t")}, reads_writes={}, body=[])
    assert _classify_edge(writer_a, writer_b) == DepKind.WAW


def test_rebuild_scope_populates_buffer_writers() -> None:
    """Every alloc/write block appears in buffer_writers in path order."""
    a = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x")}, reads_writes={}, body=[])
    b = SBlock(iter_vars=[], reads={"src": _mk_access("x")}, writes={"dst": _mk_access("y")}, reads_writes={}, body=[])
    scope = rebuild_scope([a, b])
    assert "x" in scope.buffer_writers
    assert "y" in scope.buffer_writers
    assert len(scope.buffer_writers["x"]) == 1


def test_rebuild_scope_includes_rmw_in_buffer_writers() -> None:
    """RMW blocks count as writers too."""
    rmw = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={"dst": _mk_access("psum")}, body=[])
    scope = rebuild_scope([rmw])
    assert "psum" in scope.buffer_writers
    assert len(scope.buffer_writers["psum"]) == 1


def test_dep_cache_lazy_rebuild_on_signature_mismatch() -> None:
    """DepCache rebuilds scope when signature changes."""
    cache = DepCache()
    scope_id = ScopeId(())
    a = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x")}, reads_writes={}, body=[])
    first = cache.for_scope(scope_id, [a])
    b = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x")}, reads_writes={}, body=[])
    """Same structure — should return cached."""
    cached = cache.for_scope(scope_id, [a])
    assert cached is first
    """Different children — signature mismatch forces rebuild."""
    fresh = cache.for_scope(scope_id, [a, b])
    assert fresh is not first


def test_subtree_signature_distinguishes_iter_vars() -> None:
    """subtree_signature changes when iter_vars change."""
    from nkigym.codegen.ir import IterVar
    from nkigym.ops.base import AxisRole

    block1 = SBlock(iter_vars=[IterVar(0, "d0", 4, AxisRole.PARALLEL)], reads={}, writes={}, reads_writes={}, body=[])
    block2 = SBlock(iter_vars=[IterVar(1, "d0", 4, AxisRole.PARALLEL)], reads={}, writes={}, reads_writes={}, body=[])
    assert subtree_signature(block1) != subtree_signature(block2)
