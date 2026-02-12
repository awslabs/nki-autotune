"""Unit tests for nkigym.tiling.dim_tracker module.

Tests the _DimTracker union-find data structure and TracedOp dataclass
directly, covering path compression, unification semantics, canonical
ordering, intermediate name generation, and operation recording.

Run with: pytest test/test_dim_tracker.py -v
"""

from nkigym.tiling.dim_tracker import TracedOp, _DimTracker


class TestTracedOp:
    """Tests for TracedOp dataclass."""

    def test_fields(self) -> None:
        """Verify TracedOp stores op_name, inputs, and output."""
        op = TracedOp(op_name="matmul", inputs=["a", "b"], output="c")
        assert op.op_name == "matmul"
        assert op.inputs == ["a", "b"]
        assert op.output == "c"

    def test_equality(self) -> None:
        """Verify two TracedOps with same fields are equal."""
        op1 = TracedOp(op_name="load", inputs=["a"], output="t0")
        op2 = TracedOp(op_name="load", inputs=["a"], output="t0")
        assert op1 == op2

    def test_inequality(self) -> None:
        """Verify TracedOps with different fields are not equal."""
        op1 = TracedOp(op_name="load", inputs=["a"], output="t0")
        op2 = TracedOp(op_name="store", inputs=["a"], output="t0")
        assert op1 != op2


class TestNewDim:
    """Tests for _DimTracker.new_dim()."""

    def test_creates_sequential_ids(self) -> None:
        """Dimensions are created with sequential IDs d0, d1, d2."""
        tracker = _DimTracker()
        d0 = tracker.new_dim(128)
        d1 = tracker.new_dim(256)
        d2 = tracker.new_dim(512)
        assert d0 == "d0"
        assert d1 == "d1"
        assert d2 == "d2"

    def test_records_size(self) -> None:
        """Dimension size is stored in dim_sizes."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(256)
        assert tracker.dim_sizes["d0"] == 128
        assert tracker.dim_sizes["d1"] == 256

    def test_records_order(self) -> None:
        """Dimensions are recorded in creation order."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(256)
        tracker.new_dim(64)
        assert tracker.dim_order == ["d0", "d1", "d2"]

    def test_parent_is_self(self) -> None:
        """Each new dimension is its own parent (root of its set)."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        assert tracker._parent["d0"] == "d0"


class TestFind:
    """Tests for _DimTracker.find() with path compression."""

    def test_find_root_returns_self(self) -> None:
        """Finding the root of an unmerged dimension returns itself."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        assert tracker.find("d0") == "d0"

    def test_find_after_unify(self) -> None:
        """Finding a unified dimension returns the canonical root."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d0", "d1")
        assert tracker.find("d1") == "d0"

    def test_path_compression(self) -> None:
        """Path compression flattens parent pointers after find.

        Create chain d2 -> d1 -> d0. After find(d2), d2 should point
        directly to d0 (skipping d1).
        """
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d0", "d1")
        tracker.unify("d1", "d2")

        assert tracker.find("d2") == "d0"
        assert tracker._parent["d2"] == "d0"

    def test_path_compression_long_chain(self) -> None:
        """Path compression on a longer chain d4 -> d3 -> d2 -> d1 -> d0."""
        tracker = _DimTracker()
        for _ in range(5):
            tracker.new_dim(128)
        tracker.unify("d0", "d1")
        tracker.unify("d1", "d2")
        tracker.unify("d2", "d3")
        tracker.unify("d3", "d4")

        assert tracker.find("d4") == "d0"
        assert tracker._parent["d4"] == "d0"
        assert tracker._parent["d3"] == "d0"
        assert tracker._parent["d2"] == "d0"


class TestUnify:
    """Tests for _DimTracker.unify()."""

    def test_earlier_dim_becomes_root(self) -> None:
        """The earlier dimension (by creation order) becomes the canonical root."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d1", "d0")
        assert tracker.find("d1") == "d0"

    def test_unify_already_unified_is_noop(self) -> None:
        """Unifying two already-unified dimensions is a no-op."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d0", "d1")
        tracker.unify("d0", "d1")
        assert tracker.find("d1") == "d0"
        assert tracker.find("d0") == "d0"

    def test_unify_same_dim_is_noop(self) -> None:
        """Unifying a dimension with itself is a no-op."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.unify("d0", "d0")
        assert tracker.find("d0") == "d0"

    def test_unify_three_dims(self) -> None:
        """Three dimensions unified into one equivalence class."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d0", "d1")
        tracker.unify("d1", "d2")
        assert tracker.find("d0") == "d0"
        assert tracker.find("d1") == "d0"
        assert tracker.find("d2") == "d0"

    def test_unify_two_chains(self) -> None:
        """Merging two separate chains picks the earlier root.

        Chain 1: d0 <- d1
        Chain 2: d2 <- d3
        After unify(d1, d3): all map to d0.
        """
        tracker = _DimTracker()
        for _ in range(4):
            tracker.new_dim(128)
        tracker.unify("d0", "d1")
        tracker.unify("d2", "d3")
        tracker.unify("d1", "d3")
        assert tracker.find("d3") == "d0"
        assert tracker.find("d2") == "d0"


class TestGetCanonicalDims:
    """Tests for _DimTracker.get_canonical_dims()."""

    def test_no_unification(self) -> None:
        """Without unification, canonical dims are the original dims."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(256)
        assert tracker.get_canonical_dims(["d0", "d1"]) == ["d0", "d1"]

    def test_unified_dims_map_to_same_canonical(self) -> None:
        """Unified dimensions resolve to the same canonical ID."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d0", "d1")
        assert tracker.get_canonical_dims(["d0", "d1"]) == ["d0", "d0"]

    def test_mixed_unified_and_free(self) -> None:
        """Mix of unified and non-unified dimensions."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.new_dim(256)
        tracker.unify("d0", "d1")
        assert tracker.get_canonical_dims(["d0", "d1", "d2"]) == ["d0", "d0", "d2"]


class TestGetCanonicalOrder:
    """Tests for _DimTracker.get_canonical_order()."""

    def test_no_unification(self) -> None:
        """Without unification, canonical order is creation order."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(256)
        tracker.new_dim(64)
        assert tracker.get_canonical_order() == ["d0", "d1", "d2"]

    def test_one_unification(self) -> None:
        """Unified dimension is excluded from canonical order."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.new_dim(256)
        tracker.unify("d0", "d1")
        assert tracker.get_canonical_order() == ["d0", "d2"]

    def test_multiple_chains(self) -> None:
        """Multiple unification chains produce correct canonical order.

        d0 <- d2, d1 <- d3: canonical order is [d0, d1].
        """
        tracker = _DimTracker()
        for _ in range(4):
            tracker.new_dim(128)
        tracker.unify("d0", "d2")
        tracker.unify("d1", "d3")
        assert tracker.get_canonical_order() == ["d0", "d1"]

    def test_all_unified(self) -> None:
        """All dimensions unified into one gives single canonical dim."""
        tracker = _DimTracker()
        for _ in range(4):
            tracker.new_dim(128)
        tracker.unify("d0", "d1")
        tracker.unify("d1", "d2")
        tracker.unify("d2", "d3")
        assert tracker.get_canonical_order() == ["d0"]


class TestNewIntermediateName:
    """Tests for _DimTracker.new_intermediate_name()."""

    def test_sequential_names(self) -> None:
        """Intermediate names are generated sequentially."""
        tracker = _DimTracker()
        assert tracker.new_intermediate_name() == "tensor_0"
        assert tracker.new_intermediate_name() == "tensor_1"
        assert tracker.new_intermediate_name() == "tensor_2"

    def test_counter_independent_of_dims(self) -> None:
        """Intermediate name counter is independent of dimension counter."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(256)
        assert tracker.new_intermediate_name() == "tensor_0"
        tracker.new_dim(64)
        assert tracker.new_intermediate_name() == "tensor_1"


class TestRecordOp:
    """Tests for _DimTracker.record_op()."""

    def test_records_single_op(self) -> None:
        """Recording an op appends to the ops list."""
        tracker = _DimTracker()
        tracker.record_op("matmul", ["a", "b"], "c")
        assert len(tracker.ops) == 1
        assert tracker.ops[0].op_name == "matmul"
        assert tracker.ops[0].inputs == ["a", "b"]
        assert tracker.ops[0].output == "c"

    def test_records_multiple_ops(self) -> None:
        """Multiple ops are recorded in order."""
        tracker = _DimTracker()
        tracker.record_op("load", ["a"], "t0")
        tracker.record_op("matmul", ["t0", "t1"], "t2")
        tracker.record_op("store", ["t2"], "output")
        assert len(tracker.ops) == 3
        assert tracker.ops[0].op_name == "load"
        assert tracker.ops[1].op_name == "matmul"
        assert tracker.ops[2].op_name == "store"


class TestRepr:
    """Tests for _DimTracker.__repr__()."""

    def test_repr_no_unification(self) -> None:
        """Repr shows each dim with its size."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(256)
        result = repr(tracker)
        assert "d0=128" in result
        assert "d1=256" in result

    def test_repr_with_unification(self) -> None:
        """Repr shows unified dims with (=canonical) suffix."""
        tracker = _DimTracker()
        tracker.new_dim(128)
        tracker.new_dim(128)
        tracker.unify("d0", "d1")
        result = repr(tracker)
        assert "d0=128" in result
        assert "d1=128 (=d0)" in result
