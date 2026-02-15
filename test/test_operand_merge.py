"""Tests for operand merge analysis and transform.

This module tests the ``OperandMergeTransform`` class, which identifies
adjacent load and operation statements operating on contiguous slices of the
same tensor and merges them into wider operations.

Run with: pytest test/test_operand_merge.py -v
"""

import numpy as np
import pytest
from conftest import make_random_array
from operand_merge_golden import (
    tiled_activation_2x,
    tiled_activation_single,
    tiled_adjacent_4x,
    tiled_adjacent_loads_2x,
    tiled_different_partition_slices,
    tiled_different_source_tensors,
    tiled_matmul_exceeds_n_limit,
    tiled_matmul_m_dim_merge,
    tiled_matmul_m_exceeds_limit,
    tiled_matmul_n_at_limit,
    tiled_matmul_post_reuse_1x2,
    tiled_matmul_post_reuse_1x4,
    tiled_matmul_post_reuse_2x2,
    tiled_no_adjacent_loads,
    tiled_single_subgraph,
    tiled_subscript_loads_2x,
    tiled_tensor_scalar_2x,
    tiled_tensor_tensor_2x,
    tiled_tensor_tensor_diff_ops,
)

import nkigym
from nkigym.ir import Program, Statement, callable_to_ir, ir_to_callable
from nkigym.ops import ALLOC_F32_OP, LOAD_OP, NC_MATMUL_OP, STORE_OP
from nkigym.tiling import generate_tiled_ir
from nkigym.transforms import DataReuseTransform
from nkigym.transforms.operand_merge import MergeOpportunity, OperandMergeTransform
from nkigym.utils.source import callable_to_source, source_to_callable

_merge = OperandMergeTransform()


def _analyze(func):
    """Analyze a callable for merge opportunities via IR round-trip."""
    return _merge.analyze_ir(callable_to_ir(func))


def _transform(func, opp):
    """Apply a merge opportunity to a callable via IR round-trip."""
    program = callable_to_ir(func)
    new_program = _merge.transform_ir(program, opp)
    return ir_to_callable(new_program)


@pytest.fixture
def operand_merge() -> OperandMergeTransform:
    """Fixture providing an OperandMergeTransform instance.

    Returns:
        A reusable transform instance for calling analyze/transform.
    """
    return OperandMergeTransform()


class TestAnalyzeLoadMerge:
    """Tests for load merge opportunity detection.

    Verifies that ``analyze()`` correctly identifies pairs of load statements
    that access contiguous slices of the same tensor on exactly one free
    dimension and can be merged. When a load is consumed as a bare Name
    argument, the transform subscripts those consumers with the original
    slice range to preserve correctness.
    """

    def test_subscript_loads_found(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze finds 1 load opportunity for subscript-consumed loads.

        ``tiled_subscript_loads_2x`` has b[0:128, 0:128] and b[0:128, 128:256]
        consumed via subscript in nc_matmul, so load merging is safe.
        """
        opportunities = _analyze(tiled_subscript_loads_2x)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1

    def test_bare_name_loads_found(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze finds load merging even when loads are consumed as bare Name args.

        ``tiled_adjacent_loads_2x`` has adjacent b loads passed directly to
        nc_matmul as bare Name args. Load merging is now allowed; the
        transform will subscript the bare consumers with the original slices.
        """
        opportunities = _analyze(tiled_adjacent_loads_2x)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1

    def test_no_adjacent_loads(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze returns empty for non-adjacent loads.

        ``tiled_no_adjacent_loads`` has b[0:128, 0:128] and b[0:128, 256:384]
        which have a gap (128:256 missing) and are not adjacent.
        """
        opportunities = _analyze(tiled_no_adjacent_loads)
        assert len(opportunities) == 0

    def test_single_subgraph_no_opportunity(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze returns empty for single-subgraph function.

        ``tiled_single_subgraph`` has only one load per source tensor,
        so no pairs can be formed.
        """
        opportunities = _analyze(tiled_single_subgraph)
        assert len(opportunities) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify the MergeOpportunity fields for a load opportunity.

        Checks op_type, stmt_a < stmt_b, merged_slice covers both original
        slices, and differing_operand_idx points to the free dimension.
        """
        opportunities = _analyze(tiled_subscript_loads_2x)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1
        opp = load_opps[0]

        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "load"
        assert opp.stmt_a < opp.stmt_b
        assert opp.merged_slice == (0, 256)
        assert opp.differing_operand_idx == 1


class TestAnalyzeNoOpportunity:
    """Tests for cases where no merge opportunities should be found.

    Verifies that ``analyze()`` correctly rejects loads from different
    source tensors and loads with mismatched partition dimension slices.
    """

    def test_different_source_tensors(self, operand_merge: OperandMergeTransform) -> None:
        """Verify loads from different tensors do not merge.

        ``tiled_different_source_tensors`` has one load from ``a`` and one
        from ``b``, so they cannot be merged.
        """
        opportunities = _analyze(tiled_different_source_tensors)
        assert len(opportunities) == 0

    def test_different_partition_slices(self, operand_merge: OperandMergeTransform) -> None:
        """Verify loads with different partition slices do not merge.

        ``tiled_different_partition_slices`` has a[0:128, 0:128] and
        a[128:256, 0:128] which differ on the partition dimension (dim 0),
        not the free dimension. They should not be grouped together.
        """
        opportunities = _analyze(tiled_different_partition_slices)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 0


class TestAnalyzeOpMerge:
    """Tests for nc_matmul operation merging detection.

    Verifies that ``analyze()`` finds nc_matmul merge opportunities on
    post-data-reuse functions and respects hardware tile limits.
    """

    def test_nc_matmul_merge_found(self, operand_merge: OperandMergeTransform) -> None:
        """Post-reuse 1x2 should find nc_matmul opportunity.

        ``tiled_matmul_post_reuse_1x2`` has two nc_matmul calls sharing
        tensor_0 (a load) and differing on the b operand (0:128 vs 128:256).
        """
        opps = _analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_nc_matmul_merge_1x4(self, operand_merge: OperandMergeTransform) -> None:
        """Post-reuse 1x4 should find nc_matmul opportunities.

        ``tiled_matmul_post_reuse_1x4`` has 4 nc_matmul calls with shared a
        and adjacent b loads spanning 0:512.
        """
        opps = _analyze(tiled_matmul_post_reuse_1x4)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) >= 1

    def test_nc_matmul_n_limit_rejection(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N > 512 should not produce opportunities exceeding limit.

        ``tiled_matmul_exceeds_n_limit`` has 5 adjacent b loads spanning
        0:640. No single nc_matmul opportunity should have a merged slice
        wider than 512.
        """
        opps = _analyze(tiled_matmul_exceeds_n_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        for opp in op_opps:
            merged_size = opp.merged_slice[1] - opp.merged_slice[0]
            assert merged_size <= 512

    def test_bare_name_loads_produce_load_and_op_opps(self, operand_merge: OperandMergeTransform) -> None:
        """Adjacent loads consumed as bare Name args produce both load and op opportunities.

        ``tiled_adjacent_loads_2x`` has adjacent b loads passed directly to
        nc_matmul. Load merging is now allowed (bare arg guard removed),
        so both load-level and op-level opportunities are found.
        """
        opps = _analyze(tiled_adjacent_loads_2x)
        load_opps = [o for o in opps if o.op_type == "load"]
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(load_opps) == 1
        assert load_opps[0].merged_slice == (0, 256)
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_four_adjacent_produces_nc_matmul_opps(self, operand_merge: OperandMergeTransform) -> None:
        """Four adjacent loads with bare Name args produce nc_matmul opportunities.

        ``tiled_adjacent_4x`` has 4 nc_matmul calls with adjacent b loads.
        Greedy pairing should produce at least 2 nc_matmul merge opportunities.
        """
        opportunities = _analyze(tiled_adjacent_4x)
        op_opps = [o for o in opportunities if o.op_type == "nc_matmul"]
        assert len(op_opps) >= 2
        for opp in op_opps:
            assert opp.stmt_a < opp.stmt_b


class TestTransformSource:
    """Tests for transform() source code correctness.

    Verifies that applying a merge opportunity produces source code
    with widened slices and removed absorbed statements.
    """

    def test_load_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """Load merge should widen the kept load and remove absorbed.

        Uses ``tiled_subscript_loads_2x`` where loads are consumed via subscript.
        After merging b[0:128, 0:128] and b[0:128, 128:256], the kept
        load should become b[0:128, 0:256] and tensor_4 should be removed.
        """
        opps = _analyze(tiled_subscript_loads_2x)
        load_opps = [o for o in opps if o.op_type == "load"]
        assert len(load_opps) == 1
        result = _transform(tiled_subscript_loads_2x, load_opps[0])
        source = callable_to_source(result)
        assert "0:256" in source
        assert "tensor_4" not in source

    def test_nc_matmul_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge should widen load, remove absorbed chain, widen store.

        After merging the two nc_matmul ops in post_reuse_1x2, only one
        nc_matmul call should remain with the b load widened to 0:256.
        """
        opps = _analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        result = _transform(tiled_matmul_post_reuse_1x2, op_opps[0])
        source = callable_to_source(result)
        assert "0:256" in source
        assert source.count("nc_matmul") == 1

    def test_nc_matmul_merge_removes_absorbed_chain(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge should remove the absorbed op's load and store.

        After merging, the absorbed nc_matmul, its dedicated b load, and
        its output store should all be removed from the source.
        """
        opps = _analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_post_reuse_1x2, op_opps[0])
        source = callable_to_source(result)
        assert "tensor_5" not in source
        assert "tensor_4" not in source


class TestLoadMergeSubscripts:
    """Tests for load merge consumer subscripting.

    Verifies that after load merging, bare Name consumers are rewritten
    to Subscript expressions using the original slice ranges.
    """

    def test_load_merge_subscripts_consumers(self, operand_merge: OperandMergeTransform) -> None:
        """After load merging on 2x2 IR, consumers appear as subscript expressions.

        ``tiled_matmul_post_reuse_2x2`` has shared b loads (tensor_1, tensor_4)
        consumed as bare Name args. After load merging tensor_1 + tensor_4
        into tensor_1[0:128, 0:256], all nc_matmul consumers should reference
        tensor_1 via subscript (not bare Name).
        """
        opps = _analyze(tiled_matmul_post_reuse_2x2)
        load_opps = [o for o in opps if o.op_type == "load"]
        assert len(load_opps) >= 1

        func = tiled_matmul_post_reuse_2x2
        for opp in load_opps:
            func = _transform(func, opp)

        source = callable_to_source(func)
        assert "0:256" in source
        assert "tensor_1[" in source
        assert "0:128, 0:128]" in source or "0:128, 128:256]" in source

    def test_load_merge_2x2_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Load merging on 2x2 post-reuse IR preserves numerical correctness."""
        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)

        original = tiled_matmul_post_reuse_2x2(a, b)
        func = tiled_matmul_post_reuse_2x2

        opps = _analyze(func)
        load_opps = [o for o in opps if o.op_type == "load"]
        for opp in load_opps:
            func = _transform(func, opp)

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)


class TestTransformNumerical:
    """Tests for numerical correctness after transform.

    Verifies that applying merge transforms preserves the function's
    numerical output.
    """

    def test_nc_matmul_merge_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge should produce numerically identical results."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)

        opps = _analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_func = _transform(tiled_matmul_post_reuse_1x2, op_opps[0])

        original = tiled_matmul_post_reuse_1x2(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "fixture,a_shape,b_shape",
        [
            (tiled_matmul_post_reuse_1x2, (128, 128), (128, 256)),
            (tiled_matmul_post_reuse_1x4, (128, 128), (128, 512)),
            (tiled_adjacent_loads_2x, (128, 128), (128, 256)),
            (tiled_matmul_post_reuse_2x2, (128, 256), (128, 256)),
        ],
        ids=["post_reuse_1x2", "post_reuse_1x4", "adjacent_loads_2x", "post_reuse_2x2"],
    )
    def test_iterative_merge_numerical(
        self, operand_merge: OperandMergeTransform, fixture: object, a_shape: tuple, b_shape: tuple
    ) -> None:
        """Iteratively applying all opportunities should preserve correctness."""
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)

        original = fixture(a, b)
        func = fixture
        while True:
            opps = _analyze(func)
            if not opps:
                break
            func = _transform(func, opps[0])

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)


class TestAnalyzeMatmulMerge:
    """Tests for nc_matmul merging opportunity detection.

    Covers N-dimension (RHS) and M-dimension (LHS) merging, including
    at-limit and over-limit cases.
    """

    def test_same_lhs_adjacent_rhs(self, operand_merge: OperandMergeTransform) -> None:
        """Two matmuls with same LHS, adjacent RHS should find nc_matmul opportunity.

        ``tiled_matmul_post_reuse_1x2`` has shared tensor_0 (a) and
        b loads at 0:128 and 128:256. Merged N=256.
        """
        opps = _analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].differing_operand_idx == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_n_at_limit_accepted(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=512 (exactly at limit) should produce an opportunity.

        ``tiled_matmul_n_at_limit`` has b[0:128, 0:256] and b[0:128, 256:512].
        """
        opps = _analyze(tiled_matmul_n_at_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 512)

    def test_n_over_limit_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=640 (over limit) should not appear as a single opportunity.

        ``tiled_matmul_exceeds_n_limit`` has 5 adjacent b loads. No single
        nc_matmul opportunity should have merged N > 512.
        """
        opps = _analyze(tiled_matmul_exceeds_n_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        for opp in op_opps:
            assert opp.merged_slice[1] - opp.merged_slice[0] <= 512

    def test_same_rhs_adjacent_lhs_m_merge(self, operand_merge: OperandMergeTransform) -> None:
        """Two matmuls with same RHS, adjacent LHS should merge on M dimension.

        ``tiled_matmul_m_dim_merge`` has a[0:64,...] and a[64:128,...] with
        shared b. Merged M = 128, within the M=128 limit.
        """
        opps = _analyze(tiled_matmul_m_dim_merge)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].differing_operand_idx == 0
        assert op_opps[0].merged_slice == (0, 128)

    def test_m_over_limit_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged M=192 (over M=128 limit) should not produce an opportunity.

        ``tiled_matmul_m_exceeds_limit`` has a[0:128,...] and a[128:192,...]
        with shared b. Merged M = 192 > 128.
        """
        opps = _analyze(tiled_matmul_m_exceeds_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 0


class TestTransformMatmulMerge:
    """Tests for nc_matmul transform source code correctness.

    Verifies that applying an nc_matmul merge opportunity produces
    correct source with widened slices, one nc_matmul call, and the
    absorbed chain removed.
    """

    def test_same_lhs_adjacent_rhs_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on post_reuse_1x2 should produce one nc_matmul with b[0:128, 0:256].

        After transform: tensor_1 widened to b[0:128, 0:256], one nc_matmul
        remains, output store widened to output[0:128, 0:256].
        """
        opps = _analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_post_reuse_1x2, op_opps[0])
        source = callable_to_source(result)
        assert source.count("nc_matmul") == 1
        assert "0:256" in source
        assert "tensor_4" not in source
        assert "tensor_5" not in source

    def test_n_at_limit_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on N=512 at-limit should produce one nc_matmul with b[0:128, 0:512]."""
        opps = _analyze(tiled_matmul_n_at_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_n_at_limit, op_opps[0])
        source = callable_to_source(result)
        assert source.count("nkigym.nc_matmul(") == 1
        assert "0:512" in source

    def test_m_dim_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on M-dim merge should produce one nc_matmul call.

        After transform: tensor_0 widened to a[0:128, 0:128], one nc_matmul
        remains, output store widened to output[0:128, 0:128].
        """
        opps = _analyze(tiled_matmul_m_dim_merge)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_m_dim_merge, op_opps[0])
        source = callable_to_source(result)
        assert source.count("nkigym.nc_matmul(") == 1
        assert "tensor_3" not in source
        assert "tensor_4" not in source


class TestMatmulMergeNumerical:
    """Tests for nc_matmul merge numerical correctness.

    Verifies that applying nc_matmul merge transforms preserves
    numerical output across different dimension configurations.
    """

    @pytest.mark.parametrize(
        "fixture,a_shape,b_shape",
        [
            (tiled_matmul_post_reuse_1x2, (128, 128), (128, 256)),
            (tiled_matmul_n_at_limit, (128, 128), (128, 512)),
            (tiled_matmul_m_dim_merge, (128, 128), (128, 128)),
        ],
        ids=["same_lhs_adjacent_rhs", "n_at_limit", "m_dim_merge"],
    )
    def test_single_merge_numerical(
        self, operand_merge: OperandMergeTransform, fixture: object, a_shape: tuple, b_shape: tuple
    ) -> None:
        """nc_matmul merge should produce numerically identical results."""
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)

        opps = _analyze(fixture)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_func = _transform(fixture, op_opps[0])

        original = fixture(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)

    def test_iterative_1x4_full_merge(self, operand_merge: OperandMergeTransform) -> None:
        """Iteratively merging 1x4 post-reuse from 4 to 1 nc_matmul call.

        After all opportunities are applied, there should be a single
        nc_matmul with N=512 and the result should match the original.
        """
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 512), seed=43)

        original = tiled_matmul_post_reuse_1x4(a, b)
        func = tiled_matmul_post_reuse_1x4
        while True:
            opps = _analyze(func)
            if not opps:
                break
            func = _transform(func, opps[0])

        source = callable_to_source(func)
        assert source.count("nc_matmul") == 1
        assert "0:512" in source

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)


class TestAnalyzeTensorTensorMerge:
    """Tests for tensor_tensor operation merging detection.

    Verifies that ``analyze()`` finds tensor_tensor merge opportunities
    when two calls share one operand and differ on adjacent slices of the
    other, and rejects pairs with different op kwargs.
    """

    def test_same_op_adjacent_first_arg(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_tensor with same op=np.add, differing first arg.

        ``tiled_tensor_tensor_2x`` has tensor_0=a[0:128,0:128] and
        tensor_3=a[0:128,128:256] with shared tensor_1=b[0:128,0:128].
        """
        opps = _analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        assert len(tt_opps) == 1
        assert tt_opps[0].differing_operand_idx == 0
        assert tt_opps[0].merged_slice == (0, 256)

    def test_different_op_kwargs_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_tensor with different op kwargs should not merge.

        ``tiled_tensor_tensor_diff_ops`` has op=np.add vs op=np.multiply.
        """
        opps = _analyze(tiled_tensor_tensor_diff_ops)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        assert len(tt_opps) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for tensor_tensor."""
        opps = _analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        opp = tt_opps[0]
        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "tensor_tensor"
        assert opp.stmt_a < opp.stmt_b
        assert "tensor_tensor" in opp.description


class TestAnalyzeActivationMerge:
    """Tests for activation operation merging detection.

    Verifies that ``analyze()`` finds activation merge opportunities
    when two calls share the same op kwarg and differ on adjacent input
    slices, and rejects single-op cases.
    """

    def test_same_op_adjacent_input(self, operand_merge: OperandMergeTransform) -> None:
        """Two activation with same op=np.tanh, adjacent input slices.

        ``tiled_activation_2x`` has tensor_0=a[0:128,0:128] and
        tensor_2=a[0:128,128:256].
        """
        opps = _analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        assert len(act_opps) == 1
        assert act_opps[0].differing_operand_idx == 0
        assert act_opps[0].merged_slice == (0, 256)

    def test_single_op_no_opportunity(self, operand_merge: OperandMergeTransform) -> None:
        """Single activation op should produce no opportunities.

        ``tiled_activation_single`` has only one activation call.
        """
        opps = _analyze(tiled_activation_single)
        assert len(opps) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for activation."""
        opps = _analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        opp = act_opps[0]
        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "activation"
        assert opp.stmt_a < opp.stmt_b


class TestAnalyzeTensorScalarMerge:
    """Tests for tensor_scalar operation merging detection.

    Verifies that ``analyze()`` finds tensor_scalar merge opportunities
    when two calls share the same keyword args and differ on adjacent
    input slices.
    """

    def test_same_kwargs_adjacent_input(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_scalar with same op0/operand0, adjacent input slices.

        ``tiled_tensor_scalar_2x`` has tensor_0=a[0:128,0:128] and
        tensor_2=a[0:128,128:256].
        """
        opps = _analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        assert len(ts_opps) == 1
        assert ts_opps[0].differing_operand_idx == 0
        assert ts_opps[0].merged_slice == (0, 256)

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for tensor_scalar."""
        opps = _analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        opp = ts_opps[0]
        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "tensor_scalar"
        assert opp.stmt_a < opp.stmt_b
        assert "tensor_scalar" in opp.description


class TestTransformElementwise:
    """Tests for elementwise op transform source code correctness.

    Verifies that applying a merge opportunity on tensor_tensor,
    activation, and tensor_scalar ops produces correct source with widened
    slices and removed absorbed statements. These ops are not in the
    runtime, so only source-level verification is possible.
    """

    def test_tensor_tensor_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_tensor merge should produce one call with widened load.

        After transform: tensor_0 widened to a[0:128, 0:256], single
        tensor_tensor call, absorbed chain (tensor_3, tensor_5, store) removed.
        """
        opps = _analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        result = _transform(tiled_tensor_tensor_2x, tt_opps[0])
        source = callable_to_source(result)
        assert source.count("nkigym.tensor_tensor(") == 1
        assert "0:256" in source
        assert "tensor_3 = " not in source
        assert "tensor_5 = " not in source

    def test_activation_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """activation merge should produce one call with widened load.

        After transform: tensor_0 widened to a[0:128, 0:256], single
        activation call, absorbed chain (tensor_2, tensor_3, store) removed.
        """
        opps = _analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        result = _transform(tiled_activation_2x, act_opps[0])
        source = callable_to_source(result)
        assert source.count("nkigym.activation(") == 1
        assert "0:256" in source
        assert "tensor_2 = " not in source
        assert "tensor_3 = " not in source

    def test_tensor_scalar_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_scalar merge should produce one call with widened load.

        After transform: tensor_0 widened to a[0:128, 0:256], single
        tensor_scalar call, absorbed chain (tensor_2, tensor_3, store) removed.
        """
        opps = _analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        result = _transform(tiled_tensor_scalar_2x, ts_opps[0])
        source = callable_to_source(result)
        assert source.count("nkigym.tensor_scalar(") == 1
        assert "0:256" in source
        assert "tensor_2 = " not in source
        assert "tensor_3 = " not in source

    def test_tensor_tensor_preserves_kwargs(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_tensor merge should preserve the op= keyword argument."""
        opps = _analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        result = _transform(tiled_tensor_tensor_2x, tt_opps[0])
        source = callable_to_source(result)
        assert "op=np.add" in source

    def test_activation_preserves_kwargs(self, operand_merge: OperandMergeTransform) -> None:
        """activation merge should preserve the op= keyword argument."""
        opps = _analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        result = _transform(tiled_activation_2x, act_opps[0])
        source = callable_to_source(result)
        assert "op=np.tanh" in source

    def test_tensor_scalar_preserves_kwargs(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_scalar merge should preserve op0= and operand0= kwargs."""
        opps = _analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        result = _transform(tiled_tensor_scalar_2x, ts_opps[0])
        source = callable_to_source(result)
        assert "op0=np.multiply" in source
        assert "operand0=2.0" in source


class TestE2EPipeline:
    """End-to-end tests: tiling -> data_reuse -> operand_merge.

    Verifies that the full transform pipeline preserves numerical
    correctness when applied to a tiled function.
    """

    @pytest.mark.parametrize(
        "a_shape,b_shape", [((128, 128), (128, 256)), ((128, 256), (128, 256))], ids=["1x2", "2x2"]
    )
    def test_full_pipeline_matmul(self, operand_merge: OperandMergeTransform, a_shape: tuple, b_shape: tuple) -> None:
        """Full pipeline: generate_tiled_function -> DataReuseTransform -> OperandMergeTransform."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        program = generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, output_dtype=a.dtype)

        reuse = DataReuseTransform()
        for group in reuse.analyze_ir(program):
            program = reuse.transform_ir(program, group)

        merge = OperandMergeTransform()
        while True:
            opps = merge.analyze_ir(program)
            if not opps:
                break
            program = merge.transform_ir(program, opps[0])

        result = ir_to_callable(program)(a, b)
        np.testing.assert_allclose(expected, result, rtol=1e-5, atol=1e-5)


class TestOperandMergeIRDirect:
    """Tests for OperandMergeTransform.analyze_ir/transform_ir on program tuples directly."""

    def test_analyze_ir_finds_opportunities(self) -> None:
        """Verify analyze_ir returns opportunities from a program tuple."""
        program = callable_to_ir(tiled_matmul_post_reuse_1x2)
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        assert len(opps) > 0
        for opp in opps:
            assert isinstance(opp, MergeOpportunity)

    def test_transform_ir_returns_valid_program(self) -> None:
        """Verify transform_ir returns a valid program with fewer statements."""
        program = callable_to_ir(tiled_matmul_post_reuse_1x2)
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        assert len(opps) > 0

        new_program = transform.transform_ir(program, opps[0])
        assert new_program.name == program.name
        assert new_program.params == program.params
        assert new_program.return_var == program.return_var
        assert len(new_program.stmts) < len(program.stmts)

        func = ir_to_callable(new_program)
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = tiled_matmul_post_reuse_1x2(a, b)
        actual = func(a, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


class TestCheckAdjacentSlicesDifferentLengths:
    """Tests for OperandMergeTransform._check_adjacent_slices with mismatched slice tuple lengths."""

    def test_different_lengths_returns_none(self) -> None:
        """Slice tuples with different numbers of dimensions cannot be adjacent."""
        slices_a = ((0, 128),)
        slices_b = ((0, 128), (0, 128))
        assert OperandMergeTransform._check_adjacent_slices(slices_a, slices_b) is None

    def test_same_length_adjacent(self) -> None:
        """Same-length slice tuples with one adjacent dimension succeed."""
        slices_a = ((0, 128), (0, 128))
        slices_b = ((0, 128), (128, 256))
        result = OperandMergeTransform._check_adjacent_slices(slices_a, slices_b)
        assert result is not None
        dim, merged = result
        assert dim == 1
        assert merged == (0, 256)

    def test_two_differing_dims_returns_none(self) -> None:
        """Slices differing on two dimensions cannot be merged."""
        slices_a = ((0, 128), (0, 128))
        slices_b = ((128, 256), (128, 256))
        assert OperandMergeTransform._check_adjacent_slices(slices_a, slices_b) is None

    def test_non_adjacent_gap_returns_none(self) -> None:
        """Slices with a gap between them cannot be merged."""
        slices_a = ((0, 128), (0, 128))
        slices_b = ((0, 128), (256, 384))
        assert OperandMergeTransform._check_adjacent_slices(slices_a, slices_b) is None

    def test_reverse_adjacency(self) -> None:
        """Slices where B ends where A starts are also adjacent."""
        slices_a = ((0, 128), (128, 256))
        slices_b = ((0, 128), (0, 128))
        result = OperandMergeTransform._check_adjacent_slices(slices_a, slices_b)
        assert result is not None
        dim, merged = result
        assert dim == 1
        assert merged == (0, 256)


class TestCheckDependencySafe:
    """Tests for OperandMergeTransform._check_dependency_safe returning False."""

    def test_safe_when_no_intervening_usage(self) -> None:
        """Merging is safe when the absorbed variable has no usage between the two statements."""
        var_usage = {"tensor_0": [0, 5]}
        assert OperandMergeTransform._check_dependency_safe(0, 5, "tensor_0", var_usage) is True

    def test_unsafe_when_intervening_usage(self) -> None:
        """Merging is unsafe when the absorbed variable is used between idx_lo and idx_hi."""
        var_usage = {"tensor_0": [0, 3, 5]}
        assert OperandMergeTransform._check_dependency_safe(0, 5, "tensor_0", var_usage) is False

    def test_safe_when_var_not_in_usage(self) -> None:
        """Merging is safe when the absorbed variable is not in the usage index."""
        var_usage: dict[str, list[int]] = {}
        assert OperandMergeTransform._check_dependency_safe(0, 5, "tensor_0", var_usage) is True

    def test_usage_at_idx_lo_is_safe(self) -> None:
        """Usage at exactly idx_lo is not intervening (it is the defining statement)."""
        var_usage = {"tensor_0": [0, 5]}
        assert OperandMergeTransform._check_dependency_safe(0, 5, "tensor_0", var_usage) is True

    def test_usage_at_idx_hi_is_safe(self) -> None:
        """Usage at exactly idx_hi is not intervening (it is the absorbed statement)."""
        var_usage = {"tensor_0": [0, 5]}
        assert OperandMergeTransform._check_dependency_safe(0, 5, "tensor_0", var_usage) is True

    def test_dependency_blocks_compute_merge(self) -> None:
        """End-to-end: dependency between statements blocks compute merge.

        Build a program where tensor_2 (from first matmul) is used as input
        to the second matmul, so absorbing the first matmul is unsafe.
        """
        stmts: tuple[Statement, ...] = (
            Statement(ALLOC_F32_OP, (("output", ((0, 128), (0, 256))),), True),
            Statement(LOAD_OP, (("a", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((0, 128), (0, 128))), ("tensor_1", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("tensor_0", ()), ("tensor_1", ()), ("tensor_2", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((0, 128), (128, 256))), ("tensor_3", ((0, 128), (0, 128)))), True),
            Statement(
                NC_MATMUL_OP,
                (("tensor_2", ((0, 128), (0, 128))), ("tensor_3", ()), ("tensor_4", ((0, 128), (0, 128)))),
                True,
            ),
            Statement(STORE_OP, (("tensor_2", ((0, 128), (0, 128))), ("output", ((0, 128), (0, 128)))), True),
            Statement(STORE_OP, (("tensor_4", ((0, 128), (0, 128))), ("output", ((0, 128), (128, 256)))), True),
        )
        program = Program("dep_fn", ("a", "b"), stmts, "output", "def dep_fn(a, b):")
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        matmul_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(matmul_opps) == 0


class TestHasAccumulationBlocking:
    """Tests for OperandMergeTransform._has_accumulation blocking a merge."""

    def test_no_accumulation(self) -> None:
        """Variable written once by compute is not an accumulation."""
        stmts: tuple[Statement, ...] = (
            Statement(LOAD_OP, (("a", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((0, 128), (0, 128))), ("tensor_1", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("tensor_0", ()), ("tensor_1", ()), ("tensor_2", ((0, 128), (0, 128)))), True),
        )
        compute_vars = {"tensor_2"}
        assert OperandMergeTransform._has_accumulation("tensor_2", stmts, compute_vars) is False

    def test_has_accumulation(self) -> None:
        """Variable written twice by compute (reduction tiling) is an accumulation."""
        stmts: tuple[Statement, ...] = (
            Statement(LOAD_OP, (("a", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((0, 128), (0, 128))), ("tensor_1", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("tensor_0", ()), ("tensor_1", ()), ("tensor_2", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("a", ((128, 256), (0, 128))), ("tensor_3", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((128, 256), (0, 128))), ("tensor_4", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("tensor_3", ()), ("tensor_4", ()), ("tensor_2", ((0, 128), (0, 128)))), False),
        )
        compute_vars = {"tensor_2"}
        assert OperandMergeTransform._has_accumulation("tensor_2", stmts, compute_vars) is True

    def test_accumulation_blocks_merge(self) -> None:
        """Accumulating variables block compute merging at the analyze level.

        Build a reduction-tiled program with two accumulating matmuls on
        adjacent output tiles. The accumulation should prevent merging.
        """
        stmts: tuple[Statement, ...] = (
            Statement(ALLOC_F32_OP, (("output", ((0, 128), (0, 256))),), True),
            Statement(LOAD_OP, (("a", ((0, 128), (0, 128))), ("t0", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((0, 128), (0, 128))), ("t1", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("t0", ()), ("t1", ()), ("t2", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("a", ((128, 256), (0, 128))), ("t3", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((128, 256), (0, 128))), ("t4", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("t3", ()), ("t4", ()), ("t2", ((0, 128), (0, 128)))), False),
            Statement(LOAD_OP, (("a", ((0, 128), (0, 128))), ("t5", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((0, 128), (128, 256))), ("t6", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("t5", ()), ("t6", ()), ("t7", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("a", ((128, 256), (0, 128))), ("t8", ((0, 128), (0, 128)))), True),
            Statement(LOAD_OP, (("b", ((128, 256), (128, 256))), ("t9", ((0, 128), (0, 128)))), True),
            Statement(NC_MATMUL_OP, (("t8", ()), ("t9", ()), ("t7", ((0, 128), (0, 128)))), False),
            Statement(STORE_OP, (("t2", ((0, 128), (0, 128))), ("output", ((0, 128), (0, 128)))), True),
            Statement(STORE_OP, (("t7", ((0, 128), (0, 128))), ("output", ((0, 128), (128, 256)))), True),
        )
        program = Program("accum_fn", ("a", "b"), stmts, "output", "def accum_fn(a, b):")
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        matmul_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(matmul_opps) == 0


class TestThreeWayMergeChain:
    """Tests for three-way iterative merge chains.

    Verifies that after merging a pair, re-analysis can find a second
    opportunity to merge the result with a third statement.
    """

    def test_iterative_merge_three_loads(self) -> None:
        """Three adjacent loads can be merged in two iterations.

        Build a program with 3 adjacent b loads at [0:128], [128:256], [256:384].
        First iteration merges two into [0:256], second merges with [256:384]
        to get [0:384]. Uses source-level round-trip since nc_matmul N=384 < 512.
        """
        source = (
            "def three_load(a, b):\n"
            "    output = nkigym.ndarray((128, 384), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:128]\n"
            "    tensor_2 = nkigym.nc_matmul(tensor_0, tensor_1)\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
            "    tensor_3 = b[0:128, 128:256]\n"
            "    tensor_4 = nkigym.nc_matmul(tensor_0, tensor_3)\n"
            "    output[0:128, 128:256] = tensor_4[0:128, 0:128]\n"
            "    tensor_5 = b[0:128, 256:384]\n"
            "    tensor_6 = nkigym.nc_matmul(tensor_0, tensor_5)\n"
            "    output[0:128, 256:384] = tensor_6[0:128, 0:128]\n"
            "    return output\n"
        )

        original_func = source_to_callable(source, "three_load")
        program = callable_to_ir(original_func)
        transform = OperandMergeTransform()

        iteration_count = 0
        while True:
            opps = transform.analyze_ir(program)
            if not opps:
                break
            program = transform.transform_ir(program, opps[0])
            iteration_count += 1

        assert iteration_count >= 2

        func = ir_to_callable(program)
        source_final = callable_to_source(func)
        assert source_final.count("nc_matmul") == 1

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 384), seed=43)
        expected = original_func(a, b)
        actual = func(a, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
