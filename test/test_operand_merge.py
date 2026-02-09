"""Tests for operand merge analysis and transform.

This module tests the ``OperandMergeTransform`` class, which identifies
adjacent load and operation statements operating on contiguous slices of the
same tensor and merges them into wider operations.

Test classes:
- ``TestAnalyzeLoadMerge``: Tests for load merge opportunity detection.
- ``TestAnalyzeNoOpportunity``: Tests for cases with no merge opportunities.
- ``TestAnalyzeOpMerge``: Tests for nc_matmul operation merging detection.
- ``TestTransformSource``: Tests for transform() source code correctness.
- ``TestLoadMergeSubscripts``: Tests for load merge consumer subscripting.
- ``TestTransformNumerical``: Tests for numerical correctness after transform.
- ``TestAnalyzeMatmulMerge``: Tests for nc_matmul dimension merging.
- ``TestTransformMatmulMerge``: Tests for nc_matmul transform source.
- ``TestMatmulMergeNumerical``: Tests for nc_matmul numerical correctness.
- ``TestMatmulHardwareLimits``: Tests for nc_matmul hardware limit enforcement.
- ``TestAnalyzeTensorTensorMerge``: Tests for tensor_tensor merging.
- ``TestAnalyzeActivationMerge``: Tests for activation merging.
- ``TestAnalyzeTensorScalarMerge``: Tests for tensor_scalar merging.
- ``TestTransformElementwise``: Tests for elementwise op transform source.
- ``TestE2EPipeline``: End-to-end tests with full tiling/reuse/merge pipeline.

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
from nkigym.tiling import generate_tiled_function
from nkigym.transforms import DataReuseTransform
from nkigym.transforms.operand_merge import MergeOpportunity, OperandMergeTransform
from nkigym.utils.source import get_source


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
        opportunities = operand_merge.analyze(tiled_subscript_loads_2x)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1

    def test_bare_name_loads_found(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze finds load merging even when loads are consumed as bare Name args.

        ``tiled_adjacent_loads_2x`` has adjacent b loads passed directly to
        nc_matmul as bare Name args. Load merging is now allowed; the
        transform will subscript the bare consumers with the original slices.
        """
        opportunities = operand_merge.analyze(tiled_adjacent_loads_2x)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1

    def test_no_adjacent_loads(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze returns empty for non-adjacent loads.

        ``tiled_no_adjacent_loads`` has b[0:128, 0:128] and b[0:128, 256:384]
        which have a gap (128:256 missing) and are not adjacent.
        """
        opportunities = operand_merge.analyze(tiled_no_adjacent_loads)
        assert len(opportunities) == 0

    def test_single_subgraph_no_opportunity(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze returns empty for single-subgraph function.

        ``tiled_single_subgraph`` has only one load per source tensor,
        so no pairs can be formed.
        """
        opportunities = operand_merge.analyze(tiled_single_subgraph)
        assert len(opportunities) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify the MergeOpportunity fields for a load opportunity.

        Checks op_type, stmt_a < stmt_b, merged_slice covers both original
        slices, and differing_operand_idx points to the free dimension.
        """
        opportunities = operand_merge.analyze(tiled_subscript_loads_2x)
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
        opportunities = operand_merge.analyze(tiled_different_source_tensors)
        assert len(opportunities) == 0

    def test_different_partition_slices(self, operand_merge: OperandMergeTransform) -> None:
        """Verify loads with different partition slices do not merge.

        ``tiled_different_partition_slices`` has a[0:128, 0:128] and
        a[128:256, 0:128] which differ on the partition dimension (dim 0),
        not the free dimension. They should not be grouped together.
        """
        opportunities = operand_merge.analyze(tiled_different_partition_slices)
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
        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_nc_matmul_merge_1x4(self, operand_merge: OperandMergeTransform) -> None:
        """Post-reuse 1x4 should find nc_matmul opportunities.

        ``tiled_matmul_post_reuse_1x4`` has 4 nc_matmul calls with shared a
        and adjacent b loads spanning 0:512.
        """
        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x4)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) >= 1

    def test_nc_matmul_n_limit_rejection(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N > 512 should not produce opportunities exceeding limit.

        ``tiled_matmul_exceeds_n_limit`` has 5 adjacent b loads spanning
        0:640. No single nc_matmul opportunity should have a merged slice
        wider than 512.
        """
        opps = operand_merge.analyze(tiled_matmul_exceeds_n_limit)
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
        opps = operand_merge.analyze(tiled_adjacent_loads_2x)
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
        opportunities = operand_merge.analyze(tiled_adjacent_4x)
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
        opps = operand_merge.analyze(tiled_subscript_loads_2x)
        load_opps = [o for o in opps if o.op_type == "load"]
        assert len(load_opps) == 1
        result = operand_merge.transform(tiled_subscript_loads_2x, load_opps[0])
        source = get_source(result)
        assert "0:256" in source
        assert "tensor_4" not in source

    def test_nc_matmul_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge should widen load, remove absorbed chain, widen store.

        After merging the two nc_matmul ops in post_reuse_1x2, only one
        nc_matmul call should remain with the b load widened to 0:256.
        """
        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        result = operand_merge.transform(tiled_matmul_post_reuse_1x2, op_opps[0])
        source = get_source(result)
        assert "0:256" in source
        assert source.count("nc_matmul") == 1

    def test_nc_matmul_merge_removes_absorbed_chain(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge should remove the absorbed op's load and store.

        After merging, the absorbed nc_matmul, its dedicated b load, and
        its output store should all be removed from the source.
        """
        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = operand_merge.transform(tiled_matmul_post_reuse_1x2, op_opps[0])
        source = get_source(result)
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
        opps = operand_merge.analyze(tiled_matmul_post_reuse_2x2)
        load_opps = [o for o in opps if o.op_type == "load"]
        assert len(load_opps) >= 1

        func = tiled_matmul_post_reuse_2x2
        for opp in load_opps:
            func = operand_merge.transform(func, opp)

        source = get_source(func)
        assert "0:256" in source
        assert "tensor_1[" in source
        assert "0:128, 0:128]" in source or "0:128, 128:256]" in source

    def test_load_merge_2x2_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Load merging on 2x2 post-reuse IR preserves numerical correctness."""
        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)

        original = tiled_matmul_post_reuse_2x2(a, b)
        func = tiled_matmul_post_reuse_2x2

        opps = operand_merge.analyze(func)
        load_opps = [o for o in opps if o.op_type == "load"]
        for opp in load_opps:
            func = operand_merge.transform(func, opp)

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed)


class TestTransformNumerical:
    """Tests for numerical correctness after transform.

    Verifies that applying merge transforms preserves the function's
    numerical output.
    """

    def test_nc_matmul_merge_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge should produce numerically identical results."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)

        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_func = operand_merge.transform(tiled_matmul_post_reuse_1x2, op_opps[0])

        original = tiled_matmul_post_reuse_1x2(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed)

    def test_iterative_post_reuse_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Iteratively applying all opportunities on post-reuse should preserve correctness."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)

        original = tiled_matmul_post_reuse_1x2(a, b)
        func = tiled_matmul_post_reuse_1x2
        while True:
            opps = operand_merge.analyze(func)
            if not opps:
                break
            func = operand_merge.transform(func, opps[0])

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed)

    def test_iterative_post_reuse_1x4_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Iteratively merging 1x4 post-reuse should preserve correctness."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 512), seed=43)

        original = tiled_matmul_post_reuse_1x4(a, b)
        func = tiled_matmul_post_reuse_1x4
        while True:
            opps = operand_merge.analyze(func)
            if not opps:
                break
            func = operand_merge.transform(func, opps[0])

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed)

    def test_nc_matmul_merge_adjacent_loads_2x_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Op-level merge of adjacent_loads_2x should preserve correctness.

        ``tiled_adjacent_loads_2x`` produces nc_matmul opportunities (not load).
        Iteratively applying all ops should yield numerically identical results.
        """
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)

        original = tiled_adjacent_loads_2x(a, b)
        func = tiled_adjacent_loads_2x
        while True:
            opps = operand_merge.analyze(func)
            if not opps:
                break
            func = operand_merge.transform(func, opps[0])

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed)

    def test_iterative_post_reuse_2x2_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Iteratively merging 2x2 post-reuse should preserve correctness.

        The 2x2 case has shared loads on both sides. Load merging first
        subscripts the bare consumers, then op merging merges the matmul
        pairs. All opportunities are applied iteratively until none remain.
        """
        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)

        original = tiled_matmul_post_reuse_2x2(a, b)
        func = tiled_matmul_post_reuse_2x2
        while True:
            opps = operand_merge.analyze(func)
            if not opps:
                break
            func = operand_merge.transform(func, opps[0])

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed)


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
        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].differing_operand_idx == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_n_at_limit_accepted(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=512 (exactly at limit) should produce an opportunity.

        ``tiled_matmul_n_at_limit`` has b[0:128, 0:256] and b[0:128, 256:512].
        """
        opps = operand_merge.analyze(tiled_matmul_n_at_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 512)

    def test_n_over_limit_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=640 (over limit) should not appear as a single opportunity.

        ``tiled_matmul_exceeds_n_limit`` has 5 adjacent b loads. No single
        nc_matmul opportunity should have merged N > 512.
        """
        opps = operand_merge.analyze(tiled_matmul_exceeds_n_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        for opp in op_opps:
            assert opp.merged_slice[1] - opp.merged_slice[0] <= 512

    def test_same_rhs_adjacent_lhs_m_merge(self, operand_merge: OperandMergeTransform) -> None:
        """Two matmuls with same RHS, adjacent LHS should merge on M dimension.

        ``tiled_matmul_m_dim_merge`` has a[0:64,...] and a[64:128,...] with
        shared b. Merged M = 128, within the M=128 limit.
        """
        opps = operand_merge.analyze(tiled_matmul_m_dim_merge)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].differing_operand_idx == 0
        assert op_opps[0].merged_slice == (0, 128)

    def test_m_over_limit_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged M=192 (over M=128 limit) should not produce an opportunity.

        ``tiled_matmul_m_exceeds_limit`` has a[0:128,...] and a[128:192,...]
        with shared b. Merged M = 192 > 128.
        """
        opps = operand_merge.analyze(tiled_matmul_m_exceeds_limit)
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
        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = operand_merge.transform(tiled_matmul_post_reuse_1x2, op_opps[0])
        source = get_source(result)
        assert source.count("nc_matmul") == 1
        assert "0:256" in source
        assert "tensor_4" not in source
        assert "tensor_5" not in source

    def test_n_at_limit_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on N=512 at-limit should produce one nc_matmul with b[0:128, 0:512]."""
        opps = operand_merge.analyze(tiled_matmul_n_at_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = operand_merge.transform(tiled_matmul_n_at_limit, op_opps[0])
        source = get_source(result)
        assert source.count("nkigym.nc_matmul(") == 1
        assert "0:512" in source

    def test_m_dim_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on M-dim merge should produce one nc_matmul call.

        After transform: tensor_0 widened to a[0:128, 0:128], one nc_matmul
        remains, output store widened to output[0:128, 0:128].
        """
        opps = operand_merge.analyze(tiled_matmul_m_dim_merge)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = operand_merge.transform(tiled_matmul_m_dim_merge, op_opps[0])
        source = get_source(result)
        assert source.count("nkigym.nc_matmul(") == 1
        assert "tensor_3" not in source
        assert "tensor_4" not in source


class TestMatmulMergeNumerical:
    """Tests for nc_matmul merge numerical correctness.

    Verifies that applying nc_matmul merge transforms preserves
    numerical output across different dimension configurations.
    """

    def test_same_lhs_adjacent_rhs_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """nc_matmul merge on N dimension should produce identical results."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)

        opps = operand_merge.analyze(tiled_matmul_post_reuse_1x2)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_func = operand_merge.transform(tiled_matmul_post_reuse_1x2, op_opps[0])

        original = tiled_matmul_post_reuse_1x2(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed)

    def test_n_at_limit_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """N=512 at-limit merge should produce identical results."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 512), seed=43)

        opps = operand_merge.analyze(tiled_matmul_n_at_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_func = operand_merge.transform(tiled_matmul_n_at_limit, op_opps[0])

        original = tiled_matmul_n_at_limit(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed)

    def test_m_dim_merge_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """M-dimension merge should produce identical results."""
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 128), seed=43)

        opps = operand_merge.analyze(tiled_matmul_m_dim_merge)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_func = operand_merge.transform(tiled_matmul_m_dim_merge, op_opps[0])

        original = tiled_matmul_m_dim_merge(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed)

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
            opps = operand_merge.analyze(func)
            if not opps:
                break
            func = operand_merge.transform(func, opps[0])

        source = get_source(func)
        assert source.count("nc_matmul") == 1
        assert "0:512" in source

        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed)


class TestMatmulHardwareLimits:
    """Tests for nc_matmul hardware limit enforcement.

    Verifies that ``analyze()`` respects N=512 and M=128 limits and
    does not produce opportunities that exceed them.
    """

    def test_n_512_accepted(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=512 is exactly at the limit and should be accepted."""
        opps = operand_merge.analyze(tiled_matmul_n_at_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice[1] - op_opps[0].merged_slice[0] == 512

    def test_n_640_all_within_limit(self, operand_merge: OperandMergeTransform) -> None:
        """5 adjacent loads (N=640 total): every opportunity must have N <= 512."""
        opps = operand_merge.analyze(tiled_matmul_exceeds_n_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        for opp in op_opps:
            merged_n = opp.merged_slice[1] - opp.merged_slice[0]
            assert merged_n <= 512, f"Merged N={merged_n} exceeds limit 512"

    def test_m_128_accepted(self, operand_merge: OperandMergeTransform) -> None:
        """Merged M=128 is exactly at the limit and should be accepted."""
        opps = operand_merge.analyze(tiled_matmul_m_dim_merge)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice[1] - op_opps[0].merged_slice[0] == 128

    def test_m_192_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged M=192 exceeds M=128 limit and should not produce an opportunity."""
        opps = operand_merge.analyze(tiled_matmul_m_exceeds_limit)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 0


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
        opps = operand_merge.analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        assert len(tt_opps) == 1
        assert tt_opps[0].differing_operand_idx == 0
        assert tt_opps[0].merged_slice == (0, 256)

    def test_different_op_kwargs_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_tensor with different op kwargs should not merge.

        ``tiled_tensor_tensor_diff_ops`` has op=np.add vs op=np.multiply.
        """
        opps = operand_merge.analyze(tiled_tensor_tensor_diff_ops)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        assert len(tt_opps) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for tensor_tensor."""
        opps = operand_merge.analyze(tiled_tensor_tensor_2x)
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
        opps = operand_merge.analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        assert len(act_opps) == 1
        assert act_opps[0].differing_operand_idx == 0
        assert act_opps[0].merged_slice == (0, 256)

    def test_single_op_no_opportunity(self, operand_merge: OperandMergeTransform) -> None:
        """Single activation op should produce no opportunities.

        ``tiled_activation_single`` has only one activation call.
        """
        opps = operand_merge.analyze(tiled_activation_single)
        assert len(opps) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for activation."""
        opps = operand_merge.analyze(tiled_activation_2x)
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
        opps = operand_merge.analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        assert len(ts_opps) == 1
        assert ts_opps[0].differing_operand_idx == 0
        assert ts_opps[0].merged_slice == (0, 256)

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for tensor_scalar."""
        opps = operand_merge.analyze(tiled_tensor_scalar_2x)
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
        opps = operand_merge.analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        result = operand_merge.transform(tiled_tensor_tensor_2x, tt_opps[0])
        source = get_source(result)
        assert source.count("nkigym.tensor_tensor(") == 1
        assert "0:256" in source
        assert "tensor_3 = " not in source
        assert "tensor_5 = " not in source

    def test_activation_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """activation merge should produce one call with widened load.

        After transform: tensor_0 widened to a[0:128, 0:256], single
        activation call, absorbed chain (tensor_2, tensor_3, store) removed.
        """
        opps = operand_merge.analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        result = operand_merge.transform(tiled_activation_2x, act_opps[0])
        source = get_source(result)
        assert source.count("nkigym.activation(") == 1
        assert "0:256" in source
        assert "tensor_2 = " not in source
        assert "tensor_3 = " not in source

    def test_tensor_scalar_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_scalar merge should produce one call with widened load.

        After transform: tensor_0 widened to a[0:128, 0:256], single
        tensor_scalar call, absorbed chain (tensor_2, tensor_3, store) removed.
        """
        opps = operand_merge.analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        result = operand_merge.transform(tiled_tensor_scalar_2x, ts_opps[0])
        source = get_source(result)
        assert source.count("nkigym.tensor_scalar(") == 1
        assert "0:256" in source
        assert "tensor_2 = " not in source
        assert "tensor_3 = " not in source

    def test_tensor_tensor_preserves_kwargs(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_tensor merge should preserve the op= keyword argument."""
        opps = operand_merge.analyze(tiled_tensor_tensor_2x)
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        result = operand_merge.transform(tiled_tensor_tensor_2x, tt_opps[0])
        source = get_source(result)
        assert "op=np.add" in source

    def test_activation_preserves_kwargs(self, operand_merge: OperandMergeTransform) -> None:
        """activation merge should preserve the op= keyword argument."""
        opps = operand_merge.analyze(tiled_activation_2x)
        act_opps = [o for o in opps if o.op_type == "activation"]
        result = operand_merge.transform(tiled_activation_2x, act_opps[0])
        source = get_source(result)
        assert "op=np.tanh" in source

    def test_tensor_scalar_preserves_kwargs(self, operand_merge: OperandMergeTransform) -> None:
        """tensor_scalar merge should preserve op0= and operand0= kwargs."""
        opps = operand_merge.analyze(tiled_tensor_scalar_2x)
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        result = operand_merge.transform(tiled_tensor_scalar_2x, ts_opps[0])
        source = get_source(result)
        assert "op0=np.multiply" in source
        assert "operand0=2.0" in source


class TestE2EPipeline:
    """End-to-end tests: tiling -> data_reuse -> operand_merge.

    Verifies that the full transform pipeline preserves numerical
    correctness when applied to a tiled function.
    """

    def test_full_pipeline_matmul_1x2(self, operand_merge: OperandMergeTransform) -> None:
        """Full pipeline on (128,128)x(128,256) matmul (1x2 tile grid).

        Steps: generate_tiled_function -> DataReuseTransform -> OperandMergeTransform.
        The 1x2 case has one shared a load and two adjacent b loads, which
        merges cleanly without shared-load aliasing issues.
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = matmul(a, b)

        tiled = generate_tiled_function(matmul, {"a": (128, 128), "b": (128, 256)})

        reuse = DataReuseTransform()
        for group in reuse.analyze(tiled):
            tiled = reuse.transform(tiled, group)

        while True:
            opps = operand_merge.analyze(tiled)
            if not opps:
                break
            tiled = operand_merge.transform(tiled, opps[0])

        result = tiled(a, b)
        np.testing.assert_allclose(expected, result, rtol=1e-5)

    def test_full_pipeline_matmul_2x2(self, operand_merge: OperandMergeTransform) -> None:
        """Full pipeline on (128,256)x(128,256) matmul fully merges via load+op passes.

        After data reuse, shared loads are merged first (with bare
        consumers subscripted), then op merging merges the matmul pairs.
        The result is numerically correct.
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)
        expected = matmul(a, b)

        tiled = generate_tiled_function(matmul, {"a": (128, 256), "b": (128, 256)})

        reuse = DataReuseTransform()
        for group in reuse.analyze(tiled):
            tiled = reuse.transform(tiled, group)

        while True:
            opps = operand_merge.analyze(tiled)
            if not opps:
                break
            tiled = operand_merge.transform(tiled, opps[0])

        result = tiled(a, b)
        np.testing.assert_allclose(expected, result, rtol=1e-5)
