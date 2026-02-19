"""Tests for operand merge analysis and transform.

This module tests the ``OperandMergeTransform`` class, which identifies
adjacent load and operation statements operating on contiguous slices of the
same tensor and merges them into wider operations.

Run with: pytest test/test_operand_merge.py -v
"""

import numpy as np
import pytest
from conftest import _golden_to_program, _ref, _slice_ref, make_random_array

import nkigym
from nkigym.ir import GymProgram, GymStatement, TensorRef, program_to_source, source_to_program
from nkigym.tiling import tile_program
from nkigym.transforms import DataReuseTransform
from nkigym.transforms.operand_merge import MergeOpportunity, OperandMergeTransform
from nkigym.utils import callable_to_source, source_to_callable

_merge = OperandMergeTransform()


def _make_tiled_matmul_program(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> GymProgram:
    """Generate a tiled matmul program via the pipeline.

    Args:
        a_shape: Shape of the a input.
        b_shape: Shape of the b input.

    Returns:
        Tiled GymProgram.
    """

    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute matrix multiplication."""
        return nkigym.nc_matmul(a, b)

    source = callable_to_source(matmul)
    program = source_to_program(source, {"a": a_shape, "b": b_shape}, np.float32)
    return tile_program(program)


def _apply_reuse(program: GymProgram) -> GymProgram:
    """Apply DataReuseTransform exhaustively.

    Args:
        program: GymProgram to transform.

    Returns:
        GymProgram with all data reuse applied.
    """
    reuse = DataReuseTransform()
    while True:
        pairs = reuse.analyze_ir(program)
        if not pairs:
            break
        program = reuse.transform_ir(program, pairs[0])
    return program


def _analyze(
    golden_func: object, params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]]
) -> list[MergeOpportunity]:
    """Analyze a golden function for merge opportunities."""
    program = _golden_to_program(golden_func, params, input_shapes, np.float32)
    return _merge.analyze_ir(program)


def _transform(
    golden_func: object, params: tuple[str, ...], input_shapes: dict[str, tuple[int, ...]], opp: MergeOpportunity
) -> GymProgram:
    """Apply a merge opportunity to a golden function."""
    program = _golden_to_program(golden_func, params, input_shapes, np.float32)
    return _merge.transform_ir(program, opp)


AB_SHAPES = {"a": (128, 128), "b": (128, 256)}
AB_PARAMS = ("a", "b")


@pytest.fixture
def operand_merge() -> OperandMergeTransform:
    """Fixture providing an OperandMergeTransform instance."""
    return OperandMergeTransform()


class TestAnalyzeLoadMerge:
    """Tests for load merge opportunity detection."""

    def test_subscript_loads_found(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze finds 1 load opportunity for subscript-consumed loads."""
        from operand_merge_golden import tiled_subscript_loads_2x

        opportunities = _analyze(tiled_subscript_loads_2x, AB_PARAMS, AB_SHAPES)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1

    def test_bare_name_loads_found(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze finds load merging when loads are consumed as bare Name args."""
        from operand_merge_golden import tiled_adjacent_loads_2x

        opportunities = _analyze(tiled_adjacent_loads_2x, AB_PARAMS, AB_SHAPES)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1

    def test_no_adjacent_loads(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze returns empty for non-adjacent loads."""
        from operand_merge_golden import tiled_no_adjacent_loads

        opportunities = _analyze(tiled_no_adjacent_loads, AB_PARAMS, {"a": (128, 128), "b": (128, 384)})
        assert len(opportunities) == 0

    def test_single_subgraph_no_opportunity(self, operand_merge: OperandMergeTransform) -> None:
        """Verify analyze returns empty for single-subgraph function."""
        from operand_merge_golden import tiled_single_subgraph

        opportunities = _analyze(tiled_single_subgraph, AB_PARAMS, {"a": (128, 128), "b": (128, 128)})
        assert len(opportunities) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify the MergeOpportunity fields for a load opportunity."""
        from operand_merge_golden import tiled_subscript_loads_2x

        opportunities = _analyze(tiled_subscript_loads_2x, AB_PARAMS, AB_SHAPES)
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 1
        opp = load_opps[0]

        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "load"
        assert opp.stmt_a < opp.stmt_b
        assert opp.merged_slice == (0, 256)
        assert opp.differing_operand_idx == 1


class TestAnalyzeNoOpportunity:
    """Tests for cases where no merge opportunities should be found."""

    def test_different_source_tensors(self, operand_merge: OperandMergeTransform) -> None:
        """Verify loads from different tensors do not merge."""
        from operand_merge_golden import tiled_different_source_tensors

        opportunities = _analyze(tiled_different_source_tensors, AB_PARAMS, {"a": (128, 128), "b": (128, 128)})
        assert len(opportunities) == 0

    def test_different_partition_slices(self, operand_merge: OperandMergeTransform) -> None:
        """Verify loads with different partition slices do not merge."""
        from operand_merge_golden import tiled_different_partition_slices

        opportunities = _analyze(tiled_different_partition_slices, AB_PARAMS, {"a": (256, 128), "b": (128, 128)})
        load_opps = [opp for opp in opportunities if opp.op_type == "load"]
        assert len(load_opps) == 0


class TestAnalyzeOpMerge:
    """Tests for nc_matmul operation merging detection."""

    def test_nc_matmul_merge_1x4(self, operand_merge: OperandMergeTransform) -> None:
        """Post-reuse 1x4 should find nc_matmul opportunities."""
        from operand_merge_golden import tiled_matmul_post_reuse_1x4

        opps = _analyze(tiled_matmul_post_reuse_1x4, AB_PARAMS, {"a": (128, 128), "b": (128, 512)})
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) >= 1

    def test_bare_name_loads_produce_load_and_op_opps(self, operand_merge: OperandMergeTransform) -> None:
        """Adjacent loads consumed as bare Name args produce both load and op opportunities."""
        from operand_merge_golden import tiled_adjacent_loads_2x

        opps = _analyze(tiled_adjacent_loads_2x, AB_PARAMS, AB_SHAPES)
        load_opps = [o for o in opps if o.op_type == "load"]
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(load_opps) == 1
        assert load_opps[0].merged_slice == (0, 256)
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_four_adjacent_produces_nc_matmul_opps(self, operand_merge: OperandMergeTransform) -> None:
        """Four adjacent loads with bare Name args produce nc_matmul opportunities."""
        from operand_merge_golden import tiled_adjacent_4x

        opportunities = _analyze(tiled_adjacent_4x, AB_PARAMS, {"a": (128, 128), "b": (128, 512)})
        op_opps = [o for o in opportunities if o.op_type == "nc_matmul"]
        assert len(op_opps) >= 2
        for opp in op_opps:
            assert opp.stmt_a < opp.stmt_b


class TestTransformSource:
    """Tests for transform() source code correctness."""

    def test_load_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """Load merge should widen the kept load and remove absorbed."""
        from operand_merge_golden import tiled_subscript_loads_2x

        opps = _analyze(tiled_subscript_loads_2x, AB_PARAMS, AB_SHAPES)
        load_opps = [o for o in opps if o.op_type == "load"]
        assert len(load_opps) == 1
        result = _transform(tiled_subscript_loads_2x, AB_PARAMS, AB_SHAPES, load_opps[0])
        source = program_to_source(result)
        assert "0:256" in source
        assert "tensor_4" not in source


class TestLoadMergeSubscripts:
    """Tests for load merge consumer subscripting."""

    def test_load_merge_subscripts_consumers(self, operand_merge: OperandMergeTransform) -> None:
        """After load merging on 2x2 IR, consumers appear as subscript expressions."""
        from operand_merge_golden import tiled_matmul_post_reuse_2x2

        shapes = {"a": (128, 256), "b": (128, 256)}
        opps = _analyze(tiled_matmul_post_reuse_2x2, AB_PARAMS, shapes)
        load_opps = [o for o in opps if o.op_type == "load"]
        assert len(load_opps) >= 1

        program = _golden_to_program(tiled_matmul_post_reuse_2x2, AB_PARAMS, shapes, np.float32)
        for opp in load_opps:
            program = _merge.transform_ir(program, opp)

        source = program_to_source(program)
        assert "0:256" in source
        assert "tensor_1[" in source
        assert "0:128, 0:128]" in source or "0:128, 128:256]" in source

    def test_load_merge_2x2_numerical(self, operand_merge: OperandMergeTransform) -> None:
        """Load merging on 2x2 post-reuse IR preserves numerical correctness."""
        from operand_merge_golden import tiled_matmul_post_reuse_2x2

        a = make_random_array((128, 256), seed=42)
        b = make_random_array((128, 256), seed=43)

        original = tiled_matmul_post_reuse_2x2(a, b)
        shapes = {"a": (128, 256), "b": (128, 256)}
        program = _golden_to_program(tiled_matmul_post_reuse_2x2, AB_PARAMS, shapes, np.float32)

        opps = _merge.analyze_ir(program)
        load_opps = [o for o in opps if o.op_type == "load"]
        for opp in load_opps:
            program = _merge.transform_ir(program, opp)

        func = source_to_callable(program_to_source(program), program.name)
        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)


class TestTransformNumerical:
    """Tests for numerical correctness after transform."""

    @pytest.mark.parametrize(
        "fixture_name,a_shape,b_shape",
        [
            ("tiled_matmul_post_reuse_1x2", (128, 128), (128, 256)),
            ("tiled_matmul_post_reuse_1x4", (128, 128), (128, 512)),
            ("tiled_adjacent_loads_2x", (128, 128), (128, 256)),
            ("tiled_matmul_post_reuse_2x2", (128, 256), (128, 256)),
        ],
        ids=["post_reuse_1x2", "post_reuse_1x4", "adjacent_loads_2x", "post_reuse_2x2"],
    )
    def test_iterative_merge_numerical(
        self, operand_merge: OperandMergeTransform, fixture_name: str, a_shape: tuple, b_shape: tuple
    ) -> None:
        """Iteratively applying all opportunities should preserve correctness."""
        import operand_merge_golden

        fixture = getattr(operand_merge_golden, fixture_name)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)

        original = fixture(a, b)
        shapes = {"a": a_shape, "b": b_shape}
        program = _golden_to_program(fixture, AB_PARAMS, shapes, np.float32)
        while True:
            opps = _merge.analyze_ir(program)
            if not opps:
                break
            program = _merge.transform_ir(program, opps[0])

        func = source_to_callable(program_to_source(program), program.name)
        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)


class TestAnalyzeMatmulMerge:
    """Tests for nc_matmul merging opportunity detection."""

    def test_same_lhs_adjacent_rhs(self, operand_merge: OperandMergeTransform) -> None:
        """Two matmuls with same LHS, adjacent RHS should find nc_matmul opportunity."""
        from operand_merge_golden import tiled_matmul_post_reuse_1x2

        opps = _analyze(tiled_matmul_post_reuse_1x2, AB_PARAMS, AB_SHAPES)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].differing_operand_idx == 1
        assert op_opps[0].merged_slice == (0, 256)

    def test_n_at_limit_accepted(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=512 (exactly at limit) should produce an opportunity."""
        from operand_merge_golden import tiled_matmul_n_at_limit

        opps = _analyze(tiled_matmul_n_at_limit, AB_PARAMS, {"a": (128, 128), "b": (128, 512)})
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].merged_slice == (0, 512)

    def test_n_over_limit_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged N=640 (over limit) should not appear as a single opportunity."""
        from operand_merge_golden import tiled_matmul_exceeds_n_limit

        opps = _analyze(tiled_matmul_exceeds_n_limit, AB_PARAMS, {"a": (128, 128), "b": (128, 640)})
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        for opp in op_opps:
            assert opp.merged_slice[1] - opp.merged_slice[0] <= 512

    def test_same_rhs_adjacent_lhs_m_merge(self, operand_merge: OperandMergeTransform) -> None:
        """Two matmuls with same RHS, adjacent LHS should merge on M dimension."""
        from operand_merge_golden import tiled_matmul_m_dim_merge

        opps = _analyze(tiled_matmul_m_dim_merge, AB_PARAMS, {"a": (128, 128), "b": (128, 128)})
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 1
        assert op_opps[0].differing_operand_idx == 0
        assert op_opps[0].merged_slice == (0, 128)

    def test_m_over_limit_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Merged M=192 (over M=128 limit) should not produce an opportunity."""
        from operand_merge_golden import tiled_matmul_m_exceeds_limit

        opps = _analyze(tiled_matmul_m_exceeds_limit, AB_PARAMS, {"a": (128, 192), "b": (128, 128)})
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(op_opps) == 0


class TestTransformMatmulMerge:
    """Tests for nc_matmul transform source code correctness."""

    def test_same_lhs_adjacent_rhs_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on post_reuse_1x2 should produce one nc_matmul with b[0:128, 0:256]."""
        from operand_merge_golden import tiled_matmul_post_reuse_1x2

        opps = _analyze(tiled_matmul_post_reuse_1x2, AB_PARAMS, AB_SHAPES)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_post_reuse_1x2, AB_PARAMS, AB_SHAPES, op_opps[0])
        source = program_to_source(result)
        assert source.count("nc_matmul") == 1
        assert "0:256" in source
        assert "tensor_4" not in source
        assert "tensor_5" not in source

    def test_n_at_limit_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on N=512 at-limit should produce one nc_matmul with b[0:128, 0:512]."""
        from operand_merge_golden import tiled_matmul_n_at_limit

        shapes = {"a": (128, 128), "b": (128, 512)}
        opps = _analyze(tiled_matmul_n_at_limit, AB_PARAMS, shapes)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_n_at_limit, AB_PARAMS, shapes, op_opps[0])
        source = program_to_source(result)
        assert source.count("nkigym.nc_matmul(") == 1
        assert "0:512" in source

    def test_m_dim_merge_source(self, operand_merge: OperandMergeTransform) -> None:
        """Transform on M-dim merge should produce one nc_matmul call."""
        from operand_merge_golden import tiled_matmul_m_dim_merge

        shapes = {"a": (128, 128), "b": (128, 128)}
        opps = _analyze(tiled_matmul_m_dim_merge, AB_PARAMS, shapes)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result = _transform(tiled_matmul_m_dim_merge, AB_PARAMS, shapes, op_opps[0])
        source = program_to_source(result)
        assert source.count("nkigym.nc_matmul(") == 1
        assert "tensor_3" not in source
        assert "tensor_4" not in source


class TestMatmulMergeNumerical:
    """Tests for nc_matmul merge numerical correctness."""

    @pytest.mark.parametrize(
        "fixture_name,a_shape,b_shape",
        [
            ("tiled_matmul_post_reuse_1x2", (128, 128), (128, 256)),
            ("tiled_matmul_n_at_limit", (128, 128), (128, 512)),
            ("tiled_matmul_m_dim_merge", (128, 128), (128, 128)),
        ],
        ids=["same_lhs_adjacent_rhs", "n_at_limit", "m_dim_merge"],
    )
    def test_single_merge_numerical(
        self, operand_merge: OperandMergeTransform, fixture_name: str, a_shape: tuple, b_shape: tuple
    ) -> None:
        """nc_matmul merge should produce numerically identical results."""
        import operand_merge_golden

        fixture = getattr(operand_merge_golden, fixture_name)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)

        shapes = {"a": a_shape, "b": b_shape}
        opps = _analyze(fixture, AB_PARAMS, shapes)
        op_opps = [o for o in opps if o.op_type == "nc_matmul"]
        result_program = _transform(fixture, AB_PARAMS, shapes, op_opps[0])
        result_func = source_to_callable(program_to_source(result_program), result_program.name)

        original = fixture(a, b)
        transformed = result_func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)

    def test_iterative_1x4_full_merge(self, operand_merge: OperandMergeTransform) -> None:
        """Iteratively merging 1x4 post-reuse from 4 to 1 nc_matmul call."""
        from operand_merge_golden import tiled_matmul_post_reuse_1x4

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 512), seed=43)

        original = tiled_matmul_post_reuse_1x4(a, b)
        shapes = {"a": (128, 128), "b": (128, 512)}
        program = _golden_to_program(tiled_matmul_post_reuse_1x4, AB_PARAMS, shapes, np.float32)
        while True:
            opps = _merge.analyze_ir(program)
            if not opps:
                break
            program = _merge.transform_ir(program, opps[0])

        source = program_to_source(program)
        assert source.count("nc_matmul") == 1
        assert "0:512" in source

        func = source_to_callable(program_to_source(program), program.name)
        transformed = func(a, b)
        np.testing.assert_allclose(original, transformed, rtol=1e-5, atol=1e-5)


class TestAnalyzeTensorTensorMerge:
    """Tests for tensor_tensor operation merging detection."""

    def test_same_op_adjacent_first_arg(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_tensor with same op=np.add, differing first arg."""
        from operand_merge_golden import tiled_tensor_tensor_2x

        opps = _analyze(tiled_tensor_tensor_2x, AB_PARAMS, {"a": (128, 256), "b": (128, 128)})
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        assert len(tt_opps) == 1
        assert tt_opps[0].differing_operand_idx == 0
        assert tt_opps[0].merged_slice == (0, 256)

    def test_different_op_kwargs_rejected(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_tensor with different op kwargs should not merge."""
        from operand_merge_golden import tiled_tensor_tensor_diff_ops

        opps = _analyze(tiled_tensor_tensor_diff_ops, AB_PARAMS, {"a": (128, 256), "b": (128, 128)})
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        assert len(tt_opps) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for tensor_tensor."""
        from operand_merge_golden import tiled_tensor_tensor_2x

        opps = _analyze(tiled_tensor_tensor_2x, AB_PARAMS, {"a": (128, 256), "b": (128, 128)})
        tt_opps = [o for o in opps if o.op_type == "tensor_tensor"]
        opp = tt_opps[0]
        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "tensor_tensor"
        assert opp.stmt_a < opp.stmt_b
        assert "tensor_tensor" in opp.description


class TestAnalyzeActivationMerge:
    """Tests for activation operation merging detection."""

    def test_same_op_adjacent_input(self, operand_merge: OperandMergeTransform) -> None:
        """Two activation with same op=np.tanh, adjacent input slices."""
        from operand_merge_golden import tiled_activation_2x

        opps = _analyze(tiled_activation_2x, ("a",), {"a": (128, 256)})
        act_opps = [o for o in opps if o.op_type == "activation"]
        assert len(act_opps) == 1
        assert act_opps[0].differing_operand_idx == 0
        assert act_opps[0].merged_slice == (0, 256)

    def test_single_op_no_opportunity(self, operand_merge: OperandMergeTransform) -> None:
        """Single activation op should produce no opportunities."""
        from operand_merge_golden import tiled_activation_single

        opps = _analyze(tiled_activation_single, ("a",), {"a": (128, 128)})
        assert len(opps) == 0

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for activation."""
        from operand_merge_golden import tiled_activation_2x

        opps = _analyze(tiled_activation_2x, ("a",), {"a": (128, 256)})
        act_opps = [o for o in opps if o.op_type == "activation"]
        opp = act_opps[0]
        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "activation"
        assert opp.stmt_a < opp.stmt_b


class TestAnalyzeTensorScalarMerge:
    """Tests for tensor_scalar operation merging detection."""

    def test_same_kwargs_adjacent_input(self, operand_merge: OperandMergeTransform) -> None:
        """Two tensor_scalar with same op0/operand0, adjacent input slices."""
        from operand_merge_golden import tiled_tensor_scalar_2x

        opps = _analyze(tiled_tensor_scalar_2x, ("a",), {"a": (128, 256)})
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        assert len(ts_opps) == 1
        assert ts_opps[0].differing_operand_idx == 0
        assert ts_opps[0].merged_slice == (0, 256)

    def test_opportunity_fields(self, operand_merge: OperandMergeTransform) -> None:
        """Verify MergeOpportunity fields for tensor_scalar."""
        from operand_merge_golden import tiled_tensor_scalar_2x

        opps = _analyze(tiled_tensor_scalar_2x, ("a",), {"a": (128, 256)})
        ts_opps = [o for o in opps if o.op_type == "tensor_scalar"]
        opp = ts_opps[0]
        assert isinstance(opp, MergeOpportunity)
        assert opp.op_type == "tensor_scalar"
        assert opp.stmt_a < opp.stmt_b
        assert "tensor_scalar" in opp.description


class TestTransformElementwise:
    """Tests for elementwise op transform source code correctness."""

    @pytest.mark.parametrize(
        "fixture_name,params,shapes,op_type,op_call,removed_vars",
        [
            (
                "tiled_tensor_tensor_2x",
                AB_PARAMS,
                {"a": (128, 256), "b": (128, 128)},
                "tensor_tensor",
                "nkigym.tensor_tensor(",
                ["tensor_3", "tensor_5"],
            ),
            (
                "tiled_activation_2x",
                ("a",),
                {"a": (128, 256)},
                "activation",
                "nkigym.activation(",
                ["tensor_2", "tensor_3"],
            ),
            (
                "tiled_tensor_scalar_2x",
                ("a",),
                {"a": (128, 256)},
                "tensor_scalar",
                "nkigym.tensor_scalar(",
                ["tensor_2", "tensor_3"],
            ),
        ],
        ids=["tensor_tensor", "activation", "tensor_scalar"],
    )
    def test_merge_source(
        self,
        operand_merge: OperandMergeTransform,
        fixture_name: str,
        params: tuple[str, ...],
        shapes: dict[str, tuple[int, ...]],
        op_type: str,
        op_call: str,
        removed_vars: list[str],
    ) -> None:
        """Elementwise merge should produce one call with widened load and remove absorbed vars."""
        import operand_merge_golden

        fixture = getattr(operand_merge_golden, fixture_name)
        opps = _analyze(fixture, params, shapes)
        typed_opps = [o for o in opps if o.op_type == op_type]
        result = _transform(fixture, params, shapes, typed_opps[0])
        source = program_to_source(result)
        assert source.count(op_call) == 1
        assert "0:256" in source
        for var in removed_vars:
            assert f"{var} = " not in source

    @pytest.mark.parametrize(
        "fixture_name,params,shapes,op_type,expected_kwargs",
        [
            ("tiled_tensor_tensor_2x", AB_PARAMS, {"a": (128, 256), "b": (128, 128)}, "tensor_tensor", ["op=np.add"]),
            ("tiled_activation_2x", ("a",), {"a": (128, 256)}, "activation", ["op=np.tanh"]),
            ("tiled_tensor_scalar_2x", ("a",), {"a": (128, 256)}, "tensor_scalar", ["op0=np.multiply", "operand0=2.0"]),
        ],
        ids=["tensor_tensor", "activation", "tensor_scalar"],
    )
    def test_preserves_kwargs(
        self,
        operand_merge: OperandMergeTransform,
        fixture_name: str,
        params: tuple[str, ...],
        shapes: dict[str, tuple[int, ...]],
        op_type: str,
        expected_kwargs: list[str],
    ) -> None:
        """Elementwise merge should preserve operation-specific keyword arguments."""
        import operand_merge_golden

        fixture = getattr(operand_merge_golden, fixture_name)
        opps = _analyze(fixture, params, shapes)
        typed_opps = [o for o in opps if o.op_type == op_type]
        result = _transform(fixture, params, shapes, typed_opps[0])
        source = program_to_source(result)
        for kwarg in expected_kwargs:
            assert kwarg in source


class TestE2EPipeline:
    """End-to-end tests: tiling -> data_reuse -> operand_merge."""

    @pytest.mark.parametrize(
        "a_shape,b_shape", [((128, 128), (128, 256)), ((128, 256), (128, 256))], ids=["1x2", "2x2"]
    )
    def test_full_pipeline_matmul(self, operand_merge: OperandMergeTransform, a_shape: tuple, b_shape: tuple) -> None:
        """Full pipeline: tile_program -> DataReuseTransform -> OperandMergeTransform."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        source = callable_to_source(matmul)
        program = source_to_program(source, {"a": a_shape, "b": b_shape}, np.float32)
        program = tile_program(program)

        reuse = DataReuseTransform()
        for group in reuse.analyze_ir(program):
            program = reuse.transform_ir(program, group)

        merge = OperandMergeTransform()
        while True:
            opps = merge.analyze_ir(program)
            if not opps:
                break
            program = merge.transform_ir(program, opps[0])

        result = source_to_callable(program_to_source(program), program.name)(a, b)
        np.testing.assert_allclose(expected, result, rtol=1e-5, atol=1e-5)


class TestOperandMergeIRDirect:
    """Tests for OperandMergeTransform.analyze_ir/transform_ir on program tuples directly."""

    def test_analyze_ir_finds_opportunities(self) -> None:
        """Verify analyze_ir returns opportunities from a program tuple."""
        from operand_merge_golden import tiled_matmul_post_reuse_1x2

        program = _golden_to_program(tiled_matmul_post_reuse_1x2, AB_PARAMS, AB_SHAPES, np.float32)
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        assert len(opps) > 0
        for opp in opps:
            assert isinstance(opp, MergeOpportunity)

    def test_transform_ir_returns_valid_program(self) -> None:
        """Verify transform_ir returns a valid program with fewer statements."""
        from operand_merge_golden import tiled_matmul_post_reuse_1x2

        program = _golden_to_program(tiled_matmul_post_reuse_1x2, AB_PARAMS, AB_SHAPES, np.float32)
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        assert len(opps) > 0

        new_program = transform.transform_ir(program, opps[0])
        assert new_program.name == program.name
        assert new_program.params == program.params
        assert new_program.return_var == program.return_var
        assert len(new_program.stmts) < len(program.stmts)

        func = source_to_callable(program_to_source(new_program), new_program.name)
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
        """End-to-end: dependency between statements blocks compute merge."""
        stmts: tuple[GymStatement, ...] = (
            GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output", (128, 256))),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (128, 128), ((0, 128), (0, 128)))),), _ref("tensor_0", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (128, 256), ((0, 128), (0, 128)))),), _ref("tensor_1", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_1", (128, 128)))),
                _ref("tensor_2", (128, 128)),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("b", (128, 256), ((0, 128), (128, 256)))),),
                _ref("tensor_3", (128, 128)),
            ),
            GymStatement(
                "nc_matmul",
                (
                    ("stationary", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", _ref("tensor_3", (128, 128))),
                ),
                _ref("tensor_4", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_2", (128, 128))),
                    ("dst", _slice_ref("output", (128, 256), ((0, 128), (0, 128)))),
                ),
                _slice_ref("output", (128, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_4", (128, 128))),
                    ("dst", _slice_ref("output", (128, 256), ((0, 128), (128, 256)))),
                ),
                _slice_ref("output", (128, 256), ((0, 128), (128, 256))),
            ),
        )
        program = GymProgram("dep_fn", ("a", "b"), (("a", (128, 128)), ("b", (128, 256))), stmts, "output", np.float32)
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        matmul_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(matmul_opps) == 0


class TestHasAccumulationBlocking:
    """Tests for OperandMergeTransform._has_accumulation blocking a merge."""

    def test_no_accumulation(self) -> None:
        """Variable written once by compute is not an accumulation."""
        stmts: tuple[GymStatement, ...] = (
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (128, 128), ((0, 128), (0, 128)))),), _ref("tensor_0", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (128, 128), ((0, 128), (0, 128)))),), _ref("tensor_1", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_1", (128, 128)))),
                _ref("tensor_2", (128, 128)),
            ),
        )
        compute_vars = {"tensor_2"}
        assert OperandMergeTransform._has_accumulation("tensor_2", stmts, compute_vars) is False

    def test_has_accumulation(self) -> None:
        """Variable written twice by compute (reduction tiling) is an accumulation."""
        stmts: tuple[GymStatement, ...] = (
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (256, 128), ((0, 128), (0, 128)))),), _ref("tensor_0", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (256, 128), ((0, 128), (0, 128)))),), _ref("tensor_1", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_1", (128, 128)))),
                _ref("tensor_2", (128, 128)),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("a", (256, 128), ((128, 256), (0, 128)))),),
                _ref("tensor_3", (128, 128)),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("b", (256, 128), ((128, 256), (0, 128)))),),
                _ref("tensor_4", (128, 128)),
            ),
            GymStatement(
                "nc_matmul",
                (
                    ("stationary", _ref("tensor_3", (128, 128))),
                    ("moving", _ref("tensor_4", (128, 128))),
                    ("acc", _ref("tensor_2", (128, 128))),
                ),
                _ref("tensor_2", (128, 128)),
            ),
        )
        compute_vars = {"tensor_2"}
        assert OperandMergeTransform._has_accumulation("tensor_2", stmts, compute_vars) is True

    def test_accumulation_blocks_merge(self) -> None:
        """Accumulating variables block compute merging at the analyze level."""
        stmts: tuple[GymStatement, ...] = (
            GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output", (128, 256))),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (256, 128), ((0, 128), (0, 128)))),), _ref("t0", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (256, 128), ((0, 128), (0, 128)))),), _ref("t1", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("t0", (128, 128))), ("moving", _ref("t1", (128, 128)))),
                _ref("t2", (128, 128)),
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (256, 128), ((128, 256), (0, 128)))),), _ref("t3", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (256, 128), ((128, 256), (0, 128)))),), _ref("t4", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (
                    ("stationary", _ref("t3", (128, 128))),
                    ("moving", _ref("t4", (128, 128))),
                    ("acc", _ref("t2", (128, 128))),
                ),
                _ref("t2", (128, 128)),
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (256, 128), ((0, 128), (0, 128)))),), _ref("t5", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (256, 128), ((0, 128), (128, 256)))),), _ref("t6", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("t5", (128, 128))), ("moving", _ref("t6", (128, 128)))),
                _ref("t7", (128, 128)),
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (256, 128), ((128, 256), (0, 128)))),), _ref("t8", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (256, 128), ((128, 256), (128, 256)))),), _ref("t9", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (
                    ("stationary", _ref("t8", (128, 128))),
                    ("moving", _ref("t9", (128, 128))),
                    ("acc", _ref("t7", (128, 128))),
                ),
                _ref("t7", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (("src", _ref("t2", (128, 128))), ("dst", _slice_ref("output", (128, 256), ((0, 128), (0, 128))))),
                _slice_ref("output", (128, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                "np_store",
                (("src", _ref("t7", (128, 128))), ("dst", _slice_ref("output", (128, 256), ((0, 128), (128, 256))))),
                _slice_ref("output", (128, 256), ((0, 128), (128, 256))),
            ),
        )
        program = GymProgram(
            "accum_fn", ("a", "b"), (("a", (256, 128)), ("b", (256, 128))), stmts, "output", np.float32
        )
        transform = OperandMergeTransform()
        opps = transform.analyze_ir(program)
        matmul_opps = [o for o in opps if o.op_type == "nc_matmul"]
        assert len(matmul_opps) == 0


class TestThreeWayMergeChain:
    """Tests for three-way iterative merge chains."""

    def test_iterative_merge_three_loads(self) -> None:
        """Three adjacent loads can be merged in two iterations."""
        stmts: tuple[GymStatement, ...] = (
            GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output", (128, 384))),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (128, 128), ((0, 128), (0, 128)))),), _ref("tensor_0", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (128, 384), ((0, 128), (0, 128)))),), _ref("tensor_1", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_1", (128, 128)))),
                _ref("tensor_2", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_2", (128, 128))),
                    ("dst", _slice_ref("output", (128, 384), ((0, 128), (0, 128)))),
                ),
                _slice_ref("output", (128, 384), ((0, 128), (0, 128))),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("b", (128, 384), ((0, 128), (128, 256)))),),
                _ref("tensor_3", (128, 128)),
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_3", (128, 128)))),
                _ref("tensor_4", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_4", (128, 128))),
                    ("dst", _slice_ref("output", (128, 384), ((0, 128), (128, 256)))),
                ),
                _slice_ref("output", (128, 384), ((0, 128), (128, 256))),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("b", (128, 384), ((0, 128), (256, 384)))),),
                _ref("tensor_5", (128, 128)),
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_5", (128, 128)))),
                _ref("tensor_6", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_6", (128, 128))),
                    ("dst", _slice_ref("output", (128, 384), ((0, 128), (256, 384)))),
                ),
                _slice_ref("output", (128, 384), ((0, 128), (256, 384))),
            ),
        )
        program = GymProgram(
            "three_load", ("a", "b"), (("a", (128, 128)), ("b", (128, 384))), stmts, "output", np.float32
        )
        transform = OperandMergeTransform()

        iteration_count = 0
        while True:
            opps = transform.analyze_ir(program)
            if not opps:
                break
            program = transform.transform_ir(program, opps[0])
            iteration_count += 1

        assert iteration_count >= 2

        source = program_to_source(program)
        assert source.count("nc_matmul") == 1

        func = source_to_callable(program_to_source(program), program.name)
        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 384), seed=43)

        expected_stmts: tuple[GymStatement, ...] = (
            GymStatement("np_empty", (("dtype", "np.float32"),), _ref("output", (128, 384))),
            GymStatement(
                "np_slice", (("src", _slice_ref("a", (128, 128), ((0, 128), (0, 128)))),), _ref("tensor_0", (128, 128))
            ),
            GymStatement(
                "np_slice", (("src", _slice_ref("b", (128, 384), ((0, 128), (0, 128)))),), _ref("tensor_1", (128, 128))
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_1", (128, 128)))),
                _ref("tensor_2", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_2", (128, 128))),
                    ("dst", _slice_ref("output", (128, 384), ((0, 128), (0, 128)))),
                ),
                _slice_ref("output", (128, 384), ((0, 128), (0, 128))),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("b", (128, 384), ((0, 128), (128, 256)))),),
                _ref("tensor_3", (128, 128)),
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_3", (128, 128)))),
                _ref("tensor_4", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_4", (128, 128))),
                    ("dst", _slice_ref("output", (128, 384), ((0, 128), (128, 256)))),
                ),
                _slice_ref("output", (128, 384), ((0, 128), (128, 256))),
            ),
            GymStatement(
                "np_slice",
                (("src", _slice_ref("b", (128, 384), ((0, 128), (256, 384)))),),
                _ref("tensor_5", (128, 128)),
            ),
            GymStatement(
                "nc_matmul",
                (("stationary", _ref("tensor_0", (128, 128))), ("moving", _ref("tensor_5", (128, 128)))),
                _ref("tensor_6", (128, 128)),
            ),
            GymStatement(
                "np_store",
                (
                    ("src", _ref("tensor_6", (128, 128))),
                    ("dst", _slice_ref("output", (128, 384), ((0, 128), (256, 384)))),
                ),
                _slice_ref("output", (128, 384), ((0, 128), (256, 384))),
            ),
        )
        original_prog = GymProgram(
            "three_load", ("a", "b"), (("a", (128, 128)), ("b", (128, 384))), expected_stmts, "output", np.float32
        )
        original_func = source_to_callable(program_to_source(original_prog), original_prog.name)
        expected = original_func(a, b)
        actual = func(a, b)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
