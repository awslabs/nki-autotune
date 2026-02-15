"""Tests for IR conversion utilities (callable_to_ir, ir_to_source, ir_to_callable)."""

import numpy as np
import pytest
from conftest import assert_arrays_close, make_random_array, normalize_source

from nkigym.ir import Program, Statement, callable_to_ir, ir_to_callable, ir_to_source
from nkigym.ops import (
    ALLOC_F32_OP,
    ALLOC_F64_OP,
    LOAD_OP,
    NC_MATMUL_OP,
    STORE_OP,
    ActivationOp,
    AllocOp,
    LoadOp,
    MatmulOp,
    NcTransposeOp,
    StoreOp,
    TensorScalarOp,
    TensorTensorOp,
)
from nkigym.ops.neuron_op import NeuronOp
from nkigym.tiling import generate_tiled_ir

CAN_MERGE_OPERAND_DIM_CASES = [
    (0, 0, 128, True, "lhs_K_at_limit"),
    (0, 0, 129, False, "lhs_K_over"),
    (0, 1, 128, True, "lhs_M_at_limit"),
    (0, 1, 129, False, "lhs_M_over"),
    (1, 0, 128, True, "rhs_K_at_limit"),
    (1, 0, 129, False, "rhs_K_over"),
    (1, 1, 512, True, "rhs_N_at_limit"),
    (1, 1, 513, False, "rhs_N_over"),
    (2, 0, 128, True, "dst_M_at_limit"),
    (2, 1, 512, True, "dst_N_at_limit"),
]

SOURCE_ROUND_TRIP_SHAPES = [
    ("128x128_128x128", (128, 128), (128, 128)),
    ("128x256_128x256", (128, 256), (128, 256)),
    ("128x256_128x128", (128, 256), (128, 128)),
    ("256x256_256x256", (256, 256), (256, 256)),
]

NUMERICAL_ROUND_TRIP_SHAPES = [
    ("128x128_128x128", (128, 128), (128, 128)),
    ("128x256_128x256", (128, 256), (128, 256)),
    ("256x256_256x256", (256, 256), (256, 256)),
    ("128x512_128x128", (128, 512), (128, 128)),
    ("512x128_512x128", (512, 128), (512, 128)),
]

TENSOR_TENSOR_SOURCE = (
    "def test_fn(a, b):\n"
    "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
    "    tensor_0 = a[0:128, 0:128]\n"
    "    tensor_1 = b[0:128, 0:128]\n"
    "    tensor_2 = nkigym.tensor_tensor(tensor_0, tensor_1, op=np.add)\n"
    "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
    "    return output\n"
)

ACTIVATION_SOURCE = (
    "def test_fn(a):\n"
    "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
    "    tensor_0 = a[0:128, 0:128]\n"
    "    tensor_1 = nkigym.activation(tensor_0, op=np.tanh)\n"
    "    output[0:128, 0:128] = tensor_1[0:128, 0:128]\n"
    "    return output\n"
)

S128 = (0, 128)
SL128 = (S128, S128)


def _alloc_stmt(var: str, shape: tuple[int, ...], op: AllocOp = ALLOC_F32_OP) -> Statement:
    """Build an alloc Statement."""
    return Statement(op, ((var, tuple((0, s) for s in shape)),), True)


def _load_stmt(src: str, src_sl: tuple, dst: str, dst_sl: tuple) -> Statement:
    """Build a load Statement."""
    return Statement(LOAD_OP, ((src, src_sl), (dst, dst_sl)), True)


def _store_stmt(src: str, src_sl: tuple, dst: str, dst_sl: tuple) -> Statement:
    """Build a store Statement."""
    return Statement(STORE_OP, ((src, src_sl), (dst, dst_sl)), True)


class TestOpMetadata:
    """Tests for NeuronOp attributes, tile limits, and registry."""

    def test_matmul_metadata(self) -> None:
        """Verify MatmulOp derived properties from OperandDesc."""
        assert NC_MATMUL_OP.operand_names == ("stationary", "moving", "dst")
        assert NC_MATMUL_OP.read_positions == (0, 1)
        assert NC_MATMUL_OP.write_positions == (2,)
        assert NC_MATMUL_OP.tile_limits == {"K": 128, "M": 128, "N": 512}

    def test_framework_op_metadata(self) -> None:
        """Verify load, store, alloc ops have correct metadata."""
        assert LOAD_OP.op_name == "load"
        assert LOAD_OP.operand_names == ("src", "dst")
        assert STORE_OP.op_name == "store"
        assert STORE_OP.operand_names == ("src", "dst")
        assert ALLOC_F32_OP.op_name == "alloc_float32"
        assert ALLOC_F32_OP.dtype == np.float32
        assert ALLOC_F64_OP.op_name == "alloc_float64"
        assert ALLOC_F64_OP.dtype == np.float64

    def test_no_tile_limits_always_mergeable(self) -> None:
        """Ops without tile_limits accept any merge size."""
        assert LOAD_OP.can_merge_dim(0, 9999) is True
        assert STORE_OP.can_merge_dim(0, 9999) is True
        assert ALLOC_F32_OP.can_merge_dim(0, 9999) is True

    @pytest.mark.parametrize(
        "operand_idx,dim,new_size,expected",
        [c[:4] for c in CAN_MERGE_OPERAND_DIM_CASES],
        ids=[c[4] for c in CAN_MERGE_OPERAND_DIM_CASES],
    )
    def test_matmul_can_merge_operand_dim(self, operand_idx: int, dim: int, new_size: int, expected: bool) -> None:
        """Verify can_merge_operand_dim maps operand+dim to correct tile limit."""
        assert NC_MATMUL_OP.can_merge_operand_dim(operand_idx, dim, new_size) is expected

    def test_registry(self) -> None:
        """Verify NeuronOp.get returns correct classes for all registered ops."""
        assert NeuronOp.get("nc_matmul") is MatmulOp
        assert NeuronOp.get("tensor_tensor") is TensorTensorOp
        assert NeuronOp.get("activation") is ActivationOp
        assert NeuronOp.get("nc_transpose") is NcTransposeOp
        assert NeuronOp.get("tensor_scalar") is TensorScalarOp
        for name in ("nc_matmul", "tensor_tensor", "activation", "nc_transpose", "tensor_scalar"):
            assert name in NeuronOp.all_ops()

    def test_registry_unknown_raises(self) -> None:
        """Verify NeuronOp.get raises KeyError for unknown op names."""
        with pytest.raises(KeyError, match="Unknown op"):
            NeuronOp.get("nonexistent_op")


class TestCallableToIR:
    """Tests for callable_to_ir and generate_tiled_ir parsing."""

    def test_single_tile_structure(self, matmul_func) -> None:
        """Verify IR structure and op singletons for a single-tile matmul."""
        program = generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32)
        assert program.name == "matmul"
        assert program.params == ("a", "b")
        assert program.return_var == "output"
        assert isinstance(program.stmts[0].op, AllocOp)
        assert program.stmts[0].op is ALLOC_F32_OP
        for stmt in program.stmts:
            assert stmt.first_write is True

    def test_multi_tile_op_types_and_singletons(self, matmul_func) -> None:
        """Verify all expected op types and singletons for multi-tile matmul."""
        program = generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32)
        op_types = {type(stmt.op) for stmt in program.stmts}
        assert {AllocOp, LoadOp, StoreOp, MatmulOp} <= op_types
        for stmt in program.stmts:
            if isinstance(stmt.op, LoadOp):
                assert stmt.op is LOAD_OP
            elif isinstance(stmt.op, StoreOp):
                assert stmt.op is STORE_OP
            elif isinstance(stmt.op, MatmulOp):
                assert stmt.op == NC_MATMUL_OP

    def test_alloc_dtype_singletons(self, matmul_func) -> None:
        """Verify alloc uses correct dtype singleton."""
        p32 = generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32)
        p64 = generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float64)
        assert p32.stmts[0].op is ALLOC_F32_OP
        assert p64.stmts[0].op is ALLOC_F64_OP

    def test_reduction_tiling_and_aug_assign(self, matmul_func) -> None:
        """Verify reduction tiling produces correct stmts and augmented assignment."""
        program = generate_tiled_ir(matmul_func, {"a": (256, 128), "b": (256, 128)}, np.float32)
        parsed = callable_to_ir(ir_to_callable(program))
        compute_stmts = [s for s in parsed.stmts if isinstance(s.op, MatmulOp)]
        assert len(compute_stmts) == 2
        assert len({s.operands[-1][0] for s in compute_stmts}) == 1
        assert any(not s.first_write for s in compute_stmts)

    def test_elementwise_ops_parsed(self) -> None:
        """Verify tensor_tensor and activation are parsed correctly."""

        def tt_fn(a, b):
            """Test function."""

        tt_fn.__source__ = TENSOR_TENSOR_SOURCE
        tt_stmts = [s for s in callable_to_ir(tt_fn).stmts if isinstance(s.op, TensorTensorOp)]
        assert len(tt_stmts) == 1
        assert ("op", "np.add") in tt_stmts[0].op.kwargs_repr

        def act_fn(a):
            """Test function."""

        act_fn.__source__ = ACTIVATION_SOURCE
        act_stmts = [s for s in callable_to_ir(act_fn).stmts if isinstance(s.op, ActivationOp)]
        assert len(act_stmts) == 1
        assert ("op", "np.tanh") in act_stmts[0].op.kwargs_repr

    def test_load_dst_slices_zero_based(self, matmul_func) -> None:
        """Verify load destination slices start at 0."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32))
        for stmt in callable_to_ir(tiled).stmts:
            if isinstance(stmt.op, LoadOp):
                for start, _ in stmt.operands[1][1]:
                    assert start == 0

    def test_error_bad_subscript(self) -> None:
        """Verify ValueError for non-slice subscript."""

        def bad_func(a):
            """Bad function."""

        bad_func.__source__ = "def bad_func(a):\n    tensor_0 = a[0, 0:128]\n    return tensor_0\n"
        with pytest.raises(ValueError, match="Unsupported subscript type"):
            callable_to_ir(bad_func)

    def test_error_bad_shape(self) -> None:
        """Verify ValueError for non-integer shape."""

        def bad_func(a):
            """Bad function."""

        bad_func.__source__ = (
            "def bad_func(a):\n    output = nkigym.ndarray((128, 3.14), dtype=np.float32)\n    return output\n"
        )
        with pytest.raises(ValueError, match="Shape element must be an integer"):
            callable_to_ir(bad_func)

    def test_error_unknown_op(self) -> None:
        """Verify ValueError for unknown operation."""

        def bad_func(a):
            """Bad function."""

        bad_func.__source__ = (
            "def bad_func(a):\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = nkigym.unknown_op(tensor_0)\n"
            "    return tensor_1\n"
        )
        with pytest.raises(ValueError, match="Unknown operation"):
            callable_to_ir(bad_func)


class TestIRRoundTrip:
    """Tests for ir_to_source, ir_to_callable, and round-trip correctness."""

    @pytest.mark.parametrize(
        "name,a_shape,b_shape", SOURCE_ROUND_TRIP_SHAPES, ids=[s[0] for s in SOURCE_ROUND_TRIP_SHAPES]
    )
    def test_source_round_trip(self, name: str, a_shape: tuple, b_shape: tuple, matmul_func) -> None:
        """Verify ir_to_source(callable_to_ir(func)) produces equivalent source."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": a_shape, "b": b_shape}, np.float32))
        assert normalize_source(ir_to_source(callable_to_ir(tiled))) == normalize_source(tiled.__source__)

    @pytest.mark.parametrize(
        "name,a_shape,b_shape", NUMERICAL_ROUND_TRIP_SHAPES, ids=[s[0] for s in NUMERICAL_ROUND_TRIP_SHAPES]
    )
    def test_numerical_round_trip(self, name: str, a_shape: tuple, b_shape: tuple, matmul_func) -> None:
        """Verify ir_to_callable(callable_to_ir(func)) is numerically equivalent."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": a_shape, "b": b_shape}, np.float32))
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = tiled(a, b)
        actual = ir_to_callable(callable_to_ir(tiled))(a, b)
        assert_arrays_close(actual, expected)

    def test_double_matmul_round_trip(self, double_matmul_func) -> None:
        """Verify round-trip for chained matmul: D = (A @ B) @ C."""
        shapes = {"a": (128, 128), "b": (128, 128), "c": (128, 128)}
        tiled = ir_to_callable(generate_tiled_ir(double_matmul_func, shapes, np.float32))
        a, b, c = (make_random_array((128, 128), seed=s) for s in (42, 43, 44))
        expected = tiled(a, b, c)
        actual = ir_to_callable(callable_to_ir(tiled))(a, b, c)
        assert_arrays_close(actual, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preserved(self, dtype, matmul_func) -> None:
        """Verify dtype appears in rendered source and numerical round-trip works."""
        shapes = {"a": (128, 128), "b": (128, 128)}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, shapes, dtype))
        source = ir_to_source(callable_to_ir(tiled))
        assert f"np.{np.dtype(dtype).name}" in source
        a = make_random_array((128, 128), seed=42, dtype=dtype)
        b = make_random_array((128, 128), seed=43, dtype=dtype)
        expected = tiled(a, b)
        actual = ir_to_callable(callable_to_ir(tiled))(a, b)
        assert_arrays_close(actual, expected)

    def test_preamble_preserved(self, matmul_func) -> None:
        """Verify preamble and docstring survive round-trip."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        source = ir_to_source(callable_to_ir(tiled))
        assert "def matmul(a, b):" in source
        assert "return output" in source
        compile(source, "<test>", "exec")

        def my_func(a, b):
            """My docstring."""

        my_func.__source__ = (
            "def my_func(a, b):\n"
            '    """My docstring."""\n'
            "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    output[0:128, 0:128] = tensor_0[0:128, 0:128]\n"
            "    return output\n"
        )
        rendered = ir_to_source(callable_to_ir(my_func))
        assert "def my_func(a, b):" in rendered
        assert '"""My docstring."""' in rendered

    def test_elementwise_source_round_trip(self) -> None:
        """Verify elementwise op survives source round-trip."""

        def test_fn(a, b):
            """Test function."""

        test_fn.__source__ = TENSOR_TENSOR_SOURCE
        rendered = ir_to_source(callable_to_ir(test_fn))
        assert "nkigym.tensor_tensor(" in rendered
        assert "op=np.add" in rendered

    def test_accumulation_renders_plus_equals(self, matmul_func) -> None:
        """Verify augmented assignment renders as += in source."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (256, 128), "b": (256, 128)}, np.float32))
        assert "+=" in ir_to_source(callable_to_ir(tiled))

    def test_program_is_hashable(self, matmul_func) -> None:
        """Verify program tuples are hashable and distinct for different shapes."""
        p1 = callable_to_ir(
            ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        )
        p2 = callable_to_ir(
            ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32))
        )
        d = {p1: "small", p2: "large"}
        assert len(d) == 2
        assert d[p1] == "small"

    def test_handcrafted_ir_to_source(self) -> None:
        """Verify ir_to_source renders hand-built alloc, load, store correctly."""
        alloc = Program("fn", ("a",), (_alloc_stmt("output", (128, 256)),), "output", "def fn(a):")
        assert "nkigym.ndarray((128, 256), dtype=np.float32)" in ir_to_source(alloc)

        load = Program("fn", ("a",), (_load_stmt("a", SL128, "t0", SL128),), "output", "def fn(a):")
        assert "t0 = a[0:128, 0:128]" in ir_to_source(load)

        store = Program("fn", ("a",), (_store_stmt("t0", SL128, "output", SL128),), "output", "def fn(a):")
        assert "output[0:128, 0:128] = t0[0:128, 0:128]" in ir_to_source(store)

    def test_edge_cases(self) -> None:
        """Verify ir_to_source/ir_to_callable handle edge cases correctly."""
        empty = Program("empty_fn", ("a",), (), "a", "def empty_fn(a):")
        src = ir_to_source(empty)
        assert "def empty_fn(a):" in src
        assert "return a" in src
        compile(src, "<test>", "exec")

        alloc_only = Program("alloc_fn", (), (_alloc_stmt("output", (128, 128)),), "output", "def alloc_fn():")
        assert "nkigym.ndarray((128, 128), dtype=np.float32)" in ir_to_source(alloc_only)
        compile(ir_to_source(alloc_only), "<test>", "exec")

        typed = "def typed_fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:"
        typed_prog = Program("typed_fn", ("a", "b"), (_load_stmt("a", SL128, "t0", SL128),), "t0", typed)
        assert typed in ir_to_source(typed_prog)

        no_preamble = Program("my_fn", ("x", "y"), (_alloc_stmt("output", (128, 128)),), "output", "")
        assert "def my_fn(x, y):" in ir_to_source(no_preamble)

        passthrough = Program("passthrough", ("a",), (), "a", "def passthrough(a):")
        func = ir_to_callable(passthrough)
        a = make_random_array((128, 128), seed=42)
        np.testing.assert_array_equal(func(a), a)
