"""Tests for IR conversion utilities (callable_to_ir, ir_to_source, ir_to_callable).

Verifies:
- NKIOp attribute extensions and can_merge_dim behavior
- callable_to_ir parsing of tiled functions
- ir_to_source rendering and round-trip correctness
- ir_to_callable numerical equivalence
- Program tuples are hashable (usable as dict keys)
"""

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
    AllocOp,
    ElementwiseOp,
    LoadOp,
    NKIMatmul,
    StoreOp,
)
from nkigym.tiling import generate_tiled_ir


class TestNKIOpAttributes:
    """Tests for extended NKIOp attributes and can_merge_dim."""

    def test_nc_matmul_operand_names(self):
        """Verify NKIMatmul has correct operand_names."""
        assert NC_MATMUL_OP.operand_names == ("lhs", "rhs", "dst")

    def test_nc_matmul_read_positions(self):
        """Verify NKIMatmul has correct read_positions."""
        assert NC_MATMUL_OP.read_positions == (0, 1)

    def test_nc_matmul_write_positions(self):
        """Verify NKIMatmul has correct write_positions."""
        assert NC_MATMUL_OP.write_positions == (2,)

    def test_nc_matmul_tile_limits(self):
        """Verify NKIMatmul has correct tile_limits."""
        assert NC_MATMUL_OP.tile_limits == {"M": 128, "K": 128, "N": 512}

    def test_nc_matmul_can_merge_dim_m_at_limit(self):
        """Verify can_merge_dim returns True for M at exactly 128."""
        assert NC_MATMUL_OP.can_merge_dim(0, 128) is True

    def test_nc_matmul_can_merge_dim_m_over_limit(self):
        """Verify can_merge_dim returns False for M over 128."""
        assert NC_MATMUL_OP.can_merge_dim(0, 129) is False

    def test_nc_matmul_can_merge_dim_k_at_limit(self):
        """Verify can_merge_dim returns True for K at exactly 128."""
        assert NC_MATMUL_OP.can_merge_dim(1, 128) is True

    def test_nc_matmul_can_merge_dim_k_over_limit(self):
        """Verify can_merge_dim returns False for K over 128."""
        assert NC_MATMUL_OP.can_merge_dim(1, 256) is False

    def test_nc_matmul_can_merge_dim_n_at_limit(self):
        """Verify can_merge_dim returns True for N at exactly 512."""
        assert NC_MATMUL_OP.can_merge_dim(2, 512) is True

    def test_nc_matmul_can_merge_dim_n_over_limit(self):
        """Verify can_merge_dim returns False for N over 512."""
        assert NC_MATMUL_OP.can_merge_dim(2, 513) is False

    def test_nc_matmul_can_merge_dim_out_of_range(self):
        """Verify can_merge_dim returns True for dim beyond tile_limits."""
        assert NC_MATMUL_OP.can_merge_dim(5, 9999) is True

    def test_load_op_metadata(self):
        """Verify LoadOp has correct metadata."""
        assert LOAD_OP.op_name == "load"
        assert LOAD_OP.operand_names == ("src", "dst")
        assert LOAD_OP.read_positions == (0,)
        assert LOAD_OP.write_positions == (1,)
        assert LOAD_OP.tile_limits == {}

    def test_load_op_can_merge_dim_always_true(self):
        """Verify LoadOp can_merge_dim returns True (no tile limits)."""
        assert LOAD_OP.can_merge_dim(0, 9999) is True

    def test_store_op_metadata(self):
        """Verify StoreOp has correct metadata."""
        assert STORE_OP.op_name == "store"
        assert STORE_OP.operand_names == ("src", "dst")
        assert STORE_OP.read_positions == (0,)
        assert STORE_OP.write_positions == (1,)
        assert STORE_OP.tile_limits == {}

    def test_store_op_can_merge_dim_always_true(self):
        """Verify StoreOp can_merge_dim returns True (no tile limits)."""
        assert STORE_OP.can_merge_dim(0, 9999) is True

    def test_alloc_f32_metadata(self):
        """Verify ALLOC_F32_OP has correct metadata."""
        assert ALLOC_F32_OP.op_name == "alloc_float32"
        assert ALLOC_F32_OP.operand_names == ("tensor",)
        assert ALLOC_F32_OP.read_positions == ()
        assert ALLOC_F32_OP.write_positions == (0,)
        assert ALLOC_F32_OP.dtype == np.float32

    def test_alloc_f64_metadata(self):
        """Verify ALLOC_F64_OP has correct metadata."""
        assert ALLOC_F64_OP.op_name == "alloc_float64"
        assert ALLOC_F64_OP.operand_names == ("tensor",)
        assert ALLOC_F64_OP.read_positions == ()
        assert ALLOC_F64_OP.write_positions == (0,)
        assert ALLOC_F64_OP.dtype == np.float64

    def test_alloc_op_can_merge_dim_always_true(self):
        """Verify AllocOp can_merge_dim returns True (no tile limits)."""
        assert ALLOC_F32_OP.can_merge_dim(0, 9999) is True

    @pytest.mark.parametrize(
        "operand_idx,dim,new_size,expected",
        [
            (0, 0, 128, True),
            (0, 0, 129, False),
            (0, 1, 128, True),
            (0, 1, 129, False),
            (1, 0, 128, True),
            (1, 0, 129, False),
            (1, 1, 512, True),
            (1, 1, 513, False),
            (2, 0, 128, True),
            (2, 1, 512, True),
        ],
        ids=[
            "lhs_K_at_limit",
            "lhs_K_over",
            "lhs_M_at_limit",
            "lhs_M_over",
            "rhs_K_at_limit",
            "rhs_K_over",
            "rhs_N_at_limit",
            "rhs_N_over",
            "dst_M_at_limit",
            "dst_N_at_limit",
        ],
    )
    def test_nc_matmul_can_merge_operand_dim(self, operand_idx: int, dim: int, new_size: int, expected: bool) -> None:
        """Verify can_merge_operand_dim maps operand+dim to correct tile limit."""
        assert NC_MATMUL_OP.can_merge_operand_dim(operand_idx, dim, new_size) is expected

    def test_load_op_can_merge_operand_dim_always_true(self) -> None:
        """Verify LoadOp can_merge_operand_dim returns True (no tile limits)."""
        assert LOAD_OP.can_merge_operand_dim(0, 0, 9999) is True

    def test_alloc_op_can_merge_operand_dim_always_true(self) -> None:
        """Verify AllocOp can_merge_operand_dim returns True (no tile limits)."""
        assert ALLOC_F32_OP.can_merge_operand_dim(0, 0, 9999) is True


class TestCallableToIR:
    """Tests for callable_to_ir parsing."""

    def test_single_tile_matmul_structure(self, matmul_func):
        """Verify IR structure for a single-tile matmul."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)

        assert program.name == "matmul"
        assert program.params == ("a", "b")
        assert program.return_var == "output"
        assert len(program.stmts) > 0
        assert isinstance(program.stmts[0][0], AllocOp)

    def test_multi_tile_op_types(self, matmul_func):
        """Verify all expected op types present for multi-tile matmul."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float64))
        program = callable_to_ir(tiled)

        op_types = {type(op) for op, _ in program.stmts}
        assert AllocOp in op_types
        assert LoadOp in op_types
        assert StoreOp in op_types
        assert NKIMatmul in op_types

    def test_alloc_singleton_f32(self, matmul_func):
        """Verify alloc op uses ALLOC_F32_OP singleton for float32."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        assert program.stmts[0][0] is ALLOC_F32_OP

    def test_alloc_singleton_f64(self, matmul_func):
        """Verify alloc op uses ALLOC_F64_OP singleton for float64."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float64))
        program = callable_to_ir(tiled)
        assert program.stmts[0][0] is ALLOC_F64_OP

    def test_compute_op_is_registry_singleton(self, matmul_func):
        """Verify compute ops reference NC_MATMUL_OP singleton."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        compute_stmts = [s for s in program.stmts if isinstance(s[0], NKIMatmul)]
        assert len(compute_stmts) > 0
        for op, _ in compute_stmts:
            assert op is NC_MATMUL_OP

    def test_load_op_is_singleton(self, matmul_func):
        """Verify load ops reference the LOAD_OP singleton."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32))
        program = callable_to_ir(tiled)
        load_stmts = [s for s in program.stmts if isinstance(s[0], LoadOp)]
        assert len(load_stmts) > 0
        for op, _ in load_stmts:
            assert op is LOAD_OP

    def test_store_op_is_singleton(self, matmul_func):
        """Verify store ops reference the STORE_OP singleton."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32))
        program = callable_to_ir(tiled)
        store_stmts = [s for s in program.stmts if isinstance(s[0], StoreOp)]
        assert len(store_stmts) > 0
        for op, _ in store_stmts:
            assert op is STORE_OP

    def test_reduction_tiling_stmt_count(self, matmul_func):
        """Verify correct number of compute stmts for reduction tiling."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (256, 128), "b": (256, 128)}, np.float64))
        program = callable_to_ir(tiled)
        compute_stmts = [s for s in program.stmts if isinstance(s[0], NKIMatmul)]
        assert len(compute_stmts) == 2

    def test_load_operand_slices(self, matmul_func):
        """Verify load operand slices are correct (src absolute, dst 0-based)."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32))
        program = callable_to_ir(tiled)
        load_stmts = [s for s in program.stmts if isinstance(s[0], LoadOp)]

        first_load = load_stmts[0]
        src_var, src_slices = first_load[1][0]
        dst_var, dst_slices = first_load[1][1]

        assert src_var in ("a", "b")
        for start, stop in dst_slices:
            assert start == 0


class TestCallableToIRErrors:
    """Tests for callable_to_ir error paths."""

    def test_unsupported_subscript_type(self) -> None:
        """Verify ValueError for non-slice subscript elements."""

        def bad_func(a):
            """Bad function."""
            tensor_0 = a[0, 0:128]
            return tensor_0

        bad_func.__source__ = "def bad_func(a):\n    tensor_0 = a[0, 0:128]\n    return tensor_0\n"
        with pytest.raises(ValueError, match="Unsupported subscript type"):
            callable_to_ir(bad_func)

    def test_non_integer_shape_element(self) -> None:
        """Verify ValueError for non-integer shape in ndarray call."""
        source = (
            "def bad_func(a):\n" "    output = nkigym.ndarray((128, 3.14), dtype=np.float32)\n" "    return output\n"
        )

        def bad_func(a):
            """Bad function."""

        bad_func.__source__ = source
        with pytest.raises(ValueError, match="Shape element must be an integer"):
            callable_to_ir(bad_func)

    def test_unknown_operation_name(self) -> None:
        """Verify ValueError for unknown nkigym operation."""
        source = (
            "def bad_func(a):\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = nkigym.unknown_op(tensor_0)\n"
            "    return tensor_1\n"
        )

        def bad_func(a):
            """Bad function."""

        bad_func.__source__ = source
        with pytest.raises(ValueError, match="Unknown operation"):
            callable_to_ir(bad_func)


class TestElementwiseIR:
    """Tests for ElementwiseOp parsing and round-trip through IR."""

    def test_tensor_tensor_parsed_as_elementwise_op(self) -> None:
        """Verify tensor_tensor call is parsed as ElementwiseOp."""
        source = (
            "def test_fn(a, b):\n"
            "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:128]\n"
            "    tensor_2 = nkigym.tensor_tensor(tensor_0, tensor_1, op=np.add)\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
            "    return output\n"
        )

        def test_fn(a, b):
            """Test function."""

        test_fn.__source__ = source
        program = callable_to_ir(test_fn)
        compute_stmts = [s for s in program.stmts if isinstance(s[0], ElementwiseOp)]
        assert len(compute_stmts) == 1
        assert compute_stmts[0][0].op_name == "tensor_tensor"
        assert ("op", "np.add") in compute_stmts[0][0].kwargs_repr

    def test_activation_parsed_as_elementwise_op(self) -> None:
        """Verify activation call is parsed as ElementwiseOp."""
        source = (
            "def test_fn(a):\n"
            "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = nkigym.activation(tensor_0, op=np.tanh)\n"
            "    output[0:128, 0:128] = tensor_1[0:128, 0:128]\n"
            "    return output\n"
        )

        def test_fn(a):
            """Test function."""

        test_fn.__source__ = source
        program = callable_to_ir(test_fn)
        compute_stmts = [s for s in program.stmts if isinstance(s[0], ElementwiseOp)]
        assert len(compute_stmts) == 1
        assert compute_stmts[0][0].op_name == "activation"
        assert ("op", "np.tanh") in compute_stmts[0][0].kwargs_repr

    def test_elementwise_source_round_trip(self) -> None:
        """Verify elementwise op survives callable_to_ir -> ir_to_source round-trip."""
        source = (
            "def test_fn(a, b):\n"
            "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    tensor_1 = b[0:128, 0:128]\n"
            "    tensor_2 = nkigym.tensor_tensor(tensor_0, tensor_1, op=np.add)\n"
            "    output[0:128, 0:128] = tensor_2[0:128, 0:128]\n"
            "    return output\n"
        )

        def test_fn(a, b):
            """Test function."""

        test_fn.__source__ = source
        program = callable_to_ir(test_fn)
        rendered = ir_to_source(program)
        assert "nkigym.tensor_tensor(" in rendered
        assert "op=np.add" in rendered


class TestAugAssign:
    """Tests for _parse_aug_assign handling of augmented assignment."""

    def test_aug_assign_produces_nc_matmul_stmt(self, matmul_func) -> None:
        """Verify augmented assignment (+=) is parsed with correct op and accumulator."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (256, 128), "b": (256, 128)}, np.float32))
        program = callable_to_ir(tiled)

        compute_stmts = [s for s in program.stmts if isinstance(s[0], NKIMatmul)]
        assert len(compute_stmts) >= 2

        dst_vars = [s[1][-1][0] for s in compute_stmts]
        assert len(set(dst_vars)) == 1

    def test_aug_assign_round_trip_source(self, matmul_func) -> None:
        """Verify augmented assignment survives source round-trip with += syntax."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (256, 128), "b": (256, 128)}, np.float32))
        program = callable_to_ir(tiled)
        source = ir_to_source(program)
        assert "+=" in source


class TestGenerateTiledIR:
    """Tests for generate_tiled_ir producing correct program tuples directly."""

    def test_program_structure(self, matmul_func) -> None:
        """Verify generate_tiled_ir returns a well-formed program tuple."""
        program = generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32)

        assert program.name == "matmul"
        assert program.params == ("a", "b")
        assert program.return_var == "output"
        assert len(program.stmts) > 0

    def test_op_types_present(self, matmul_func) -> None:
        """Verify expected op types in generated IR."""
        program = generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32)

        op_types = {type(op) for op, _ in program.stmts}
        assert AllocOp in op_types
        assert LoadOp in op_types
        assert StoreOp in op_types
        assert NKIMatmul in op_types

    def test_first_stmt_is_alloc(self, matmul_func) -> None:
        """Verify first statement is always an allocation."""
        program = generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32)
        assert isinstance(program.stmts[0][0], AllocOp)

    def test_ir_to_callable_numerical(self, matmul_func) -> None:
        """Verify ir_to_callable on generate_tiled_ir output is numerically correct."""
        a_shape, b_shape = (128, 256), (128, 256)
        program = generate_tiled_ir(matmul_func, {"a": a_shape, "b": b_shape}, np.float32)
        func = ir_to_callable(program)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul_func(a, b)
        actual = func(a, b)
        assert_arrays_close(actual, expected)


class TestPreamblePreservation:
    """Tests for preamble (def line + docstring) preservation through IR."""

    def test_preamble_in_program(self, matmul_func) -> None:
        """Verify program.preamble contains the original def line."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        assert "def matmul(a, b):" in program.preamble

    def test_preamble_round_trip(self) -> None:
        """Verify preamble with type annotations and docstring survives round-trip."""
        source = (
            "def my_func(a, b):\n"
            '    """My docstring."""\n'
            "    output = nkigym.ndarray((128, 128), dtype=np.float32)\n"
            "    tensor_0 = a[0:128, 0:128]\n"
            "    output[0:128, 0:128] = tensor_0[0:128, 0:128]\n"
            "    return output\n"
        )

        def my_func(a, b):
            """My docstring."""

        my_func.__source__ = source
        program = callable_to_ir(my_func)
        rendered = ir_to_source(program)
        assert "def my_func(a, b):" in rendered
        assert '"""My docstring."""' in rendered


class TestIRToSource:
    """Tests for ir_to_source rendering."""

    def test_renders_valid_python(self, matmul_func):
        """Verify ir_to_source produces syntactically valid Python."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        source = ir_to_source(program)
        compile(source, "<test>", "exec")

    def test_contains_function_def(self, matmul_func):
        """Verify rendered source contains the function definition."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        source = ir_to_source(program)
        assert "def matmul(a, b):" in source

    def test_contains_return(self, matmul_func):
        """Verify rendered source contains return statement."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        source = ir_to_source(program)
        assert "return output" in source

    def test_handcrafted_alloc_stmt(self):
        """Verify ir_to_source renders AllocOp correctly from hand-built program."""
        stmts: tuple[Statement, ...] = ((ALLOC_F32_OP, (("output", ((0, 128), (0, 256))),)),)
        program = Program("test_fn", ("a",), stmts, "output", "def test_fn(a):")
        source = ir_to_source(program)
        assert "output = nkigym.ndarray((128, 256), dtype=np.float32)" in source

    def test_handcrafted_load_stmt(self):
        """Verify ir_to_source renders LoadOp correctly from hand-built program."""
        stmts: tuple[Statement, ...] = ((LOAD_OP, (("a", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),)
        program = Program("test_fn", ("a",), stmts, "output", "def test_fn(a):")
        source = ir_to_source(program)
        assert "tensor_0 = a[0:128, 0:128]" in source

    def test_handcrafted_store_stmt(self):
        """Verify ir_to_source renders StoreOp correctly from hand-built program."""
        stmts: tuple[Statement, ...] = (
            (STORE_OP, (("tensor_0", ((0, 128), (0, 128))), ("output", ((0, 128), (0, 128))))),
        )
        program = Program("test_fn", ("a",), stmts, "output", "def test_fn(a):")
        source = ir_to_source(program)
        assert "output[0:128, 0:128] = tensor_0[0:128, 0:128]" in source


class TestRoundTrip:
    """Tests for round-trip correctness."""

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [((128, 128), (128, 128)), ((128, 256), (128, 256)), ((128, 256), (128, 128)), ((256, 256), (256, 256))],
        ids=["128x128_128x128", "128x256_128x256", "128x256_128x128", "256x256_256x256"],
    )
    def test_source_round_trip(self, a_shape, b_shape, matmul_func):
        """Verify ir_to_source(callable_to_ir(func)) produces equivalent source."""
        input_shapes = {"a": a_shape, "b": b_shape}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, input_shapes, np.float32))
        original_source = tiled.__source__

        program = callable_to_ir(tiled)
        round_tripped_source = ir_to_source(program)

        assert normalize_source(round_tripped_source) == normalize_source(original_source)

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [
            ((128, 128), (128, 128)),
            ((128, 256), (128, 256)),
            ((256, 256), (256, 256)),
            ((128, 512), (128, 128)),
            ((512, 128), (512, 128)),
        ],
        ids=["128x128_128x128", "128x256_128x256", "256x256_256x256", "128x512_128x128", "512x128_512x128"],
    )
    def test_numerical_round_trip(self, a_shape, b_shape, matmul_func):
        """Verify ir_to_callable(callable_to_ir(func)) produces numerically equivalent results."""
        input_shapes = {"a": a_shape, "b": b_shape}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, input_shapes, np.float32))

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = tiled(a, b)

        program = callable_to_ir(tiled)
        round_tripped = ir_to_callable(program)
        actual = round_tripped(a, b)

        assert_arrays_close(actual, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_preserved_round_trip(self, dtype, matmul_func):
        """Verify dtype is preserved through round-trip."""
        input_shapes = {"a": (128, 128), "b": (128, 128)}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, input_shapes, dtype))

        program = callable_to_ir(tiled)
        source = ir_to_source(program)

        dtype_name = np.dtype(dtype).name
        assert f"np.{dtype_name}" in source

    def test_reduction_round_trip(self, matmul_func):
        """Verify round-trip for matmul with reduction tiling (K > 128)."""
        input_shapes = {"a": (256, 128), "b": (256, 128)}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, input_shapes, np.float64))

        a = make_random_array((256, 128), seed=42)
        b = make_random_array((256, 128), seed=43)
        expected = tiled(a, b)

        program = callable_to_ir(tiled)
        round_tripped = ir_to_callable(program)
        actual = round_tripped(a, b)

        assert_arrays_close(actual, expected)

    def test_double_matmul_round_trip(self, double_matmul_func):
        """Verify round-trip for double matmul: D = (A @ B) @ C."""
        input_shapes = {"a": (128, 128), "b": (128, 128), "c": (128, 128)}
        tiled = ir_to_callable(generate_tiled_ir(double_matmul_func, input_shapes, np.float32))

        a = make_random_array((128, 128), seed=42)
        b = make_random_array((128, 128), seed=43)
        c = make_random_array((128, 128), seed=44)
        expected = tiled(a, b, c)

        program = callable_to_ir(tiled)
        round_tripped = ir_to_callable(program)
        actual = round_tripped(a, b, c)

        assert_arrays_close(actual, expected)

    def test_f64_numerical_round_trip(self, matmul_func):
        """Verify numerical round-trip for float64."""
        input_shapes = {"a": (128, 128), "b": (128, 128)}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, input_shapes, np.float64))

        a = make_random_array((128, 128), seed=42, dtype=np.float64)
        b = make_random_array((128, 128), seed=43, dtype=np.float64)
        expected = tiled(a, b)

        program = callable_to_ir(tiled)
        round_tripped = ir_to_callable(program)
        actual = round_tripped(a, b)

        assert_arrays_close(actual, expected)

    def test_program_is_hashable(self, matmul_func):
        """Verify program tuples are hashable (usable as dict keys)."""
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        program = callable_to_ir(tiled)
        d = {program: "test_value"}
        assert d[program] == "test_value"

    def test_program_dict_key_different_shapes(self, matmul_func):
        """Verify different programs produce different dict keys."""
        tiled_1 = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 128), "b": (128, 128)}, np.float32))
        tiled_2 = ir_to_callable(generate_tiled_ir(matmul_func, {"a": (128, 256), "b": (128, 256)}, np.float32))
        p1 = callable_to_ir(tiled_1)
        p2 = callable_to_ir(tiled_2)
        d = {p1: "small", p2: "large"}
        assert len(d) == 2
        assert d[p1] == "small"
        assert d[p2] == "large"


class TestIREdgeCases:
    """Tests for edge cases in IR conversion."""

    def test_empty_statements_program(self) -> None:
        """Verify ir_to_source handles a program with no statements."""
        program = Program("empty_fn", ("a",), (), "a", "def empty_fn(a):")
        source = ir_to_source(program)
        assert "def empty_fn(a):" in source
        assert "return a" in source
        compiled = compile(source, "<test>", "exec")
        assert compiled is not None

    def test_alloc_only_program(self) -> None:
        """Verify ir_to_source handles a program with only an allocation."""
        stmts: tuple[Statement, ...] = ((ALLOC_F32_OP, (("output", ((0, 128), (0, 128))),)),)
        program = Program("alloc_fn", (), stmts, "output", "def alloc_fn():")
        source = ir_to_source(program)
        assert "nkigym.ndarray((128, 128), dtype=np.float32)" in source
        assert "return output" in source
        compiled = compile(source, "<test>", "exec")
        assert compiled is not None

    def test_preamble_with_type_annotations(self) -> None:
        """Verify ir_to_source preserves preamble with type annotations."""
        preamble = "def typed_fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:"
        stmts: tuple[Statement, ...] = ((LOAD_OP, (("a", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),)
        program = Program("typed_fn", ("a", "b"), stmts, "tensor_0", preamble)
        source = ir_to_source(program)
        assert "def typed_fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:" in source
        assert "def typed_fn(a, b):" not in source

    def test_preamble_with_multiline_docstring(self) -> None:
        """Verify ir_to_source preserves preamble with multiline docstring."""
        preamble = 'def doc_fn(a):\n    """First line.\n\n    Second line.\n    """'
        stmts: tuple[Statement, ...] = ((LOAD_OP, (("a", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),)
        program = Program("doc_fn", ("a",), stmts, "tensor_0", preamble)
        source = ir_to_source(program)
        assert '"""First line.' in source
        assert "Second line." in source

    def test_empty_preamble_generates_def_line(self) -> None:
        """Verify ir_to_source generates def line when preamble is empty."""
        stmts: tuple[Statement, ...] = ((ALLOC_F32_OP, (("output", ((0, 128), (0, 128))),)),)
        program = Program("my_fn", ("x", "y"), stmts, "output", "")
        source = ir_to_source(program)
        assert "def my_fn(x, y):" in source

    def test_ir_to_callable_empty_statements(self) -> None:
        """Verify ir_to_callable produces a working function for empty statements.

        The function just returns the named variable (a parameter).
        """
        program = Program("passthrough", ("a",), (), "a", "def passthrough(a):")
        func = ir_to_callable(program)
        a = make_random_array((128, 128), seed=42)
        result = func(a)
        np.testing.assert_array_equal(result, a)

    def test_empty_program_to_source(self) -> None:
        """Verify ir_to_source on a program with params but empty statements."""
        program = Program("identity", ("x", "y"), (), "x", "def identity(x, y):")
        source = ir_to_source(program)
        assert "def identity(x, y):" in source
        assert "return x" in source
        compiled = compile(source, "<test>", "exec")
        assert compiled is not None

    def test_preamble_in_source(self) -> None:
        """Verify preamble content appears verbatim in ir_to_source output."""
        preamble = "def fn_with_preamble(a):\n    import math\n"
        stmts: tuple[Statement, ...] = ((ALLOC_F32_OP, (("output", ((0, 128), (0, 128))),)),)
        program = Program("fn_with_preamble", ("a",), stmts, "output", preamble)
        source = ir_to_source(program)
        assert "import math" in source
        assert "def fn_with_preamble(a):" in source
