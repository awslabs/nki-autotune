"""Tests for gym_to_nki IR-based lowering.

Verifies that lower_ir_to_nki produces correct NKI kernel code from
IR programs.
"""

import numpy as np
import pytest

import nkigym
from nkigym.codegen.gym_to_nki import lower_ir_to_nki
from nkigym.ir import Program, callable_to_ir, ir_to_callable
from nkigym.ops import ALLOC_F32_OP, ALLOC_F64_OP, LOAD_OP, NC_MATMUL_OP, STORE_OP, ElementwiseOp
from nkigym.tiling import generate_tiled_ir


def _make_single_matmul_program() -> Program:
    """Build a hand-crafted IR program for a single matmul tile.

    The program represents:
        output = alloc((128, 128), float32)
        tensor_0 = lhsT[0:128, 0:128]
        tensor_1 = rhs[0:128, 0:128]
        tensor_2 = nc_matmul(tensor_0, tensor_1)
        output[0:128, 0:128] = tensor_2
        return output
    """
    stmts = (
        (ALLOC_F32_OP, (("output", ((0, 128), (0, 128))),)),
        (LOAD_OP, (("lhsT", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),
        (LOAD_OP, (("rhs", ((0, 128), (0, 128))), ("tensor_1", ((0, 128), (0, 128))))),
        (
            NC_MATMUL_OP,
            (
                ("tensor_0", ((0, 128), (0, 128))),
                ("tensor_1", ((0, 128), (0, 128))),
                ("tensor_2", ((0, 128), (0, 128))),
            ),
        ),
        (STORE_OP, (("tensor_2", ((0, 128), (0, 128))), ("output", ((0, 128), (0, 128))))),
    )
    return Program("tiled_func", ("lhsT", "rhs"), stmts, "output", "def tiled_func(lhsT, rhs):")


def _make_accumulate_program() -> Program:
    """Build a hand-crafted IR program with nc_matmul accumulation.

    The program represents a reduction-tiled matmul where two nc_matmul
    calls accumulate into the same destination.
    """
    stmts = (
        (ALLOC_F32_OP, (("output", ((0, 128), (0, 128))),)),
        (LOAD_OP, (("lhsT", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),
        (LOAD_OP, (("rhs", ((0, 128), (0, 128))), ("tensor_1", ((0, 128), (0, 128))))),
        (
            NC_MATMUL_OP,
            (
                ("tensor_0", ((0, 128), (0, 128))),
                ("tensor_1", ((0, 128), (0, 128))),
                ("tensor_2", ((0, 128), (0, 128))),
            ),
        ),
        (LOAD_OP, (("lhsT", ((128, 256), (0, 128))), ("tensor_3", ((0, 128), (0, 128))))),
        (LOAD_OP, (("rhs", ((128, 256), (0, 128))), ("tensor_4", ((0, 128), (0, 128))))),
        (
            NC_MATMUL_OP,
            (
                ("tensor_3", ((0, 128), (0, 128))),
                ("tensor_4", ((0, 128), (0, 128))),
                ("tensor_2", ((0, 128), (0, 128))),
            ),
        ),
        (STORE_OP, (("tensor_2", ((0, 128), (0, 128))), ("output", ((0, 128), (0, 128))))),
    )
    return Program("tiled_func", ("lhsT", "rhs"), stmts, "output", "def tiled_func(lhsT, rhs):")


class TestLowerIrToNki:
    """Tests for lower_ir_to_nki with hand-built and generated programs."""

    def test_imports_present(self) -> None:
        """Verify the generated code contains required NKI imports."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "import nki" in nki_code
        assert "import nki.isa as nisa" in nki_code
        assert "import nki.language as nl" in nki_code
        assert "import numpy as np" in nki_code

    def test_function_header(self) -> None:
        """Verify the generated function has @nki.jit decorator and correct name."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "@nki.jit" in nki_code
        assert "def nki_tiled_func(lhsT, rhs):" in nki_code

    def test_alloc_generates_hbm(self) -> None:
        """Verify AllocOp generates HBM ndarray allocation."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "output = nl.ndarray(shape=(128, 128), dtype=lhsT.dtype, buffer=nl.shared_hbm)" in nki_code

    def test_load_generates_sbuf_and_dma(self) -> None:
        """Verify LoadOp generates SBUF allocation and DMA copy."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "tensor_0 = nl.ndarray(shape=(128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)" in nki_code
        assert "nisa.dma_copy(dst=tensor_0[0:128, 0:128], src=lhsT[0:128, 0:128])" in nki_code

    def test_matmul_generates_psum_alloc(self) -> None:
        """Verify first nc_matmul generates PSUM buffer allocation."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "nl.zeros(" in nki_code
        assert "buffer=nl.psum" in nki_code

    def test_store_with_psum_generates_tensor_copy(self) -> None:
        """Verify store of PSUM tensor generates tensor_copy to SBUF first."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "tensor_2_sbuf = nl.ndarray(shape=(128, 128)" in nki_code
        assert "nisa.tensor_copy(dst=tensor_2_sbuf, src=tensor_2[0:128, 0:128]" in nki_code
        assert "nisa.dma_copy(dst=output[0:128, 0:128], src=tensor_2_sbuf)" in nki_code

    def test_return_var(self) -> None:
        """Verify the return statement uses the correct variable."""
        program = _make_single_matmul_program()
        nki_code = lower_ir_to_nki(program)

        assert "    return output" in nki_code

    def test_accumulate_uses_nisa_nc_matmul(self) -> None:
        """Verify second nc_matmul to same dst uses nisa.nc_matmul (accumulate).

        The first nc_matmul also generates nisa.nc_matmul (via generate_nki),
        but paired with a nl.zeros allocation. The second (accumulate) call
        should be a standalone nisa.nc_matmul with no allocation.
        """
        program = _make_accumulate_program()
        nki_code = lower_ir_to_nki(program)

        assert "nl.zeros(" in nki_code
        assert "nisa.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])" in nki_code

    def test_float64_alloc(self) -> None:
        """Verify float64 alloc generates correct HBM allocation."""
        stmts = (
            (ALLOC_F64_OP, (("output", ((0, 64), (0, 256))),)),
            (STORE_OP, (("tensor_0", ()), ("output", ((0, 64), (0, 256))))),
        )
        program = Program("f64_func", ("x",), stmts, "output", "def f64_func(x):")
        nki_code = lower_ir_to_nki(program)

        assert "output = nl.ndarray(shape=(64, 256), dtype=x.dtype, buffer=nl.shared_hbm)" in nki_code

    def test_generated_tiled_program(self) -> None:
        """Verify lowering works with a program from generate_tiled_function."""

        def matmul_func(lhsT, rhs):
            """Single tile matmul kernel."""
            return nkigym.nc_matmul(lhsT, rhs)

        input_shapes = {"lhsT": (128, 128), "rhs": (128, 128)}
        tiled = ir_to_callable(generate_tiled_ir(matmul_func, input_shapes, np.float32))
        nki_code = lower_ir_to_nki(callable_to_ir(tiled))

        assert "@nki.jit" in nki_code
        assert "nl.ndarray" in nki_code
        assert "nisa.dma_copy" in nki_code
        assert "return output" in nki_code


class TestLowerErrors:
    """Tests for error handling in lower_ir_to_nki."""

    def test_elementwise_raises_not_implemented(self) -> None:
        """Verify ElementwiseOp raises NotImplementedError."""
        ew_op = ElementwiseOp("tensor_tensor", (("op", "np.add"),))
        stmts = (
            (LOAD_OP, (("x", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),
            (
                ew_op,
                (
                    ("tensor_0", ((0, 128), (0, 128))),
                    ("tensor_1", ((0, 128), (0, 128))),
                    ("tensor_2", ((0, 128), (0, 128))),
                ),
            ),
        )
        program = Program("ew_func", ("x",), stmts, "tensor_2", "def ew_func(x):")

        with pytest.raises(NotImplementedError, match="ElementwiseOp.*tensor_tensor.*not yet supported"):
            lower_ir_to_nki(program)

    def test_store_non_psum_skips_tensor_copy(self) -> None:
        """Verify store of non-PSUM tensor does not generate tensor_copy."""
        stmts = (
            (ALLOC_F32_OP, (("output", ((0, 128), (0, 128))),)),
            (LOAD_OP, (("x", ((0, 128), (0, 128))), ("tensor_0", ((0, 128), (0, 128))))),
            (STORE_OP, (("tensor_0", ((0, 128), (0, 128))), ("output", ((0, 128), (0, 128))))),
        )
        program = Program("copy_func", ("x",), stmts, "output", "def copy_func(x):")
        nki_code = lower_ir_to_nki(program)

        assert "tensor_copy" not in nki_code
        assert "nisa.dma_copy(dst=output[0:128, 0:128], src=tensor_0[0:128, 0:128])" in nki_code
