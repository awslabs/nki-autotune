"""End-to-end integration tests for the nkigym pipeline.

Exercises the full pipeline: function definition -> tiling -> transform
search -> codegen lowering, verifying numerical correctness and structural
properties at each stage.

Run with: pytest test/test_e2e_pipeline.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest
from conftest import assert_arrays_close, make_random_array

import nkigym
from nkigym.codegen import lower_ir_to_nki, roll_loops
from nkigym.ir import ir_to_callable, ir_to_source
from nkigym.ops import NKIMatmul
from nkigym.search import search
from nkigym.tiling import generate_tiled_ir
from nkigym.transforms import DataReuseTransform, OperandMergeTransform
from nkigym.utils.source import get_source


class TestMatmulE2E:
    """End-to-end tests for matmul through the full pipeline."""

    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [((128, 128), (128, 128)), ((128, 256), (128, 256)), ((128, 256), (128, 128))],
        ids=["1x1_1x1", "1x2_1x2", "1x2_1x1"],
    )
    def test_tile_transform_codegen(self, a_shape: tuple, b_shape: tuple) -> None:
        """Full pipeline: define -> tile -> data_reuse -> operand_merge -> codegen.

        Args:
            a_shape: Shape of first input matrix.
            b_shape: Shape of second input matrix.
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        program = generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32)
        tiled = ir_to_callable(program)
        assert_arrays_close(tiled(a, b), expected)

        reuse = DataReuseTransform()
        for group in reuse.analyze_ir(program):
            program = reuse.transform_ir(program, group)
        assert_arrays_close(ir_to_callable(program)(a, b), expected)

        merge = OperandMergeTransform()
        while True:
            opps = merge.analyze_ir(program)
            if not opps:
                break
            program = merge.transform_ir(program, opps[0])
        assert_arrays_close(ir_to_callable(program)(a, b), expected)

        nki_source = lower_ir_to_nki(program)
        assert "nki.language" in nki_source or "nl." in nki_source
        compiled = compile(nki_source, "<test>", "exec")
        assert compiled is not None

    def test_search_produces_lowerable_variants(self, tmp_path: Path) -> None:
        """Search over tiled matmul produces unique variants that lower to valid NKI."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (128, 128), (128, 256)
        kernel_kwargs = {"a": make_random_array(a_shape, seed=10), "b": make_random_array(b_shape, seed=11)}

        transforms = [DataReuseTransform(), OperandMergeTransform()]
        variants = search(matmul, transforms, math.inf, 42, 0, tmp_path, kernel_kwargs)

        assert len(variants) >= 1
        sources = {ir_to_source(v) for v in variants}
        assert len(sources) == len(variants)

        for variant in variants:
            nki_source = lower_ir_to_nki(variant)
            compiled = compile(nki_source, "<test>", "exec")
            assert compiled is not None

    def test_search_min_depth_variants_are_correct(self, tmp_path: Path) -> None:
        """Search with min_depth=1 returns numerically correct fully-transformed variants."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (128, 256), (128, 256)
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)
        kernel_kwargs = {"a": make_random_array(a_shape, seed=10), "b": make_random_array(b_shape, seed=11)}

        transforms = [DataReuseTransform()]
        variants = search(matmul, transforms, math.inf, 42, 1, tmp_path, kernel_kwargs)

        assert len(variants) >= 1
        for variant in variants:
            assert_arrays_close(ir_to_callable(variant)(a, b), expected)


class TestReductionTilingE2E:
    """End-to-end tests for reduction tiling (K > 128) through the full pipeline."""

    @pytest.mark.parametrize(
        "a_shape,b_shape", [((256, 128), (256, 128)), ((384, 128), (384, 128))], ids=["k256", "k384"]
    )
    def test_reduction_tile_transform_codegen(self, a_shape: tuple, b_shape: tuple) -> None:
        """Full pipeline with reduction tiling: define -> tile -> IR -> codegen.

        Reduction tiling splits the K dimension into 128-wide tiles with
        accumulation (+=). Verifies correctness and NKI lowering.

        Args:
            a_shape: Shape of first input matrix (K, M).
            b_shape: Shape of second input matrix (K, N).
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        program = generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32)
        tiled = ir_to_callable(program)
        assert_arrays_close(tiled(a, b), expected)

        source = get_source(tiled)
        assert "+=" in source

        compute_stmts = [s for s in program.stmts if isinstance(s[0], NKIMatmul)]
        k_tiles = a_shape[0] // 128
        assert len(compute_stmts) == k_tiles

        nki_source = lower_ir_to_nki(program)
        assert "nki.language" in nki_source or "nl." in nki_source
        compiled = compile(nki_source, "<test>", "exec")
        assert compiled is not None

    def test_reduction_with_multi_n_tiles(self, tmp_path: Path) -> None:
        """Reduction tiling combined with N-dimension tiling and transforms.

        Shape (256, 128) x (256, 256) has K=256 (2 reduction tiles) and
        N=256 (2 N tiles), producing multiple subgraphs that can be
        data-reused and operand-merged.
        """

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (256, 128), (256, 256)
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)
        kernel_kwargs = {"a": make_random_array(a_shape, seed=10), "b": make_random_array(b_shape, seed=11)}

        transforms = [DataReuseTransform(), OperandMergeTransform()]
        variants = search(matmul, transforms, 5, 42, 0, tmp_path, kernel_kwargs)

        assert len(variants) >= 1
        for variant in variants:
            assert_arrays_close(ir_to_callable(variant)(a, b), expected)

            nki_source = lower_ir_to_nki(variant)
            compiled = compile(nki_source, "<test>", "exec")
            assert compiled is not None


class TestIRPipelineE2E:
    """End-to-end tests using the IR-based pipeline (no callable bridge)."""

    def test_ir_pipeline_matmul(self) -> None:
        """Full IR pipeline: generate_tiled_ir -> analyze_ir -> transform_ir -> lower_ir_to_nki."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (128, 256), (128, 256)
        a = make_random_array(a_shape, seed=42)
        b = make_random_array(b_shape, seed=43)
        expected = matmul(a, b)

        program = generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32)

        func = ir_to_callable(program)
        assert_arrays_close(func(a, b), expected)

        reuse = DataReuseTransform()
        for group in reuse.analyze_ir(program):
            program = reuse.transform_ir(program, group)

        merge = OperandMergeTransform()
        while True:
            opps = merge.analyze_ir(program)
            if not opps:
                break
            program = merge.transform_ir(program, opps[0])

        func = ir_to_callable(program)
        assert_arrays_close(func(a, b), expected)

        nki_source = lower_ir_to_nki(program)
        assert "nki.language" in nki_source or "nl." in nki_source
        compiled = compile(nki_source, "<test>", "exec")
        assert compiled is not None

    def test_ir_pipeline_loop_rolling(self) -> None:
        """Full IR pipeline with loop rolling on a tiled nkigym function."""

        def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Compute matrix multiplication."""
            return nkigym.nc_matmul(a, b)

        a_shape, b_shape = (256, 256), (256, 256)

        tiled = ir_to_callable(generate_tiled_ir(matmul, {"a": a_shape, "b": b_shape}, np.float32))
        rolled_func = roll_loops(tiled)
        rolled_source = get_source(rolled_func)

        assert "for " in rolled_source
        compiled = compile(rolled_source, "<test>", "exec")
        assert compiled is not None
