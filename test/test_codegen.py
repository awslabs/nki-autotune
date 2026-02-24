"""Tests for GymProgram-to-NKI kernel code generation."""

import ast
import os
import re
from pathlib import Path

import numpy as np
import pytest
from golden.autotune import NEURON_DEVICES_AVAILABLE
from nkipy.runtime import BaremetalExecutor

import nkigym
from autotune.compiler.compile import TensorStub, compile_kernel, create_spike_kernel, run_spike_kernel
from nkigym.codegen import lower_to_nki
from nkigym.ir import GymProgram, GymStatement, TensorRef, source_to_program
from nkigym.search.benchmark import benchmark_variants
from nkigym.tiling import tile_program
from nkigym.utils import callable_to_source


def _has_line(code: str, pattern: str) -> bool:
    """Check if any line in code matches a regex pattern."""
    return any(re.search(pattern, line) for line in code.splitlines())


def _count_occurrences(code: str, pattern: str) -> int:
    """Count how many lines match a regex pattern."""
    return sum(1 for line in code.splitlines() if re.search(pattern, line))


_T = TensorRef
_S = GymStatement
_P = GymProgram
_128 = (0, 128)
_FULL_128x128 = (_128, _128)

SINGLE_TILE_MATMUL = _P(
    "single_matmul",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 128))),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("output", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("a", (128, 128), _FULL_128x128)),), _T("_t0", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (128, 128), _FULL_128x128)),), _T("_t1", (128, 128), _FULL_128x128)),
        _S(
            "nc_matmul",
            (("stationary", _T("_t0", (128, 128), _FULL_128x128)), ("moving", _T("_t1", (128, 128), _FULL_128x128))),
            _T("_t2", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t2", (128, 128), _FULL_128x128)), ("dst", _T("output", (128, 128), _FULL_128x128))),
            _T("output", (128, 128), _FULL_128x128),
        ),
    ),
    "output",
    np.float32,
)

ACCUM_MATMUL = _P(
    "accum_matmul",
    ("a", "b"),
    (("a", (256, 128)), ("b", (256, 128))),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("output", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("a", (256, 128), _FULL_128x128)),), _T("_t0", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (256, 128), _FULL_128x128)),), _T("_t1", (128, 128), _FULL_128x128)),
        _S(
            "nc_matmul",
            (("stationary", _T("_t0", (128, 128), _FULL_128x128)), ("moving", _T("_t1", (128, 128), _FULL_128x128))),
            _T("_t2", (128, 128), _FULL_128x128),
        ),
        _S("np_slice", (("src", _T("a", (256, 128), ((128, 256), _128))),), _T("_t3", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (256, 128), ((128, 256), _128))),), _T("_t4", (128, 128), _FULL_128x128)),
        _S(
            "nc_matmul",
            (
                ("stationary", _T("_t3", (128, 128), _FULL_128x128)),
                ("moving", _T("_t4", (128, 128), _FULL_128x128)),
                ("acc", _T("_t2", (128, 128), _FULL_128x128)),
            ),
            _T("_t5", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t5", (128, 128), _FULL_128x128)), ("dst", _T("output", (128, 128), _FULL_128x128))),
            _T("output", (128, 128), _FULL_128x128),
        ),
    ),
    "output",
    np.float32,
)

ELEMENTWISE_PROGRAM = _P(
    "ewise_fn",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 128))),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("output", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("a", (128, 128), _FULL_128x128)),), _T("_t0", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (128, 128), _FULL_128x128)),), _T("_t1", (128, 128), _FULL_128x128)),
        _S(
            "tensor_tensor",
            (
                ("data1", _T("_t0", (128, 128), _FULL_128x128)),
                ("data2", _T("_t1", (128, 128), _FULL_128x128)),
                ("op", "np.multiply"),
            ),
            _T("_t2", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t2", (128, 128), _FULL_128x128)), ("dst", _T("output", (128, 128), _FULL_128x128))),
            _T("output", (128, 128), _FULL_128x128),
        ),
    ),
    "output",
    np.float32,
)

TENSOR_SCALAR_PROGRAM = _P(
    "ts_fn",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 1))),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("out", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("a", (128, 128), _FULL_128x128)),), _T("_t0", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (128, 1), ((0, 128), (0, 1)))),), _T("_t1", (128, 1), ((0, 128), (0, 1)))),
        _S(
            "tensor_scalar",
            (
                ("data", _T("_t0", (128, 128), _FULL_128x128)),
                ("operand0", _T("_t1", (128, 1), ((0, 128), (0, 1)))),
                ("op", "np.add"),
            ),
            _T("_t2", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t2", (128, 128), _FULL_128x128)), ("dst", _T("out", (128, 128), _FULL_128x128))),
            _T("out", (128, 128), _FULL_128x128),
        ),
    ),
    "out",
    np.float32,
)

ACTIVATION_PROGRAM = _P(
    "act_fn",
    ("a",),
    (("a", (128, 128)),),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("out", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("a", (128, 128), _FULL_128x128)),), _T("_t0", (128, 128), _FULL_128x128)),
        _S(
            "activation",
            (("data", _T("_t0", (128, 128), _FULL_128x128)), ("op", "np.tanh")),
            _T("_t1", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t1", (128, 128), _FULL_128x128)), ("dst", _T("out", (128, 128), _FULL_128x128))),
            _T("out", (128, 128), _FULL_128x128),
        ),
    ),
    "out",
    np.float32,
)

_64 = (0, 64)

TRANSPOSE_PROGRAM = _P(
    "trans_fn",
    ("a",),
    (("a", (128, 64)),),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("out", (64, 128), (_64, _128))),
        _S("np_slice", (("src", _T("a", (128, 64), (_128, _64))),), _T("_t0", (128, 64), (_128, _64))),
        _S("nc_transpose", (("data", _T("_t0", (128, 64), (_128, _64))),), _T("_t1", (64, 128), (_64, _128))),
        _S(
            "np_store",
            (("src", _T("_t1", (64, 128), (_64, _128))), ("dst", _T("out", (64, 128), (_64, _128)))),
            _T("out", (64, 128), (_64, _128)),
        ),
    ),
    "out",
    np.float32,
)

MULTI_TILE_PROGRAM = _P(
    "multi_tile",
    ("a", "b"),
    (("a", (128, 128)), ("b", (128, 256))),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("output", (128, 256), (_128, (0, 256)))),
        _S("np_slice", (("src", _T("a", (128, 128), _FULL_128x128)),), _T("_t0", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (128, 256), (_128, _128))),), _T("_t1", (128, 128), _FULL_128x128)),
        _S(
            "nc_matmul",
            (("stationary", _T("_t0", (128, 128), _FULL_128x128)), ("moving", _T("_t1", (128, 128), _FULL_128x128))),
            _T("_t2", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t2", (128, 128), _FULL_128x128)), ("dst", _T("output", (128, 256), (_128, _128)))),
            _T("output", (128, 256), (_128, _128)),
        ),
        _S("np_slice", (("src", _T("a", (128, 128), _FULL_128x128)),), _T("_t3", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("b", (128, 256), (_128, (128, 256)))),), _T("_t4", (128, 128), _FULL_128x128)),
        _S(
            "nc_matmul",
            (("stationary", _T("_t3", (128, 128), _FULL_128x128)), ("moving", _T("_t4", (128, 128), _FULL_128x128))),
            _T("_t5", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t5", (128, 128), _FULL_128x128)), ("dst", _T("output", (128, 256), (_128, (128, 256))))),
            _T("output", (128, 256), (_128, (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

_256 = (0, 256)
_128_256 = (128, 256)

MERGED_LOAD_MATMUL = _P(
    "merged_matmul",
    ("a", "b"),
    (("a", (256, 256)), ("b", (256, 256))),
    (
        _S("np_empty", (("dtype", "np.float32"),), _T("output", (128, 128), _FULL_128x128)),
        _S("np_slice", (("src", _T("a", (256, 256), (_128, _256))),), _T("_t0", (128, 256), (_128, _256))),
        _S("np_slice", (("src", _T("b", (256, 256), (_128, _256))),), _T("_t1", (128, 256), (_128, _256))),
        _S(
            "nc_matmul",
            (("stationary", _T("_t0", (128, 128), _FULL_128x128)), ("moving", _T("_t1", (128, 128), _FULL_128x128))),
            _T("_t2", (128, 128), _FULL_128x128),
        ),
        _S(
            "nc_matmul",
            (
                ("stationary", _T("_t0", (128, 128), (_128, _128_256))),
                ("moving", _T("_t1", (128, 128), (_128, _128_256))),
                ("acc", _T("_t2", (128, 128), _FULL_128x128)),
            ),
            _T("_t3", (128, 128), _FULL_128x128),
        ),
        _S(
            "np_store",
            (("src", _T("_t3", (128, 128), _FULL_128x128)), ("dst", _T("output", (128, 128), _FULL_128x128))),
            _T("output", (128, 128), _FULL_128x128),
        ),
    ),
    "output",
    np.float32,
)


def test_simple_matmul_structure() -> None:
    """Single-tile matmul: verify imports, decorator, signature, structure."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)

    assert _has_line(code, r"import nki$")
    assert _has_line(code, r"import nki\.language as nl")
    assert _has_line(code, r"import nki\.isa as nisa")
    assert _has_line(code, r"@nki\.jit")
    assert _has_line(code, r"def single_matmul\(a, b\):")
    assert _has_line(code, r"return output")
    assert _has_line(code, r"nl\.ndarray\(.*, buffer=nl\.shared_hbm\)")
    assert _count_occurrences(code, r"nisa\.dma_copy") >= 2
    assert _has_line(code, r"nl\.ndarray\(.*, buffer=nl\.psum\)")
    assert _has_line(code, r"nisa\.nc_matmul\(dst=")
    assert _has_line(code, r"nisa\.tensor_copy\(dst=")


def test_matmul_accumulation() -> None:
    """Two-tile reduction: first allocs PSUM, second accumulates."""
    code = lower_to_nki(ACCUM_MATMUL)

    assert _count_occurrences(code, r"nl\.ndarray\(.*, buffer=nl\.psum\)") == 1
    assert _count_occurrences(code, r"nisa\.nc_matmul\(dst=") == 2
    assert _count_occurrences(code, r"nisa\.tensor_copy\(dst=") == 1


def test_np_empty_lowering() -> None:
    """Output allocation maps to nl.ndarray with shared_hbm buffer."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)

    assert _has_line(code, r"output = nl\.ndarray\(.*, buffer=nl\.shared_hbm\)")


def test_np_slice_lowering() -> None:
    """Input slice maps to SBUF alloc + nisa.dma_copy."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)

    assert _has_line(code, r"_t0 = nl\.ndarray\(.*, buffer=nl\.sbuf\)")
    assert _has_line(code, r"nisa\.dma_copy\(dst=_t0, src=a")


def test_psum_staging() -> None:
    """Store of matmul result stages through SBUF via tensor_copy."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)

    assert _has_line(code, r"_staging_0 = nl\.ndarray\(.*, buffer=nl\.sbuf\)")
    assert _has_line(code, r"nisa\.tensor_copy\(dst=_staging_0, src=_t2\[0:128, 0:128\]\)")
    assert _has_line(code, r"nisa\.dma_copy\(dst=output\[0:128, 0:128\], src=_staging_0\)")


def test_sbuf_direct_store() -> None:
    """Store of element-wise result uses direct dma_copy (no staging)."""
    code = lower_to_nki(ELEMENTWISE_PROGRAM)

    assert not _has_line(code, r"nisa\.tensor_copy")
    assert _has_line(code, r"nisa\.dma_copy\(dst=output\[0:128, 0:128\], src=_t2\[0:128, 0:128\]\)")


def test_tensor_tensor_op() -> None:
    """tensor_tensor produces nisa.tensor_tensor with mapped op."""
    code = lower_to_nki(ELEMENTWISE_PROGRAM)

    assert _has_line(
        code, r"nisa\.tensor_tensor\(dst=_t2, data1=_t0\[0:128, 0:128\], data2=_t1\[0:128, 0:128\], op=nl\.multiply\)"
    )


def test_tensor_scalar_op() -> None:
    """tensor_scalar produces nisa.tensor_scalar with mapped op."""
    code = lower_to_nki(TENSOR_SCALAR_PROGRAM)

    assert _has_line(
        code, r"nisa\.tensor_scalar\(dst=_t2, data=_t0\[0:128, 0:128\], op0=nl\.add, operand0=_t1\[0:128, 0:1\]\)"
    )


def test_activation_op() -> None:
    """activation produces nisa.activation with mapped function."""
    code = lower_to_nki(ACTIVATION_PROGRAM)

    assert _has_line(code, r"nisa\.activation\(dst=_t1, op=nl\.tanh, data=_t0\[0:128, 0:128\]\)")


def test_nc_transpose_op() -> None:
    """nc_transpose produces nisa.nc_transpose."""
    code = lower_to_nki(TRANSPOSE_PROGRAM)

    assert _has_line(code, r"nisa\.nc_transpose\(dst=_t1, data=_t0\[0:128, 0:64\]\)")


def test_pipeline_integration() -> None:
    """End-to-end: user function -> tile -> lower -> valid Python AST."""

    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute matrix multiplication."""
        return nkigym.nc_matmul(a, b)

    source = callable_to_source(matmul)
    program = source_to_program(source, {"a": (128, 128), "b": (128, 128)}, np.float32)
    tiled = tile_program(program)
    code = lower_to_nki(tiled)

    ast.parse(code)
    assert _has_line(code, r"@nki\.jit")
    assert _has_line(code, r"nisa\.nc_matmul")
    assert _has_line(code, r"return")


def test_multi_tile_parallel() -> None:
    """Multiple parallel tile positions: correct slice indices."""
    code = lower_to_nki(MULTI_TILE_PROGRAM)

    assert _count_occurrences(code, r"nisa\.nc_matmul\(dst=") == 2
    assert _count_occurrences(code, r"nisa\.tensor_copy\(dst=") == 2
    assert _has_line(code, r"dst=output\[0:128, 0:128\]")
    assert _has_line(code, r"dst=output\[0:128, 128:256\]")


def test_merged_load_subscript() -> None:
    """Merged load: nc_matmul operands subscript into wider SBUF tile."""
    code = lower_to_nki(MERGED_LOAD_MATMUL)

    assert _has_line(code, r"nl\.ndarray\(\(128, 256\), .*, buffer=nl\.sbuf\)")
    assert _has_line(code, r"stationary=_t0\[0:128, 0:128\]")
    assert _has_line(code, r"moving=_t1\[0:128, 0:128\]")
    assert _has_line(code, r"stationary=_t0\[0:128, 128:256\]")
    assert _has_line(code, r"moving=_t1\[0:128, 128:256\]")


@pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices")
def test_nki_matmul_compiles(tmp_path: Path) -> None:
    """Lowered NKI matmul kernel compiles to NEFF successfully."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)
    kernel_path = tmp_path / "nki_matmul.py"
    kernel_path.write_text(code)

    a = np.zeros((128, 128), dtype=np.float32)
    b = np.zeros((128, 128), dtype=np.float32)

    neff_path = compile_kernel(
        kernel_name=(str(kernel_path), "single_matmul"),
        input_tensors={"a": a, "b": b},
        output_tensors=[("output", (128, 128), np.float32)],
        kernel_kwargs={},
        compiler_flags="",
        output_dir=str(tmp_path),
    )

    assert os.path.exists(neff_path)
    assert neff_path.endswith(".neff")


@pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices")
def test_nki_matmul_hardware(tmp_path: Path) -> None:
    """Compile lowered NKI matmul, run on Neuron, verify numerical output."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)
    kernel_path = tmp_path / "nki_matmul.py"
    kernel_path.write_text(code)

    rng = np.random.default_rng(42)
    a = rng.standard_normal((128, 128)).astype(np.float32)
    b = rng.standard_normal((128, 128)).astype(np.float32)
    input_tensors = {"a": a, "b": b}
    kernel_name = (str(kernel_path), "single_matmul")

    neff_path = compile_kernel(
        kernel_name=kernel_name,
        input_tensors=input_tensors,
        output_tensors=[("output", (128, 128), np.float32)],
        kernel_kwargs={},
        compiler_flags="",
        output_dir=str(tmp_path),
    )

    output_stubs = [TensorStub(shape=(128, 128), dtype=np.float32, name="output")]
    spike_kernel = create_spike_kernel(neff_path, kernel_name, input_tensors, output_stubs, {})

    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    with BaremetalExecutor(verbose=0) as spike:
        _, outputs = run_spike_kernel(spike, spike_kernel, input_tensors, neff_path, {})

    expected = a.T @ b
    np.testing.assert_allclose(outputs[0], expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices")
def test_nki_elementwise_hardware(tmp_path: Path) -> None:
    """Compile lowered NKI elementwise kernel, run on Neuron, verify output."""
    code = lower_to_nki(ELEMENTWISE_PROGRAM)
    kernel_path = tmp_path / "nki_ewise.py"
    kernel_path.write_text(code)

    rng = np.random.default_rng(42)
    a = rng.standard_normal((128, 128)).astype(np.float32)
    b = rng.standard_normal((128, 128)).astype(np.float32)
    input_tensors = {"a": a, "b": b}
    kernel_name = (str(kernel_path), "ewise_fn")

    neff_path = compile_kernel(
        kernel_name=kernel_name,
        input_tensors=input_tensors,
        output_tensors=[("output", (128, 128), np.float32)],
        kernel_kwargs={},
        compiler_flags="",
        output_dir=str(tmp_path),
    )

    output_stubs = [TensorStub(shape=(128, 128), dtype=np.float32, name="output")]
    spike_kernel = create_spike_kernel(neff_path, kernel_name, input_tensors, output_stubs, {})

    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    with BaremetalExecutor(verbose=0) as spike:
        _, outputs = run_spike_kernel(spike, spike_kernel, input_tensors, neff_path, {})

    expected = a * b
    np.testing.assert_allclose(outputs[0], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not NEURON_DEVICES_AVAILABLE, reason="Requires Neuron devices")
def test_benchmark_variants_pipeline(tmp_path: Path) -> None:
    """Lower matmul variant, benchmark via autotune backend, verify results."""
    code = lower_to_nki(SINGLE_TILE_MATMUL)
    (tmp_path / "nki").mkdir()
    (tmp_path / "nki" / "nki_d0_v0.py").write_text(code)

    rng = np.random.default_rng(42)
    a, b = rng.standard_normal((128, 128)).astype(np.float32), rng.standard_normal((128, 128)).astype(np.float32)

    results = benchmark_variants(
        cache_dir=tmp_path,
        func_name="single_matmul",
        kernel_kwargs={"a": a, "b": b},
        output_name="output",
        output_shape=(128, 128),
        warmup=2,
        iters=3,
    )

    assert len(results.workloads) > 0
