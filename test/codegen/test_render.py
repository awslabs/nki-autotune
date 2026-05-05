"""Tests for the eager renderer."""

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


@nkigym_kernel
def _matmul_lhsT_rhs(lhs_T, rhs):
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_SPECS = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}


@nkigym_kernel
def _transpose_kernel(x):
    xs = NKILoad()(data=x)
    y = NKITranspose()(data=xs)
    out = NKIStore()(data=y)
    return out


@nkigym_kernel
def _dma_transpose_kernel(x):
    xs = NKILoad()(data=x)
    y = NKIDMATranspose()(data=xs)
    out = NKIStore()(data=y)
    return out


@nkigym_kernel
def _activation_kernel(x):
    xs = NKILoad()(data=x)
    y = NKIActivation(op="tanh", scale=0.5, bias=0.0)(data=xs)
    out = NKIStore()(data=y)
    return out


@nkigym_kernel
def _tensor_scalar_kernel(x, v):
    xs = NKILoad()(data=x)
    vs = NKILoad()(data=v)
    y = NKITensorScalar(op="multiply")(data=xs, operand0=vs)
    out = NKIStore()(data=y)
    return out


@nkigym_kernel
def _rms_kernel(x):
    xs = NKILoad()(data=x)
    m = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 128, bias=1e-6)(data=xs)
    out = NKIStore()(data=m)
    return out


EPS = 1e-6


@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=EPS)(data=lhs_sbuf)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


def test_render_emits_header_and_allocations() -> None:
    """Kernel header lists imports, decorator, signature, asserts, HBM output, SBUF allocs."""
    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    try:
        src = render(g)
    except ValueError:
        src = _header_only(g)
    assert "import nki" in src
    assert "@nki.jit" in src
    assert "def _matmul_lhsT_rhs(lhs_T, rhs):" in src
    assert "assert lhs_T.shape == (2048, 2048)" in src
    assert "assert rhs.shape == (2048, 2048)" in src
    assert "hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)" in src
    assert "sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src
    assert "sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src
    assert "sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src
    assert "sbuf_out = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)" in src


def _header_only(g) -> str:
    """Run the renderer's header path without op emission (Task-4 scope)."""
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _Writer,
    )

    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, g)
    w.indent()
    _emit_param_asserts(w, g)
    _emit_hbm_output(w, g)
    _emit_sbuf_allocations(w, g)
    return w.getvalue()


def test_sbuf_tile_slice_2d() -> None:
    """Per-tile slice for a 2D SBUF tensor uses i_block_<p> and i_block_<f>."""
    from nkigym.codegen.render import _sbuf_tile_slice

    slice_expr = _sbuf_tile_slice("sbuf_lhs", ("d0", "d1"), p_tile=128, f_tile=128)
    assert slice_expr == "sbuf_lhs[0:128, i_block_d0, i_block_d1 * 128 : i_block_d1 * 128 + 128]"


def test_sbuf_tile_slice_1d() -> None:
    """Per-tile slice for a 1D SBUF tensor (e.g. activation_reduce output) uses 0:1 on free."""
    from nkigym.codegen.render import _sbuf_tile_slice

    slice_expr = _sbuf_tile_slice("sbuf_rms", ("d0",), p_tile=128, f_tile=1)
    assert slice_expr == "sbuf_rms[0:128, i_block_d0, 0:1]"


def test_hbm_tile_slice() -> None:
    """HBM tile slice uses p_tile and f_tile offsets."""
    from nkigym.codegen.render import _hbm_tile_slice

    slice_expr = _hbm_tile_slice("lhs", ("d0", "d1"), p_tile=128, f_tile=128)
    assert slice_expr == "lhs[i_block_d0 * 128 : i_block_d0 * 128 + 128, i_block_d1 * 128 : i_block_d1 * 128 + 128]"


def test_render_load_store_kernel() -> None:
    """A bare NKILoad + NKIStore kernel renders and simulates correctly."""
    import nki
    import numpy as np

    @nkigym_kernel
    def passthrough(x):
        xs = NKILoad()(data=x)
        out = NKIStore()(data=xs)
        return out

    specs = {"x": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(passthrough, specs)
    src = render(g)
    assert "nisa.dma_copy(" in src
    assert "for i_block_" in src

    """CPU-sim the rendered kernel at fp32 and compare to the numpy identity."""
    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["passthrough"]
    rng = np.random.default_rng(0)
    x_in = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x_in)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x_in, atol=1e-5, rtol=1e-5)


def test_render_matmul_lhsT_rhs() -> None:
    """Full lhs_T @ rhs kernel renders and simulates to the numpy reference."""
    import nki
    import numpy as np

    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    src = render(g)
    assert "nisa.nc_matmul(" in src
    assert "nl.psum" in src
    assert "nisa.tensor_copy(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_matmul_lhsT_rhs"]

    rng = np.random.default_rng(0)
    lhs_T = rng.standard_normal((2048, 2048)).astype(np.float32)
    rhs = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = nki.simulate(kernel)(lhs_T=lhs_T, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = lhs_T.T @ rhs
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)


def test_render_transpose() -> None:
    """Stand-alone NKITranspose kernel renders and simulates correctly."""
    import nki
    import numpy as np

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_transpose_kernel, specs)
    src = render(g)
    assert "nisa.nc_transpose(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_transpose_kernel"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x.T, atol=1e-5)


def test_render_dma_transpose() -> None:
    """NKIDMATranspose emits nisa.dma_transpose."""
    import nki
    import numpy as np

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_dma_transpose_kernel, specs)
    src = render(g)
    assert "nisa.dma_transpose(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_dma_transpose_kernel"]
    x = np.random.default_rng(1).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x.T, atol=1e-5)


def test_render_activation() -> None:
    """NKIActivation emits nisa.activation with scale/bias plumbed through."""
    import nki
    import numpy as np

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_activation_kernel, specs)
    src = render(g)
    assert "nisa.activation(" in src
    assert "scale=0.5" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_activation_kernel"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, np.tanh(x * 0.5), atol=1e-4)


def test_render_tensor_scalar() -> None:
    """NKITensorScalar emits nisa.tensor_scalar.

    Uses a second load for operand0 (a 1D vector) so we don't depend on
    NKIActivationReduce (landing in Task 6e).
    """
    specs = {"x": ((128, 128), "bfloat16"), "v": ((128,), "float32")}
    g = parse_and_resolve(_tensor_scalar_kernel, specs)
    src = render(g)
    assert "nisa.tensor_scalar(" in src


def test_render_activation_reduce_rmsnorm() -> None:
    """activation_reduce with post_op=rsqrt emits memset + accumulate loop + post_op."""
    import nki
    import numpy as np

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)
    assert "nisa.memset(" in src
    assert "nisa.activation_reduce(" in src
    assert "nisa.activation(" in src
    assert "op=nl.rsqrt" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_rms_kernel"]
    x = np.random.default_rng(0).standard_normal((128, 128)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = 1.0 / np.sqrt(np.mean(x * x, axis=1) + 1e-6)
    assert np.allclose(actual.reshape(-1), expected, atol=5e-4, rtol=5e-4)


def test_render_rmsnorm_matmul_end_to_end() -> None:
    """The full rmsnorm+matmul DAG renders and matches the numpy reference."""
    import nki
    import numpy as np

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(_rmsnorm_matmul, specs)
    src = render(g)

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_rmsnorm_matmul"]

    rng = np.random.default_rng(42)
    lhs = rng.standard_normal((128, 256)).astype(np.float32)
    rhs = rng.standard_normal((256, 512)).astype(np.float32)
    actual = nki.simulate(kernel)(lhs=lhs, rhs=rhs)
    if isinstance(actual, tuple):
        actual = actual[0]
    m = np.mean(lhs * lhs, axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + EPS)
    expected = (lhs * rms_inv) @ rhs
    assert np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
