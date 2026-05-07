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
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
    m = NKIActivation(op="rsqrt", scale=1 / 128, bias=1e-6)(data=sum_sq)
    out = NKIStore()(data=m)
    return out


EPS = 1e-6


@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=EPS)(data=sum_sq)
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
    """Return tensor (``out``) lives only in HBM — no sbuf_out allocation."""
    assert "sbuf_out =" not in src


def _header_only(g) -> str:
    """Run the renderer's header path without op emission (Task-4 scope)."""
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _Writer,
    )

    forest = build_canonical_forest(g)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, g)
    w.indent()
    _emit_param_asserts(w, g)
    _emit_hbm_output(w, g)
    _emit_sbuf_allocations(w, g, forest)
    return w.getvalue()


def test_sbuf_tile_slice_2d() -> None:
    """Per-tile slice for a 2D SBUF tensor uses ``(i_<d>_0 + i_<d>_1)``.

    ``total_slots == raw_trip_product`` on each axis, so the modulo
    wrap collapses to the raw sum (no cosmetic ``% N``).
    """
    from nkigym.codegen.render import _sbuf_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"], "d1": ["i_d1_0", "i_d1_1"]}
    trips = {"d0": [16, 1], "d1": [8, 1]}
    slice_expr = _sbuf_tile_slice(
        "sbuf_lhs",
        ("d0", "d1"),
        p_tile=128,
        f_tile=128,
        path_names=names,
        path_trips=trips,
        total_slots_p=16,
        total_slots_f=8,
    )
    assert slice_expr == (
        "sbuf_lhs[0:128, i_d0_0 + i_d0_1, " "(i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128]"
    )


def test_sbuf_tile_slice_1d() -> None:
    """Per-tile slice for a 1D SBUF tensor uses ``(i_<p>_0 + i_<p>_1)`` on P."""
    from nkigym.codegen.render import _sbuf_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"]}
    trips = {"d0": [4, 1]}
    slice_expr = _sbuf_tile_slice(
        "sbuf_rms", ("d0",), p_tile=128, f_tile=1, path_names=names, path_trips=trips, total_slots_p=4, total_slots_f=1
    )
    assert slice_expr == "sbuf_rms[0:128, i_d0_0 + i_d0_1, 0:1]"


def test_hbm_tile_slice() -> None:
    """HBM tile slice uses ``(i_<d>_0 + i_<d>_1) * tile`` offsets."""
    from nkigym.codegen.render import _hbm_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"], "d1": ["i_d1_0", "i_d1_1"]}
    trips = {"d0": [16, 1], "d1": [16, 1]}
    slice_expr = _hbm_tile_slice(
        "lhs",
        ("d0", "d1"),
        p_tile=128,
        f_tile=128,
        path_names=names,
        path_trips=trips,
        total_slots_p=16,
        total_slots_f=16,
    )
    assert slice_expr == (
        "lhs[(i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128, "
        "(i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128]"
    )


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
    assert "for i_d0_0 in range" in src

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
    """activation_reduce with post_op=rsqrt emits activation_reduce + tensor_reduce + post_op."""
    import nki
    import numpy as np

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)
    assert "nisa.tensor_reduce(" in src
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


def test_sbuf_tile_slice_2d_canonical_form() -> None:
    """Canonical 2N-form: i_<d>_0 + i_<d>_1 with parenthesised compound on free axis."""
    from nkigym.codegen.render import _sbuf_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"], "d1": ["i_d1_0", "i_d1_1"]}
    trips = {"d0": [16, 1], "d1": [8, 1]}
    slice_expr = _sbuf_tile_slice(
        "sbuf_lhs",
        ("d0", "d1"),
        p_tile=128,
        f_tile=128,
        path_names=names,
        path_trips=trips,
        total_slots_p=16,
        total_slots_f=8,
    )
    assert slice_expr == (
        "sbuf_lhs[0:128, i_d0_0 + i_d0_1, " "(i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128]"
    )


def test_sbuf_tile_slice_1d_canonical_form() -> None:
    """1D tensor uses only partition-axis path ordinals, free-axis literal 0:1."""
    from nkigym.codegen.render import _sbuf_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"]}
    trips = {"d0": [4, 1]}
    slice_expr = _sbuf_tile_slice(
        "sbuf_rms", ("d0",), p_tile=128, f_tile=1, path_names=names, path_trips=trips, total_slots_p=4, total_slots_f=1
    )
    assert slice_expr == "sbuf_rms[0:128, i_d0_0 + i_d0_1, 0:1]"


def test_hbm_tile_slice_parenthesises_compound_before_multiplication() -> None:
    """HBM slice wraps (i_<d>_0 + i_<d>_1) in parens before the * tile multiplication."""
    from nkigym.codegen.render import _hbm_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"], "d1": ["i_d1_0", "i_d1_1"]}
    trips = {"d0": [16, 1], "d1": [16, 1]}
    slice_expr = _hbm_tile_slice(
        "lhs",
        ("d0", "d1"),
        p_tile=128,
        f_tile=128,
        path_names=names,
        path_trips=trips,
        total_slots_p=16,
        total_slots_f=16,
    )
    assert slice_expr == (
        "lhs[(i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128, "
        "(i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128]"
    )


def test_swapped_dst_tile_slice_swaps_p_and_f_for_transpose() -> None:
    """Transpose dst: partition slot = src_f's ordinals; free slot = src_p's ordinals."""
    from nkigym.codegen.render import _swapped_dst_tile_slice

    names = {"d0": ["i_d0_0", "i_d0_1"], "d1": ["i_d1_0", "i_d1_1"]}
    trips = {"d0": [16, 1], "d1": [16, 1]}
    slice_expr = _swapped_dst_tile_slice(
        dst_name="lhs_T",
        src_p_axis="d0",
        src_f_axis="d1",
        tile=128,
        path_names=names,
        path_trips=trips,
        total_slots_p=16,
        total_slots_f=16,
    )
    assert slice_expr == (
        "sbuf_lhs_T[0:128, i_d1_0 + i_d1_1, " "(i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128]"
    )


def test_slot_expr_collapses_when_tail_product_is_one() -> None:
    """Canonical form trips [t_0, 1]: term i_<d>_0 has tail=1, so no ' * N' multiplier."""
    from nkigym.codegen.render import _slot_expr

    names = {"d0": ["i_d0_0", "i_d0_1"]}
    trips = {"d0": [16, 1]}
    expr = _slot_expr(names, trips, "d0", total_slots=16)
    assert expr == "i_d0_0 + i_d0_1"


def test_slot_expr_multi_split_uses_tail_products() -> None:
    """If trips [t_0, t_1, 1], outermost term carries '* t_1' (tail product)."""
    from nkigym.codegen.render import _slot_expr

    names = {"d0": ["i_d0_0", "i_d0_1", "i_d0_2"]}
    trips = {"d0": [4, 2, 1]}
    expr = _slot_expr(names, trips, "d0", total_slots=8)
    assert expr == "i_d0_0 * 2 + i_d0_1 + i_d0_2"


def test_slot_expr_raises_when_dim_has_no_ancestors() -> None:
    """If path_names[dim] is empty or missing, caller has no open loop for that dim — raise."""
    from nkigym.codegen.render import _slot_expr

    try:
        _slot_expr({"d0": []}, {"d0": []}, "d0", total_slots=1)
    except ValueError as exc:
        assert "d0" in str(exc)
    else:
        raise AssertionError("_slot_expr did not raise on missing ancestors")


def test_walker_emits_for_headers_with_path_ordinal_names() -> None:
    """The walker emits 'for i_<d>_<ordinal>' headers with the correct trip count."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.codegen.render import _BODY_EMITTERS, _Writer, render_forest
    from nkigym.ops.base import AxisRole

    @nkigym_kernel
    def _walker_test_kernel(x):
        y = NKILoad()(data=x)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = parse_and_resolve(_walker_test_kernel, specs)
    """Synthesise a minimal forest with a single LoopNode pair + a marker BodyLeaf.
    Omit node.name so the walker falls back to position-derived names — this
    test verifies the fallback path works for hand-built test forests."""
    marker_tree = LoopNode(
        "d0", 4, AxisRole.PARALLEL, [LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="_marker_")])]
    )

    def _marker_emitter(w, op_graph, op, path_names, path_trips, forest):
        """Test emitter — prints the names and trips it sees at body time."""
        _ = forest
        d0_names = path_names.get("d0", [])
        d0_trips = path_trips.get("d0", [])
        w.line(f"MARK(d0_names={d0_names!r}, trips={d0_trips!r})")

    _BODY_EMITTERS[("NKILoad", "_marker_")] = _marker_emitter

    try:
        w = _Writer()
        render_forest(w, g, [marker_tree])
        src = w.getvalue()
        assert "for i_d0_0 in range(4):" in src
        assert "for i_d0_1 in range(1):" in src
        assert "MARK(d0_names=['i_d0_0', 'i_d0_1'], trips=[4, 1])" in src
    finally:
        """Clean up the test registration so it doesn't pollute other tests."""
        del _BODY_EMITTERS[("NKILoad", "_marker_")]


def test_register_body_decorator_stores_emitter_keyed_on_op_kind_and_phase() -> None:
    """_register_body returns the function and stores it in _BODY_EMITTERS."""
    from nkigym.codegen.render import _BODY_EMITTERS, _register_body

    @_register_body("TestOp", "test_phase")
    def _emit_test(w, op_graph, op, path_names, path_trips, forest):
        _ = forest
        w.line("TEST")

    try:
        assert _BODY_EMITTERS[("TestOp", "test_phase")] is _emit_test
    finally:
        del _BODY_EMITTERS[("TestOp", "test_phase")]


def test_walker_raises_on_missing_emitter() -> None:
    """If no emitter is registered for (op_kind, phase), walker raises with a clear message."""
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
    from nkigym.codegen.render import _Writer, render_forest
    from nkigym.ops.base import AxisRole

    @nkigym_kernel
    def _no_emitter_kernel(x):
        y = NKILoad()(data=x)
        out = NKIStore()(data=y)
        return out

    specs = {"x": ((128, 256), "bfloat16")}
    g = parse_and_resolve(_no_emitter_kernel, specs)
    tree = LoopNode("d0", 1, AxisRole.PARALLEL, [BodyLeaf(op_idx=0, phase="_nonexistent_phase_")])
    w = _Writer()
    try:
        render_forest(w, g, [tree])
    except ValueError as exc:
        assert "NKILoad" in str(exc)
        assert "_nonexistent_phase_" in str(exc)
    else:
        raise AssertionError("render_forest did not raise on missing emitter")


@nkigym_kernel
def _passthrough_kernel(x):
    """Passthrough for walker test."""
    xs = NKILoad()(data=x)
    out = NKIStore()(data=xs)
    return out


def test_render_forest_load_store_cpu_sim_matches() -> None:
    """Rendering a passthrough kernel via the walker and simulating it matches numpy."""
    import nki
    import numpy as np

    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _hbm_name,
        _Writer,
        render_forest,
    )

    specs = {"x": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(_passthrough_kernel, specs)
    forest = build_canonical_forest(g)

    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, g)
    w.indent()
    _emit_param_asserts(w, g)
    _emit_hbm_output(w, g)
    _emit_sbuf_allocations(w, g, forest)
    render_forest(w, g, forest)
    w.line(f"return {_hbm_name(g.return_name)}")
    w.dedent()
    src = w.getvalue()

    assert "for i_d0_0 in range" in src
    assert "for i_d0_1 in range(1):" in src
    assert "nisa.dma_copy(" in src

    src_fp32 = src.replace("nl.bfloat16", "nl.float32")
    ns: dict = {}
    exec(src_fp32, ns)
    kernel = ns["_passthrough_kernel"]
    rng = np.random.default_rng(0)
    x_in = rng.standard_normal((2048, 2048)).astype(np.float32)
    actual = nki.simulate(kernel)(x=x_in)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert np.allclose(actual, x_in, atol=1e-5, rtol=1e-5)


def _render_via_walker(op_graph) -> str:
    """Helper: render a full kernel through the forest walker.

    Exactly equivalent to today's ``render(op_graph)`` default path, but
    uses ``render_forest`` + a canonical forest instead of the legacy
    per-op emitters. When Task C5 swaps the default ``render`` to this
    code path, this helper can be replaced with a direct call.
    """
    from nkigym.codegen.loop_forest import build_canonical_forest
    from nkigym.codegen.render import (
        _emit_hbm_output,
        _emit_imports,
        _emit_param_asserts,
        _emit_sbuf_allocations,
        _emit_signature,
        _hbm_name,
        _Writer,
        render_forest,
    )

    forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph, forest)
    render_forest(w, op_graph, forest)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()


def test_render_forest_matmul_cpu_sim_matches() -> None:
    """Full lhs_T @ rhs kernel rendered through the walker simulates correctly."""
    import nki
    import numpy as np

    from nkigym.codegen.graph import parse_and_resolve

    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    src = _render_via_walker(g)
    assert "nisa.nc_matmul(" in src
    assert "nisa.memset(psum_tile" in src
    assert "nisa.tensor_copy(" in src
    assert "for i_d0_0 in range" in src
    assert "for i_d0_1 in range(1):" in src

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


def test_render_forest_activation_reduce_cpu_sim_matches() -> None:
    """RMS kernel rendered through the walker simulates correctly."""
    import nki
    import numpy as np

    from nkigym.codegen.graph import parse_and_resolve

    specs = {"x": ((128, 128), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = _render_via_walker(g)
    assert "nisa.tensor_reduce(" in src
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


def test_render_forest_rmsnorm_matmul_cpu_sim_matches() -> None:
    """Full rmsnorm+matmul DAG rendered via walker matches numpy end-to-end."""
    import nki
    import numpy as np

    from nkigym.codegen.graph import parse_and_resolve

    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    g = parse_and_resolve(_rmsnorm_matmul, specs)
    src = _render_via_walker(g)
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


def test_render_emits_op_local_buffer_allocations_at_top() -> None:
    """Renderer emits one nl.ndarray per op-local buffer, after tensor allocations."""
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """scratch: (p_tile=128, 1, num_f_tiles*f_tile=4*512=2048) — f_tile-wide per call."""
    assert "sbuf_local_0 = nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.sbuf)" in src
    """slot_vec: (p_tile=128, 1, num_f_tiles=4)."""
    assert "sbuf_local_1 = nl.ndarray((128, 1, 4), dtype=nl.float32, buffer=nl.sbuf)" in src


def test_render_reduce_step_writes_slot_and_omits_tensor_tensor_merge() -> None:
    """reduce_step emits activation_reduce(reduce_res=slot_vec[:, :, f_idx:f_idx+1]) with no merge.

    Pattern 2 doesn't need tensor_tensor or tmp_red — each call writes
    a distinct slot directly.
    """
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """activation_reduce writes to sbuf_local_1 (slot_vec); dst is sbuf_local_0 (scratch)."""
    assert "nisa.activation_reduce(" in src
    assert "reduce_res=sbuf_local_1" in src
    """No tmp_red allocation, no tensor_tensor merge."""
    assert "tmp_red" not in src
    assert "nisa.tensor_tensor(" not in src
    """The legacy reducer_init memset on sbuf_<reduce_output> must also be gone."""
    assert "nisa.memset(sbuf_m" not in src


def test_render_reduce_close_emits_tensor_reduce_on_slot_vec() -> None:
    """reduce_close folds slot_vec into the op's (P, 1) output via nisa.tensor_reduce."""
    specs = {"x": ((128, 2048), "bfloat16")}
    g = parse_and_resolve(_rms_kernel, specs)
    src = render(g)

    """tensor_reduce closes the slot vector along its free axis."""
    assert "nisa.tensor_reduce(" in src
    """Must reduce axis=2 of (p_tile, 1, num_f_tiles) shape."""
    assert "axis=2" in src
    """Output is sbuf_m (the op's reduce output), slot is sbuf_local_1."""
    assert "sbuf_m" in src
    assert "sbuf_local_1" in src
