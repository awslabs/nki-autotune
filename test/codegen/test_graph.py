"""Tests for OpGraph dataclasses and dim resolution."""

from nkigym.codegen.graph import DimInfo, OpGraph, Tensor, _parse_ast, _ParsedOpRaw
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.transpose import NKITranspose


def test_tensor_fields() -> None:
    """Tensor carries name, dim_ids, shape, dtype, origin."""
    t = Tensor(name="lhs", dim_ids=("d0", "d1"), shape=(2048, 2048), dtype="bfloat16", origin="param")
    assert t.name == "lhs"
    assert t.dim_ids == ("d0", "d1")
    assert t.shape == (2048, 2048)
    assert t.dtype == "bfloat16"
    assert t.origin == "param"


def test_dim_info_fields() -> None:
    """DimInfo carries dim_id, total_size, tile_size, num_tiles."""
    d = DimInfo(dim_id="d0", total_size=2048, tile_size=128, num_tiles=16)
    assert d.num_tiles == 16


def test_op_graph_empty_per_op_attrs() -> None:
    """OpGraph initialises per_op_attrs as an empty dict."""
    g = OpGraph(func_name="f", param_names=[], return_name="out", tensors={}, dims={}, ops=[], per_op_attrs={})
    assert g.per_op_attrs == {}


@nkigym_kernel
def _matmul_func(lhs, rhs):
    """Inline-defined nkigym function for AST tests."""
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    lhs_T = NKITranspose()(data=lhs_sbuf)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


def test_parse_ast_captures_ops_in_source_order() -> None:
    """_parse_ast returns one entry per NKIOp call, in source order."""
    raws, return_name = _parse_ast(_matmul_func)
    assert all(isinstance(r, _ParsedOpRaw) for r in raws)
    kinds = [r.op_cls.__name__ for r in raws]
    assert kinds == ["NKILoad", "NKILoad", "NKITranspose", "NKIMatmul", "NKIStore"]
    assert return_name == "out"


def test_parse_ast_captures_operand_names() -> None:
    """Name-valued kwargs become operand_names entries."""
    raws, _ = _parse_ast(_matmul_func)
    assert raws[0].operand_names == {"data": "lhs"}
    assert raws[3].operand_names == {"stationary": "lhs_T", "moving": "rhs_sbuf"}
    assert raws[3].output_names == ["prod"]


EPS = 1e-6


@nkigym_kernel
def _rms_func(lhs):
    """Function for testing literal kwargs parsing."""
    lhs_sbuf = NKILoad()(data=lhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    out = NKIStore()(data=sum_sq)
    return out


def test_parse_ast_captures_literal_kwargs() -> None:
    """Constructor + call-site literal kwargs merge into op_kwargs."""
    raws, _ = _parse_ast(_rms_func)
    kwargs = raws[1].op_kwargs
    assert kwargs["op"] == "square"
    assert kwargs["reduce_op"] == "add"
    assert "post_op" not in kwargs
    assert "scale" not in kwargs
    assert "bias" not in kwargs


def test_parse_and_resolve_simple_matmul() -> None:
    """lhs_T (K, M) @ moving (K, N) → output (M, N) unifies correctly."""
    from nkigym.codegen.graph import parse_and_resolve

    @nkigym_kernel
    def kernel(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    assert g.param_names == ["lhs_T", "rhs"]
    assert g.return_name == "out"
    """K-axis of lhs_T and rhs must unify (both first dim)."""
    assert g.tensors["lhs_T"].dim_ids[0] == g.tensors["rhs"].dim_ids[0]
    assert g.tensors["prod"].dim_ids == (g.tensors["lhs_T"].dim_ids[1], g.tensors["rhs"].dim_ids[1])
    """Matmul tile limits produce K=128, M=128, N=512 after min() with sizes."""
    k_dim = g.tensors["lhs_T"].dim_ids[0]
    m_dim = g.tensors["lhs_T"].dim_ids[1]
    n_dim = g.tensors["rhs"].dim_ids[1]
    assert g.dims[k_dim].tile_size == 128
    assert g.dims[m_dim].tile_size == 128
    assert g.dims[n_dim].tile_size == 512
    assert g.dims[k_dim].num_tiles == 16
    assert g.dims[n_dim].num_tiles == 4


def test_parse_and_resolve_tensor_origins() -> None:
    """Params tag 'param', intermediates 'intermediate', last store output 'return'."""
    from nkigym.codegen.graph import parse_and_resolve

    @nkigym_kernel
    def kernel(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    assert g.tensors["lhs_T"].origin == "param"
    assert g.tensors["rhs"].origin == "param"
    assert g.tensors["lhs_T_sbuf"].origin == "intermediate"
    assert g.tensors["prod"].origin == "intermediate"
    assert g.tensors["out"].origin == "return"


def test_parse_and_resolve_activation_reduce_output_dtype() -> None:
    """NKIActivationReduce's output (P,) is pinned to float32."""
    from nkigym.codegen.graph import parse_and_resolve

    @nkigym_kernel
    def kernel(lhs):
        lhs_sbuf = NKILoad()(data=lhs)
        rms = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
        out = NKIStore()(data=rms)
        return out

    specs = {"lhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    assert g.tensors["rms"].dtype == "float32"
    assert g.tensors["rms"].dim_ids == g.tensors["lhs"].dim_ids[:1]


def test_parse_and_resolve_touched_dims_ordering() -> None:
    """touched_dims lists partition dim first, then free, then reducing."""
    from nkigym.codegen.graph import parse_and_resolve

    @nkigym_kernel
    def kernel(lhs_T, rhs):
        lhs_T_sbuf = NKILoad()(data=lhs_T)
        rhs_sbuf = NKILoad()(data=rhs)
        prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
        out = NKIStore()(data=prod)
        return out

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(kernel, specs)
    matmul_op = next(op for op in g.ops if op.op_cls.__name__ == "NKIMatmul")
    """Matmul partition = M (from stationary's second axis), frees = N, reducing = K."""
    k_dim = g.tensors["lhs_T_sbuf"].dim_ids[0]
    m_dim = g.tensors["lhs_T_sbuf"].dim_ids[1]
    n_dim = g.tensors["rhs_sbuf"].dim_ids[1]
    assert matmul_op.touched_dims == (m_dim, n_dim, k_dim)


def test_op_local_buffers_defaults_to_empty() -> None:
    """NKIOp.OP_LOCAL_BUFFERS defaults to an empty dict so existing ops are unaffected."""
    from nkigym.ops.base import NKIOp
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul

    assert NKIOp.OP_LOCAL_BUFFERS == {}
    assert NKIMatmul.OP_LOCAL_BUFFERS == {}
    assert NKILoad.OP_LOCAL_BUFFERS == {}


def test_parse_and_resolve_registers_op_local_derived_dims_for_activation_reduce() -> None:
    """After parse_and_resolve, an F_slot derived dim exists for each activation_reduce op.

    The derived dim's tile_size is 1 and num_tiles matches the source F dim's num_tiles.
    """
    from nkigym.codegen.graph import parse_and_resolve

    @nkigym_kernel
    def _rms(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
        out = NKIStore()(data=m)
        return out

    g = parse_and_resolve(_rms, {"x": ((128, 2048), "bfloat16")})
    ar_op = next(o for o in g.ops if o.op_cls.__name__ == "NKIActivationReduce")
    f_dim_id = ar_op.axis_map["F"]
    f_info = g.dims[f_dim_id]
    assert f_info.tile_size == 512
    assert f_info.num_tiles == 4

    """A derived F_slot dim must be registered for this op instance."""
    f_slot_dim_ids = [d for d in g.dims if d.startswith(f_dim_id) and d.endswith("_f_slot")]
    assert len(f_slot_dim_ids) == 1
    slot_info = g.dims[f_slot_dim_ids[0]]
    assert slot_info.tile_size == 1
    assert slot_info.num_tiles == f_info.num_tiles
    assert slot_info.total_size == f_info.num_tiles


def test_parsed_op_resolves_op_local_buffer_names_for_activation_reduce() -> None:
    """ParsedOp.op_local_buffers maps logical → (emitted_name, location, dtype, shape).

    Naming convention: sbuf_local_<id> / psum_local_<id>, id assigned per
    op instance in encounter order across OP_LOCAL_BUFFERS iteration order.
    """
    from nkigym.codegen.graph import parse_and_resolve
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.load import NKILoad
    from nkigym.ops.store import NKIStore

    @nkigym_kernel
    def _rms(x):
        xs = NKILoad()(data=x)
        m = NKIActivationReduce(op="square", reduce_op="add")(data=xs)
        out = NKIStore()(data=m)
        return out

    g = parse_and_resolve(_rms, {"x": ((128, 2048), "bfloat16")})
    ar_op = next(o for o in g.ops if o.op_cls.__name__ == "NKIActivationReduce")

    """Buffers dict is keyed by logical name declared on the op."""
    assert set(ar_op.op_local_buffers.keys()) == {"scratch", "slot_vec"}

    scratch = ar_op.op_local_buffers["scratch"]
    assert scratch.emitted_name == "sbuf_local_0"
    assert scratch.location == "sbuf"
    assert scratch.dtype == "float32"
    """Shape: (p_tile, 1, num_f_tiles * f_tile) — P dim contributes p_tile
    plus singleton block dim; F contributes num_f_tiles*f_tile."""
    assert scratch.shape == (128, 1, 2048)

    slot = ar_op.op_local_buffers["slot_vec"]
    assert slot.emitted_name == "sbuf_local_1"
    assert slot.location == "sbuf"
    assert slot.dtype == "float32"
    """Shape: (p_tile, 1, num_f_tiles) — F_slot.num_tiles=4, tile_size=1."""
    assert slot.shape == (128, 1, 4)


def test_activation_reduce_rejects_removed_kwargs() -> None:
    """NKIActivationReduce mirrors nisa.activation_reduce kwargs; post_op/scale/bias removed."""
    import numpy as np
    import pytest

    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.base import _RoleArray

    xs_np = np.ones((128, 2048), dtype=np.float32)
    xs = xs_np.view(_RoleArray)
    xs.role = "sbuf"

    """Valid call: only op + reduce_op."""
    NKIActivationReduce(op="square", reduce_op="add")(data=xs)

    """Removed kwargs must raise TypeError with a clear message."""
    with pytest.raises(TypeError):
        NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt")(data=xs)
    with pytest.raises(TypeError):
        NKIActivationReduce(op="square", reduce_op="add", scale=0.5)(data=xs)
    with pytest.raises(TypeError):
        NKIActivationReduce(op="square", reduce_op="add", bias=1e-6)(data=xs)


def test_parse_and_resolve_attaches_dep_graph() -> None:
    """parse_and_resolve populates OpGraph.dep with producer/consumer edges."""
    from nkigym.codegen.graph import parse_and_resolve

    specs = {"lhs": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(_matmul_func, specs)

    """dep field is populated."""
    assert hasattr(g, "dep")
    assert g.dep.producer
    assert g.dep.consumers

    """lhs_sbuf has one producer: the first Load op."""
    load_lhs = g.ops[0]
    assert "lhs_sbuf" in g.dep.producer
    assert g.dep.producer["lhs_sbuf"] == load_lhs.idx

    """lhs_T consumes lhs_sbuf."""
    assert "lhs_sbuf" in g.dep.consumers
    consumers = g.dep.consumers["lhs_sbuf"]
    assert len(consumers) == 1
    """lhs_T is produced by the Transpose op."""
    transpose_op = g.ops[2]
    assert transpose_op.idx in consumers
