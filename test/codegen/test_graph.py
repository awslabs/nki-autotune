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
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 2048, bias=EPS)(
        data=lhs_sbuf
    )
    out = NKIStore()(data=rms_inv)
    return out


def test_parse_ast_captures_literal_kwargs() -> None:
    """Constructor + call-site literal kwargs merge into op_kwargs."""
    raws, _ = _parse_ast(_rms_func)
    kwargs = raws[1].op_kwargs
    assert kwargs["op"] == "square"
    assert kwargs["reduce_op"] == "add"
    assert kwargs["post_op"] == "rsqrt"
    assert kwargs["scale"] == 1 / 2048
    assert kwargs["bias"] == EPS


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
