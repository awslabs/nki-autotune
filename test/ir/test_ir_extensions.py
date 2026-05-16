"""Tests for the extended IR fields (location, dtype, kwargs, axis_map, return_name)."""

from nkigym.ir import ISANode, RootNode, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

K, M, N = 2048, 2048, 2048
_INPUT_SPECS: dict[str, tuple[int, ...]] = {"lhs_T": (K, M), "rhs": (K, N)}


@nkigym_kernel
def _matmul_fixture(lhs_T, rhs):
    """``lhs_T.T @ rhs`` fixture shared across tests in this file."""
    sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
    NKILoad()(src=rhs, dst=sbuf_rhs)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def _isa_leaves(ir) -> list[ISANode]:
    """Return every :class:`ISANode` payload in pre-order."""
    return [ir.tree.data(nid) for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]


def _leaves_by_op(ir, op_name: str) -> list[ISANode]:
    """Filter ISA leaves by their op class name."""
    return [leaf for leaf in _isa_leaves(ir) if leaf.op_cls.__name__ == op_name]


def test_memset_leaf_carries_value_kwarg():
    """NKIMemset(value=0.0) call-site kwargs are captured onto ISANode.kwargs."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    memsets = _leaves_by_op(ir, "NKIMemset")
    assert len(memsets) == 1
    assert memsets[0].kwargs == {"value": 0.0}


def test_non_config_leaves_have_empty_kwargs():
    """Load/Store/Matmul/TensorCopy take no non-operand kwargs in this fixture."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for op_name in ("NKILoad", "NKIStore", "NKIMatmul", "NKITensorCopy"):
        for leaf in _leaves_by_op(ir, op_name):
            assert leaf.kwargs == {}, f"{op_name} leaf kwargs={leaf.kwargs}"


def test_alloc_leaves_have_empty_kwargs():
    """NKIAlloc leaves never carry kwargs — alloc params live on TensorDims."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        assert leaf.kwargs == {}


def test_matmul_leaf_carries_axis_map():
    """NKIMatmul leaf axis_map resolves every abstract axis to a concrete dim."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    matmuls = _leaves_by_op(ir, "NKIMatmul")
    assert len(matmuls) == 1
    axis_map = matmuls[0].axis_map
    assert set(axis_map) == {"K", "M", "N"}
    assert all(isinstance(v, str) and v.startswith("d") for v in axis_map.values())


def test_alloc_leaves_carry_location():
    """NKIAlloc leaves carry the declared ``location`` (shared_hbm/sbuf/psum)."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    expected = {
        "sbuf_lhs_T": "sbuf",
        "sbuf_rhs": "sbuf",
        "psum_acc": "psum",
        "sbuf_prod": "sbuf",
        "hbm_out": "shared_hbm",
    }
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        tname = leaf.writes[0]
        assert leaf.location == expected[tname], f"{tname}.location={leaf.location!r}"


def test_non_alloc_leaves_have_no_location():
    """Only :class:`NKIAlloc` carries a location — compute-op leaves leave it ``None``."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for leaf in _isa_leaves(ir):
        if leaf.op_cls is NKIAlloc:
            continue
        assert leaf.location is None, f"{leaf.op_cls.__name__}.location={leaf.location!r}"


def test_alloc_leaves_carry_dtype():
    """NKIAlloc leaves carry the declared ``dtype``."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    expected = {
        "sbuf_lhs_T": "bfloat16",
        "sbuf_rhs": "bfloat16",
        "psum_acc": "float32",
        "sbuf_prod": "bfloat16",
        "hbm_out": "bfloat16",
    }
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        tname = leaf.writes[0]
        assert leaf.dtype == expected[tname], f"{tname}.dtype={leaf.dtype!r}"


def test_non_alloc_leaves_have_no_dtype():
    """Only :class:`NKIAlloc` carries a dtype — compute-op leaves leave it ``None``."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for leaf in _isa_leaves(ir):
        if leaf.op_cls is NKIAlloc:
            continue
        assert leaf.dtype is None, f"{leaf.op_cls.__name__}.dtype={leaf.dtype!r}"


def test_alloc_leaves_carry_axis_map():
    """NKIAlloc leaves carry an axis_map zipping NKIAlloc.OPERAND_AXES['dst'] to the tensor's dim_ids."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    expected_dst_axes = NKIAlloc.OPERAND_AXES["dst"]
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        assert len(leaf.writes) == 1
        tensor = ir.tensors[leaf.writes[0]]
        assert set(leaf.axis_map) == set(expected_dst_axes[: len(tensor.dim_ids)])
        for abstract, concrete in leaf.axis_map.items():
            idx = expected_dst_axes.index(abstract)
            assert concrete == tensor.dim_ids[idx]


def test_alloc_axis_map_keys_match_tensorize_sizes_keys():
    """An alloc's axis_map and tensorize_sizes are both keyed by the same abstract axes."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for leaf in _leaves_by_op(ir, "NKIAlloc"):
        assert set(leaf.axis_map) == set(leaf.tensorize_sizes)


def test_intermediate_tensor_location_and_dtype():
    """NKIAlloc kwargs flow through to TensorDims.location / .dtype."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    expected = {
        "sbuf_lhs_T": ("sbuf", "bfloat16"),
        "sbuf_rhs": ("sbuf", "bfloat16"),
        "psum_acc": ("psum", "float32"),
        "sbuf_prod": ("sbuf", "bfloat16"),
        "hbm_out": ("shared_hbm", "bfloat16"),
    }
    for name, (loc, dt) in expected.items():
        t = ir.tensors[name]
        assert t.location == loc, f"{name}.location={t.location!r}"
        assert t.dtype == dt, f"{name}.dtype={t.dtype!r}"


def test_param_tensor_location_is_hbm():
    """Kernel params live in HBM; the location is fixed by the role lattice."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for name in ("lhs_T", "rhs"):
        assert ir.tensors[name].location == "shared_hbm"


def test_param_tensor_dtype_inferred_from_load_dst():
    """Param dtype is inferred from the SBUF buffer it's loaded into."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    for name in ("lhs_T", "rhs"):
        assert ir.tensors[name].dtype == "bfloat16"


import pytest

from nkigym.ir.dimension_analysis import analyze_dimensions


def test_analysis_ops_includes_allocs_in_source_order():
    """analyze_dimensions emits _OpRecord entries for NKIAlloc interleaved with compute ops in source order."""
    analysis = analyze_dimensions(_matmul_fixture, _INPUT_SPECS)
    op_kinds = [op.op_cls.__name__ for op in analysis.ops]
    assert op_kinds == [
        "NKIAlloc",
        "NKIAlloc",
        "NKIAlloc",
        "NKIAlloc",
        "NKIAlloc",
        "NKILoad",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKIStore",
    ]


def test_alloc_op_record_axis_map():
    """NKIAlloc _OpRecord has axis_map zipped from OPERAND_AXES['dst'] to dim_ids."""
    analysis = analyze_dimensions(_matmul_fixture, _INPUT_SPECS)
    alloc_ops = [op for op in analysis.ops if op.op_cls is NKIAlloc]
    dst_axes = NKIAlloc.OPERAND_AXES["dst"]
    for op in alloc_ops:
        tensor_name = op.operand_names["dst"]
        tensor = analysis.tensors[tensor_name]
        expected_axis_map = {dst_axes[i]: dim_id for i, dim_id in enumerate(tensor.dim_ids)}
        assert op.axis_map == expected_axis_map


def test_alloc_op_record_has_empty_kwargs():
    """NKIAlloc _OpRecord.kwargs is empty — alloc params live on TensorDims."""
    analysis = analyze_dimensions(_matmul_fixture, _INPUT_SPECS)
    for op in analysis.ops:
        if op.op_cls is NKIAlloc:
            assert op.kwargs == {}


def test_return_name_is_parsed():
    """analyse_dimensions captures the top-level ``return <Name>`` identifier."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert ir.return_name == "hbm_out"


def test_missing_return_raises():
    """A kernel that does not end with ``return <Name>`` fails analysis."""

    @nkigym_kernel
    def no_return(x):
        sbuf_x = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
        NKILoad()(src=x, dst=sbuf_x)

    with pytest.raises(ValueError, match="return"):
        analyze_dimensions(no_return, {"x": (128, 128)})


def test_envelope_fields_are_flat():
    """KernelIR exposes signature + dim_sizes + tensors directly — no .analysis wrapper."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert ir.func_name == "_matmul_fixture"
    assert ir.param_names == ["lhs_T", "rhs"]
    assert ir.return_name == "hbm_out"
    assert set(ir.tensors) == {"lhs_T", "rhs", "sbuf_lhs_T", "sbuf_rhs", "psum_acc", "sbuf_prod", "hbm_out"}
    assert set(ir.dim_sizes.values()) == {K, M, N}
    assert not hasattr(ir, "analysis")
    assert not hasattr(ir, "ops")


def test_tree_has_no_dim_sizes_attribute():
    """KernelTree is a pure schedule tree — dim_sizes lives on KernelIR."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert not hasattr(ir.tree, "dim_sizes")


def test_tree_num_nodes_matches_graph_node_count():
    """``KernelTree.num_nodes`` reports the total node count in the underlying graph."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    assert ir.tree.num_nodes == ir.tree.graph.number_of_nodes()


def test_empty_tree_num_nodes_is_one():
    """A freshly-constructed tree contains only the dummy root, so ``num_nodes == 1``."""
    from nkigym.ir import KernelTree

    assert KernelTree().num_nodes == 1


def test_num_nodes_grows_with_add_node():
    """Each ``add_node`` call increments ``num_nodes`` by one."""
    from nkigym.ir import KernelTree

    tree = KernelTree()
    before = tree.num_nodes
    tree.add_node(RootNode(), parent=tree.root)
    assert tree.num_nodes == before + 1


def test_removed_symbols_are_not_exported_from_package():
    """DimensionAnalysis / OpAxes / analyze_dimensions must not be package-level imports."""
    import nkigym.ir as ir_pkg

    for removed in ("DimensionAnalysis", "OpAxes", "analyze_dimensions"):
        assert not hasattr(ir_pkg, removed), f"nkigym.ir.{removed} should have been removed"


def test_op_record_is_private_in_dimension_analysis_module():
    """_OpRecord is the tracer-local replacement for the old OpAxes — it must not be re-exported."""
    import nkigym.ir.dimension_analysis as dm

    assert not hasattr(dm, "OpAxes")
    assert hasattr(dm, "_OpRecord")


def test_dump_writes_envelope_markdown(tmp_path):
    """KernelIR.dump emits envelope.md alongside tree.* and dependency.*."""
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    ir.dump(tmp_path)
    envelope = (tmp_path / "envelope.md").read_text(encoding="utf-8")
    assert "# `_matmul_fixture`" in envelope
    assert "`lhs_T`" in envelope and "`rhs`" in envelope
    assert "Returns**: `hbm_out`" in envelope
    assert "## Dim sizes" in envelope
    assert "| `d0` | 2048 |" in envelope
    assert "## Tensors" in envelope
    assert "| `psum_acc` | intermediate | `psum` | `float32` |" in envelope
    assert "| `lhs_T` | param | `shared_hbm` | `bfloat16` |" in envelope
