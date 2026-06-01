"""Tests for the extended IR fields (location, dtype, kwargs, axis_map, return_name)."""

from test.transforms._fixtures import INPUT_SPECS, f_matmul

from nkigym.ir import ForNode, ISANode, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore


@nkigym_kernel
def _reduce_then_activate(x):
    """reduce → elementwise activation on the (P,) vector → store (rmsnorm-ish)."""
    sx = NKILoad()(src=x)
    red = NKIActivationReduce(op="square", reduce_op="add")(data=sx)
    act = NKIActivation(op="rsqrt")(data=red)
    out = NKIStore()(src=act)
    return out


def test_trace_synthesizes_1d_output_for_elementwise_on_reduce():
    """An elementwise op consuming a 1D (P,) reduce output yields a 1D output, not (P,F)."""
    from nkigym.ir.dimension_analysis import analyze_dimensions

    analysis = analyze_dimensions(_reduce_then_activate, {"x": ((128, 512), "bfloat16")})
    """The activation output 'act' must be 1D (P,) — its F axis is absent from the op's axis map."""
    assert len(analysis.tensors["act"].dim_ids) == 1
    assert analysis.dim_sizes[analysis.tensors["act"].dim_ids[0]] == 128


def test_build_initial_ir_handles_elementwise_on_1d_reduce():
    """The full canonical build (not just the trace) must handle a (P,) reduce → elementwise chain."""
    from nkigym.ir import build_initial_ir

    ir = build_initial_ir(_reduce_then_activate, {"x": ((128, 512), "bfloat16")})
    """The activation buffer 'act' is 1D (P,)=(128,); the build must not KeyError on the absent F axis."""
    assert ir.all_buffers()["act"].shape == (128,)


def _isa_leaves(ir) -> list[ISANode]:
    """Return every :class:`ISANode` payload in pre-order."""
    return [ir.tree.data(nid) for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]


def _leaves_by_op(ir, op_name: str) -> list[ISANode]:
    """Filter ISA leaves by their op class name."""
    return [leaf for leaf in _isa_leaves(ir) if leaf.op_cls.__name__ == op_name]


def test_memset_leaf_carries_value_kwarg():
    """The synthesized NKIMemset leaf carries ``value=0.0`` on ISANode.kwargs."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    memsets = _leaves_by_op(ir, "NKIMemset")
    assert len(memsets) == 1
    assert memsets[0].kwargs == {"value": 0.0}


def test_non_config_leaves_have_empty_kwargs():
    """Load/Store/Matmul/TensorCopy take no non-operand kwargs in this fixture."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    for op_name in ("NKILoad", "NKIStore", "NKIMatmul", "NKITensorCopy"):
        for leaf in _leaves_by_op(ir, op_name):
            assert leaf.kwargs == {}, f"{op_name} leaf kwargs={leaf.kwargs}"


import pytest

from nkigym.ir.dimension_analysis import analyze_dimensions


def test_return_name_is_parsed():
    """analyse_dimensions captures the top-level ``return <Name>`` identifier."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    assert ir.return_name == "hbm_out"


def test_missing_return_raises():
    """A kernel that does not end with ``return <Name>`` fails analysis."""

    @nkigym_kernel
    def no_return(x):
        sbuf_x = NKILoad()(src=x)

    with pytest.raises(ValueError, match="return"):
        analyze_dimensions(no_return, {"x": ((128, 128), "bfloat16")})


def test_tree_has_no_dim_sizes_attribute():
    """KernelTree is a pure schedule tree — dim_sizes lives on KernelIR."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    assert not hasattr(ir.tree, "dim_sizes")


def test_tree_num_nodes_matches_graph_node_count():
    """``KernelTree.num_nodes`` reports the total node count in the underlying graph."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
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
    tree.add_node(ForNode(loop_var="i", extent=2), parent=tree.root)
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


def test_envelope_markdown_format():
    """KernelIR._render_envelope_md emits signature and buffers table."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    envelope = ir._render_envelope_md()
    assert "# `f_matmul`" in envelope
    assert "`lhs_T`" in envelope and "`rhs`" in envelope
    assert "**Returns**: `hbm_out`" in envelope
    assert "## Buffers" in envelope
    """Buffers table includes location, dtype, shape for all allocated buffers."""
    assert "| Name | Location | Dtype | Shape |" in envelope
    """Check at least one intermediate buffer appears."""
    assert "psum" in envelope or "sbuf" in envelope
    assert "bfloat16" in envelope or "float32" in envelope


def test_itervar_constructor_and_equality():
    """IterVar is a frozen dataclass with structural equality."""
    from nkigym.ir.tree import IterVar
    from nkigym.ops.base import AxisRole

    a = IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL)
    b = IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL)
    c = IterVar(axis="M", dom=(0, 2048), role=AxisRole.ACCUMULATION)
    assert a == b
    assert a != c


def test_buffer_constructor_and_equality():
    """Buffer is a frozen dataclass with structural equality."""
    from nkigym.ir.tree import Buffer

    a = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    b = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    assert a == b


def test_bufferregion_constructor_and_equality():
    """BufferRegion is a frozen dataclass with structural equality on tensor + ranges."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion

    a = BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),))
    b = BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),))
    assert a == b


def test_blocknode_constructor_minimal():
    """A BlockNode with no iter_vars is the legal empty case (e.g. the synthetic root block)."""
    from nkigym.ir.tree import BlockNode

    node = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=())
    assert node.iter_vars == ()
    assert node.alloc_buffers == ()
    assert node.annotations == {}


def test_blocknode_constructor_full():
    """A BlockNode with iter_vars and a region carries every field."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, IterVar
    from nkigym.ops.base import AxisRole

    block = BlockNode(
        iter_vars=(
            IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL),
            IterVar(axis="N", dom=(0, 2048), role=AxisRole.PARALLEL),
        ),
        iter_values=(Var(name="i_M"), Var(name="i_N")),
        reads=(),
        writes=(BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),)),),
        alloc_buffers=(Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum"),),
    )
    assert len(block.iter_vars) == 2
    assert len(block.alloc_buffers) == 1


def test_blocknode_default_annotations_is_fresh_dict():
    """Two default-constructed BlockNodes don't share an annotations dict."""
    from nkigym.ir.tree import BlockNode

    a = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    b = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    a.annotations["k"] = 1
    assert "k" not in b.annotations


def test_kerneltree_blocks_helper_yields_blocknode_nids():
    """blocks() walks the tree in pre-order and yields nids whose payload is a BlockNode."""
    from nkigym.ir.tree import BlockNode, KernelTree

    tree = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    nid_a = tree.add_node(block, parent=tree.root)
    nid_b = tree.add_node(block, parent=tree.root)

    """Root is also a BlockNode now."""
    out = list(tree.blocks())
    assert set(out) == {tree.root, nid_a, nid_b}


def test_kerneltree_blocks_skips_non_block_payloads():
    """blocks() ignores ForNode and ISANode payloads."""
    from nkigym.ir.tree import BlockNode, ForNode, KernelTree

    tree = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    block_nid = tree.add_node(block, parent=tree.root)
    tree.add_node(ForNode(loop_var="i_d0_0", extent=2), parent=block_nid)

    """Root is also a BlockNode, so blocks() returns root + block_nid."""
    out = list(tree.blocks())
    assert tree.root in out and block_nid in out


def test_ir_module_exports_new_payloads():
    """The ir package re-exports the new payload classes."""
    from nkigym.ir import BlockNode, Buffer, BufferRegion, IterVar  # noqa: F401


def test_fornode_payload_uses_loop_var_and_extent():
    """ForNode carries (loop_var, extent), not (dim, trip)."""
    from nkigym.ir.tree import ForNode

    node = ForNode(loop_var="i_M_0", extent=16)
    assert node.loop_var == "i_M_0"
    assert node.extent == 16
    """Old field names must NOT be present."""
    assert not hasattr(node, "dim")
    assert not hasattr(node, "trip")


def test_isanode_payload_uses_operand_bindings():
    """ISANode carries (op_cls, operand_bindings, kwargs); legacy fields removed."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion, ISANode
    from nkigym.ops.matmul import NKIMatmul

    bindings = {
        "stationary": BufferRegion(
            tensor="sbuf_lhs_T", ranges=((Var(name="vK"), Const(value=1)), (Var(name="vM"), Const(value=128)))
        ),
        "moving": BufferRegion(
            tensor="sbuf_rhs", ranges=((Var(name="vK"), Const(value=1)), (Var(name="vN"), Const(value=512)))
        ),
        "dst": BufferRegion(
            tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)), (Var(name="vN"), Const(value=512)))
        ),
    }
    node = ISANode(op_cls=NKIMatmul, operand_bindings=bindings, kwargs={})
    assert node.op_cls is NKIMatmul
    assert set(node.operand_bindings) == {"stationary", "moving", "dst"}
    """Legacy fields must NOT be present."""
    for old in ("reads", "writes", "rmw", "axis_map", "tensorize_sizes", "location", "dtype"):
        assert not hasattr(node, old), f"ISANode unexpectedly carries legacy field {old!r}"


def test_kernelir_envelope_is_slim():
    """KernelIR drops dim_sizes and tensors fields."""
    import dataclasses

    from nkigym.ir.ir import KernelIR

    field_names = {f.name for f in dataclasses.fields(KernelIR)}
    assert field_names == {"func_name", "param_names", "return_name", "tree", "dependency", "param_buffers"}


def test_kernelir_helper_methods_exposed():
    """KernelIR exposes all_buffers / buffer / axis_extent."""
    from nkigym.ir.ir import KernelIR

    assert callable(KernelIR.all_buffers)
    assert callable(KernelIR.buffer)
    assert callable(KernelIR.axis_extent)


def test_canonical_matmul_emits_root_block_with_alloc_buffers():
    """The canonical 2048**3 matmul tree's root child is a single BlockNode whose alloc_buffers
    list every kernel-lifetime tensor, including psum_prod (canonical scope = root)."""
    from nkigym.ir.tree import BlockNode

    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    """Tree.root is now the root block directly (no intermediate RootNode)."""
    root_block = ir.tree.data(ir.tree.root)
    assert isinstance(root_block, BlockNode)
    """Buffer scope: kernel-lifetime tensors are placed at LCA of their users.
    Most buffers are multi-use (at root); hbm_out is single-use (at its writer block)."""
    all_buffers = set()
    for nid in ir.tree.blocks():
        blk = ir.tree.data(nid)
        all_buffers.update(buf.name for buf in blk.alloc_buffers)
    assert {"sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"} <= all_buffers


def test_dump_tree_runs_on_canonical_ir(tmp_path):
    """dump_tree on the canonical matmul IR produces tree.mmd and tree.png."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree_visualize import dump_tree

    ir = build_canonical_ir()
    dump_tree(ir.tree, tmp_path)
    assert (tmp_path / "tree.mmd").exists()
    """The png is generated by mmdc; if mmdc is unavailable on this host the dump should still succeed
    but with a warning. We only check the mmd file unconditionally."""


def test_trace_synthesizes_ssa_outputs():
    """The SSA trace names each op's output from its assignment LHS and infers dims/dtype/location."""
    from test.transforms._fixtures import INPUT_SPECS, f_matmul

    from nkigym.ir.dimension_analysis import analyze_dimensions

    analysis = analyze_dimensions(f_matmul, INPUT_SPECS)
    t = analysis.tensors
    assert set(t) == {"lhs_T", "rhs", "sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"}
    assert t["psum_prod"].location == "psum"
    assert t["psum_prod"].dtype == "bfloat16"
    assert t["sbuf_prod"].location == "sbuf"
    assert t["sbuf_prod"].dtype == "bfloat16"
    assert t["hbm_out"].location == "shared_hbm"
    assert analysis.dim_sizes[t["psum_prod"].dim_ids[0]] == 2048
    assert analysis.dim_sizes[t["psum_prod"].dim_ids[1]] == 2048


def test_param_dtype_seeded_from_spec():
    """analyze_dimensions reads param dtype from the (shape, dtype) spec, not from buffer inference.

    The spec dtype (``float16``) deliberately diverges from the dtype of the
    sbuf buffers the params load into (``bfloat16``). The deleted inference
    path would have yielded ``bfloat16``; spec-seeding must yield ``float16``.
    """
    from test.transforms._fixtures import f_matmul

    analysis = analyze_dimensions(f_matmul, {"lhs_T": ((2048, 2048), "float16"), "rhs": ((2048, 2048), "float16")})
    assert analysis.tensors["lhs_T"].dtype == "float16"
    assert analysis.tensors["rhs"].dtype == "float16"
