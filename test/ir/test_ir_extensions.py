"""Tests for the extended IR fields (location, dtype, kwargs, axis_map, return_name)."""

from nkigym.ir import ForNode, ISANode, build_initial_ir
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
    ir = build_initial_ir(_matmul_fixture, _INPUT_SPECS)
    envelope = ir._render_envelope_md()
    assert "# `_matmul_fixture`" in envelope
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
    from nkigym.ir import build_initial_ir
    from nkigym.ir.tree import BlockNode
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_copy import NKITensorCopy

    K = M = N = 2048

    @nkigym_kernel
    def f_matmul(lhs_T, rhs):
        sbuf_lhs_T = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
        sbuf_rhs = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
        psum_prod = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="shared_hbm", shape=(M, N), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=sbuf_lhs_T)
        NKILoad()(src=rhs, dst=sbuf_rhs)
        NKIMemset(value=0.0)(dst=psum_prod)
        NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs, dst=psum_prod)
        NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    ir = build_initial_ir(f_matmul, {"lhs_T": (K, M), "rhs": (K, N)})
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
