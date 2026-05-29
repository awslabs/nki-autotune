"""Unit tests for the ``label()`` methods on ``nkigym.ir.tree`` payload classes."""

from nkigym.ops.base import AxisRole


def test_itervar_label_abbreviates_role_and_shows_dom():
    """IterVar.label() is ``axis(ROL lo..hi)`` with the role's first 3 letters."""
    from nkigym.ir.tree import IterVar

    par = IterVar(axis="d0", dom=(0, 2048), role=AxisRole.PARALLEL)
    acc = IterVar(axis="K", dom=(0, 128), role=AxisRole.ACCUMULATION)
    assert par.label() == "d0(PAR 0..2048)"
    assert acc.label() == "K(ACC 0..128)"


def test_buffer_label_shows_shape_dtype_location():
    """Buffer.label() is ``name (s0,s1) dtype@location``."""
    from nkigym.ir.tree import Buffer

    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    assert buf.label() == "sbuf_lhs_T (2048,2048) bfloat16@sbuf"


def test_bufferregion_label_renders_lo_and_width():
    """BufferRegion.label() shows each axis as ``lo : +width`` (ranges store (lo, width))."""
    from nkigym.ir.expr import Const, Mul, Var
    from nkigym.ir.tree import BufferRegion

    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    assert region.label() == "sbuf_lhs_T[i_d0_0 : +128, i_d1_0 * 128 : +128]"


def test_bufferregion_label_single_axis():
    """A single-axis region renders without a comma separator."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion

    region = BufferRegion(tensor="t", ranges=((Var(name="v"), Const(value=512)),))
    assert region.label() == "t[v : +512]"


def test_fornode_label():
    """ForNode.label() is ``Loop <loop_var> extent=<extent>``."""
    from nkigym.ir.tree import ForNode

    assert ForNode(loop_var="i_d0_0", extent=16).label() == "Loop i_d0_0 extent=16"


def test_isanode_label_includes_op_bindings_and_kwargs():
    """ISANode.label() lists the op name, per-slot region labels, and kwargs."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion, ISANode
    from nkigym.ops.memset import NKIMemset

    node = ISANode(
        op_cls=NKIMemset,
        operand_bindings={"dst": BufferRegion(tensor="psum_prod", ranges=((Var(name="v"), Const(value=128)),))},
        kwargs={"value": 0.0},
    )
    text = node.label()
    assert text.startswith("NKIMemset")
    assert "dst=psum_prod[v : +128]" in text
    assert "kwargs={'value': 0.0}" in text


def test_isanode_label_omits_empty_kwargs_and_bindings():
    """With no bindings and no kwargs, the label is just the op name."""
    from nkigym.ir.tree import ISANode
    from nkigym.ops.matmul import NKIMatmul

    assert ISANode(op_cls=NKIMatmul, operand_bindings={}, kwargs={}).label() == "NKIMatmul"


def test_blocknode_label_empty_root_shows_all_fields_with_empty_marker():
    """The root block (no iter_vars) still renders all six field labels, each empty as ∅."""
    from nkigym.ir.tree import BlockNode

    text = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()).label()
    assert text.startswith("BlockNode")
    for field_name in ("iter_vars:", "iter_values:", "reads:", "writes:", "allocs:", "annotations:"):
        assert field_name in text, f"missing field line {field_name!r}"
    """Every field is empty, so ∅ must appear on each value position."""
    assert text.count("∅") == 6


def test_blocknode_label_full_renders_every_field_content():
    """A populated block surfaces iter_var/iter_value/read/write/alloc content."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, IterVar

    block = BlockNode(
        iter_vars=(
            IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL),
            IterVar(axis="N", dom=(0, 2048), role=AxisRole.PARALLEL),
        ),
        iter_values=(Var(name="i_M"), Var(name="i_N")),
        reads=(BufferRegion(tensor="sbuf_in", ranges=((Var(name="i_M"), Const(value=128)),)),),
        writes=(BufferRegion(tensor="psum_prod", ranges=((Var(name="i_N"), Const(value=512)),)),),
        alloc_buffers=(Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum"),),
    )
    text = block.label()
    assert "M(PAR 0..2048) N(PAR 0..2048)" in text
    assert "M=i_M" in text and "N=i_N" in text
    assert "sbuf_in[i_M : +128]" in text
    assert "psum_prod[i_N : +512]" in text
    assert "psum_prod (2048,2048) float32@psum" in text
    """annotations defaulted empty → exactly one ∅ (the annotations line)."""
    assert text.count("∅") == 1


def test_blocknode_label_renders_non_empty_annotations():
    """A populated annotations dict renders on the annotations line instead of ∅."""
    from nkigym.ir.tree import BlockNode

    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=(), annotations={"skip": True})
    text = block.label()
    assert "annotations: {'skip': True}" in text
    """Only the five other empty fields show ∅; the annotations line carries the dict."""
    assert text.count("∅") == 5


def test_mermaid_node_labels_escape_brackets_and_newlines():
    """Every Mermaid node label is self-contained: no raw newline, no literal [ or ] inside the quoted text."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree_visualize import _to_mermaid

    mmd = _to_mermaid(build_canonical_ir().tree)
    for line in mmd.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("n") and '"' in stripped):
            continue
        inner = stripped[stripped.index('"') + 1 : stripped.rindex('"')]
        assert "\n" not in inner, f"raw newline in label: {stripped!r}"
        assert "[" not in inner and "]" not in inner, f"unescaped bracket in label: {stripped!r}"


def test_mermaid_shows_blocknode_iter_values_and_regions():
    """The rendered tree now surfaces BlockNode fields that the old summary omitted."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree_visualize import _to_mermaid

    mmd = _to_mermaid(build_canonical_ir().tree)
    assert "iter_values:" in mmd
    assert "reads:" in mmd and "writes:" in mmd
    assert "annotations:" in mmd
    """Region brackets are HTML-escaped, so the entity appears instead of a literal [."""
    assert "&#91;" in mmd
