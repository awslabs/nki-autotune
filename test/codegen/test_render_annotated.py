"""Unit tests for render_annotated — forest-node comments inline in NKI source."""

from test.codegen._rmsnorm_matmul_fixture import INPUT_SPECS, f_nkigym

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.render import render, render_annotated


def test_render_annotated_is_superset_of_render() -> None:
    """Stripping comment lines from render_annotated output yields the plain render."""
    op_graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(op_graph)
    plain = render(op_graph, forest=forest)
    annotated = render_annotated(op_graph, forest=forest)
    stripped = "\n".join(line for line in annotated.splitlines() if not line.strip().startswith("# ")) + "\n"
    assert stripped == plain


def test_render_annotated_includes_loopnode_comment_above_every_for_header() -> None:
    """Every `for ... in range(...)` line has a preceding `# LoopNode(...)` comment."""
    op_graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(op_graph)
    annotated = render_annotated(op_graph, forest=forest).splitlines()
    for i, line in enumerate(annotated):
        if line.strip().startswith("for ") and " in range(" in line:
            prev = annotated[i - 1].strip()
            assert prev.startswith("# LoopNode("), f"missing annotation above: {line!r}"


def test_render_annotated_includes_bodyleaf_comment_above_top_level_body_dispatch() -> None:
    """At least one `# BodyLeaf(` comment appears somewhere in the annotated output."""
    op_graph = parse_and_resolve(f_nkigym, INPUT_SPECS)
    forest = build_canonical_forest(op_graph)
    annotated = render_annotated(op_graph, forest=forest)
    assert "# BodyLeaf(" in annotated
    assert 'phase="psum_init"' in annotated
    assert 'phase="compute"' in annotated
    assert 'phase="drain"' in annotated
