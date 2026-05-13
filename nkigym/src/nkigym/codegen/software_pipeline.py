"""Sub-pass for the ``software_pipeline_depth`` annotation key.

Currently a no-op validator: walks the forest and visits every
annotated ``ForNode`` without changing emission. Loops with ``depth > 1``
render identically to ``depth == 1`` today — prologue / body / epilogue
emission is scheduled for the Bug B fix followup spec.

The pass still runs on every render so later enhancements (e.g.
precomputing required multi-buffer widths from pipeline stage offsets)
can slot in without re-threading the render pipeline.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock


def apply_software_pipeline(module: KernelIR) -> None:
    """Walk every forest root; observe ``software_pipeline_depth`` annotations.

    No mutation yet — see module docstring.

    Args:
        module: KernelIR to visit.
    """
    for root in module.body:
        _walk(root)


def _walk(node: ForNode | SBlock) -> None:
    """Depth-first walk over the schedule tree. SBlocks terminate descent."""
    if isinstance(node, SBlock):
        return
    depth = node.annotations.get("software_pipeline_depth", 1)
    _ = depth
    for child in node.children:
        _walk(child)
