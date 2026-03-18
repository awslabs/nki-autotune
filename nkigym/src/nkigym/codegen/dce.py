"""Dead code elimination for NKIKernel: single-pass backward liveness + filtering."""

from nkigym.codegen.types import NKIBlock, NKIKernel

_OUTPUT_NAME = "output"


def dce(kernel: NKIKernel) -> NKIKernel:
    """Remove dead statements via single-pass backward liveness.

    Combined backward scan propagates liveness and filters dead stmts
    in one pass (previously two separate scans).  Preserves block
    object identity when all stmts in a block are live, enabling
    id-based caching in transform analysis.

    Args:
        kernel: The NKI kernel to optimize.

    Returns:
        New kernel with dead statements removed and empty blocks dropped.
    """
    live_names: set[str] = {_OUTPUT_NAME}
    new_blocks: list[NKIBlock] = []
    for block in reversed(kernel.blocks):
        live_body = _backward_filter(block.body, live_names)
        if live_body is block.body:
            new_blocks.append(block)
        elif live_body:
            new_blocks.append(block._replace(body=live_body))
    new_blocks.reverse()
    return kernel._replace(blocks=tuple(new_blocks))


def _backward_filter(body: tuple, live_names: set[str]) -> tuple:
    """Backward scan: propagate liveness and collect live stmts.

    Returns the original body tuple when all stmts are live, preserving
    object identity for block-level caching in transform analysis.

    Args:
        body: Block body tuple (stmts in topological order).
        live_names: Mutable set of live tensor names (updated in place).

    Returns:
        Tuple of live stmts, or original body if unchanged.
    """
    live_stmts: list = []
    for stmt in reversed(body):
        if stmt.dst_name() in live_names:
            live_stmts.append(stmt)
            for name in stmt.input_names():
                live_names.add(name)
    result = body
    if len(live_stmts) != len(body):
        live_stmts.reverse()
        result = tuple(live_stmts)
    return result
