"""Dead code elimination for NKIKernel: backward transitive liveness from output stores."""

import dataclasses

from nkigym.codegen.types import NKIBlock, NKIKernel, _stmt_tensor_names
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp

_OUTPUT_NAME = "output"


def dce(kernel: NKIKernel) -> NKIKernel:
    """Remove dead statements from an NKIKernel via backward liveness analysis.

    Roots: all NKIDmaCopy with dst.name == "output" are always live.
    Backward walk: live stmt -> mark input tensor names live -> mark their producers live.

    Args:
        kernel: The NKI kernel to optimize.

    Returns:
        New kernel with dead statements removed and empty blocks dropped.
    """
    live_names = _find_live_names(kernel)
    new_blocks = _filter_blocks(kernel, live_names)
    return kernel._replace(blocks=tuple(new_blocks))


def _find_live_names(kernel: NKIKernel) -> set[str]:
    """Find all live tensor names via backward reachability from output stores.

    Args:
        kernel: The NKI kernel to analyze.

    Returns:
        Set of tensor names that are transitively live.
    """
    producers: dict[str, list[tuple[int, int]]] = {}
    all_stmts: list[tuple[int, int, NKIOp]] = []
    for bi, block in enumerate(kernel.blocks):
        for si, stmt in enumerate(block.body):
            all_stmts.append((bi, si, stmt))
            dst_name = _get_dst_name(stmt)
            if dst_name:
                producers.setdefault(dst_name, []).append((bi, si))

    live_set: set[tuple[int, int]] = set()
    live_names: set[str] = {_OUTPUT_NAME}
    worklist: list[tuple[int, int]] = []
    for bi, si, stmt in all_stmts:
        dst_name = _get_dst_name(stmt)
        if dst_name == _OUTPUT_NAME:
            _enqueue(bi, si, live_set, worklist)

    _propagate(worklist, all_stmts, producers, live_set, live_names)
    return live_names


def _propagate(
    worklist: list[tuple[int, int]],
    all_stmts: list[tuple[int, int, NKIOp]],
    producers: dict[str, list[tuple[int, int]]],
    live_set: set[tuple[int, int]],
    live_names: set[str],
) -> None:
    """Backward propagation of liveness through producers.

    Args:
        worklist: Initial live (block_idx, stmt_idx) pairs.
        all_stmts: All statements with their block/stmt indices.
        producers: Map from tensor name to producer locations.
        live_set: Mutable set of live statement locations.
        live_names: Mutable set of live tensor names.
    """
    stmt_map = {(bi, si): stmt for bi, si, stmt in all_stmts}
    while worklist:
        bi, si = worklist.pop()
        stmt = stmt_map[(bi, si)]
        for name in _stmt_input_names(stmt):
            live_names.add(name)
            for prod_loc in producers.get(name, []):
                _enqueue(prod_loc[0], prod_loc[1], live_set, worklist)


def _enqueue(bi: int, si: int, live_set: set[tuple[int, int]], worklist: list[tuple[int, int]]) -> None:
    """Add a statement to the worklist if not already live.

    Args:
        bi: Block index.
        si: Statement index.
        live_set: Mutable set of live locations.
        worklist: Mutable worklist.
    """
    key = (bi, si)
    if key not in live_set:
        live_set.add(key)
        worklist.append(key)


def _filter_blocks(kernel: NKIKernel, live_names: set[str]) -> list[NKIBlock]:
    """Filter dead statements from blocks and drop empty blocks.

    Args:
        kernel: The NKI kernel.
        live_names: Set of live tensor names.

    Returns:
        List of non-empty blocks with dead statements removed.
    """
    new_blocks: list[NKIBlock] = []
    for block in kernel.blocks:
        live_body = tuple(s for s in block.body if _stmt_is_live(s, live_names))
        if live_body:
            new_blocks.append(block._replace(body=live_body))
    return new_blocks


def _get_dst_name(stmt: NKIOp) -> str:
    """Extract the destination tensor name from a statement.

    Args:
        stmt: An NKI statement.

    Returns:
        Destination name, or empty string if none.
    """
    result = ""
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if fld.name == "dst" and isinstance(val, str):
            result = val
        elif fld.name == "dst" and isinstance(val, TensorRef):
            result = val.name
    return result


def _stmt_input_names(stmt: NKIOp) -> list[str]:
    """Extract input (non-dst) tensor names from a statement.

    Args:
        stmt: An NKI statement.

    Returns:
        List of input variable names.
    """
    all_names = _stmt_tensor_names(stmt)
    dst_name = _get_dst_name(stmt)
    return [n for n in all_names if n != dst_name]


def _stmt_is_live(stmt: NKIOp, live_names: set[str]) -> bool:
    """Check if a statement produces a live tensor name.

    Args:
        stmt: An NKI statement.
        live_names: Set of live tensor names.

    Returns:
        True if the statement's destination is in live_names.
    """
    dst_name = _get_dst_name(stmt)
    return dst_name in live_names
