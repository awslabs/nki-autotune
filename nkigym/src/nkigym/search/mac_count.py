"""MAC counting from GymProgram IR.

Sums multiply-accumulate operations across all ``nc_matmul`` statements.
Since all search variants are equivalent transformations of the same
computation, the MAC count is invariant and only needs to be computed
once from the root program.
"""

from nkigym.ir import GymProgram, TensorRef


def compute_mac_count(program: GymProgram) -> int:
    """Count total MACs from nc_matmul statements in a GymProgram.

    For each nc_matmul with stationary=[K, M] and moving=[K, N],
    MACs = K * M * N.

    Args:
        program: The GymProgram IR to analyze.

    Returns:
        Total multiply-accumulate count.
    """
    total = 0
    for stmt in program.stmts:
        if stmt.op != "nc_matmul":
            continue
        stat_shape = _get_operand_shape(stmt, "stationary")
        mov_shape = _get_operand_shape(stmt, "moving")
        if stat_shape and mov_shape:
            total += stat_shape[0] * stat_shape[1] * mov_shape[1]
    return total


def _get_operand_shape(stmt: "GymStatement", name: str) -> tuple[int, ...]:
    """Extract the shape of a named operand from a statement's kwargs.

    Args:
        stmt: The GymStatement to inspect.
        name: The keyword argument name (e.g. ``"stationary"``).

    Returns:
        The operand shape, or empty tuple if not found.
    """
    result: tuple[int, ...] = ()
    for key, val in stmt.kwargs:
        if key == name and isinstance(val, TensorRef):
            result = val.shape
            break
    return result
