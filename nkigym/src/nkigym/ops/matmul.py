"""Matrix multiplication: dst = stationary.T @ moving."""

import numpy as np

from nkigym.codegen.context import get_kwarg
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import GymOp, Tensor


class MatmulOp(GymOp):
    """Matrix multiplication contracting the K dimension.

    Computes ``stationary.T @ moving`` where stationary is [K, M]
    and moving is [K, N], producing output [M, N].
    """

    op_name = "nc_matmul"
    inputs = (Tensor("stationary", ("K", "M")), Tensor("moving", ("K", "N")))
    outputs = (Tensor("result", ("M", "N")),)
    tile_limits = {"K": 128, "M": 128, "N": 512}

    def simulate(self, stationary: np.ndarray, moving: np.ndarray, **kwargs: object) -> np.ndarray:
        """Compute stationary.T @ moving, optionally accumulating.

        Args:
            stationary: Left-hand side array of shape [K, M].
            moving: Right-hand side array of shape [K, N].
            acc: Accumulation buffer to add the result into.

        Returns:
            Result array of shape [M, N], or acc + result if acc is provided.
        """
        acc = kwargs.get("acc")
        result = np.matmul(stationary.T, moving)
        if acc is not None:
            result = acc + result
        return result

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Compute output shape: [K,M] x [K,N] -> [M,N].

        Args:
            input_shapes: Tuple of (stationary_shape, moving_shape).

        Returns:
            Shape tuple (M, N).
        """
        return (input_shapes[0][1], input_shapes[1][1])

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower nc_matmul to PSUM alloc + nisa.nc_matmul, or accumulate.

        Args:
            stmt: The nc_matmul statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        stat_ref = get_kwarg(stmt, "stationary")
        mov_ref = get_kwarg(stmt, "moving")
        acc_ref = get_kwarg(stmt, "acc")

        if not isinstance(stat_ref, TensorRef):
            raise ValueError("nc_matmul missing stationary operand")
        if not isinstance(mov_ref, TensorRef):
            raise ValueError("nc_matmul missing moving operand")

        stat_name = ctx.subscript(stat_ref)
        mov_name = ctx.subscript(mov_ref)
        out_name = stmt.output.name
        out_sub = ctx.subscript(stmt.output)
        ctx.buffers[out_name] = "PSUM"

        lines: list[str] = []
        if isinstance(acc_ref, TensorRef):
            canonical = ctx.resolve(acc_ref.name)
            acc_sub = ctx.subscript(acc_ref)
            ctx.aliases[out_name] = canonical
            ctx.alias_offsets[out_name] = tuple(s for s, _ in acc_ref.slices)
            lines = [f"nisa.nc_matmul(dst={acc_sub}, stationary={stat_name}, moving={mov_name})"]
        else:
            shape_str = repr(stmt.output.shape)
            lines = [
                f"{out_name} = nl.ndarray({shape_str}, dtype=nl.float32, buffer=nl.psum)",
                f"nisa.nc_matmul(dst={out_sub}, stationary={stat_name}, moving={mov_name})",
            ]
        return lines
