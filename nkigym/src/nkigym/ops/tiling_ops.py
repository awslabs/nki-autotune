"""GymOps for tiling infrastructure: allocation, slicing, and storing."""

import numpy as np

from nkigym.codegen.context import get_kwarg
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import GymOp, Tensor


class EmptyOp(GymOp):
    """Allocate an uninitialized output tensor via np.empty.

    kwargs: shape0, shape1, dtype (all string literals).
    """

    op_name = "np_empty"
    inputs = ()
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(self, *, shape0: object, shape1: object, dtype: object, **kwargs: object) -> np.ndarray:
        """Allocate an empty array.

        Args:
            shape0: First dimension size (resolved to int).
            shape1: Second dimension size (resolved to int).
            dtype: Numpy dtype (resolved from string like ``np.float32``).

        Returns:
            Uninitialized numpy array of the given shape and dtype.
        """
        return np.empty((int(shape0), int(shape1)), dtype=dtype)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for EmptyOp since shape comes from kwargs."""
        raise NotImplementedError("EmptyOp shape is determined by kwargs, not input shapes")

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower np_empty to ``nl.ndarray(..., buffer=nl.shared_hbm)``.

        Args:
            stmt: The np_empty statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        dtype = get_kwarg(stmt, "dtype")
        if dtype is None:
            raise ValueError(f"np_empty for '{stmt.output.name}' missing dtype kwarg")
        dtype_str = str(dtype).replace("np.", "nl.")
        shape_str = repr(stmt.output.shape)
        name = stmt.output.name
        ctx.buffers[name] = "SHARED_HBM"
        return [f"{name} = nl.ndarray({shape_str}, dtype={dtype_str}, buffer=nl.shared_hbm)"]


class SliceOp(GymOp):
    """Extract a contiguous slice from a tensor.

    kwargs: src (variable ref), start0, stop0, start1, stop1 (int literals).
    """

    op_name = "np_slice"
    inputs = (Tensor("src", ("P", "F")),)
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(
        self, src: np.ndarray, *, start0: object, stop0: object, start1: object, stop1: object, **kwargs: object
    ) -> np.ndarray:
        """Extract a 2D slice from the source array.

        Args:
            src: Source numpy array.
            start0: Row start index.
            stop0: Row stop index.
            start1: Column start index.
            stop1: Column stop index.

        Returns:
            Sliced view of the source array.
        """
        return src[int(start0) : int(stop0), int(start1) : int(stop1)]

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for SliceOp since shape comes from kwargs."""
        raise NotImplementedError("SliceOp shape is determined by kwargs, not input shapes")

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower np_slice to SBUF alloc + dma_copy.

        Args:
            stmt: The np_slice statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        if not isinstance(src_ref, TensorRef):
            raise ValueError(f"np_slice for '{stmt.output.name}' has non-TensorRef src")

        src_buffer = ctx.buffer_of(src_ref.name)
        out_name = stmt.output.name
        shape_str = repr(stmt.output.shape)
        src_subscript = ctx.subscript(src_ref)

        ctx.buffers[out_name] = "SBUF"
        lines = [f"{out_name} = {src_subscript}"]
        if src_buffer != "SBUF":
            lines = [
                f"{out_name} = nl.ndarray({shape_str}, dtype=nl.float32, buffer=nl.sbuf)",
                f"nisa.dma_copy(dst={out_name}, src={src_subscript})",
            ]
        return lines


class StoreOp(GymOp):
    """Write a source tile into a destination tensor at a slice position.

    kwargs: src (variable ref), dst (variable ref),
            start0, stop0, start1, stop1 (int literals).
    The statement output should be the same variable as dst.
    """

    op_name = "np_store"
    inputs = (Tensor("src", ("P", "F")), Tensor("dst", ("P", "F")))
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        *,
        start0: object,
        stop0: object,
        start1: object,
        stop1: object,
        **kwargs: object,
    ) -> np.ndarray:
        """Write src into a slice of dst.

        Args:
            src: Source tile array.
            dst: Destination array to write into.
            start0: Row start index in dst.
            stop0: Row stop index in dst.
            start1: Column start index in dst.
            stop1: Column stop index in dst.

        Returns:
            The dst array (mutated in place).
        """
        dst[int(start0) : int(stop0), int(start1) : int(stop1)] = src
        return dst

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Not used for StoreOp since shape comes from kwargs."""
        raise NotImplementedError("StoreOp shape is determined by kwargs, not input shapes")

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower np_store to dma_copy, with PSUM staging if needed.

        Args:
            stmt: The np_store statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        src_ref = get_kwarg(stmt, "src")
        dst_ref = get_kwarg(stmt, "dst")

        if not isinstance(src_ref, TensorRef) or not isinstance(dst_ref, TensorRef):
            raise ValueError("np_store missing src or dst operand")

        src_subscript = ctx.subscript(src_ref)
        dst_subscript = ctx.subscript(dst_ref)

        resolved_src = ctx.resolve(src_ref.name)
        src_buffer = ctx.buffers.get(resolved_src, "SBUF")

        lines: list[str] = [f"nisa.dma_copy(dst={dst_subscript}, src={src_subscript})"]
        if src_buffer == "PSUM":
            staging_name = f"_staging_{ctx.staging_counter}"
            ctx.staging_counter += 1
            shape_str = repr(src_ref.shape)
            lines = [
                f"{staging_name} = nl.ndarray({shape_str}, dtype=nl.float32, buffer=nl.sbuf)",
                f"nisa.tensor_copy(dst={staging_name}, src={src_subscript})",
                f"nisa.dma_copy(dst={dst_subscript}, src={staging_name})",
            ]
        return lines
