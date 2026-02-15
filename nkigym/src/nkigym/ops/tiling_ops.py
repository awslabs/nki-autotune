"""GymOps for tiling infrastructure: allocation, slicing, and storing."""

import numpy as np

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
