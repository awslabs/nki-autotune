"""Golden programs -- basic and adjacent merge cases."""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


BEFORE_NO_ADJACENT_LOADS = GymProgram(
    "tiled_no_adjacent_loads",
    _kw({"a": (128, 128), "b": (128, 384)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 384), ((0, 128), (0, 384)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 384), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 384), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 384), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 384), ((0, 128), (256, 384)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 384), ((0, 128), (256, 384)))),
            ),
            TensorRef("output", (128, 384), ((0, 128), (256, 384))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_SINGLE_SUBGRAPH = GymProgram(
    "tiled_single_subgraph",
    _kw({"a": (128, 128), "b": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_DIFFERENT_SOURCE_TENSORS = GymProgram(
    "tiled_different_source_tensors",
    _kw({"a": (128, 128), "b": (128, 128), "c": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("c", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_DIFFERENT_PARTITION_SLICES = GymProgram(
    "tiled_different_partition_slices",
    _kw({"a": (256, 128), "b": (128, 128)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 128), ((0, 256), (0, 128)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (256, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (256, 128), ((128, 256), (0, 128)))),
            ),
            TensorRef("output", (256, 128), ((128, 256), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)

BEFORE_ADJACENT_LOADS_2X = GymProgram(
    "tiled_adjacent_loads_2x",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_ADJACENT_LOADS_2X_PARTIAL = GymProgram(
    "tiled_adjacent_loads_2x",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
            ),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256)))),
            ),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (128, 256))),
        ),
    ),
    "output",
    np.float32,
)

AFTER_ADJACENT_LOADS_2X = GymProgram(
    "tiled_adjacent_loads_2x",
    _kw({"a": (128, 128), "b": (128, 256)}),
    (
        GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("b", (128, 256), ((0, 128), (0, 256)))),),
            TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            MatmulOp,
            (
                ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                ("moving", TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            StoreOp,
            (
                ("src", TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256)))),
                ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            ),
            TensorRef("output", (128, 256), ((0, 128), (0, 256))),
        ),
        GymStatement(
            LoadOp,
            (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    "output",
    np.float32,
)
