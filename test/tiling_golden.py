"""Golden values for dimension analysis tests.

This module contains expected DimensionAnalysis results and generated source code
strings for various matmul configurations. Used by test_tiling.py for validation.
"""

from nkigym.tiling import DimensionAnalysis, DimInfo, TensorSliceInfo

EXPECTED_SINGLE_MATMUL = {
    ((128, 128), (128, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 128, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (128, 128), "b": (128, 128), "output": (128, 128)},
        tile_counts={"d0": 1, "d2": 1},
        num_subgraphs=1,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "output": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
        },
        output="output",
        reduction_tile_counts={"d1": 1},
    ),
    ((256, 128), (128, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (256, 128), "b": (128, 128), "output": (256, 128)},
        tile_counts={"d0": 2, "d2": 1},
        num_subgraphs=2,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1]), 1: TensorSliceInfo([128, 0], [128, 128], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1]), 1: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 1},
    ),
    ((256, 128), (128, 256)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 256, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (256, 128), "b": (128, 256), "output": (256, 256)},
        tile_counts={"d0": 2, "d2": 2},
        num_subgraphs=4,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([128, 128], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 1},
    ),
    ((128, 256), (256, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 128, "parallel"),
            "d1": DimInfo("d1", 256, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (128, 256), "b": (256, 128), "output": (128, 128)},
        tile_counts={"d0": 1, "d2": 1},
        num_subgraphs=1,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 256], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [256, 128], [1, 1])},
            "output": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
        },
        output="output",
        reduction_tile_counts={"d1": 2},
    ),
    ((512, 128), (128, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 512, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (512, 128), "b": (128, 128), "output": (512, 128)},
        tile_counts={"d0": 4, "d2": 1},
        num_subgraphs=4,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 1},
    ),
    ((256, 256), (256, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 256, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (256, 256), "b": (256, 128), "output": (256, 128)},
        tile_counts={"d0": 2, "d2": 1},
        num_subgraphs=2,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 256], [1, 1]), 1: TensorSliceInfo([128, 0], [128, 256], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [256, 128], [1, 1]), 1: TensorSliceInfo([0, 0], [256, 128], [1, 1])},
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 2},
    ),
    ((256, 256), (256, 256)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 256, "reduction"),
            "d2": DimInfo("d2", 256, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (256, 256), "b": (256, 256), "output": (256, 256)},
        tile_counts={"d0": 2, "d2": 2},
        num_subgraphs=4,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
                3: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                3: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([128, 128], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 2},
    ),
    ((128, 512), (512, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 128, "parallel"),
            "d1": DimInfo("d1", 512, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (128, 512), "b": (512, 128), "output": (128, 128)},
        tile_counts={"d0": 1, "d2": 1},
        num_subgraphs=1,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 512], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [512, 128], [1, 1])},
            "output": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
        },
        output="output",
        reduction_tile_counts={"d1": 4},
    ),
    ((512, 256), (256, 512)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 512, "parallel"),
            "d1": DimInfo("d1", 256, "reduction"),
            "d2": DimInfo("d2", 512, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (512, 256), "b": (256, 512), "output": (512, 512)},
        tile_counts={"d0": 4, "d2": 4},
        num_subgraphs=16,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                2: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                3: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                4: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
                5: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
                6: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
                7: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
                8: TensorSliceInfo([256, 0], [128, 256], [1, 1]),
                9: TensorSliceInfo([256, 0], [128, 256], [1, 1]),
                10: TensorSliceInfo([256, 0], [128, 256], [1, 1]),
                11: TensorSliceInfo([256, 0], [128, 256], [1, 1]),
                12: TensorSliceInfo([384, 0], [128, 256], [1, 1]),
                13: TensorSliceInfo([384, 0], [128, 256], [1, 1]),
                14: TensorSliceInfo([384, 0], [128, 256], [1, 1]),
                15: TensorSliceInfo([384, 0], [128, 256], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
                2: TensorSliceInfo([0, 256], [256, 128], [1, 1]),
                3: TensorSliceInfo([0, 384], [256, 128], [1, 1]),
                4: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                5: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
                6: TensorSliceInfo([0, 256], [256, 128], [1, 1]),
                7: TensorSliceInfo([0, 384], [256, 128], [1, 1]),
                8: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                9: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
                10: TensorSliceInfo([0, 256], [256, 128], [1, 1]),
                11: TensorSliceInfo([0, 384], [256, 128], [1, 1]),
                12: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                13: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
                14: TensorSliceInfo([0, 256], [256, 128], [1, 1]),
                15: TensorSliceInfo([0, 384], [256, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 256], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 384], [128, 128], [1, 1]),
                4: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                5: TensorSliceInfo([128, 128], [128, 128], [1, 1]),
                6: TensorSliceInfo([128, 256], [128, 128], [1, 1]),
                7: TensorSliceInfo([128, 384], [128, 128], [1, 1]),
                8: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                9: TensorSliceInfo([256, 128], [128, 128], [1, 1]),
                10: TensorSliceInfo([256, 256], [128, 128], [1, 1]),
                11: TensorSliceInfo([256, 384], [128, 128], [1, 1]),
                12: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
                13: TensorSliceInfo([384, 128], [128, 128], [1, 1]),
                14: TensorSliceInfo([384, 256], [128, 128], [1, 1]),
                15: TensorSliceInfo([384, 384], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 2},
    ),
    ((512, 128), (128, 512)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 512, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 512, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (512, 128), "b": (128, 512), "output": (512, 512)},
        tile_counts={"d0": 4, "d2": 4},
        num_subgraphs=16,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                4: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                5: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                6: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                7: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                8: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                9: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                10: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                11: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                12: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
                13: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
                14: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
                15: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 256], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 384], [128, 128], [1, 1]),
                4: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                5: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                6: TensorSliceInfo([0, 256], [128, 128], [1, 1]),
                7: TensorSliceInfo([0, 384], [128, 128], [1, 1]),
                8: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                9: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                10: TensorSliceInfo([0, 256], [128, 128], [1, 1]),
                11: TensorSliceInfo([0, 384], [128, 128], [1, 1]),
                12: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                13: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                14: TensorSliceInfo([0, 256], [128, 128], [1, 1]),
                15: TensorSliceInfo([0, 384], [128, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 256], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 384], [128, 128], [1, 1]),
                4: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                5: TensorSliceInfo([128, 128], [128, 128], [1, 1]),
                6: TensorSliceInfo([128, 256], [128, 128], [1, 1]),
                7: TensorSliceInfo([128, 384], [128, 128], [1, 1]),
                8: TensorSliceInfo([256, 0], [128, 128], [1, 1]),
                9: TensorSliceInfo([256, 128], [128, 128], [1, 1]),
                10: TensorSliceInfo([256, 256], [128, 128], [1, 1]),
                11: TensorSliceInfo([256, 384], [128, 128], [1, 1]),
                12: TensorSliceInfo([384, 0], [128, 128], [1, 1]),
                13: TensorSliceInfo([384, 128], [128, 128], [1, 1]),
                14: TensorSliceInfo([384, 256], [128, 128], [1, 1]),
                15: TensorSliceInfo([384, 384], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 1},
    ),
    ((128, 1024), (1024, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2"],
        dim_info={
            "d0": DimInfo("d0", 128, "parallel"),
            "d1": DimInfo("d1", 1024, "reduction"),
            "d2": DimInfo("d2", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "output": ["d0", "d2"]},
        tensor_shapes={"a": (128, 1024), "b": (1024, 128), "output": (128, 128)},
        tile_counts={"d0": 1, "d2": 1},
        num_subgraphs=1,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 1024], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [1024, 128], [1, 1])},
            "output": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
        },
        output="output",
        reduction_tile_counts={"d1": 8},
    ),
}

EXPECTED_DOUBLE_MATMUL = {
    ((128, 128), (128, 128), (128, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2", "d3"],
        dim_info={
            "d0": DimInfo("d0", 128, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 128, "reduction"),
            "d3": DimInfo("d3", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "c": ["d2", "d3"], "output": ["d0", "d3"]},
        tensor_shapes={"a": (128, 128), "b": (128, 128), "c": (128, 128), "output": (128, 128)},
        tile_counts={"d0": 1, "d3": 1},
        num_subgraphs=1,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "c": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "output": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
        },
        output="output",
        reduction_tile_counts={"d1": 1, "d2": 1},
    ),
    ((256, 128), (128, 128), (128, 128)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2", "d3"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 128, "reduction"),
            "d3": DimInfo("d3", 128, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "c": ["d2", "d3"], "output": ["d0", "d3"]},
        tensor_shapes={"a": (256, 128), "b": (128, 128), "c": (128, 128), "output": (256, 128)},
        tile_counts={"d0": 2, "d3": 1},
        num_subgraphs=2,
        slice_params={
            "a": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1]), 1: TensorSliceInfo([128, 0], [128, 128], [1, 1])},
            "b": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1]), 1: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "c": {0: TensorSliceInfo([0, 0], [128, 128], [1, 1]), 1: TensorSliceInfo([0, 0], [128, 128], [1, 1])},
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 1, "d2": 1},
    ),
    ((256, 128), (128, 128), (128, 256)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2", "d3"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 128, "reduction"),
            "d2": DimInfo("d2", 128, "reduction"),
            "d3": DimInfo("d3", 256, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "c": ["d2", "d3"], "output": ["d0", "d3"]},
        tensor_shapes={"a": (256, 128), "b": (128, 128), "c": (128, 256), "output": (256, 256)},
        tile_counts={"d0": 2, "d3": 2},
        num_subgraphs=4,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
            },
            "c": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([128, 128], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 1, "d2": 1},
    ),
    ((256, 256), (256, 256), (256, 256)): DimensionAnalysis(
        dim_order=["d0", "d1", "d2", "d3"],
        dim_info={
            "d0": DimInfo("d0", 256, "parallel"),
            "d1": DimInfo("d1", 256, "reduction"),
            "d2": DimInfo("d2", 256, "reduction"),
            "d3": DimInfo("d3", 256, "parallel"),
        },
        tensor_dims={"a": ["d0", "d1"], "b": ["d1", "d2"], "c": ["d2", "d3"], "output": ["d0", "d3"]},
        tensor_shapes={"a": (256, 256), "b": (256, 256), "c": (256, 256), "output": (256, 256)},
        tile_counts={"d0": 2, "d3": 2},
        num_subgraphs=4,
        slice_params={
            "a": {
                0: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                1: TensorSliceInfo([0, 0], [128, 256], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
                3: TensorSliceInfo([128, 0], [128, 256], [1, 1]),
            },
            "b": {
                0: TensorSliceInfo([0, 0], [256, 256], [1, 1]),
                1: TensorSliceInfo([0, 0], [256, 256], [1, 1]),
                2: TensorSliceInfo([0, 0], [256, 256], [1, 1]),
                3: TensorSliceInfo([0, 0], [256, 256], [1, 1]),
            },
            "c": {
                0: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
                2: TensorSliceInfo([0, 0], [256, 128], [1, 1]),
                3: TensorSliceInfo([0, 128], [256, 128], [1, 1]),
            },
            "output": {
                0: TensorSliceInfo([0, 0], [128, 128], [1, 1]),
                1: TensorSliceInfo([0, 128], [128, 128], [1, 1]),
                2: TensorSliceInfo([128, 0], [128, 128], [1, 1]),
                3: TensorSliceInfo([128, 128], [128, 128], [1, 1]),
            },
        },
        output="output",
        reduction_tile_counts={"d1": 2, "d2": 2},
    ),
}

GOLDEN_SINGLE_MATMUL_SOURCE = {
    (
        (128, 128),
        (128, 128),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    return output
""",
    (
        (256, 128),
        (128, 128),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = np.matmul(tensor_3, tensor_4)
    output[128:256, 0:128] = tensor_5

    return output
""",
    (
        (256, 128),
        (128, 256),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = np.matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5

    tensor_6 = a[128:256, 0:128]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = np.matmul(tensor_6, tensor_7)
    output[128:256, 0:128] = tensor_8

    tensor_9 = a[128:256, 0:128]
    tensor_10 = b[0:128, 128:256]
    tensor_11 = np.matmul(tensor_9, tensor_10)
    output[128:256, 128:256] = tensor_11

    return output
""",
    (
        (128, 256),
        (256, 128),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[128:256, 0:128]
    tensor_2 += np.matmul(tensor_3, tensor_4)
    output[0:128, 0:128] = tensor_2

    return output
""",
    (
        (256, 256),
        (256, 256),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[128:256, 0:128]
    tensor_2 += np.matmul(tensor_3, tensor_4)
    output[0:128, 0:128] = tensor_2

    tensor_5 = a[0:128, 0:128]
    tensor_6 = b[0:128, 128:256]
    tensor_7 = np.matmul(tensor_5, tensor_6)
    tensor_8 = a[0:128, 128:256]
    tensor_9 = b[128:256, 128:256]
    tensor_7 += np.matmul(tensor_8, tensor_9)
    output[0:128, 128:256] = tensor_7

    tensor_10 = a[128:256, 0:128]
    tensor_11 = b[0:128, 0:128]
    tensor_12 = np.matmul(tensor_10, tensor_11)
    tensor_13 = a[128:256, 128:256]
    tensor_14 = b[128:256, 0:128]
    tensor_12 += np.matmul(tensor_13, tensor_14)
    output[128:256, 0:128] = tensor_12

    tensor_15 = a[128:256, 0:128]
    tensor_16 = b[0:128, 128:256]
    tensor_17 = np.matmul(tensor_15, tensor_16)
    tensor_18 = a[128:256, 128:256]
    tensor_19 = b[128:256, 128:256]
    tensor_17 += np.matmul(tensor_18, tensor_19)
    output[128:256, 128:256] = tensor_17

    return output
""",
    (
        (512, 128),
        (128, 512),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((512, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    output[0:128, 0:128] = tensor_2

    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = np.matmul(tensor_3, tensor_4)
    output[0:128, 128:256] = tensor_5

    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 256:384]
    tensor_8 = np.matmul(tensor_6, tensor_7)
    output[0:128, 256:384] = tensor_8

    tensor_9 = a[0:128, 0:128]
    tensor_10 = b[0:128, 384:512]
    tensor_11 = np.matmul(tensor_9, tensor_10)
    output[0:128, 384:512] = tensor_11

    tensor_12 = a[128:256, 0:128]
    tensor_13 = b[0:128, 0:128]
    tensor_14 = np.matmul(tensor_12, tensor_13)
    output[128:256, 0:128] = tensor_14

    tensor_15 = a[128:256, 0:128]
    tensor_16 = b[0:128, 128:256]
    tensor_17 = np.matmul(tensor_15, tensor_16)
    output[128:256, 128:256] = tensor_17

    tensor_18 = a[128:256, 0:128]
    tensor_19 = b[0:128, 256:384]
    tensor_20 = np.matmul(tensor_18, tensor_19)
    output[128:256, 256:384] = tensor_20

    tensor_21 = a[128:256, 0:128]
    tensor_22 = b[0:128, 384:512]
    tensor_23 = np.matmul(tensor_21, tensor_22)
    output[128:256, 384:512] = tensor_23

    tensor_24 = a[256:384, 0:128]
    tensor_25 = b[0:128, 0:128]
    tensor_26 = np.matmul(tensor_24, tensor_25)
    output[256:384, 0:128] = tensor_26

    tensor_27 = a[256:384, 0:128]
    tensor_28 = b[0:128, 128:256]
    tensor_29 = np.matmul(tensor_27, tensor_28)
    output[256:384, 128:256] = tensor_29

    tensor_30 = a[256:384, 0:128]
    tensor_31 = b[0:128, 256:384]
    tensor_32 = np.matmul(tensor_30, tensor_31)
    output[256:384, 256:384] = tensor_32

    tensor_33 = a[256:384, 0:128]
    tensor_34 = b[0:128, 384:512]
    tensor_35 = np.matmul(tensor_33, tensor_34)
    output[256:384, 384:512] = tensor_35

    tensor_36 = a[384:512, 0:128]
    tensor_37 = b[0:128, 0:128]
    tensor_38 = np.matmul(tensor_36, tensor_37)
    output[384:512, 0:128] = tensor_38

    tensor_39 = a[384:512, 0:128]
    tensor_40 = b[0:128, 128:256]
    tensor_41 = np.matmul(tensor_39, tensor_40)
    output[384:512, 128:256] = tensor_41

    tensor_42 = a[384:512, 0:128]
    tensor_43 = b[0:128, 256:384]
    tensor_44 = np.matmul(tensor_42, tensor_43)
    output[384:512, 256:384] = tensor_44

    tensor_45 = a[384:512, 0:128]
    tensor_46 = b[0:128, 384:512]
    tensor_47 = np.matmul(tensor_45, tensor_46)
    output[384:512, 384:512] = tensor_47

    return output
""",
    (
        (128, 1024),
        (1024, 128),
    ): """\
def tiled_matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[128:256, 0:128]
    tensor_2 += np.matmul(tensor_3, tensor_4)
    tensor_5 = a[0:128, 256:384]
    tensor_6 = b[256:384, 0:128]
    tensor_2 += np.matmul(tensor_5, tensor_6)
    tensor_7 = a[0:128, 384:512]
    tensor_8 = b[384:512, 0:128]
    tensor_2 += np.matmul(tensor_7, tensor_8)
    tensor_9 = a[0:128, 512:640]
    tensor_10 = b[512:640, 0:128]
    tensor_2 += np.matmul(tensor_9, tensor_10)
    tensor_11 = a[0:128, 640:768]
    tensor_12 = b[640:768, 0:128]
    tensor_2 += np.matmul(tensor_11, tensor_12)
    tensor_13 = a[0:128, 768:896]
    tensor_14 = b[768:896, 0:128]
    tensor_2 += np.matmul(tensor_13, tensor_14)
    tensor_15 = a[0:128, 896:1024]
    tensor_16 = b[896:1024, 0:128]
    tensor_2 += np.matmul(tensor_15, tensor_16)
    output[0:128, 0:128] = tensor_2

    return output
""",
}

GOLDEN_DOUBLE_MATMUL_SOURCE = {
    (
        (128, 128),
        (128, 128),
        (128, 128),
    ): """\
def tiled_double_matmul(a, b, c):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    tensor_3 = c[0:128, 0:128]
    tensor_4 = np.matmul(tensor_2, tensor_3)
    output[0:128, 0:128] = tensor_4

    return output
""",
    (
        (256, 128),
        (128, 128),
        (128, 128),
    ): """\
def tiled_double_matmul(a, b, c):
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    tensor_3 = c[0:128, 0:128]
    tensor_4 = np.matmul(tensor_2, tensor_3)
    output[0:128, 0:128] = tensor_4

    tensor_5 = a[128:256, 0:128]
    tensor_6 = b[0:128, 0:128]
    tensor_7 = np.matmul(tensor_5, tensor_6)
    tensor_8 = c[0:128, 0:128]
    tensor_9 = np.matmul(tensor_7, tensor_8)
    output[128:256, 0:128] = tensor_9

    return output
""",
    (
        (256, 128),
        (128, 128),
        (128, 256),
    ): """\
def tiled_double_matmul(a, b, c):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = np.matmul(tensor_0, tensor_1)
    tensor_3 = c[0:128, 0:128]
    tensor_4 = np.matmul(tensor_2, tensor_3)
    output[0:128, 0:128] = tensor_4

    tensor_5 = a[0:128, 0:128]
    tensor_6 = b[0:128, 0:128]
    tensor_7 = np.matmul(tensor_5, tensor_6)
    tensor_8 = c[0:128, 128:256]
    tensor_9 = np.matmul(tensor_7, tensor_8)
    output[0:128, 128:256] = tensor_9

    tensor_10 = a[128:256, 0:128]
    tensor_11 = b[0:128, 0:128]
    tensor_12 = np.matmul(tensor_10, tensor_11)
    tensor_13 = c[0:128, 0:128]
    tensor_14 = np.matmul(tensor_12, tensor_13)
    output[128:256, 0:128] = tensor_14

    tensor_15 = a[128:256, 0:128]
    tensor_16 = b[0:128, 0:128]
    tensor_17 = np.matmul(tensor_15, tensor_16)
    tensor_18 = c[0:128, 128:256]
    tensor_19 = np.matmul(tensor_17, tensor_18)
    output[128:256, 128:256] = tensor_19

    return output
""",
}
