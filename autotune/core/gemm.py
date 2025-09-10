# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Dict, Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.core.gemm_config import GEMMConfig
from autotune.core.tensor import HBMTensor, SBUFTensor, TileCoordinates


class MetaGEMM:
    """
    Position reference:
    position = 0
    loop_0:
        position = 1
        loop_1:
            position = 2
            loop_2:
                position = 3
            position = 2
        position = 1
    position = 0
    """

    def __init__(self, transposed_lhs: bool, loop_order: str, lhs_position: int, rhs_position: int) -> None:
        """
        Requirements:
        - Loop order must contain exactly the characters 'M', 'N', and 'K'.
        - Load LHS position: | M | K | = 3 choices. Always try to stay <= N level if possible.
        - Load RHS position: | N | K | = 3 choices. Always try to stay <= M level if possible.
        - Matmul is dependent on MNK axes, has to happen in the innermost loop, position = 3.
        - Result init and save must be K_position, on the same level.
        It is pointless to further hoist result init and save out of the M,N parallel axes.
        There are 3! loop orders for MNK loops. There are hence in total, 3 * 3 * 6 = 54 kernel templates.
        """
        self.transposed_lhs = transposed_lhs
        self.loop_order = str_to_dict(loop_order)
        assert sorted(loop_order) == sorted(
            "MNK"
        ), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
        self.axes = {"lhs": ("K", "M") if self.transposed_lhs else ("M", "K"), "rhs": ("K", "N"), "result": ("M", "N")}
        self.op_positions: Dict[str, int] = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, self.axes["lhs"])
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, self.axes["rhs"])
        self.op_positions["result"] = self.loop_order["K"]
        self.op_positions["save"] = self.loop_order["K"]

    def _get_loop_range(self, position: int) -> int:
        """Check if any tensor operations at position > current will use this axis"""
        axes_used = set()
        for op_name in self.op_positions:
            op_pos = self.op_positions[op_name]
            if op_pos > position and op_name in self.axes:
                axes_used.update(self.axes[op_name])
        axis = self.loop_order[position]
        if axis in axes_used:
            trip_count = getattr(self.gemm_config, f"NUM_BLOCK_{axis}")
        else:
            trip_count = 1
        return trip_count

    def _parse_absolute_position(self, relative_position: int, axes: Tuple[str, ...]):
        """
        Convert relative_position to absolute_position.
        Relative position is wrt the axes.
        |0| A0 |1| A1 |2|
        - As small as possible to reduce redundant ops.

        Example:
            If axes=('M','K') with positions [1,3] and relative_position=1:
            Returns 2 (must be > 1 and <= 3, so minimum is 2)
        """
        axis_positions = sorted([self.loop_order[axis] for axis in axes])
        if relative_position == 0:
            absolute_position = 0
        elif relative_position == 1:
            absolute_position = axis_positions[0] + 1
        elif relative_position == 2:
            absolute_position = axis_positions[1] + 1
        else:
            raise Exception(f"relative_position {relative_position} is out of bound for axes {axes}.")
        return absolute_position

    def __call__(self, lhs: tensor, rhs: tensor, NUM_BLOCK_M: int, NUM_BLOCK_N: int, NUM_BLOCK_K: int) -> Any:
        """

        Args:
            lhs (tensor): _description_
            rhs (tensor): _description_
            NUM_BLOCK_M (int): _description_
            NUM_BLOCK_N (int): _description_
            NUM_BLOCK_K (int): _description_

        Returns:
            Any: _description_
        """
        self.gemm_config = GEMMConfig()
        self.gemm_config(
            transposed_lhs=self.transposed_lhs,
            lhs_shape=lhs.shape,
            rhs_shape=rhs.shape,
            NUM_BLOCK_M=NUM_BLOCK_M,
            NUM_BLOCK_N=NUM_BLOCK_N,
            NUM_BLOCK_K=NUM_BLOCK_K,
        )
        if self.transposed_lhs:
            self.lhs_hbm = HBMTensor(lhs, axes=("K", "M"))
        else:
            self.lhs_hbm = HBMTensor(lhs, axes=("M", "K"))
        self.rhs_hbm = HBMTensor(rhs, axes=("K", "N"))
        self.result_hbm = nl.ndarray((self.gemm_config.M, self.gemm_config.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
        loop_vars = {}
        self.maybe_init(curr_position=0, loop_vars=loop_vars)
        for block_id_0 in nl.affine_range(self._get_loop_range(position=0)):
            loop_vars[self.loop_order[0]] = block_id_0
            self.maybe_init(curr_position=1, loop_vars=loop_vars)
            for block_id_1 in nl.affine_range(self._get_loop_range(position=1)):
                loop_vars[self.loop_order[1]] = block_id_1
                self.maybe_init(curr_position=2, loop_vars=loop_vars)
                for block_id_2 in nl.affine_range(self._get_loop_range(position=2)):
                    loop_vars[self.loop_order[2]] = block_id_2
                    self.maybe_init(curr_position=3, loop_vars=loop_vars)
                    matmul_tiles(
                        self.lhs_tiles, self.rhs_tiles, self.result_tiles, tile_transposed_lhs=not self.transposed_lhs
                    )
                del loop_vars[self.loop_order[2]]
                self.maybe_store(curr_position=2)
            del loop_vars[self.loop_order[1]]
            self.maybe_store(curr_position=1)
        del loop_vars[self.loop_order[0]]
        self.maybe_store(curr_position=0)
        return self.result_hbm

    def maybe_init(self, curr_position: int, loop_vars: Dict):
        if self.op_positions["lhs"] == curr_position:
            lhs_tile_sizes: Dict[str, int] = {}
            lhs_tile_coordinates = TileCoordinates()
            for axis in self.axes["lhs"]:
                lhs_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if axis in loop_vars:
                    start_tile_index = loop_vars[axis] * getattr(self.gemm_config, f"TILES_IN_BLOCK_{axis}")
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                lhs_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
            self.lhs_tiles = SBUFTensor(
                par_axis=self.axes["lhs"][0], tile_sizes=lhs_tile_sizes, tile_coordinates=lhs_tile_coordinates
            )
            self.lhs_tiles.load(source=self.lhs_hbm)
            if not self.transposed_lhs:
                self.lhs_tiles.tile_transpose()
        if self.op_positions["rhs"] == curr_position:
            rhs_tile_sizes: Dict[str, int] = {}
            rhs_tile_coordinates = TileCoordinates()
            for axis in self.axes["rhs"]:
                rhs_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if axis in loop_vars:
                    start_tile_index = loop_vars[axis] * getattr(self.gemm_config, f"TILES_IN_BLOCK_{axis}")
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                rhs_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
            self.rhs_tiles = SBUFTensor(
                par_axis=self.axes["rhs"][0], tile_sizes=rhs_tile_sizes, tile_coordinates=rhs_tile_coordinates
            )
            self.rhs_tiles.load(source=self.rhs_hbm)
        if self.op_positions["result"] == curr_position:
            result_tile_sizes = {}
            result_tile_coordinates = TileCoordinates()
            for axis in self.axes["result"]:
                result_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if axis in loop_vars:
                    start_tile_index = loop_vars[axis] * getattr(self.gemm_config, f"TILES_IN_BLOCK_{axis}")
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                result_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
            self.result_tiles = SBUFTensor(
                par_axis=self.axes["result"][0], tile_sizes=result_tile_sizes, tile_coordinates=result_tile_coordinates
            )
            self.result_tiles.init_as_zero(self.result_hbm.dtype)

    def maybe_store(self, curr_position: int):
        if self.op_positions["save"] == curr_position:
            self.result_tiles.save_to_hbm(self.result_hbm)


def str_to_dict(loop_order: str) -> Dict:
    """
    Convert a loop order string into a bidirectional mapping dictionary.

    Args:
        loop_order (str): String representing loop order (e.g., "MNK", "KNM").

    Returns:
        Dict: Bidirectional mapping where indices map to characters and vice versa.

    Example:
        >>> str_to_dict("MNK")
        {0: 'M', 1: 'N', 2: 'K', 'M': 0, 'N': 1, 'K': 2}
    """
    loop_order_dict = {}
    for position, axis in enumerate(loop_order):
        loop_order_dict[position] = axis
        loop_order_dict[axis] = position
    return loop_order_dict


@nki.jit
def lhsT_rhs_meta_gemm(lhs: tensor, rhs: tensor, config: GEMMConfig):
    transposed_lhs = True
    gemm_instance = MetaGEMM(transposed_lhs, loop_order, lhs_position, rhs_position)
    return gemm_instance(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K)


@nki.jit
def lhs_rhs_meta_gemm(lhs: tensor, rhs: tensor, config: GEMMConfig):
    transposed_lhs = False
    gemm_instance = MetaGEMM(transposed_lhs, loop_order, lhs_position, rhs_position)
    return gemm_instance(lhs, rhs, NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K)
