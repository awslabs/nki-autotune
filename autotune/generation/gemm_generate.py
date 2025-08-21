# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from itertools import permutations
from typing import Any, Dict, Tuple

import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.core.tensor import HBMTensor, SBUFTensor
from autotune.generation.generate import generate_configs
from autotune.modules.dma import save_result_block
from autotune.modules.matmul import GEMMCompatibility, matmul_tensors


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
        self.mm = GEMMCompatibility(transposed_lhs=transposed_lhs)

        print(self.op_positions)
        print(self.loop_order)

    def _parse_absolute_position(self, relative_position: int, axes: Tuple[str, ...]):
        """
        Convert relative_position to absolute_position.
        Relative position is wrt the axes.
        |0| P0 |1| P1 |2|
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
        self.mm((lhs, rhs), {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K})
        if self.transposed_lhs:
            lhs_hbm = HBMTensor(lhs, axes=("K", "M"))
        else:
            lhs_hbm = HBMTensor(lhs, axes=("M", "K"))
        rhs_hbm = HBMTensor(rhs, axes=("K", "N"))
        result_hbm = nl.ndarray((self.mm.M, self.mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
        loop_vars = {}
        self.maybe_init(curr_position=0, loop_vars=loop_vars)
        for block_id_0 in nl.affine_range(getattr(self.mm, f"NUM_BLOCK_{self.loop_order[0]}")):
            loop_vars[self.loop_order[0]] = block_id_0
            self.maybe_init(curr_position=1, loop_vars=loop_vars)
            for block_id_1 in nl.affine_range(getattr(self.mm, f"NUM_BLOCK_{self.loop_order[1]}")):
                loop_vars[self.loop_order[1]] = block_id_1
                self.maybe_init(curr_position=2, loop_vars=loop_vars)
                for block_id_2 in nl.affine_range(getattr(self.mm, f"NUM_BLOCK_{self.loop_order[2]}")):
                    loop_vars[self.loop_order[2]] = block_id_2
                    self.maybe_init(curr_position=3, loop_vars=loop_vars)
                    matmul_tensors()
                del loop_vars[self.loop_order[2]]
                self.maybe_store(curr_position=2, loop_vars=loop_vars)
            del loop_vars[self.loop_order[1]]
            self.maybe_store(curr_position=1, loop_vars=loop_vars)
        del loop_vars[self.loop_order[0]]
        self.maybe_store(curr_position=0, loop_vars=loop_vars)

    def maybe_init(self, curr_position: int, loop_vars: Dict):
        if self.op_positions["lhs"] == curr_position:
            lhs_tile_sizes = {}
            lhs_num_tiles = {}
            for axis in self.axes["lhs"]:
                lhs_tile_sizes[axis] = getattr(self.mm, f"TILE_{axis}")
                if axis in loop_vars:
                    lhs_num_tiles[axis] = getattr(self.mm, f"TILES_IN_BLOCK_{axis}")
                else:
                    lhs_num_tiles[axis] = getattr(self.mm, f"TILES_IN_{axis}")
            lhs_tiles = SBUFTensor(par_axis=self.axes["lhs"][0], tile_sizes=lhs_tile_sizes, num_tiles=lhs_num_tiles)
            lhs_tiles.load()
            if not self.transposed_lhs:
                lhs_tiles.tile_transpose()
        if self.op_positions["rhs"] == curr_position:
            rhs_tile_sizes = {}
            rhs_num_tiles = {}
            for axis in self.axes["rhs"]:
                rhs_tile_sizes[axis] = getattr(self.mm, f"TILE_{axis}")
                if axis in loop_vars:
                    rhs_num_tiles[axis] = getattr(self.mm, f"TILES_IN_BLOCK_{axis}")
                else:
                    rhs_num_tiles[axis] = getattr(self.mm, f"TILES_IN_{axis}")
            rhs_tiles = SBUFTensor(par_axis=self.axes["rhs"][0], tile_sizes=rhs_tile_sizes, num_tiles=rhs_num_tiles)
            rhs_tiles.load()
        if self.op_positions["result"] == curr_position:
            result_tile_sizes = {}
            result_num_tiles = {}
            for axis in self.axes["result"]:
                result_tile_sizes[axis] = getattr(self.mm, f"TILE_{axis}")
                if axis in loop_vars:
                    result_num_tiles[axis] = getattr(self.mm, f"TILES_IN_BLOCK_{axis}")
                else:
                    result_num_tiles[axis] = getattr(self.mm, f"TILES_IN_{axis}")
            result_tiles = SBUFTensor(
                par_axis=self.axes["result"][0], tile_sizes=result_tile_sizes, num_tiles=result_num_tiles
            )

    def maybe_store(self, curr_position: int, loop_vars: Dict):
        if self.op_positions["save"] == curr_position:
            save_result_block()


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


if __name__ == "__main__":
    loop_orders = ["".join(loop_order) for loop_order in permutations("MNK")]
    lhs_positions = [0, 1, 2]
    rhs_positions = [0, 1, 2]
    template_params = {"loop_order": loop_orders, "lhs_position": lhs_positions, "rhs_position": rhs_positions}
    template_configs = generate_configs(**template_params)
    template_configs = [{"loop_order": "MKN", "lhs_position": 1, "rhs_position": 2}]
    transposed_lhs = True
    for template_id, template_config in enumerate(template_configs):
        kernel = MetaGEMM(transposed_lhs=transposed_lhs, **template_config)
