# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Tuple

from autotune.modules.matmul import GEMMCompatibility


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
        - Matmul is dependent on MNK dimensions, has to happen in the innermost loop, position = 3.
        - Result init and save must be K_position, on the same level.
        It is pointless to further hoist result init and save out of the M,N parallel dimensions.
        """
        self.loop_order = str_to_dict(loop_order)
        assert sorted(loop_order) == sorted(
            "MNK"
        ), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
        self.op_positions: Dict[str, int] = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, ["M", "K"])
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, ["K", "N"])
        self.op_positions["result"] = self.loop_order["K"]
        self.op_positions["save"] = self.loop_order["K"]
        self.transposed_lhs = transposed_lhs
        self.code_file_path = "generated_kernels/generated_kernel.py"
        self._generate_code()

    def _parse_absolute_position(self, relative_position: int, dependent_dims: List[str]):
        """
        relative_position to calculate absolute_position:
        - Must be > dependent_dim_positions[:relative_position]
        - Must be <= dependent_dim_positions[relative_position:]
        - As small as possible.
        """
        candidates = [0, 1, 2, 3]
        dependent_dim_positions = sorted([self.loop_order[dim] for dim in dependent_dims])
        valid_positions = []
        for candidate in candidates:
            valids = []
            for dependent_dim_position in dependent_dim_positions[:relative_position]:
                valids.append(candidate > dependent_dim_position)
            for dependent_dim_position in dependent_dim_positions[relative_position:]:
                valids.append(candidate <= dependent_dim_position)
            if all(valids):
                valid_positions.append(candidate)
        absolute_position = min(valid_positions)
        return absolute_position

    def _generate_code(self):
        imports = f"""
# This is auto generated kernel codes. Do not modify directly.
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT
from autotune.modules.dma import load_tensor_block, save_result_block
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import neuronxcc.nki as nki
        """
        func_header = f"""
@nki.jit
def lhs_rhs_gemm(
    lhs: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
):
        """
        common_body = f"""
    kernel_kwargs = {{"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}}
    mm = GEMMCompatibility(transposed_lhs={self.transposed_lhs})
    mm((lhs, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
        """
        loop_openings = ""
        loop_closings = ""
        print(self.op_positions)
        for loop_position in [0, 1, 2]:
            opening, closing = self._generate_loop(loop_position)
            loop_openings += opening
            loop_closings += closing
        innermost_loop_body = self._generate_innermost_body(loop_position + 1)
        return_code = f"""
    return result
        """
        total_code = "".join(
            [imports, func_header, common_body, loop_openings, innermost_loop_body, loop_closings, return_code]
        )
        with open(self.code_file_path, "w") as f:
            f.write(total_code)

    def _generate_loop(self, loop_position: int):
        opening = self._generate_opening_at_position(loop_position)
        indentation = self._get_indentation(loop_position)
        opening += f"""
    {indentation}for block_id_{self.loop_order[loop_position]} in nl.affine_range(mm.NUM_BLOCK_{self.loop_order[loop_position]}):
        """
        if self.op_positions["save"] == loop_position:
            closing = f"""
    {indentation}save_result_block(result, result_block, xxx)
        """
        else:
            closing = ""
        return opening, closing

    def _generate_innermost_body(self, loop_position: int):
        code = self._generate_opening_at_position(loop_position)
        indentation = self._get_indentation(loop_position)
        code += f"""
    {indentation}matmul_blocks_lhsT(lhs_block, rhs_block, result_block, ofs=xxx)
    """
        return code

    def _generate_opening_at_position(self, loop_position: int) -> str:
        indentation = self._get_indentation(loop_position)
        ops = []
        for op in self.op_positions:
            if self.op_positions[op] == loop_position:
                ops.append(op)
        code = ""
        if "lhs" in ops:
            code += f"""
    {indentation}lhs_block = load_tensor_block(lhs, xxx)
            """
        if "rhs" in ops:
            code += f"""
    {indentation}rhs_block = load_tensor_block(rhs, xxx)
            """
        if "result" in ops:
            code += f"""
    {indentation}result_block = nl.zeros(xxx)
            """
        return code

    def _get_indentation(self, indent_level: int):
        indentation = 4 * " "
        return indent_level * indentation


def get_block_shape(mm: GEMMCompatibility, dims: Tuple[str, str], loop_position: int) -> Tuple[int, int, int, int]:
    """
    Calculate the shape of tensor blocks in a GEMM operation.

    This function computes the shape of blocks based on the current position in nested loops
    and the specified dimensions. It determines how many blocks should be processed together
    for each dimension based on loop nesting.

    Args:
        dims (Tuple[str, str]): Tuple of dimension names to calculate shape for (e.g., ("M", "N"))
        loop_position (int): position in the nested loops (-1, 0, 1, 2)

    Returns:
        Tuple[int, int, int, int]: A 4-tuple representing the block shape:
            - First element: Tile size for the first dimension
            - Second element: Number of blocks * tiles in block for the first dimension
            - Third element: Number of blocks * tiles in block for the second dimension
            - Fourth element: Tile size for the second dimension

    Note:
        For dimensions with loop position less than curr_position, num_block is set to 1.
        For other dimensions, num_block is set to the corresponding NUM_BLOCK_{dim} value.
    """
    block_shape = [getattr(mm, f"TILE_{dims[0]}")]
    for dim in dims:
        dim_position = self.loop_order[dim]
        tiles_in_block = getattr(self.mm, f"TILES_IN_BLOCK_{dim}")
        if dim_position <= loop_position:
            num_block = 1
        else:
            num_block = getattr(self.mm, f"NUM_BLOCK_{dim}")
        num_tiles = num_block * tiles_in_block
        block_shape.append(num_tiles)
    block_shape.append(getattr(self.mm, f"TILE_{dims[1]}"))
    block_shape = tuple(block_shape)
    # print(
    #     f"get_block_shape: dependent dims {dims}. curr loop position {curr_position}. loop_order {loop_order}.\n--> block_shape{block_shape}."
    # )
    return block_shape


def str_to_dict(loop_order: str) -> Dict:
    loop_order_dict = {}
    for position, dimension in enumerate(loop_order):
        loop_order_dict[position] = dimension
        loop_order_dict[dimension] = position
    return loop_order_dict
