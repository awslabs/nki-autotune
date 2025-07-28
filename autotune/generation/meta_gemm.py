# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, Tuple


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
        self.transposed_lhs = transposed_lhs
        self.loop_order = str_to_dict(loop_order)
        assert sorted(loop_order) == sorted(
            "MNK"
        ), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
        if self.transposed_lhs:
            self.lhs_dims = ("K", "M")
        else:
            self.lhs_dims = ("M", "K")
        self.rhs_dims = ("K", "N")
        self.result_dims = ("M", "N")
        self.op_positions: Dict[str, int] = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, self.lhs_dims)
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, self.rhs_dims)
        self.op_positions["result"] = self.loop_order["K"]
        self.op_positions["save"] = self.loop_order["K"]
        self.code_file_path = "generated_kernels/generated_kernel.py"
        self._generate_code()

    def _parse_absolute_position(self, relative_position: int, dependent_dims: Tuple[str, ...]):
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
        code = ""
        if self.op_positions["lhs"] == loop_position:
            shape = self._get_block_shape(self.lhs_dims, loop_position)
            code += f"""
    {indentation}lhs_block = load_tensor_block(input_tensor=lhs,
    {indentation}   dim_0=(mm.TILE_{self.lhs_dims[0]}, {shape[0]}, {shape[1]}),
    {indentation}   dim_1=(mm.TILE_{self.lhs_dims[1]}, {shape[2]}, {shape[3]}))
            """
        if self.op_positions["rhs"] == loop_position:
            shape = self._get_block_shape(self.rhs_dims, loop_position)
            code += f"""
    {indentation}rhs_block = load_tensor_block(input_tensor=rhs,
    {indentation}   dim_0=(mm.TILE_{self.rhs_dims[0]}, {shape[0]}, {shape[1]}),
    {indentation}   dim_1=(mm.TILE_{self.rhs_dims[1]}, {shape[2]}, {shape[3]}))
            """
        if self.op_positions["result"] == loop_position:
            shape = self._get_block_shape(self.result_dims, loop_position)
            code += f"""
    {indentation}result_block = nl.zeros((mm.TILE_M, {shape[1]}, {shape[3]}, mm.TILE_N), dtype=result.dtype,buffer=nl.sbuf)
            """
        return code

    def _get_indentation(self, indent_level: int):
        indentation = 4 * " "
        return indent_level * indentation

    def _get_block_shape(self, dims: Tuple[str, str], loop_position: int) -> Tuple[str, ...]:
        """
        Calculate the shape of tensor blocks in a GEMM operation.

        This function computes the shape of blocks based on the current position in nested loops
        and the specified dimensions. It determines how many blocks should be processed together
        for each dimension based on loop nesting.

        Args:
            dims (Tuple[str, str]): Tuple of dimension names to calculate shape for (e.g., ("M", "N"))
            loop_position (int): position in the nested loops (0, 1, 2, 3)

        Returns:
            Tuple[4 * int]: (starting tile index 0, number of tiles 0, starting tile index 1, number of tiles 1)

        Note:
            If <= dimension position, starting = 0. Else starting = block_id_X.
            If <= dimension position, num_tiles = mm.TILES_IN_X. Else num_tiles = mm.TILES_IN_BLOCK_X.
        """
        block_shape = ()
        for dim in dims:
            dim_position = self.loop_order[dim]
            if loop_position <= dim_position:
                starting = "0"
                num_tiles = f"mm.TILES_IN_{dim}"
            else:
                starting = f"block_id_{dim}"
                num_tiles = f"mm.TILES_IN_BLOCK_{dim}"
            block_shape += (starting, num_tiles)
        return block_shape


def str_to_dict(loop_order: str) -> Dict:
    loop_order_dict = {}
    for position, dimension in enumerate(loop_order):
        loop_order_dict[position] = dimension
        loop_order_dict[dimension] = position
    return loop_order_dict
