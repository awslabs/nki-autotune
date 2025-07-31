# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
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

    def __init__(
        self, code_file_path: str, transposed_lhs: bool, loop_order: str, lhs_position: int, rhs_position: int
    ) -> None:
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
        self.code_file_path = code_file_path
        self._generate_code()

    def _parse_absolute_position(self, relative_position: int, dependent_dims: Tuple[str, ...]):
        """
        relative_position to calculate absolute_position:
        - Must be > dependent_dim_positions[:relative_position]
        - Must be <= dependent_dim_positions[relative_position:]
        - As small as possible to reduce redundant ops.
        """
        assert relative_position in range(
            len(dependent_dims) + 1
        ), f"relative_position {relative_position} is out of bound for dependent_dims {dependent_dims}."
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
        Path(self.code_file_path).parent.mkdir(parents=True, exist_ok=True)
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
    {indentation}save_result_block(result, result_block,
    {indentation}   ({self.result_block_shape[0]}, {self.result_block_shape[1]}),
    {indentation}   ({self.result_block_shape[2]}, {self.result_block_shape[3]})
    {indentation})
        """
        else:
            closing = ""
        return opening, closing

    def _generate_innermost_body(self, loop_position: int):
        code = self._generate_opening_at_position(loop_position)
        indentation = self._get_indentation(loop_position)
        num_tiles, block_offsets = self._calculate_block_offset()
        code += f"""
    {indentation}matmul_blocks_lhsT(
    {indentation}   lhs_block, ({block_offsets["lhs"][self.lhs_dims[0]]}, {block_offsets["lhs"][self.lhs_dims[1]]}),
    {indentation}   rhs_block, ({block_offsets["rhs"][self.rhs_dims[0]]}, {block_offsets["rhs"][self.rhs_dims[1]]}),
    {indentation}   result_block, ({block_offsets["result"][self.result_dims[0]]}, {block_offsets["result"][self.result_dims[1]]}),
    {indentation}   ({num_tiles["M"]}, {num_tiles["N"]}, {num_tiles["K"]})
    {indentation})
    """
        return code

    def _generate_opening_at_position(self, loop_position: int) -> str:
        indentation = self._get_indentation(loop_position)
        code = ""
        if self.op_positions["lhs"] == loop_position:
            self.lhs_block_shape = self._get_block_shape(self.lhs_dims, loop_position)
            code += f"""
    {indentation}lhs_block = load_tensor_block(input_tensor=lhs,
    {indentation}   dim_0=(mm.TILE_{self.lhs_dims[0]}, {self.lhs_block_shape[0]}, {self.lhs_block_shape[1]}),
    {indentation}   dim_1=(mm.TILE_{self.lhs_dims[1]}, {self.lhs_block_shape[2]}, {self.lhs_block_shape[3]}))
            """
        if self.op_positions["rhs"] == loop_position:
            self.rhs_block_shape = self._get_block_shape(self.rhs_dims, loop_position)
            code += f"""
    {indentation}rhs_block = load_tensor_block(input_tensor=rhs,
    {indentation}   dim_0=(mm.TILE_{self.rhs_dims[0]}, {self.rhs_block_shape[0]}, {self.rhs_block_shape[1]}),
    {indentation}   dim_1=(mm.TILE_{self.rhs_dims[1]}, {self.rhs_block_shape[2]}, {self.rhs_block_shape[3]}))
            """
        if self.op_positions["result"] == loop_position:
            self.result_block_shape = self._get_block_shape(self.result_dims, loop_position)
            code += f"""
    {indentation}result_block = nl.zeros((mm.TILE_{self.result_dims[0]}, {self.result_block_shape[1]}, {self.result_block_shape[3]}, mm.TILE_{self.result_dims[1]}), dtype=result.dtype,buffer=nl.sbuf)
            """
        return code

    def _get_indentation(self, indent_level: int):
        indentation = 4 * " "
        return indent_level * indentation

    def _get_block_shape(self, dims: Tuple[str, str], loop_position: int) -> Tuple[str, str, str, str]:
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
        block_shape = []
        for dim in dims:
            dim_position = self.loop_order[dim]
            if loop_position <= dim_position:
                start = "0"
                num_tiles = f"mm.TILES_IN_{dim}"
            else:
                start = f"block_id_{dim} * mm.TILES_IN_BLOCK_{dim}"
                num_tiles = f"mm.TILES_IN_BLOCK_{dim}"
            block_shape.extend([start, num_tiles])
        block_shape = tuple(block_shape)
        return block_shape

    def _calculate_block_offset(self):
        """
        For any dimension X in M, N, K:
        - block, block
            offset = 0
            num_X_tiles = TILES_IN_BLOCK_X
        - block, full
            block offset = 0
            full offset = block_idx_X * TILES_IN_BLOCK_X
            num_X_tiles = TILES_IN_BLOCK_X
        - full, full
            offset = 0
            num_X_tiles = TILES_IN_X
        """
        dim_load_types = {}
        for tensor in ["lhs", "rhs", "result"]:
            op_position = self.op_positions[tensor]
            for dim in getattr(self, f"{tensor}_dims"):
                if dim not in dim_load_types:
                    dim_load_types[dim] = {}
                dim_position = self.loop_order[dim]
                if op_position <= dim_position:
                    dim_load_types[dim][tensor] = "full"
                else:
                    dim_load_types[dim][tensor] = "block"
        num_tiles = {}
        for dim in dim_load_types:
            if all([load_type == "full" for load_type in dim_load_types[dim].values()]):
                num_tiles[dim] = f"mm.TILES_IN_{dim}"
            elif all([load_type == "block" for load_type in dim_load_types[dim].values()]):
                num_tiles[dim] = f"mm.TILES_IN_BLOCK_{dim}"
            else:
                num_tiles[dim] = f"mm.TILES_IN_BLOCK_{dim}"

        # Calculate offsets for each tensor and dimension
        block_offsets = {}
        for tensor in ["lhs", "rhs", "result"]:
            block_offsets[tensor] = {}
            for dim in getattr(self, f"{tensor}_dims"):
                if dim in dim_load_types and tensor in dim_load_types[dim]:
                    # Get all load types for this dimension
                    dim_loads = list(dim_load_types[dim].values())
                    # Get the load type for this tensor and dimension
                    tensor_load_type = dim_load_types[dim][tensor]

                    # Check if it's a mixed case (some "block" and some "full" access for this dimension)
                    is_all_block = all([load_type == "block" for load_type in dim_loads])
                    is_all_full = all([load_type == "full" for load_type in dim_loads])
                    is_mixed_case = not is_all_block and not is_all_full

                    # For "full" access in a mixed case, apply the full offset
                    if tensor_load_type == "full" and is_mixed_case:
                        block_offsets[tensor][dim] = f"block_id_{dim} * mm.TILES_IN_BLOCK_{dim}"
                    else:
                        # For all other cases, offset is 0
                        block_offsets[tensor][dim] = 0
        return num_tiles, block_offsets


def str_to_dict(loop_order: str) -> Dict:
    loop_order_dict = {}
    for position, dimension in enumerate(loop_order):
        loop_order_dict[position] = dimension
        loop_order_dict[dimension] = position
    return loop_order_dict
