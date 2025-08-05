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

    # FIXME: if a loop var is not used, the loop should be removed.

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
        There are 3! loop orders for MNK loops. There are hence in total, 3 * 3 * 6 = 54 kernel templates.
        """
        self.transposed_lhs = transposed_lhs
        self.loop_order = str_to_dict(loop_order)
        assert sorted(loop_order) == sorted(
            "MNK"
        ), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
        self.dependent_dims = {
            "lhs": ("K", "M") if self.transposed_lhs else ("M", "K"),
            "rhs": ("K", "N"),
            "result": ("M", "N"),
        }
        self.op_positions: Dict[str, int] = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, self.dependent_dims["lhs"])
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, self.dependent_dims["rhs"])
        self.op_positions["result"] = self.loop_order["K"]
        self.op_positions["save"] = self.loop_order["K"]
        self.tensor_types = self._get_tensor_types()
        self.start_tiles, self.num_tiles = self._calculate_tensor_coordinates()
        self.relative_offsets, self.sizes = self._calculate_relative_offsets()
        self.used_loop_vars = self._get_used_loop_vars()
        self.code_file_path = code_file_path
        self._generate_code()

    def _get_tensor_types(self):
        """
        For each lhs, rhs, result
        determine if it is a block or full in each of its dimensions

        If <= dimension position:
            full
        Else:
            block

        Returns:
        tensor_types[tensor][dim] = "block" | "full"
        """
        tensor_types = {}
        for tensor in ["lhs", "rhs", "result"]:
            absolute_position = self.op_positions[tensor]
            dependent_dims = self.dependent_dims[tensor]
            tensor_types[tensor] = {}
            for dim in dependent_dims:
                dim_position = self.loop_order[dim]
                if absolute_position <= dim_position:
                    tensor_types[tensor][dim] = "full"
                else:
                    tensor_types[tensor][dim] = "block"
        return tensor_types

    def _calculate_tensor_coordinates(self):
        """
        For each lhs, rhs, result, calculate:
        - Tile start
        - Number of tiles

        If full:
            start = 0.
            num_tiles = TILES_IN_X.
        If block:
            start = block_id_X * TILES_IN_BLOCK_X.
            num_tiles = TILES_IN_BLOCK_X.
        Returns:
        start_tiles[tensor][dim] = start
        num_tiles[tensor][dim] = num_tiles
        """
        start_tiles = {}
        num_tiles = {}
        for tensor in self.tensor_types:
            start_tiles[tensor] = {}
            num_tiles[tensor] = {}
            for dim in self.tensor_types[tensor]:
                access_type = self.tensor_types[tensor][dim]
                if access_type == "full":
                    start_tiles[tensor][dim] = "0"
                    num_tiles[tensor][dim] = f"mm.TILES_IN_{dim}"
                elif access_type == "block":
                    start_tiles[tensor][dim] = f"block_id_{dim} * mm.TILES_IN_BLOCK_{dim}"
                    num_tiles[tensor][dim] = f"mm.TILES_IN_BLOCK_{dim}"
                else:
                    raise Exception(
                        f"Expecting tensor access type block | full. Received {access_type} for {tensor} {dim}."
                    )
        return start_tiles, num_tiles

    def _calculate_relative_offsets(self):
        """
        For any dimension X in M, N, K:
        Calculate the offset in both tensors involving this particular dimension
        - block, block
            relative offset = 0
            size = TILES_IN_BLOCK_X
        - block, full
            block relative offset = 0
            full relative offset = block_id_X * TILES_IN_BLOCK_X
            size = TILES_IN_BLOCK_X
        - full, full
            relative offset = 0
            size = TILES_IN_X

        relative_offsets[tensor][dim] = relative offset
        """
        relative_offsets = {}
        sizes = {}
        for dim in "MNK":
            dim_tensor_types = {}
            for tensor in self.tensor_types:
                if dim in self.tensor_types[tensor]:
                    dim_tensor_types[tensor] = self.tensor_types[tensor][dim]

            block_count = sum(1 for t_type in dim_tensor_types.values() if t_type == "block")
            full_count = len(dim_tensor_types) - block_count
            is_mixed_case = block_count > 0 and full_count > 0

            if block_count == 0:
                sizes[dim] = f"mm.TILES_IN_{dim}"
            else:
                sizes[dim] = f"mm.TILES_IN_BLOCK_{dim}"

            for tensor, access_type in dim_tensor_types.items():
                if tensor not in relative_offsets:
                    relative_offsets[tensor] = {}

                if access_type == "full" and is_mixed_case:
                    relative_offsets[tensor][dim] = f"block_id_{dim} * mm.TILES_IN_BLOCK_{dim}"
                else:
                    relative_offsets[tensor][dim] = "0"
        return relative_offsets, sizes

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
        lhs_name = "lhsT" if self.transposed_lhs else "lhs"
        common_head = f"""
# This is auto generated kernel codes. Do not modify directly.
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT, matmul_blocks_lhs
from autotune.modules.dma import load_tensor_block, save_result_block
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor
import neuronxcc.nki as nki

@nki.jit
def {lhs_name}_rhs_gemm(
    lhs: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
):
    kernel_kwargs = {{"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}}
    mm = GEMMCompatibility(transposed_lhs={self.transposed_lhs})
    mm((lhs, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)"""
        indentation_level = 0
        loop_openings = ""
        loop_closings = ""
        for loop_position in [0, 1, 2]:
            opening, closing = self._generate_loop(loop_position)
            opening = self._add_indentation(opening, indentation_level)
            closing = self._add_indentation(closing, indentation_level)
            loop_openings += opening
            loop_closings += closing
            loop_var = self.loop_order[loop_position]
            if loop_var in self.used_loop_vars:
                indentation_level += 1
        innermost_loop_body = self._generate_innermost_body(loop_position + 1)
        innermost_loop_body = self._add_indentation(innermost_loop_body, indentation_level)
        return_code = f"""
    return result"""
        total_code = "".join([common_head, loop_openings, innermost_loop_body, loop_closings, return_code])
        Path(self.code_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.code_file_path, "w") as f:
            f.write(total_code)

    def _add_indentation(self, code: str, indentation_level: int) -> str:
        indentation = self._get_indentation(indentation_level)
        lines = code.split("\n")
        indented_lines = []
        for line in lines:
            indented_lines.append(f"{indentation}{line}")
        indented_code = "\n".join(indented_lines)
        return indented_code

    def _get_used_loop_vars(self):
        """Determine which loop variables are actually used in operations"""
        used_loop_vars = set()

        for tensor in self.start_tiles:
            for dim in self.start_tiles[tensor]:
                start = self.start_tiles[tensor][dim]
                relative_offset = self.relative_offsets[tensor][dim]
                if "block_id" in start or "block_id" in relative_offset:
                    used_loop_vars.add(dim)
        return used_loop_vars

    def _generate_loop(self, loop_position: int):
        opening = self._generate_opening_at_position(loop_position)
        loop_var = self.loop_order[loop_position]
        if loop_var in self.used_loop_vars:
            opening += f"""
    for block_id_{loop_var} in nl.affine_range(mm.NUM_BLOCK_{loop_var}):"""
        if self.op_positions["save"] == loop_position:
            result_dims = self.dependent_dims["result"]
            closing = f"""
    save_result_block(result, result_block,
       ({self.start_tiles["result"][result_dims[0]]}, {self.num_tiles["result"][result_dims[0]]}),
       ({self.start_tiles["result"][result_dims[1]]}, {self.num_tiles["result"][result_dims[1]]})
    )"""
        else:
            closing = ""
        return opening, closing

    def _generate_innermost_body(self, loop_position: int):
        code = self._generate_opening_at_position(loop_position)
        matmul_subroutine = "matmul_blocks_lhsT" if self.transposed_lhs else "matmul_blocks_lhs"
        code += f"""
    {matmul_subroutine}(
        lhs_block, ({self.relative_offsets["lhs"][self.dependent_dims["lhs"][0]]}, {self.relative_offsets["lhs"][self.dependent_dims["lhs"][1]]}),
        rhs_block, ({self.relative_offsets["rhs"][self.dependent_dims["rhs"][0]]}, {self.relative_offsets["rhs"][self.dependent_dims["rhs"][1]]}),
        result_block, ({self.relative_offsets["result"][self.dependent_dims["result"][0]]}, {self.relative_offsets["result"][self.dependent_dims["result"][1]]}),
        ({self.sizes["M"]}, {self.sizes["N"]}, {self.sizes["K"]})
    )"""
        return code

    def _generate_opening_at_position(self, loop_position: int) -> str:
        code = ""
        if self.op_positions["lhs"] == loop_position:
            lhs_dims = self.dependent_dims["lhs"]
            code += f"""
    lhs_block = load_tensor_block(input_tensor=lhs,
                                dim_0=(mm.TILE_{lhs_dims[0]}, {self.start_tiles["lhs"][lhs_dims[0]]}, {self.num_tiles["lhs"][lhs_dims[0]]}),
                                dim_1=(mm.TILE_{lhs_dims[1]}, {self.start_tiles["lhs"][lhs_dims[1]]}, {self.num_tiles["lhs"][lhs_dims[1]]}))"""
        if self.op_positions["rhs"] == loop_position:
            rhs_dims = self.dependent_dims["rhs"]
            code += f"""
    rhs_block = load_tensor_block(input_tensor=rhs,
                                dim_0=(mm.TILE_{rhs_dims[0]}, {self.start_tiles["rhs"][rhs_dims[0]]}, {self.num_tiles["rhs"][rhs_dims[0]]}),
                                dim_1=(mm.TILE_{rhs_dims[1]}, {self.start_tiles["rhs"][rhs_dims[1]]}, {self.num_tiles["rhs"][rhs_dims[1]]}))"""
        if self.op_positions["result"] == loop_position:
            result_dims = self.dependent_dims["result"]
            code += f"""
    result_block = nl.zeros((mm.TILE_{result_dims[0]}, {self.num_tiles["result"][result_dims[0]]}, {self.num_tiles["result"][result_dims[1]]}, mm.TILE_{result_dims[1]}),
                             dtype=result.dtype,buffer=nl.sbuf)"""
        return code

    def _get_indentation(self, indent_level: int):
        indentation = 4 * " "
        return indent_level * indentation


def str_to_dict(loop_order: str) -> Dict:
    loop_order_dict = {}
    for position, dimension in enumerate(loop_order):
        loop_order_dict[position] = dimension
        loop_order_dict[dimension] = position
    return loop_order_dict
