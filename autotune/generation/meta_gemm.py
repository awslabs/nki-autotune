# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from itertools import permutations
from pathlib import Path
from typing import Dict, Tuple

from autotune.generation.generate import generate_configs


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
        self.global_start_tiles, self.global_num_tiles = self._calculate_global_coordinates()
        self.local_start_tiles, self.local_num_tiles = self._calculate_local_coordinates()
        self.used_loop_vars = self._get_used_loop_vars()
        self.code_file_path = code_file_path
        code = self._generate_code()
        title = f"""
'''
This is auto generated kernel codes. Do not modify directly.
loop_order = {loop_order}
lhs_position = {lhs_position}. rhs_position = {rhs_position}.
'''"""
        Path(self.code_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.code_file_path, "w") as f:
            f.write(title + code)

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

    def _calculate_global_coordinates(self):
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

    def _calculate_local_coordinates(self):
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

    def _generate_code(self) -> str:
        if self.transposed_lhs:
            lhs_name = "lhsT"
        else:
            lhs_name = "lhs"
        common_head = f"""
from autotune.modules.matmul import GEMMCompatibility, matmul_tensors
from autotune.modules.dma import load_tensor_block, save_result_block
from autotune.typing import HBMTensor, SBUFTensor
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
    result = nl.ndarray((mm.M, mm.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
    lhs_hbm = HBMTensor(lhs, axes={self.dependent_dims["lhs"]})
    rhs_hbm = HBMTensor(rhs, axes={self.dependent_dims["rhs"]})"""
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
        innermost_loop_body = self._generate_matmul_body(loop_position + 1)
        innermost_loop_body = self._add_indentation(innermost_loop_body, indentation_level)
        return_code = f"""
    return result"""
        total_code = "".join([common_head, loop_openings, innermost_loop_body, loop_closings, return_code])
        return total_code

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

        for tensor in self.global_start_tiles:
            for dim in self.global_start_tiles[tensor]:
                start = self.global_start_tiles[tensor][dim]
                relative_offset = self.local_start_tiles[tensor][dim]
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
       ({self.global_start_tiles["result"][result_dims[0]]}, {self.global_start_tiles["result"][result_dims[1]]})
    )"""
        else:
            closing = ""
        return opening, closing

    def _generate_matmul_body(self, loop_position: int):
        code = self._generate_opening_at_position(loop_position)
        code += f"""
    matmul_tensors(
        lhs_block, ({self.local_start_tiles["lhs"][self.dependent_dims["lhs"][0]]}, {self.local_start_tiles["lhs"][self.dependent_dims["lhs"][1]]}),
        rhs_block, ({self.local_start_tiles["rhs"][self.dependent_dims["rhs"][0]]}, {self.local_start_tiles["rhs"][self.dependent_dims["rhs"][1]]}),
        result_block, ({self.local_start_tiles["result"][self.dependent_dims["result"][0]]}, {self.local_start_tiles["result"][self.dependent_dims["result"][1]]}),
        compute_num_tiles=({self.local_num_tiles["M"]}, {self.local_num_tiles["N"]}, {self.local_num_tiles["K"]}),
        tile_transposed_lhs={not self.transposed_lhs},
    )"""
        return code

    def _generate_opening_at_position(self, loop_position: int) -> str:
        code = ""
        if self.op_positions["lhs"] == loop_position:
            lhs_axes = self.dependent_dims["lhs"]
            tile_offsets = {axis: self.global_start_tiles["lhs"][axis] for axis in lhs_axes}
            num_tiles = {axis: self.global_num_tiles["lhs"][axis] for axis in lhs_axes}
            tile_offsets_str = format_dict_for_code(tile_offsets)
            num_tiles_str = format_dict_for_code(num_tiles)
            code += f"""
    lhs_block = SBUFTensor(tile_sizes={{"{lhs_axes[0]}": mm.TILE_{lhs_axes[0]}, "{lhs_axes[1]}": mm.TILE_{lhs_axes[1]}}})
    lhs_block.load(lhs_hbm,
        tile_offsets={tile_offsets_str},
        num_tiles={num_tiles_str})"""
            if not self.transposed_lhs:
                code += """
    lhs_block.tile_transpose()"""
        if self.op_positions["rhs"] == loop_position:
            rhs_axes = self.dependent_dims["rhs"]
            tile_offsets = {axis: self.global_start_tiles["rhs"][axis] for axis in rhs_axes}
            num_tiles = {axis: self.global_num_tiles["rhs"][axis] for axis in rhs_axes}
            tile_offsets_str = format_dict_for_code(tile_offsets)
            num_tiles_str = format_dict_for_code(num_tiles)
            code += f"""
    rhs_block = SBUFTensor(tile_sizes={{"{rhs_axes[0]}": mm.TILE_{rhs_axes[0]}, "{rhs_axes[1]}": mm.TILE_{rhs_axes[1]}}})
    rhs_block.load(rhs_hbm,
        tile_offsets={tile_offsets_str},
        num_tiles={num_tiles_str})"""
        if self.op_positions["result"] == loop_position:
            result_axes = self.dependent_dims["result"]
            result_num_tiles = {axis: self.global_num_tiles["result"][axis] for axis in result_axes}
            result_num_tiles_str = format_dict_for_code(result_num_tiles)
            code += f"""
    result_block = SBUFTensor(tile_sizes={{"{result_axes[0]}": mm.TILE_{result_axes[0]}, "{result_axes[1]}": mm.TILE_{result_axes[1]}}})
    result_block.init_as_zero(num_tiles={result_num_tiles_str}, dtype = result.dtype)"""
        return code

    def _get_indentation(self, indent_level: int):
        indentation = 4 * " "
        return indent_level * indentation


def format_dict_for_code(dict_obj: dict) -> str:
    """Format a dictionary for code generation, removing quotes from values that are code expressions"""
    items = []
    for key, value in dict_obj.items():
        # Remove quotes from values to make them code expressions rather than string literals
        items.append(f"'{key}': {value}")
    return "{" + ", ".join(items) + "}"


def str_to_dict(loop_order: str) -> Dict:
    loop_order_dict = {}
    for position, dimension in enumerate(loop_order):
        loop_order_dict[position] = dimension
        loop_order_dict[dimension] = position
    return loop_order_dict


if __name__ == "__main__":
    loop_orders = ["".join(loop_order) for loop_order in permutations("MNK")]
    lhs_positions = [0, 1, 2]
    rhs_positions = [0, 1, 2]
    template_params = {"loop_order": loop_orders, "lhs_position": lhs_positions, "rhs_position": rhs_positions}
    template_configs = generate_configs(**template_params)
    for transposed_lhs in [True, False]:
        folder_name = "lhsT_rhs_gemm" if transposed_lhs else "lhs_rhs_gemm"
        for template_id, template_config in enumerate(template_configs):
            kernel = MetaGEMM(
                code_file_path=f"generated_kernels/{folder_name}/generated_gemm_kernel_{template_id}.py",
                transposed_lhs=transposed_lhs,
                **template_config,
            )
