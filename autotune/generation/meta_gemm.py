# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Dict, List, Tuple

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.modules.dma import load_tensor_block, save_result_block
from autotune.modules.layout import get_block_ofs
from autotune.modules.matmul import GEMMCompatibility, matmul_blocks_lhsT


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
        self,
        NUM_BLOCK_M: int,
        NUM_BLOCK_N: int,
        NUM_BLOCK_K: int,
        loop_order: str,
        lhs_position: int,
        rhs_position: int,
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
        self.NUM_BLOCK_M = NUM_BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N
        self.NUM_BLOCK_K = NUM_BLOCK_K
        self.loop_order = str_to_dict(loop_order)
        assert sorted(loop_order) == sorted(
            "MNK"
        ), f"Invalid loop_order: {loop_order}. Must contain exactly M, N, and K."
        self.op_positions = {}
        self.op_positions["lhs"] = self._parse_absolute_position(lhs_position, ["M", "K"])
        self.op_positions["rhs"] = self._parse_absolute_position(rhs_position, ["K", "N"])
        self.op_positions["matmul"] = 3
        self.op_positions["result"] = self.loop_order["K"]
        self.mm = GEMMCompatibility(transposed_lhs=True)

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

    def __call__(self, lhsT: tensor, rhs: tensor):
        self.mm(
            input_tensors=(lhsT, rhs),
            kernel_kwargs={
                "NUM_BLOCK_M": self.NUM_BLOCK_M,
                "NUM_BLOCK_N": self.NUM_BLOCK_N,
                "NUM_BLOCK_K": self.NUM_BLOCK_K,
            },
        )
        self.lhsT_block_shape = self.get_block_shape(("K", "M"), self.op_positions["lhsT_block"])
        self.rhs_block_shape = self.get_block_shape(("K", "N"), self.op_positions["rhs_block"])
        self.result_block_shape = self.get_block_shape(("M", "N"), self.op_positions["result_block"])
        print(f"{self.lhsT_block_shape} @ {self.rhs_block_shape} = {self.result_block_shape}.")
        # result = nl.ndarray((self.mm.M, self.mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    def get_block_shape(self, dims: Tuple[str, str], loop_position: int) -> Tuple[int, int, int, int]:
        """
        Calculate the shape of tensor blocks in a GEMM operation.

        This function computes the shape of blocks based on the current position in nested loops
        and the specified dimensions. It determines how many blocks should be processed together
        for each dimension based on loop nesting.

        Args:
            mm (GEMMCompatibility): Object containing GEMM configuration parameters and block dimensions
            loop_order (Dict[str,int]): Str representing the order of loops (e.g., "MNK")
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
        block_shape = [getattr(self.mm, f"TILE_{dims[0]}")]
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


@nki.jit
def lhsT_rhs_gemm_general(
    lhsT: tensor,
    rhs: tensor,
    NUM_BLOCK_M: int,
    NUM_BLOCK_N: int,
    NUM_BLOCK_K: int,
    loop_order_str: str,
    tensor_positions: Dict[str, int],
):
    """
    Perform general matrix multiplication between a transposed left-hand side matrix and right-hand side matrix
    using block-based computation with customizable loop ordering and tensor load/store positions.

    This function implements a blocked GEMM operation that computes result = lhsT^T @ rhs, where tensors
    are processed in blocks to optimize memory access patterns. The loop ordering and tensor operation
    positions can be configured to explore different performance characteristics.

    Args:
        lhsT: tensor of shape (K, M) - Transposed left-hand side input matrix
        rhs: tensor of shape (K, N) - Right-hand side input matrix
        NUM_BLOCK_M: int - Number of blocks along M dimension
        NUM_BLOCK_N: int - Number of blocks along N dimension
        NUM_BLOCK_K: int - Number of blocks along K dimension
        loop_order: Dict - Dict specifying the loop ordering (e.g., "MNK")
        tensor_positions: Dict[str, int] - Positions where tensor operations occur:
            (lhsT_block, rhs_block)

    Returns:
        tensor of shape (M, N) - Result of the matrix multiplication
    FIXME: compute the global coordinate, use the global coordinate to update tensors
    """
    kernel_kwargs = {"NUM_BLOCK_M": NUM_BLOCK_M, "NUM_BLOCK_N": NUM_BLOCK_N, "NUM_BLOCK_K": NUM_BLOCK_K}
    mm = GEMMCompatibility(transposed_lhs=True)
    mm((lhsT, rhs), kernel_kwargs)
    result = nl.ndarray((mm.M, mm.N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    lhsT_block_shape = get_block_shape(mm, loop_order, ("K", "M"), op_positions["lhsT_block"])
    rhs_block_shape = get_block_shape(mm, loop_order, ("K", "N"), op_positions["rhs_block"])
    result_block_shape = get_block_shape(mm, loop_order, ("M", "N"), op_positions["result_block"])

    position = -1
    curr_block_ids = []
    if op_positions["result_block"] == position:
        result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
    for block_id_0 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[0]}")):
        position = 0
        curr_block_ids.append(block_id_0)
        if op_positions["lhsT_block"] == position:
            lhsT_block_ofs = get_block_ofs(mm, loop_order, ("K", "M"), position, curr_block_ids)
            lhsT_block = load_tensor_block(lhsT, lhsT_block_ofs, lhsT_block_shape)
        if op_positions["rhs_block"] == position:
            rhs_block_ofs = get_block_ofs(mm, loop_order, ("K", "N"), position, curr_block_ids)
            rhs_block = load_tensor_block(rhs, rhs_block_ofs, rhs_block_shape)
        if op_positions["result_block"] == position:
            result_block_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
            result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
        for block_id_1 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[1]}")):
            position = 1
            curr_block_ids.append(block_id_1)
            if op_positions["lhsT_block"] == position:
                lhsT_block_ofs = get_block_ofs(mm, loop_order, ("K", "M"), position, curr_block_ids)
                lhsT_block = load_tensor_block(lhsT, lhsT_block_ofs, lhsT_block_shape)
            if op_positions["rhs_block"] == position:
                rhs_block_ofs = get_block_ofs(mm, loop_order, ("K", "N"), position, curr_block_ids)
                rhs_block = load_tensor_block(rhs, rhs_block_ofs, rhs_block_shape)
            if op_positions["result_block"] == position:
                result_block_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
                result_block = nl.zeros(result_block_shape, dtype=result.dtype, buffer=nl.sbuf)
            for block_id_2 in nl.affine_range(getattr(mm, f"NUM_BLOCK_{loop_order[2]}")):
                position = 2
                curr_block_ids.append(block_id_2)
                if op_positions["lhsT_block"] == position:
                    lhsT_block_ofs = get_block_ofs(mm, loop_order, ("K", "M"), position, curr_block_ids)
                    lhsT_block = load_tensor_block(lhsT, lhsT_block_ofs, lhsT_block_shape)
                if op_positions["rhs_block"] == position:
                    rhs_block_ofs = get_block_ofs(mm, loop_order, ("K", "N"), position, curr_block_ids)
                    rhs_block = load_tensor_block(rhs, rhs_block_ofs, rhs_block_shape)
                if op_positions["matmul"] == position:
                    """
                    FIXME:
                    For matmul ofs, it should be the relative offset from the input tensors.
                    Not the global coordinates.
                    get_block_ofs calculates the global ofs.
                    """
                    print(f"{lhsT_block_ofs} @ {rhs_block_ofs} = {result_block_ofs}")
                    # matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
            position = 1
            curr_block_ids.pop(-1)
            if op_positions["save"] == position:
                if loop_order["M"] == 0 and loop_order["N"] == 1:
                    tile_index_ofs = (block_id_0 * mm.TILES_IN_BLOCK_M, block_id_1 * mm.TILES_IN_BLOCK_N)
                elif loop_order["M"] == 1 and loop_order["N"] == 0:
                    tile_index_ofs = (block_id_1 * mm.TILES_IN_BLOCK_M, block_id_0 * mm.TILES_IN_BLOCK_N)
                else:
                    raise Exception(f"Loop order {loop_order}. Save happened at {position}.")
                save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
            if op_positions["matmul"] == position:
                result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
                matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
        position = 0
        curr_block_ids.pop(-1)
        if op_positions["save"] == position:
            tile_index_ofs = (0, 0)
            save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
        if op_positions["matmul"] == position:
            result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
            matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
    position = -1
    curr_block_ids.pop(-1)
    # Inlined maybe_save for position -1
    if op_positions["save"] == position:
        tile_index_ofs = (0, 0)
        save_result_block(result, result_block, tile_index_ofs=tile_index_ofs)
    if op_positions["matmul"] == position:
        result_ofs = get_block_ofs(mm, loop_order, ("M", "N"), position, curr_block_ids)
        matmul_blocks_lhsT(lhsT_block, rhs_block, result_block, ofs=result_ofs)
    return result


def load_input_block(
    input_tensor, tensor_dims: Tuple[str, str], mm: GEMMCompatibility, loop_order: Dict, position: int, curr_block_ids
):
    block_shape = get_block_shape(mm, loop_order, tensor_dims, position)
    block_ofs = get_block_ofs(mm, loop_order, tensor_dims, position, curr_block_ids)
    block = load_tensor_block(input_tensor=input_tensor, ofs=block_ofs, load_shape=block_shape)
    return block
