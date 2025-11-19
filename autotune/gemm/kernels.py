# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from neuronxcc.nki.typing import tensor

from autotune.core.tensor import HBMTensor, SBUFTensor, TileCoordinates
from autotune.gemm.config import GEMMConfig
from autotune.gemm.utils import calculate_tile_overlap_ranges

MULTIBUFF = True
PSUM_BANKING = True

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

    def __init__(self, transposed_lhs: bool, config: GEMMConfig) -> None:
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
        self.gemm_config = config
        self.axes = {"lhs": ("K", "M") if self.transposed_lhs else ("M", "K"), "rhs": ("K", "N"), "result": ("M", "N")}
        self.loop_ranges = {position: self._get_loop_range(position) for position in range(3)}
        self.multibuff = 2 if MULTIBUFF else 1
        self.num_group_acc = 2
        self.lhs_tiles = None
        self.rhs_tiles = None

    def _get_loop_range(self, position: int) -> int:
        """Check if any tensor operations at position > current will use this axis"""
        axes_used = set()
        for op_name in self.gemm_config.op_positions:
            op_pos = self.gemm_config.op_positions[op_name]
            if op_pos > position and op_name in self.axes:
                axes_used.update(self.axes[op_name])
        axis = self.gemm_config.loop_order[position]
        if axis in axes_used:
            trip_count = getattr(self.gemm_config, f"NUM_BLOCK_{axis}")
        else:
            trip_count = 1
        return trip_count

    def __call__(self, lhs: tensor, rhs: tensor) -> Any:
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
        if self.transposed_lhs:
            self.lhs_hbm = HBMTensor(lhs, axes=("K", "M"))
        else:
            self.lhs_hbm = HBMTensor(lhs, axes=("M", "K"))
        self.rhs_hbm = HBMTensor(rhs, axes=("K", "N"))
        self.result_hbm = nl.ndarray((self.gemm_config.M, self.gemm_config.N), dtype=lhs.dtype, buffer=nl.shared_hbm)
        loop_vars = {}

        # Determine whether LHS + RHS will be multibuffered
        self.lhs_multibuff = self.gemm_config.lhs_rel_position > 0 and self.multibuff > 1
        self.rhs_multibuff = self.gemm_config.rhs_rel_position > 0 and self.multibuff > 1
    
        # If multibuffered --> initialize tensor one loop iteration before loading
        if self.lhs_multibuff:
            self.lhs_multibuff_init_pos = self.gemm_config.op_positions["lhs"] - 1
        if self.rhs_multibuff:
            self.rhs_multibuff_init_pos = self.gemm_config.op_positions["rhs"] - 1
        
        self.maybe_multibuff_init(curr_position=0, loop_vars=loop_vars) 
        self.maybe_init_or_load(curr_position=0, loop_vars=loop_vars)

        for block_id_0 in nl.affine_range(self.loop_ranges[0]): 

            loop_vars[self.gemm_config.loop_order[0]] = block_id_0
            self.maybe_multibuff_init(curr_position=1, loop_vars=loop_vars) 
            self.maybe_init_or_load(curr_position=1, loop_vars=loop_vars) 

            for block_id_1 in nl.affine_range(self.loop_ranges[1]):

                loop_vars[self.gemm_config.loop_order[1]] = block_id_1
                self.maybe_multibuff_init(curr_position=2, loop_vars=loop_vars)  
                self.maybe_init_or_load(curr_position=2, loop_vars=loop_vars)

                for block_id_2 in nl.affine_range(self.loop_ranges[2]): 

                    loop_vars[self.gemm_config.loop_order[2]] = block_id_2
                    self.maybe_init_or_load(curr_position=3, loop_vars=loop_vars)

                    lhs_block_id = loop_vars[self.gemm_config.loop_order[self.lhs_multibuff_init_pos]] if self.lhs_multibuff else 0
                    rhs_block_id = loop_vars[self.gemm_config.loop_order[self.rhs_multibuff_init_pos]] if self.rhs_multibuff else 0
                    
                    if MULTIBUFF and PSUM_BANKING:
                        matmul_tiles_manual_alloc(
                            self.lhs_tiles, 
                            self.rhs_tiles, 
                            self.result_tiles, 
                            tile_transposed_lhs=not self.transposed_lhs, 
                            lhs_block_id=lhs_block_id,
                            rhs_block_id=rhs_block_id,
                            num_group_acc=self.num_group_acc
                        )

                    else:
                        matmul_tiles(
                            self.lhs_tiles, 
                            self.rhs_tiles, 
                            self.result_tiles, 
                            tile_transposed_lhs=not self.transposed_lhs, 
                            lhs_block_id=lhs_block_id,
                            rhs_block_id=rhs_block_id,
                        )

                del loop_vars[self.gemm_config.loop_order[2]]
                self.maybe_store(curr_position=2) # written to hbm here (only happens 2x)

            del loop_vars[self.gemm_config.loop_order[1]]
            self.maybe_store(curr_position=1)

        del loop_vars[self.gemm_config.loop_order[0]]
        self.maybe_store(curr_position=0)

        return self.result_hbm

    def maybe_init_or_load(self, curr_position: int, loop_vars: Dict):
        if curr_position == 0:
            block_id = 0
        else:
            block_id = loop_vars[self.gemm_config.loop_order[curr_position-1]]

        if self.gemm_config.op_positions["lhs"] == curr_position:
            lhs_tile_sizes: Dict[str, int] = {}
            lhs_tile_coordinates = TileCoordinates()
            for axis in self.axes["lhs"]:
                lhs_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if axis in loop_vars:
                    start_tile_index = loop_vars[axis] * getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                    num_tiles = getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                
                if self.lhs_tiles:
                    self.lhs_tiles.tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
                else:
                    lhs_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
            
            if not self.lhs_tiles:
                self.lhs_tiles = SBUFTensor(
                    par_axis=self.axes["lhs"][0], tile_sizes=lhs_tile_sizes, tile_coordinates=lhs_tile_coordinates, multibuffering=1, 
                )
            
            self.lhs_tiles.load(source=self.lhs_hbm, block_id=block_id)
            if not self.transposed_lhs:
                self.lhs_tiles.tile_transpose(block_id=block_id)

        if self.gemm_config.op_positions["rhs"] == curr_position:
            rhs_tile_sizes: Dict[str, int] = {}
            rhs_tile_coordinates = TileCoordinates()
            for axis in self.axes["rhs"]:
                rhs_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if axis in loop_vars:
                    start_tile_index = loop_vars[axis] * getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                    num_tiles = getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")

                if self.rhs_tiles:
                    self.rhs_tiles.tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
                else:
                    rhs_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
                
            if not self.rhs_tiles:
                self.rhs_tiles = SBUFTensor(
                    par_axis=self.axes["rhs"][0], tile_sizes=rhs_tile_sizes, tile_coordinates=rhs_tile_coordinates, multibuffering=1, 
                )

            self.rhs_tiles.load(source=self.rhs_hbm, block_id=block_id)
        
        if self.gemm_config.op_positions["result"] == curr_position:
            result_tile_sizes = {}
            result_tile_coordinates = TileCoordinates()
            for axis in self.axes["result"]:
                result_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if axis in loop_vars:
                    start_tile_index = loop_vars[axis] * getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                    num_tiles = getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                result_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)
            self.result_tiles = SBUFTensor(
                par_axis=self.axes["result"][0], tile_sizes=result_tile_sizes, tile_coordinates=result_tile_coordinates, multibuffering=None,
            )
            self.result_tiles.init_as_zero(self.result_hbm.dtype)

    def maybe_multibuff_init(self, curr_position, loop_vars):
        if self.lhs_multibuff and self.lhs_multibuff_init_pos == curr_position:
            lhs_tile_sizes: Dict[str, int] = {}
            lhs_tile_coordinates = TileCoordinates()
            for axis in self.axes["lhs"]: 
                lhs_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if self.gemm_config.op_positions["lhs"] > self.gemm_config.loop_order[axis]: # Set size of tensor to that according to position of load
                    start_tile_index = 0  # Will be updated during load
                    num_tiles = getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                lhs_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)

            self.lhs_tiles = SBUFTensor(
                par_axis=self.axes["lhs"][0], tile_sizes=lhs_tile_sizes, tile_coordinates=lhs_tile_coordinates, multibuffering=self.multibuff, 
            )
            self.lhs_tiles.init_as_zero(dtype=self.lhs_hbm.tensor.dtype)

        if self.rhs_multibuff and self.rhs_multibuff_init_pos == curr_position:
            rhs_tile_sizes: Dict[str, int] = {}
            rhs_tile_coordinates = TileCoordinates()
            for axis in self.axes["rhs"]:
                rhs_tile_sizes[axis] = getattr(self.gemm_config, f"TILE_{axis}")
                if self.gemm_config.op_positions["rhs"] > self.gemm_config.loop_order[axis]:
                    start_tile_index = 0  # Will be updated during load
                    num_tiles = getattr(self.gemm_config, f"TILES_PER_BLOCK_{axis}")
                else:
                    start_tile_index = 0
                    num_tiles = getattr(self.gemm_config, f"TILES_IN_{axis}")
                rhs_tile_coordinates.add_axis(axis, start_tile_index, num_tiles)

            self.rhs_tiles = SBUFTensor(
                par_axis=self.axes["rhs"][0], tile_sizes=rhs_tile_sizes, tile_coordinates=rhs_tile_coordinates, multibuffering=self.multibuff,
            )
            self.rhs_tiles.init_as_zero(dtype=self.rhs_hbm.tensor.dtype)

    def maybe_store(self, curr_position: int):
        if self.gemm_config.op_positions["save"] == curr_position:
            self.result_tiles.save_to_hbm(self.result_hbm)

def matmul_tiles_manual_alloc(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, tile_transposed_lhs: bool, lhs_block_id: int, rhs_block_id: int, num_group_acc: int):
    """
    Perform tiled matrix multiplication between SBUF tiles.

    Computes result_tiles += matmul(lhs_tiles, rhs_tiles) for the overlapping regions within each block.

    Args:
        lhs_tiles: Left-hand side matrix tiles stored in SBUF memory
        rhs_tiles: Right-hand side matrix tiles stored in SBUF memory
        result_tiles: Output matrix tiles stored in SBUF memory where results
            will be accumulated
        tile_transposed_lhs: (bool) - Whether lhs_tiles is transposed at the tile level.
        Note that this is not the same as lhsT_tiles.
    """
    if tile_transposed_lhs:
        TILE_M, _, _, _, TILE_K = lhs_tiles.tensor.shape
    else:
        TILE_K, _, _, _, TILE_M = lhs_tiles.tensor.shape
    _TILE_K, _, _, _, TILE_N = rhs_tiles.tensor.shape
    _TILE_M, _, _, _TILE_N = result_tiles.tensor.shape
    assert (
        TILE_K == _TILE_K
    ), f"lhs_tiles {lhs_tiles.tensor.shape} TILE_K mismatch with rhs_tiles {rhs_tiles.tensor.shape}"
    assert (
        TILE_M == _TILE_M and TILE_N == _TILE_N
    ), f"result_tiles {result_tiles.tensor.shape} shape mismatch with lhs_tiles {lhs_tiles.tensor.shape} @ rhs_tiles {rhs_tiles.tensor.shape}"

    # Calculate overlapping regions using the helper function
    overlap_info = calculate_tile_overlap_ranges(lhs_tiles, rhs_tiles, result_tiles)
    num_M_tiles, num_N_tiles, num_K_tiles = overlap_info["num_tiles"]
    M_start = overlap_info["global_starts"]["M"]
    N_start = overlap_info["global_starts"]["N"]
    K_start = overlap_info["global_starts"]["K"]
    result_M_offset, result_N_offset = overlap_info["result_offsets"]

    # Iterate over tiles using nl.affine_range for hardware optimization
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_idx_N in nl.affine_range(num_N_tiles):
        global_N_tile = N_start + tile_idx_N

        # Manual SplitAccGrp (multibuffering of PSUM result_tile)
        num_M_groups = num_M_tiles // num_group_acc
        num_M_tiles_in_group = num_group_acc
        num_M_tiles_leftover = num_M_tiles % num_group_acc

        for m_group in nl.affine_range(num_M_groups):
            result_tile = nl.zeros((num_group_acc, nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)
            
            for m_tile in nl.affine_range(num_M_tiles_in_group):
                tile_idx_M = (num_M_tiles_in_group * m_group) + m_tile
                global_M_tile = M_start + tile_idx_M

                for tile_idx_K in nl.affine_range(num_K_tiles):

                    global_K_tile = K_start + tile_idx_K
                    lhs_tile = lhs_tiles.read_tile(tile_indices={"M": global_M_tile, "K": global_K_tile}, block_id=lhs_block_id)
                    rhs_tile = rhs_tiles.read_tile(tile_indices={"K": global_K_tile, "N": global_N_tile}, block_id=rhs_block_id)
                    result_tile[m_tile, idx_res.p, idx_res.x] += nisa.nc_matmul(lhs_tile, rhs_tile)
            
            # for m_tile in nl.affine_range(num_M_tiles_in_group):
            #     tile_idx_M = (num_M_tiles_in_group * m_group) + m_tile
                result_tiles.tensor[
                    idx_res.p, result_M_offset + tile_idx_M, result_N_offset + tile_idx_N, idx_res.x
                ] += result_tile[m_tile, idx_res.p, idx_res.x]
        
        if num_M_tiles_leftover > 0:
            result_tile = nl.zeros((num_M_tiles_leftover, nl.par_dim(TILE_M), TILE_N), dtype=nl.float32, buffer=nl.psum)
            
            for m_tile in nl.affine_range(num_M_tiles_leftover):
                tile_idx_M = (num_M_groups * num_M_tiles_in_group) + m_tile
                global_M_tile = M_start + tile_idx_M

                for tile_idx_K in nl.affine_range(num_K_tiles):
                    global_K_tile = K_start + tile_idx_K
                    lhs_tile = lhs_tiles.read_tile(tile_indices={"M": global_M_tile, "K": global_K_tile}, block_id=lhs_block_id)
                    rhs_tile = rhs_tiles.read_tile(tile_indices={"K": global_K_tile, "N": global_N_tile}, block_id=rhs_block_id)
                    result_tile[m_tile, idx_res.p, idx_res.x] += nisa.nc_matmul(lhs_tile, rhs_tile)
            
            # for m_tile in nl.affine_range(num_M_tiles_leftover):
            #     tile_idx_M = (num_M_groups * num_M_tiles_in_group) + m_tile
                result_tiles.tensor[
                    idx_res.p, result_M_offset + tile_idx_M, result_N_offset + tile_idx_N, idx_res.x
                ] += result_tile[m_tile, idx_res.p, idx_res.x]

def matmul_tiles(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, tile_transposed_lhs: bool, lhs_block_id: int, rhs_block_id: int):
    """
    Perform tiled matrix multiplication between SBUF tiles.

    Computes result_tiles += matmul(lhs_tiles, rhs_tiles) for the overlapping regions within each block.

    Args:
        lhs_tiles: Left-hand side matrix tiles stored in SBUF memory
        rhs_tiles: Right-hand side matrix tiles stored in SBUF memory
        result_tiles: Output matrix tiles stored in SBUF memory where results
            will be accumulated
        tile_transposed_lhs: (bool) - Whether lhs_tiles is transposed at the tile level.
        Note that this is not the same as lhsT_tiles.
    """
    if tile_transposed_lhs:
        TILE_M, _, _, TILE_K = lhs_tiles.tensor.shape
    else:
        TILE_K, _, _, TILE_M = lhs_tiles.tensor.shape
    _TILE_K, _, _, TILE_N = rhs_tiles.tensor.shape
    _TILE_M, _, _, _TILE_N = result_tiles.tensor.shape
    assert (
        TILE_K == _TILE_K
    ), f"lhs_tiles {lhs_tiles.tensor.shape} TILE_K mismatch with rhs_tiles {rhs_tiles.tensor.shape}"
    assert (
        TILE_M == _TILE_M and TILE_N == _TILE_N
    ), f"result_tiles {result_tiles.tensor.shape} shape mismatch with lhs_tiles {lhs_tiles.tensor.shape} @ rhs_tiles {rhs_tiles.tensor.shape}"

    # Calculate overlapping regions using the helper function
    overlap_info = calculate_tile_overlap_ranges(lhs_tiles, rhs_tiles, result_tiles)
    num_M_tiles, num_N_tiles, num_K_tiles = overlap_info["num_tiles"]
    M_start = overlap_info["global_starts"]["M"]
    N_start = overlap_info["global_starts"]["N"]
    K_start = overlap_info["global_starts"]["K"]
    result_M_offset, result_N_offset = overlap_info["result_offsets"]

    # Iterate over tiles using nl.affine_range for hardware optimization
    idx_res = nl.mgrid[0:TILE_M, 0:TILE_N]
    for tile_idx_M in nl.affine_range(num_M_tiles):
        global_M_tile = M_start + tile_idx_M
        for tile_idx_N in nl.affine_range(num_N_tiles):
            global_N_tile = N_start + tile_idx_N
            """
            Use PSUM buffer to accumulate into a single hardware tile
            """
            result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for tile_idx_K in nl.affine_range(num_K_tiles):
                global_K_tile = K_start + tile_idx_K
                # Read tiles using global indices (the read_tile method now handles conversion)
                lhs_tile = lhs_tiles.read_tile(tile_indices={"M": global_M_tile, "K": global_K_tile}, block_id=lhs_block_id)
                rhs_tile = rhs_tiles.read_tile(tile_indices={"K": global_K_tile, "N": global_N_tile}, block_id=rhs_block_id)
                result_tile += nisa.nc_matmul(lhs_tile, rhs_tile)
            # Store result using local indices for direct tensor access
            # FIXME: if K=1, just copy not add
            result_tiles.tensor[
                idx_res.p, result_M_offset + tile_idx_M, result_N_offset + tile_idx_N, idx_res.x
            ] += result_tile[idx_res.p, idx_res.x]


@nki.jit
def lhsT_rhs_meta_gemm(lhs: tensor, rhs: tensor, config: Dict):
    transposed_lhs = True
    gemm_config = GEMMConfig(**config)
    gemm_kernel = MetaGEMM(transposed_lhs, gemm_config)
    return gemm_kernel(lhs, rhs)


@nki.jit
def lhs_rhs_meta_gemm(lhs: tensor, rhs: tensor, config: Dict):
    transposed_lhs = False
    gemm_config = GEMMConfig(**config)
    gemm_kernel = MetaGEMM(transposed_lhs, gemm_config)
    return gemm_kernel(lhs, rhs)
