"""Template-based GEMM kernel implementations for NKI autotuning."""

from typing import Any

import nki
import nki.isa as nisa
import nki.language as nl

from autotune.gemm.config import GEMMConfig
from autotune.gemm.tile_helpers import (
    _POS_TO_INT,
    build_tile_coords,
    build_tile_sizes,
    check_loop_needed,
    get_num_blocks,
    set_block,
    unset_block,
)
from autotune.gemm.utils import calculate_tile_overlap_ranges
from autotune.tensor import HBMTensor, SBUFTensor


class MetaGEMM(nl.NKIObject):
    """Template-based GEMM kernel generator (54 templates from loop/position combos)."""

    def __init__(self, transposed_lhs: bool, config: GEMMConfig) -> None:
        """Initialize GEMM kernel template.

        Args:
            transposed_lhs: Whether the LHS matrix is transposed.
            config: GEMM configuration with blocking and loop order.
        """
        self.transposed_lhs = transposed_lhs
        self.gemm_config = config
        self.lhs_par = "K" if self.transposed_lhs else "M"
        self.lhs_free = "M" if self.transposed_lhs else "K"
        self.rhs_par = "K"
        self.rhs_free = "N"
        self.res_par = "M"
        self.res_free = "N"
        self.loop_ranges: dict[str, int] = {}
        self.loop_ranges["0"] = self._get_loop_range("0")
        self.loop_ranges["1"] = self._get_loop_range("1")
        self.loop_ranges["2"] = self._get_loop_range("2")

    def _get_loop_range(self, position: str) -> int:
        """Compute trip count for a loop at the given nesting position.

        Args:
            position: Loop nesting level ("0", "1", or "2") as string.

        Returns:
            Number of iterations for this loop level.
        """
        axis = self.gemm_config.loop_order[position]
        pos_int = _POS_TO_INT[position]
        needed = check_loop_needed(self.gemm_config.op_positions, pos_int, axis)
        if needed:
            trip_count = get_num_blocks(self.gemm_config, axis)
        else:
            trip_count = 1
        return trip_count

    def __call__(self, lhs: Any, rhs: Any) -> Any:
        """Execute the GEMM kernel template.

        Args:
            lhs: Left-hand side input tensor.
            rhs: Right-hand side input tensor.

        Returns:
            Result tensor allocated in shared HBM.
        """
        self._init_hbm_tensors(lhs, rhs)
        self._run_nested_loops()
        return self.result_hbm

    def _init_hbm_tensors(self, lhs: Any, rhs: Any) -> None:
        """Wrap input tensors and allocate output in HBM.

        Args:
            lhs: Left-hand side input tensor.
            rhs: Right-hand side input tensor.
        """
        if self.transposed_lhs:
            self.lhs_hbm = HBMTensor(lhs, "K", "M")
        else:
            self.lhs_hbm = HBMTensor(lhs, "M", "K")
        self.rhs_hbm = HBMTensor(rhs, "K", "N")
        self.result_hbm = nl.ndarray((self.gemm_config.M, self.gemm_config.N), dtype=lhs.dtype, buffer=nl.shared_hbm)

    def _run_nested_loops(self) -> None:
        """Execute the three nested loops with operation scheduling.

        Uses block_m, block_n, block_k as explicit vars (-1 = not set).
        """
        bm = -1
        bn = -1
        bk = -1
        self.maybe_init(curr_position=0, block_m=bm, block_n=bn, block_k=bk)
        for block_id_0 in nl.affine_range(self.loop_ranges["0"]):
            ids = set_block(self.gemm_config.loop_order["0"], block_id_0, bm, bn, bk)
            bm = ids["M"]
            bn = ids["N"]
            bk = ids["K"]
            self.maybe_init(curr_position=1, block_m=bm, block_n=bn, block_k=bk)
            for block_id_1 in nl.affine_range(self.loop_ranges["1"]):
                ids1 = set_block(self.gemm_config.loop_order["1"], block_id_1, bm, bn, bk)
                self._inner_loop(ids1["M"], ids1["N"], ids1["K"])
            self.maybe_store(curr_position=1)
        ids = unset_block(self.gemm_config.loop_order["0"], bm, bn, bk)
        bm = ids["M"]
        bn = ids["N"]
        bk = ids["K"]
        self.maybe_store(curr_position=0)

    def _inner_loop(self, bm: int, bn: int, bk: int) -> None:
        """Execute the innermost loop level.

        Args:
            bm: Current M block ID.
            bn: Current N block ID.
            bk: Current K block ID.
        """
        self.maybe_init(curr_position=2, block_m=bm, block_n=bn, block_k=bk)
        for block_id_2 in nl.affine_range(self.loop_ranges["2"]):
            ids = set_block(self.gemm_config.loop_order["2"], block_id_2, bm, bn, bk)
            self.maybe_init(curr_position=3, block_m=ids["M"], block_n=ids["N"], block_k=ids["K"])
            matmul_tiles(self.lhs_tiles, self.rhs_tiles, self.result_tiles, tile_transposed_lhs=not self.transposed_lhs)
        self.maybe_store(curr_position=2)

    def maybe_init(self, curr_position: int, block_m: int, block_n: int, block_k: int) -> None:
        """Conditionally load/init tensors based on operation position.

        Args:
            curr_position: Current loop nesting position.
            block_m: M block ID or -1.
            block_n: N block ID or -1.
            block_k: K block ID or -1.
        """
        if self.gemm_config.op_positions["lhs"] == curr_position:
            self._init_lhs(block_m, block_n, block_k, curr_position)
        if self.gemm_config.op_positions["rhs"] == curr_position:
            self._init_rhs(block_m, block_n, block_k, curr_position)
        if self.gemm_config.op_positions["result"] == curr_position:
            self._init_result(block_m, block_n, block_k, curr_position)

    def _init_lhs(self, block_m: int, block_n: int, block_k: int, pos: int) -> None:
        """Load LHS tiles into SBUF.

        Args:
            block_m: M block ID or -1.
            block_n: N block ID or -1.
            block_k: K block ID or -1.
            pos: Loop position for unique tensor naming.
        """
        lhs_tile_sizes = build_tile_sizes(self.lhs_par, self.lhs_free, self.gemm_config)
        lhs_coords = build_tile_coords(self.lhs_par, self.lhs_free, self.gemm_config, block_m, block_n, block_k)
        self.lhs_tiles = SBUFTensor(
            par_axis=self.lhs_par,
            free_axis=self.lhs_free,
            tile_sizes=lhs_tile_sizes,
            tile_coordinates=lhs_coords,
            name="lhs_" + str(pos) + "_" + str(block_m) + "_" + str(block_n) + "_" + str(block_k),
        )
        self.lhs_tiles.load(source=self.lhs_hbm)
        if not self.transposed_lhs:
            self.lhs_tiles.tile_transpose()

    def _init_rhs(self, block_m: int, block_n: int, block_k: int, pos: int) -> None:
        """Load RHS tiles into SBUF.

        Args:
            block_m: M block ID or -1.
            block_n: N block ID or -1.
            block_k: K block ID or -1.
            pos: Loop position for unique tensor naming.
        """
        rhs_tile_sizes = build_tile_sizes(self.rhs_par, self.rhs_free, self.gemm_config)
        rhs_coords = build_tile_coords(self.rhs_par, self.rhs_free, self.gemm_config, block_m, block_n, block_k)
        self.rhs_tiles = SBUFTensor(
            par_axis=self.rhs_par,
            free_axis=self.rhs_free,
            tile_sizes=rhs_tile_sizes,
            tile_coordinates=rhs_coords,
            name="rhs_" + str(pos) + "_" + str(block_m) + "_" + str(block_n) + "_" + str(block_k),
        )
        self.rhs_tiles.load(source=self.rhs_hbm)

    def _init_result(self, block_m: int, block_n: int, block_k: int, pos: int) -> None:
        """Initialize result tiles in SBUF.

        Args:
            block_m: M block ID or -1.
            block_n: N block ID or -1.
            block_k: K block ID or -1.
            pos: Loop position for unique tensor naming.
        """
        res_tile_sizes = build_tile_sizes(self.res_par, self.res_free, self.gemm_config)
        res_coords = build_tile_coords(self.res_par, self.res_free, self.gemm_config, block_m, block_n, block_k)
        self.result_tiles = SBUFTensor(
            par_axis=self.res_par,
            free_axis=self.res_free,
            tile_sizes=res_tile_sizes,
            tile_coordinates=res_coords,
            name="res_" + str(pos) + "_" + str(block_m) + "_" + str(block_n) + "_" + str(block_k),
        )
        self.result_tiles.init_as_zero(self.result_hbm.dtype)

    def maybe_store(self, curr_position: int) -> None:
        """Conditionally store result tiles back to HBM.

        Args:
            curr_position: Current loop nesting position.
        """
        if self.gemm_config.op_positions["save"] == curr_position:
            self.result_tiles.save_to_hbm(self.result_hbm)


def _compute_matmul_overlap(
    lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, tile_transposed_lhs: bool
) -> dict:
    """Extract tile dimensions and compute overlap ranges.

    Args:
        lhs_tiles: LHS SBUF tiles.
        rhs_tiles: RHS SBUF tiles.
        result_tiles: Result SBUF tiles.
        tile_transposed_lhs: Whether LHS is transposed at tile level.

    Returns:
        Dict with TILE_M, TILE_N, TILE_K and overlap info.
    """
    if tile_transposed_lhs:
        tile_m = lhs_tiles.tile_sizes[lhs_tiles.par_axis]
        tile_k = lhs_tiles.tile_sizes[lhs_tiles.free_axis]
    else:
        tile_k = lhs_tiles.tile_sizes[lhs_tiles.par_axis]
        tile_m = lhs_tiles.tile_sizes[lhs_tiles.free_axis]
    tile_n = rhs_tiles.tile_sizes[rhs_tiles.free_axis]
    overlap = calculate_tile_overlap_ranges(lhs_tiles, rhs_tiles, result_tiles)
    return {"TILE_M": tile_m, "TILE_N": tile_n, "TILE_K": tile_k, "overlap": overlap}


def _do_matmul_accumulate(lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, info: dict) -> None:
    """Execute tiled matmul and accumulate into result tiles.

    Args:
        lhs_tiles: Left-hand side matrix tiles in SBUF.
        rhs_tiles: Right-hand side matrix tiles in SBUF.
        result_tiles: Output matrix tiles in SBUF (accumulated).
        info: Dict with TILE_M, TILE_N, TILE_K and overlap info.
    """
    overlap = info["overlap"]
    tile_m = info["TILE_M"]
    tile_n = info["TILE_N"]

    for tile_idx_M in nl.affine_range(overlap["num_M"]):
        for tile_idx_N in nl.affine_range(overlap["num_N"]):
            _do_matmul_inner(lhs_tiles, rhs_tiles, result_tiles, overlap, tile_idx_M, tile_idx_N, tile_m, tile_n)


def _do_matmul_inner(
    lhs_tiles: SBUFTensor,
    rhs_tiles: SBUFTensor,
    result_tiles: SBUFTensor,
    overlap: dict,
    tile_idx_M: int,
    tile_idx_N: int,
    tile_m: int,
    tile_n: int,
) -> None:
    """Execute inner K-loop matmul and accumulate one result tile.

    Args:
        lhs_tiles: LHS SBUF tiles.
        rhs_tiles: RHS SBUF tiles.
        result_tiles: Result SBUF tiles.
        overlap: Overlap dict with starts, counts, and offsets.
        tile_idx_M: Current M tile index.
        tile_idx_N: Current N tile index.
        tile_m: M tile size.
        tile_n: N tile size.
    """
    M_start = overlap["M_start"]
    N_start = overlap["N_start"]
    K_start = overlap["K_start"]
    res_psum = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
    for tile_idx_K in nl.affine_range(overlap["num_K"]):
        lhs_tile = lhs_tiles.read_tile(tile_indices={"M": M_start + tile_idx_M, "K": K_start + tile_idx_K})
        rhs_tile = rhs_tiles.read_tile(tile_indices={"K": K_start + tile_idx_K, "N": N_start + tile_idx_N})
        nisa.nc_matmul(dst=res_psum[0:tile_m, 0:tile_n], stationary=lhs_tile, moving=rhs_tile)
    _accumulate_to_result(
        result_tiles, res_psum, overlap["res_M_off"] + tile_idx_M, overlap["res_N_off"] + tile_idx_N, tile_m, tile_n
    )


def _accumulate_to_result(
    result_tiles: SBUFTensor, res_psum: Any, m_off: int, n_off: int, tile_m: int, tile_n: int
) -> None:
    """Add PSUM result into the result SBUF tile (accumulate, not overwrite).

    Args:
        result_tiles: Destination SBUF tiles.
        res_psum: Source PSUM tile.
        m_off: M tile offset in result.
        n_off: N tile offset in result.
        tile_m: M tile size.
        tile_n: N tile size.
    """
    new_contrib = nl.ndarray((tile_m, tile_n), dtype=result_tiles.tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=new_contrib[0:tile_m, 0:tile_n], src=res_psum[0:tile_m, 0:tile_n])
    existing = nl.ndarray((tile_m, tile_n), dtype=result_tiles.tensor.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=existing[0:tile_m, 0:tile_n],
        src=result_tiles.tensor[0:tile_m, m_off : m_off + 1, n_off : n_off + 1, 0:tile_n],
    )
    nisa.tensor_tensor(
        dst=result_tiles.tensor[0:tile_m, m_off : m_off + 1, n_off : n_off + 1, 0:tile_n],
        data1=existing[0:tile_m, 0:tile_n],
        data2=new_contrib[0:tile_m, 0:tile_n],
        op=nl.add,
    )


def matmul_tiles(
    lhs_tiles: SBUFTensor, rhs_tiles: SBUFTensor, result_tiles: SBUFTensor, tile_transposed_lhs: bool
) -> None:
    """Perform tiled matrix multiplication between SBUF tiles.

    Computes result_tiles += matmul(lhs_tiles, rhs_tiles) for overlapping regions.

    Args:
        lhs_tiles: Left-hand side matrix tiles in SBUF.
        rhs_tiles: Right-hand side matrix tiles in SBUF.
        result_tiles: Output matrix tiles in SBUF (accumulated).
        tile_transposed_lhs: Whether lhs_tiles is transposed at tile level.
    """
    info = _compute_matmul_overlap(lhs_tiles, rhs_tiles, result_tiles, tile_transposed_lhs)
    _do_matmul_accumulate(lhs_tiles, rhs_tiles, result_tiles, info)


def _config_from_dict(config: dict) -> GEMMConfig:
    """Construct GEMMConfig from a dict without keyword expansion.

    Args:
        config: Dictionary with m_config, n_config, k_config,
            lhs_position, rhs_position, loop_order_0/1/2 keys.

    Returns:
        Initialized GEMMConfig.
    """
    return GEMMConfig(
        config["m_config"],
        config["n_config"],
        config["k_config"],
        config["lhs_position"],
        config["rhs_position"],
        config["loop_order_0"],
        config["loop_order_1"],
        config["loop_order_2"],
    )


@nki.jit
def lhsT_rhs_meta_gemm(lhs: Any, rhs: Any, config: dict) -> Any:
    """GEMM kernel with transposed LHS.

    Args:
        lhs: Transposed left-hand side tensor (K, M).
        rhs: Right-hand side tensor (K, N).
        config: GEMMConfig parameters as a dict.

    Returns:
        Result tensor (M, N).
    """
    gemm_config = _config_from_dict(config)
    gemm_kernel = MetaGEMM(True, gemm_config)
    return gemm_kernel(lhs, rhs)


@nki.jit
def lhs_rhs_meta_gemm(lhs: Any, rhs: Any, config: dict) -> Any:
    """GEMM kernel with standard LHS.

    Args:
        lhs: Left-hand side tensor (M, K).
        rhs: Right-hand side tensor (K, N).
        config: GEMMConfig parameters as a dict.

    Returns:
        Result tensor (M, N).
    """
    gemm_config = _config_from_dict(config)
    gemm_kernel = MetaGEMM(False, gemm_config)
    return gemm_kernel(lhs, rhs)
