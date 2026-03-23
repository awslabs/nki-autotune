"""Pre-computed rendering context for GEMM NKI source code generation."""


class _Ctx:
    """Pre-computed rendering context derived from a GEMM config dict.

    Attributes:
        M: Output rows.
        N: Output columns.
        K: Contraction dimension.
        transposed_lhs: Whether LHS is transposed (K, M) layout.
        lo: List of axis names at each loop level.
        axis_pos: Mapping from axis name to loop level.
        op_pos: Dict mapping operand names to absolute loop positions.
        tile: Per-axis tile sizes.
        tpb: Per-axis tiles-per-block.
        n_blocks: Per-axis block counts.
        tiles_in: Per-axis total tile counts.
        block_size: Per-axis block sizes.
    """

    def __init__(self, cfg: dict, transposed_lhs: bool) -> None:
        """Derive all static tiling/loop parameters from a config dict.

        Args:
            cfg: GEMM config dict with m_config, n_config, k_config,
                loop_order_0/1/2, lhs_position, rhs_position keys.
            transposed_lhs: Whether LHS is transposed.
        """
        self.transposed_lhs = transposed_lhs
        self._init_dims(cfg)
        self._init_loops(cfg)
        self._init_op_positions(cfg)

    def _init_dims(self, cfg: dict) -> None:
        """Extract dimension and tiling parameters.

        Args:
            cfg: GEMM config dict.
        """
        mc, nc, kc = cfg["m_config"], cfg["n_config"], cfg["k_config"]
        self.M = mc["size"]
        self.N = nc["size"]
        self.K = kc["size"]
        self.tile = {"M": mc["tile_size"], "N": nc["tile_size"], "K": kc["tile_size"]}
        self.tpb = {"M": mc["tiles_per_block"], "N": nc["tiles_per_block"], "K": kc["tiles_per_block"]}
        self.n_blocks = {"M": mc["num_blocks"], "N": nc["num_blocks"], "K": kc["num_blocks"]}
        self.tiles_in = {"M": mc["total_tiles"], "N": nc["total_tiles"], "K": kc["total_tiles"]}
        self.block_size = {"M": mc["block_size"], "N": nc["block_size"], "K": kc["block_size"]}

    def _init_loops(self, cfg: dict) -> None:
        """Build loop_order mapping and axis position lookup.

        Args:
            cfg: GEMM config dict.
        """
        self.lo = [cfg["loop_order_0"], cfg["loop_order_1"], cfg["loop_order_2"]]
        self.axis_pos = {self.lo[i]: i for i in range(3)}

    def _init_op_positions(self, cfg: dict) -> None:
        """Compute absolute positions for each operand.

        Args:
            cfg: GEMM config dict.
        """
        lhs_a, lhs_b = ("K", "M") if self.transposed_lhs else ("M", "K")
        self.op_pos = {
            "lhs": self._abs_pos(cfg["lhs_position"], lhs_a, lhs_b),
            "rhs": self._abs_pos(cfg["rhs_position"], "K", "N"),
        }
        self.op_pos["result"] = self.axis_pos["K"]
        self.op_pos["save"] = self.axis_pos["K"]

    def _abs_pos(self, rel: int, ax_a: str, ax_b: str) -> int:
        """Convert relative operand position to absolute loop position.

        Args:
            rel: Relative position (0, 1, or 2).
            ax_a: First axis name.
            ax_b: Second axis name.

        Returns:
            Absolute loop position.
        """
        pa, pb = self.axis_pos[ax_a], self.axis_pos[ax_b]
        lo, hi = min(pa, pb), max(pa, pb)
        lookup = {0: 0, 1: lo + 1, 2: hi + 1}
        return lookup[rel]

    def block_set_at(self, axis: str, position: int) -> bool:
        """Check if a block variable for axis is set at a given position.

        Args:
            axis: Axis name ("M", "N", or "K").
            position: Loop nesting position.

        Returns:
            True if the axis's loop is before this position.
        """
        return self.axis_pos[axis] < position

    def num_tiles_for(self, axis: str, position: int) -> int:
        """Get number of tiles an operand has for an axis at init position.

        Args:
            axis: Axis name.
            position: Operand's init position.

        Returns:
            TILES_PER_BLOCK if block is set, TILES_IN otherwise.
        """
        table = {True: self.tpb[axis], False: self.tiles_in[axis]}
        return table[self.block_set_at(axis, position)]

    def loop_needed(self, level: int) -> bool:
        """Check whether the loop at this level is needed.

        Args:
            level: Loop nesting level (0, 1, or 2).

        Returns:
            True if any operand needs this loop to iterate.
        """
        axis = self.lo[level]
        lhs_axes = {"M", "K"}
        rhs_axes = {"K", "N"}
        res_axes = {"M", "N"}
        lhs_needs = self.op_pos["lhs"] > level and axis in lhs_axes
        rhs_needs = self.op_pos["rhs"] > level and axis in rhs_axes
        res_needs = self.op_pos["result"] > level and axis in res_axes
        return lhs_needs or rhs_needs or res_needs

    def trip_count(self, level: int) -> int:
        """Get iteration count for a loop level.

        Args:
            level: Loop nesting level.

        Returns:
            Number of blocks if loop is needed, else 1.
        """
        table = {True: self.n_blocks[self.lo[level]], False: 1}
        return table[self.loop_needed(level)]


class _CodeBuilder:
    """Accumulates lines of generated source with indentation tracking."""

    def __init__(self) -> None:
        """Initialize empty code builder."""
        self.lines: list[str] = []
        self.indent = 0

    def line(self, text: str) -> None:
        """Append a line at current indentation.

        Args:
            text: Line content (without leading whitespace).
        """
        self.lines.append("    " * self.indent + text)

    def blank(self) -> None:
        """Append a blank line."""
        self.lines.append("")

    def build(self) -> str:
        """Return the accumulated source code.

        Returns:
            Complete source as a single string.
        """
        return "\n".join(self.lines) + "\n"
