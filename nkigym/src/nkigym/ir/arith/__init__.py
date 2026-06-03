"""Ported TVM ``arith`` substrate: Expr AST, Analyzer, simplifiers, IterMap, IntSet.

Mirrors TVM ``src/arith/`` file-for-file. Depends on nothing above it in the IR
stack; ``transforms/`` and the rest of ``ir/`` call into here, never the reverse.
"""

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import (
    EQ,
    LE,
    LT,
    Add,
    Const,
    Expr,
    FloorDiv,
    Max,
    Min,
    Mod,
    Mul,
    NonAffineError,
    Sub,
    Var,
    affine_terms,
    format_expr,
    from_affine,
    substitute,
    to_affine,
)
from nkigym.ir.arith.int_set import IntSet
from nkigym.ir.arith.iter_map import (
    IterMark,
    IterSplitExpr,
    IterSumExpr,
    detect_iter_map,
    iter_map_simplify,
    normalize_iter_map_to_expr,
)

__all__ = [
    "Add",
    "Analyzer",
    "Const",
    "EQ",
    "Expr",
    "FloorDiv",
    "IntSet",
    "IterMark",
    "IterSplitExpr",
    "IterSumExpr",
    "LE",
    "LT",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "NonAffineError",
    "Sub",
    "Var",
    "affine_terms",
    "detect_iter_map",
    "format_expr",
    "from_affine",
    "iter_map_simplify",
    "normalize_iter_map_to_expr",
    "substitute",
    "to_affine",
]
