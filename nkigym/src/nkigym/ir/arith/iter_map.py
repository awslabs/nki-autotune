"""Detect and normalize the constant-factor iter maps used by NKIGym."""

from __future__ import annotations

from dataclasses import dataclass, field

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Expr, FloorDiv, Mod, Mul, Var


@dataclass(eq=False)
class IterMark:
    """An iteration domain over a source expression."""

    source: Expr | IterSumExpr
    extent: int


@dataclass(eq=False)
class IterSplitExpr:
    """One mixed-radix digit of an iteration mark."""

    source: IterMark
    lower_factor: int = 1
    extent: int = 0
    scale: int = 1

    def __post_init__(self) -> None:
        """Use the source extent when no narrower digit is specified."""
        if self.extent == 0:
            self.extent = self.source.extent


@dataclass(eq=False)
class IterSumExpr:
    """A scaled sum of iteration digits plus a constant base."""

    args: list[IterSplitExpr] = field(default_factory=list)
    base: int = 0


class _IterMapRewriter:
    """Rewrite integer expressions into constant mixed-radix maps."""

    def __init__(self, input_iters: dict[str, tuple[int, int]]) -> None:
        """Create one mark per zero-based input iterator."""
        self.analyzer = Analyzer()
        self.marks: dict[str, IterMark] = {}
        self.fused: dict[tuple[tuple[int, int, int, int], ...], IterMark] = {}
        self.valid = True
        for name, (lower, upper) in input_iters.items():
            if lower != 0 or upper <= lower:
                self.valid = False
            else:
                self.analyzer.bind(name, lower, upper)
                self.marks[name] = IterMark(source=Var(name=name), extent=upper - lower)

    def rewrite(self, expr: Expr) -> IterSumExpr | None:
        """Rewrite one expression after applying ordinary arithmetic folding."""
        return self._visit(self.analyzer.simplify(expr))

    def _visit(self, expr: Expr) -> IterSumExpr | None:
        """Dispatch the supported constant-factor expression nodes."""
        result: IterSumExpr | None
        if isinstance(expr, Const):
            result = IterSumExpr(base=expr.value)
        elif isinstance(expr, Var):
            mark = self.marks.get(expr.name)
            result = None if mark is None else IterSumExpr(args=[IterSplitExpr(source=mark)])
        elif isinstance(expr, Add):
            result = self._add(self._visit(expr.left), self._visit(expr.right))
        elif isinstance(expr, Mul):
            result = self._multiply(expr)
        elif isinstance(expr, FloorDiv):
            result = self._split(expr.left, expr.right, quotient=True)
        elif isinstance(expr, Mod):
            result = self._split(expr.left, expr.right, quotient=False)
        else:
            result = None
        return result

    @staticmethod
    def _add(left: IterSumExpr | None, right: IterSumExpr | None) -> IterSumExpr | None:
        """Add two rewritten maps."""
        result = None
        if left is not None and right is not None:
            result = IterSumExpr(args=[*left.args, *right.args], base=left.base + right.base)
        return result

    def _multiply(self, expr: Mul) -> IterSumExpr | None:
        """Scale an iter map by a positive integer constant."""
        constant, value = self._constant_and_value(expr)
        rewritten = None if value is None else self._visit(value)
        result = None
        if constant is not None and constant > 0 and rewritten is not None:
            args = [
                IterSplitExpr(
                    source=arg.source, lower_factor=arg.lower_factor, extent=arg.extent, scale=arg.scale * constant
                )
                for arg in rewritten.args
            ]
            result = IterSumExpr(args=args, base=rewritten.base * constant)
        return result

    @staticmethod
    def _constant_and_value(expr: Mul) -> tuple[int | None, Expr | None]:
        """Separate a multiplication into its constant and mapped operands."""
        constant: int | None = None
        value: Expr | None = None
        if isinstance(expr.left, Const):
            constant, value = expr.left.value, expr.right
        elif isinstance(expr.right, Const):
            constant, value = expr.right.value, expr.left
        return constant, value

    def _split(self, left: Expr, right: Expr, quotient: bool) -> IterSumExpr | None:
        """Apply floor division or modulo to one contiguous iter map."""
        rewritten = self._visit(left)
        divisor = right.value if isinstance(right, Const) else 0
        digit = self._single_digit(rewritten)
        result = None
        if divisor > 0 and digit is not None:
            result = self._quotient(digit, divisor) if quotient else self._remainder(digit, divisor)
        return result

    def _single_digit(self, value: IterSumExpr | None) -> IterSplitExpr | None:
        """Return one digit, fusing a contiguous sum when necessary."""
        result = None
        if value is not None and value.base == 0:
            if len(value.args) == 1:
                result = value.args[0]
            else:
                extent = _contiguous_extent(value)
                if extent is not None:
                    key = tuple(sorted((id(arg.source), arg.lower_factor, arg.extent, arg.scale) for arg in value.args))
                    mark = self.fused.get(key)
                    if mark is None:
                        mark = IterMark(source=value, extent=extent)
                        self.fused[key] = mark
                    result = IterSplitExpr(source=mark)
        return result

    @staticmethod
    def _quotient(digit: IterSplitExpr, divisor: int) -> IterSumExpr | None:
        """Divide one digit without introducing padding."""
        result = None
        if digit.scale % divisor == 0:
            result = _copy_digit(digit, scale=digit.scale // divisor)
        elif divisor % digit.scale == 0:
            factor = divisor // digit.scale
            if digit.extent % factor == 0:
                result = _copy_digit(
                    digit, lower_factor=digit.lower_factor * factor, extent=digit.extent // factor, scale=1
                )
        return None if result is None else IterSumExpr(args=[result])

    @staticmethod
    def _remainder(digit: IterSplitExpr, divisor: int) -> IterSumExpr | None:
        """Take a modulo that selects a complete lower digit."""
        result = None
        if digit.scale % divisor == 0:
            result = IterSumExpr()
        elif divisor % digit.scale == 0:
            extent = divisor // digit.scale
            if digit.extent % extent == 0:
                result = IterSumExpr(args=[_copy_digit(digit, extent=extent)])
        return result

    def mapping_is_valid(self, indices: list[IterSumExpr]) -> bool:
        """Check contiguous outputs and exact, non-overlapping input coverage."""
        valid = all(not index.args or _contiguous_extent(index) is not None for index in indices)
        marks: dict[int, IterMark] = {id(mark): mark for mark in self.marks.values()}
        splits: dict[int, list[IterSplitExpr]] = {}
        visited: set[int] = set()
        if valid:
            for index in indices:
                for digit in index.args:
                    _collect_splits(digit, marks, splits, visited)
            valid = all(_splits_cover(mark, splits.get(mark_id, [])) for mark_id, mark in marks.items())
        return valid


def _copy_digit(
    digit: IterSplitExpr, *, lower_factor: int | None = None, extent: int | None = None, scale: int | None = None
) -> IterSplitExpr:
    """Copy a digit while replacing selected integer fields."""
    return IterSplitExpr(
        source=digit.source,
        lower_factor=digit.lower_factor if lower_factor is None else lower_factor,
        extent=digit.extent if extent is None else extent,
        scale=digit.scale if scale is None else scale,
    )


def _contiguous_extent(expr: IterSumExpr) -> int | None:
    """Return the extent of a gap-free mixed-radix sum."""
    expected = 1
    for digit in sorted(expr.args, key=lambda item: item.scale):
        if digit.scale != expected:
            return None
        expected *= digit.extent
    return expected


def _collect_splits(
    digit: IterSplitExpr, marks: dict[int, IterMark], splits: dict[int, list[IterSplitExpr]], visited: set[int]
) -> None:
    """Collect each mark's outgoing digits, following fused marks once."""
    mark_id = id(digit.source)
    marks[mark_id] = digit.source
    splits.setdefault(mark_id, []).append(digit)
    if mark_id not in visited:
        visited.add(mark_id)
        if isinstance(digit.source.source, IterSumExpr):
            for nested in digit.source.source.args:
                _collect_splits(nested, marks, splits, visited)


def _splits_cover(mark: IterMark, splits: list[IterSplitExpr]) -> bool:
    """Return whether digits partition a mark exactly once."""
    expected = 1
    valid = True
    for digit in sorted(splits, key=lambda item: item.lower_factor):
        if digit.lower_factor != expected:
            valid = False
            break
        expected *= digit.extent
    return valid and expected == mark.extent


def detect_iter_map(indices: list[Expr], input_iters: dict[str, tuple[int, int]]) -> list[IterSumExpr] | None:
    """Detect a padding-free, constant-factor iter map."""
    rewriter = _IterMapRewriter(input_iters)
    rewritten = [rewriter.rewrite(index) for index in indices] if rewriter.valid else []
    result = None
    if len(rewritten) == len(indices) and all(index is not None for index in rewritten):
        complete = [index for index in rewritten if index is not None]
        if rewriter.mapping_is_valid(complete):
            result = complete
    return result


def normalize_iter_map_to_expr(sum_expr: IterSumExpr) -> Expr:
    """Lower an iter-map sum to the ordinary expression AST."""
    analyzer = Analyzer()
    result: Expr = Const(value=sum_expr.base)
    for digit in sum_expr.args:
        result = Add(left=result, right=_normalize_digit(digit, analyzer))
    return analyzer.simplify(result)


def _normalize_digit(digit: IterSplitExpr, analyzer: Analyzer) -> Expr:
    """Lower one digit to floor-divide and modulo operations."""
    source = (
        normalize_iter_map_to_expr(digit.source.source)
        if isinstance(digit.source.source, IterSumExpr)
        else digit.source.source
    )
    span = digit.lower_factor * digit.extent
    if digit.lower_factor == 1 and digit.extent == digit.source.extent:
        result = source
    elif span == digit.source.extent:
        result = FloorDiv(left=source, right=Const(value=digit.lower_factor))
    else:
        result = FloorDiv(left=Mod(left=source, right=Const(value=span)), right=Const(value=digit.lower_factor))
    if digit.scale != 1:
        result = Mul(left=result, right=Const(value=digit.scale))
    return analyzer.simplify(result)


def iter_map_simplify(indices: list[Expr], input_iters: dict[str, tuple[int, int]]) -> list[Expr] | None:
    """Detect an iter map and return its normalized ordinary expressions."""
    detected = detect_iter_map(indices, input_iters)
    return None if detected is None else [normalize_iter_map_to_expr(index) for index in detected]
