"""Faithful subset of TVM's iter-affine-map analysis (``src/arith/iter_affine_map.cc``).

Detects whether a list of index expressions over named input iterators forms a
valid affine "iter map" and, if so, returns a normalized form. This is the
analysis TVM's ``IterMapSimplifyBlockBinding`` uses (via ``DetectIterMap`` /
``IterMapSimplify``) and that our Split / Fuse transforms rely on:

- Split recombination: ``i0*f + i1`` over ``i0 in [0,a), i1 in [0,b)`` is a
  single affine iter of extent ``a*b``.
- Fuse inverse: ``(fused//f, fused%f)`` over ``fused in [0,a*b)`` recovers two
  iters of extents ``a`` and ``b``.

The port mirrors TVM's structs (:class:`IterMark` / :class:`IterSplitExpr` /
:class:`IterSumExpr`) and the core of ``IterMapRewriter`` (the ``var_map_``
seeding, the Var/Add/Mul/FloorDiv/FloorMod visitors, ``TryFuseIters`` with its
``TryCombineSplitFromSameSource`` / ``FindBaseIter`` / ``FindIterWithExactScale``
helpers, ``NormalizeToIterWithOffset``, the ``IterMarkSplitCollector`` +
``CheckMapping`` / ``TryNormalizeSplits`` independence check) and
``IterMapToExprNormalizer`` for ``NormalizeIterMapToExpr``.

Scope (demand-driven, oracle-arbitrated): only the surjective, padding-free,
predicate-free, constant-factor corpus is ported. The padding machinery
(``PadDividendToDivisor`` update passes, ``padding_predicate_``), the predicate
constraint solver (``MatchBoundConstraints`` / ``NormalizeToIterOnBoundExpr`` /
``CheckConstraints``), subspace division, the bijective-only branches, and
symbolic (non-constant) scales/extents are intentionally omitted -- no oracle
case in the corpus reaches them. All extents, lower factors, and scales are
plain ``int`` because every corpus value is a constant. Correctness is gated by
the TVM oracle in ``test/ir/arith/test_iter_map.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, Expr, FloorDiv, Mod, Mul, Var, to_affine


@dataclass(eq=False)
class IterMark:
    """An iteration domain ``[0, extent)`` over ``source``.

    Mirrors ``IterMarkNode`` (``iter_affine_map.h``): ``IterMark(source, extent)``
    marks ``source`` as an iterator ranging over ``[0, extent)``. ``source`` is
    either a plain :class:`~nkigym.ir.arith.expr.Var` (a leaf input iterator) or
    a nested :class:`IterSumExpr` (a fused iterator). Identity-based equality
    (``eq=False``) mirrors TVM's ``ObjectPtrHash`` / ``ObjectPtrEqual`` keying:
    two marks are the same iterator only if they are the same object.
    """

    source: Expr | IterSumExpr
    extent: int


@dataclass(eq=False)
class IterSplitExpr:
    """A split of an iter mark, ``floormod(floordiv(mark, lower_factor), extent) * scale``.

    Mirrors ``IterSplitExprNode`` (``iter_affine_map.h``). Selects a contiguous
    band of the marked iterator: divide ``source`` down by ``lower_factor``, take
    ``extent`` values modulo, then scale. Mutable (``eq=False``) so the rewriter
    can copy-on-write ``scale`` / ``extent`` exactly as TVM mutates via
    ``CopyOnWrite``.
    """

    source: IterMark
    lower_factor: int = 1
    extent: int = 0
    scale: int = 1

    def __post_init__(self) -> None:
        """Default ``extent`` to the source mark's extent, mirroring the 1-arg ctor.

        ``IterSplitExpr::IterSplitExpr(IterMark source)`` sets ``extent`` to
        ``source->extent``; an explicit non-zero ``extent`` overrides it.
        """
        if self.extent == 0:
            self.extent = self.source.extent


@dataclass(eq=False)
class IterSumExpr:
    """A scaled sum of splits plus a base, ``sum(args) + base``.

    Mirrors ``IterSumExprNode`` (``iter_affine_map.h``): ``IterSumExpr(args,
    base)`` is the flat affine combination of :class:`IterSplitExpr` terms with an
    integer ``base`` offset.
    """

    args: list[IterSplitExpr] = field(default_factory=list)
    base: int = 0


def _to_iter_sum(expr: IterSumExpr | IterSplitExpr | Expr) -> IterSumExpr:
    """Wrap ``expr`` as an :class:`IterSumExpr`, mirroring ``ToIterSumExpr`` (~line 829).

    An :class:`IterSumExpr` passes through; an :class:`IterSplitExpr` becomes a
    one-arg sum with zero base; any other (non-iter) :class:`Expr` -- which here
    is always a constant after const-folding -- becomes an empty-args sum whose
    base carries the constant value.
    """
    result: IterSumExpr
    if isinstance(expr, IterSumExpr):
        result = expr
    elif isinstance(expr, IterSplitExpr):
        result = IterSumExpr(args=[expr], base=0)
    elif isinstance(expr, Const):
        result = IterSumExpr(args=[], base=expr.value)
    else:
        raise ValueError(f"cannot wrap non-constant non-iter expr as IterSum: {expr}")
    return result


def _const_value(expr: Expr) -> int | None:
    """Return the integer value of ``expr`` if it is a constant, else ``None``.

    Uses :func:`~nkigym.ir.arith.expr.to_affine` so a constant that has not been
    folded to a literal :class:`Const` (e.g. ``Mul(Const, Const)``) is still
    recognised; a term with any variable returns ``None``.
    """
    coeffs = to_affine(expr)
    result: int | None = None
    if set(coeffs) <= {None}:
        result = coeffs.get(None, 0)
    return result


class _IterMapRewriter:
    """Rewrites an :class:`Expr` to iter-map form, mirroring ``IterMapRewriter``.

    Holds the per-detection state of ``IterMapRewriter`` (``iter_affine_map.cc``
    ~line 173): the ``var_map_`` seeding of input iterators, the post-order
    visitor over Var/Add/Mul/FloorDiv/FloorMod, and the ``sum_fuse_map_`` /
    ``flattened_map_`` bookkeeping that ``TryFuseIters`` uses. ``errors`` is set
    non-empty (mirroring ``errors_``) whenever an expression cannot be
    represented, which aborts detection.
    """

    def __init__(self, input_iters: dict[str, tuple[int, int]]) -> None:
        """Seed ``var_map_`` from the input iterators, mirroring the constructor (~line 177).

        Each iterator ``name`` with half-open range ``[lo, hi)`` is registered.
        Only the ``lo == 0`` case (an :class:`IterSplitExpr` over a fresh
        :class:`IterMark`) is supported -- the corpus never offsets an input
        iterator, and TVM's trivial-iterator (extent 1) and non-zero-min branches
        are out of scope. TVM's rewriter also holds an ``Analyzer`` for symbolic
        ``CanProveEqual`` / ``CanProveDivisible`` checks; the constant-factor
        corpus needs none, so every such check is a direct integer comparison
        (``==`` / ``%``) and no analyzer is held here.
        """
        self._var_map: dict[str, IterSplitExpr] = {}
        self.errors: list[str] = []
        self._sum_fuse_map: list[tuple[IterSumExpr, IterMark]] = []
        for name, (lo, hi) in input_iters.items():
            if lo != 0:
                raise ValueError(f"non-zero iterator min unsupported (got [{lo},{hi}) for {name})")
            mark = IterMark(source=Var(name=name), extent=hi - lo)
            self._var_map[name] = IterSplitExpr(source=mark)

    def rewrite(self, expr: Expr) -> IterSumExpr:
        """Rewrite ``expr`` to a normalized iter sum, mirroring ``Rewrite`` (~line 206).

        Runs the direct (non-normalizing) visitor, wraps the result as an
        :class:`IterSumExpr`, then fuses it through
        :meth:`_normalize_to_iter_with_offset`.
        """
        return self._normalize_to_iter_with_offset(_to_iter_sum(self._direct_mutate(expr)))

    def _direct_mutate(self, expr: Expr) -> IterSumExpr | IterSplitExpr | Expr:
        """Dispatch the iter-map visitor on ``expr``, mirroring ``DirectMutate`` (~line 328).

        Each Var/Add/Mul/FloorDiv/FloorMod node is handled by the matching
        ``VisitExpr_`` port; other nodes (only constants in the corpus) pass
        through unchanged.
        """
        result: IterSumExpr | IterSplitExpr | Expr
        if isinstance(expr, Var):
            result = self._visit_var(expr)
        elif isinstance(expr, Add):
            result = self._visit_add(expr)
        elif isinstance(expr, Mul):
            result = self._visit_mul(expr)
        elif isinstance(expr, FloorDiv):
            result = self._visit_floordiv(expr)
        elif isinstance(expr, Mod):
            result = self._visit_floormod(expr)
        else:
            result = expr
        return result

    def _visit_var(self, op: Var) -> IterSplitExpr | Var:
        """Resolve a Var to its seeded split, mirroring ``VisitExpr_(VarNode*)`` (~line 1554)."""
        result: IterSplitExpr | Var = self._var_map.get(op.name, op)
        return result

    def _visit_add(self, op: Add) -> IterSumExpr | Expr:
        """Rewrite an Add, mirroring ``VisitExpr_(AddNode*)`` (~line 1561).

        Recurses both operands. If neither contains an iter map the node is left
        as a plain affine :class:`Expr`. Otherwise the left operand is wrapped as
        an :class:`IterSumExpr` and the right is folded in via :meth:`_add_to_lhs`
        (a constant goes into ``base``).
        """
        a = self._direct_mutate(op.left)
        b = self._direct_mutate(op.right)
        result: IterSumExpr | Expr
        if not _is_iter(a) and not _is_iter(b):
            result = Add(left=_as_expr(a), right=_as_expr(b))
        else:
            ret = _to_iter_sum(a)
            if not _is_iter(b):
                b_val = _const_value(_as_expr(b))
                if b_val is None:
                    raise ValueError(f"non-constant non-iter Add operand: {b}")
                ret.base += b_val
            else:
                self._add_to_lhs(ret, _to_iter_sum(b), 1)
            result = ret
        return result

    def _visit_mul(self, op: Mul) -> IterSumExpr | IterSplitExpr | Expr:
        """Rewrite a Mul, mirroring ``VisitExpr_(MulNode*)`` (~line 1629).

        Recurses both operands. Two non-iter operands stay a plain :class:`Expr`;
        two iter operands are a product of iterators and cannot be represented
        (logged as an error). Otherwise the (single) iter operand is scaled by the
        constant operand: an :class:`IterSumExpr` scales every arg and its base,
        an :class:`IterSplitExpr` scales its ``scale``.
        """
        a = self._direct_mutate(op.left)
        b = self._direct_mutate(op.right)
        a_iter = isinstance(a, (IterSumExpr, IterSplitExpr))
        b_iter = isinstance(b, (IterSumExpr, IterSplitExpr))
        result: IterSumExpr | IterSplitExpr | Expr
        if not a_iter and not b_iter:
            result = Mul(left=_as_expr(a), right=_as_expr(b))
        elif a_iter and b_iter:
            self.errors.append("Product of two iterators cannot be represented as an IterMap")
            result = Mul(left=op.left, right=op.right)
        elif isinstance(a, (IterSumExpr, IterSplitExpr)):
            result = self._scale_iter_by(a, b)
        elif isinstance(b, (IterSumExpr, IterSplitExpr)):
            result = self._scale_iter_by(b, a)
        else:
            raise ValueError("unreachable: one operand is an iter-map node")
        return result

    def _scale_iter_by(
        self, iter_side: IterSumExpr | IterSplitExpr, const_side: IterSumExpr | IterSplitExpr | Expr
    ) -> IterSumExpr | IterSplitExpr:
        """Scale ``iter_side`` by the constant value of ``const_side``.

        Helper for the single-iter-operand branch of :meth:`_visit_mul`: extracts
        the integer constant from ``const_side`` (raising if it is not constant)
        and applies :meth:`_scale_iter`.
        """
        const_val = _const_value(_as_expr(const_side))
        if const_val is None:
            raise ValueError(f"non-constant Mul scale: {const_side}")
        return self._scale_iter(iter_side, const_val)

    def _scale_iter(self, iter_side: IterSumExpr | IterSplitExpr, scale: int) -> IterSumExpr | IterSplitExpr:
        """Scale an iter operand by ``scale``, mirroring the Mul tail (~line 1660).

        An :class:`IterSumExpr` has every arg's ``scale`` and its ``base``
        multiplied (``MulToLhs``); an :class:`IterSplitExpr` has its own ``scale``
        multiplied.
        """
        result: IterSumExpr | IterSplitExpr
        if isinstance(iter_side, IterSumExpr):
            for arg in iter_side.args:
                arg.scale *= scale
            iter_side.base *= scale
            result = iter_side
        else:
            iter_side.scale *= scale
            result = iter_side
        return result

    def _visit_floordiv(self, op: FloorDiv) -> IterSumExpr | IterSplitExpr | Expr:
        """Rewrite a FloorDiv, mirroring ``VisitExpr_(FloorDivNode*)`` (~line 1942).

        Recurses both operands. A non-iter / non-iter node stays a plain
        :class:`Expr`; dividing by an iterator is rejected. Otherwise the dividend
        is preprocessed to a single fused split and divided by the constant via
        :meth:`_split_floordiv_const`.
        """
        a = self._direct_mutate(op.left)
        b = self._direct_mutate(op.right)
        result: IterSumExpr | IterSplitExpr | Expr
        if not _is_iter(a) and not _is_iter(b):
            result = FloorDiv(left=_as_expr(a), right=_as_expr(b))
        elif _is_iter(b):
            self.errors.append("Cannot represent as an IterMap: the divisor may not be an iterator")
            result = FloorDiv(left=op.left, right=op.right)
        else:
            result = self._divmod_dispatch(a, b, op, is_floordiv=True)
        return result

    def _visit_floormod(self, op: Mod) -> IterSumExpr | IterSplitExpr | Expr:
        """Rewrite a FloorMod, mirroring ``VisitExpr_(FloorModNode*)`` (~line 2026).

        Recurses both operands. A non-iter / non-iter node stays a plain
        :class:`Expr`; modding by an iterator is rejected. Otherwise the dividend
        is preprocessed to a single fused split and reduced by the constant via
        :meth:`_split_floormod_const`.
        """
        a = self._direct_mutate(op.left)
        b = self._direct_mutate(op.right)
        result: IterSumExpr | IterSplitExpr | Expr
        if not _is_iter(a) and not _is_iter(b):
            result = Mod(left=_as_expr(a), right=_as_expr(b))
        elif _is_iter(b):
            self.errors.append("Cannot represent as an IterMap: the divisor may not be an iterator")
            result = Mod(left=op.left, right=op.right)
        else:
            result = self._divmod_dispatch(a, b, op, is_floordiv=False)
        return result

    def _divmod_dispatch(
        self,
        a: IterSumExpr | IterSplitExpr | Expr,
        b: IterSumExpr | IterSplitExpr | Expr,
        op: FloorDiv | Mod,
        is_floordiv: bool,
    ) -> IterSumExpr | IterSplitExpr | Expr:
        """Preprocess the dividend then apply the constant div/mod split.

        Shared tail of the FloorDiv / FloorMod visitors: preprocess ``a`` to a
        single fused :class:`IterSplitExpr` (:meth:`_preprocess_dividend`) and
        divide / mod it by the constant ``b`` via :meth:`_split_floordiv_const` /
        :meth:`_split_floormod_const`. On any failure the original node Expr is
        returned (an error has been logged).
        """
        divisor = _const_value(_as_expr(b))
        if divisor is None:
            raise ValueError(f"non-constant div/mod divisor: {b}")
        preprocessed = self._preprocess_dividend(a)
        result: IterSumExpr | IterSplitExpr | Expr
        if preprocessed is None:
            result = op
        else:
            split, base = preprocessed
            split_result: IterSumExpr | IterSplitExpr | None
            if is_floordiv:
                split_result = self._split_floordiv_const(split, base, divisor)
            else:
                split_result = self._split_floormod_const(split, base, divisor)
            result = op if split_result is None else split_result
        return result

    def _preprocess_dividend(self, dividend: IterSumExpr | IterSplitExpr | Expr) -> tuple[IterSplitExpr, int] | None:
        """Reduce a dividend to one fused split + base, mirroring ``PreprocessDividend`` (~line 1673).

        An :class:`IterSplitExpr` is already a single split (base 0). An
        :class:`IterSumExpr` with one arg passes through; with several args it is
        fused via :meth:`_try_fuse_iters` (which must collapse to a single arg).
        Returns ``None`` (after logging) if it cannot be written as one fused
        split.
        """
        result: tuple[IterSplitExpr, int] | None
        if isinstance(dividend, IterSplitExpr):
            result = (dividend, 0)
        elif isinstance(dividend, IterSumExpr):
            if len(dividend.args) == 1:
                result = (dividend.args[0], dividend.base)
            else:
                fused = self._try_fuse_iters(dividend, allow_early_skip=True)
                if fused is None or len(fused.args) != 1:
                    self.errors.append("Dividend can't be written as a single fused IterSum")
                    result = None
                else:
                    result = (fused.args[0], fused.base)
        else:
            raise ValueError(f"unsupported dividend: {dividend}")
        return result

    def _split_floordiv_const(self, lhs: IterSplitExpr, base: int, rhs: int) -> IterSumExpr | IterSplitExpr | None:
        """``(lhs + base) // rhs`` for constant ``rhs``, mirroring ``SplitFloorDivConst`` (~line 1852).

        Ports the no-padding corpus path. ``rhs == 1`` returns the split (plus
        base). For ``scale != 1`` the divisible sub-cases rescale; the
        non-divisible scale is rejected. With ``scale == 1`` and a left pad of
        ``base % rhs == 0`` (the corpus always has ``base == 0``), an evenly
        divisible extent yields ``IterSplit(source, lower_factor*rhs,
        extent//rhs, scale)``; the ``lower_factor == 1`` full-extent case yields
        ``IterSplit(source, rhs, ceildiv(extent, rhs), scale)``.
        """
        result = self._floordiv_handle_scale(lhs, base, rhs)
        if isinstance(result, _Unhandled):
            lhs, base, rhs = result.lhs, result.base, result.rhs
            result = self._floordiv_after_scale(lhs, base, rhs)
        return result

    def _floordiv_handle_scale(
        self, lhs: IterSplitExpr, base: int, rhs: int
    ) -> IterSumExpr | IterSplitExpr | _Unhandled | None:
        """Resolve the ``scale != 1`` / ``rhs == 1`` cases of ``SplitFloorDivConst``.

        Returns the finished expression, ``None`` on the unrepresentable
        non-divisible-scale branch (after logging), or an :class:`_Unhandled`
        carrying the (possibly rescaled) ``lhs`` / ``base`` / ``rhs`` for the
        scale-1 tail in :meth:`_floordiv_after_scale`.
        """
        result: IterSumExpr | IterSplitExpr | _Unhandled | None
        if rhs == 1:
            result = lhs if base == 0 else IterSumExpr(args=[lhs], base=base)
        elif lhs.scale != 1:
            result = self._floordiv_scale_ne_one(lhs, base, rhs)
        else:
            result = _Unhandled(lhs=lhs, base=base, rhs=rhs)
        return result

    def _floordiv_scale_ne_one(
        self, lhs: IterSplitExpr, base: int, rhs: int
    ) -> IterSumExpr | IterSplitExpr | _Unhandled | None:
        """Handle ``scale != 1`` in ``SplitFloorDivConst`` (~line 1866).

        ``floordiv(x*c1*c2, c2) = x*c1`` (scale divisible by rhs, base 0) and its
        base-divisible variant rescale the split. The ``rhs`` divisible by
        ``scale`` sub-cases reduce ``rhs`` / ``base`` and continue with scale 1
        (returned as :class:`_Unhandled`). A non-divisible scale is rejected.
        """
        result: IterSumExpr | IterSplitExpr | _Unhandled | None
        if rhs != 0 and lhs.scale % rhs == 0 and base == 0:
            lhs.scale //= rhs
            result = lhs
        elif rhs != 0 and lhs.scale % rhs == 0 and base % rhs == 0:
            lhs.scale //= rhs
            result = IterSumExpr(args=[lhs], base=base // rhs)
        elif lhs.scale != 0 and rhs % lhs.scale == 0 and base == 0:
            result = _Unhandled(lhs=_with_scale(lhs, 1), base=0, rhs=rhs // lhs.scale)
        elif lhs.scale != 0 and rhs % lhs.scale == 0 and base % lhs.scale == 0:
            result = _Unhandled(lhs=_with_scale(lhs, 1), base=base // lhs.scale, rhs=rhs // lhs.scale)
        else:
            self.errors.append("Cannot represent as IterMap: numerator scale and divisor incompatible")
            result = None
        return result

    def _floordiv_after_scale(self, lhs: IterSplitExpr, base: int, rhs: int) -> IterSumExpr | IterSplitExpr | None:
        """Scale-1 tail of ``SplitFloorDivConst`` (~lines 1894-1939), no padding.

        With ``left_pad = base % rhs`` (zero in the corpus), an evenly divisible
        extent gives ``IterSplit(source, lower_factor*rhs, extent//rhs, scale)``;
        the ``lower_factor == 1`` full-extent case gives ``IterSplit(source, rhs,
        ceildiv(extent, rhs), scale)``. The ``new_base = (base - left_pad) // rhs``
        wraps the split in an :class:`IterSumExpr` when non-zero.
        """
        left_pad = base % rhs
        new_split = self._floordiv_new_split(lhs, rhs)
        result: IterSumExpr | IterSplitExpr | None
        if new_split is None:
            result = None
        else:
            new_base = (base - left_pad) // rhs
            result = new_split if new_base == 0 else IterSumExpr(args=[new_split], base=new_base)
        return result

    def _floordiv_new_split(self, lhs: IterSplitExpr, rhs: int) -> IterSplitExpr | None:
        """Build the divided split for the scale-1 path (~lines 1911-1931), no padding.

        Evenly divisible extent: ``lower_factor*rhs``, ``extent//rhs``. Else the
        ``lower_factor == 1`` and full-extent case: ``lower_factor=rhs``,
        ``ceildiv(extent, rhs)``. The general (``IterMark`` re-wrapping) branch is
        out of corpus scope and rejected.
        """
        result: IterSplitExpr | None
        if lhs.extent % rhs == 0:
            result = IterSplitExpr(
                source=lhs.source, lower_factor=lhs.lower_factor * rhs, extent=lhs.extent // rhs, scale=lhs.scale
            )
        elif lhs.lower_factor == 1 and lhs.extent == lhs.source.extent:
            result = IterSplitExpr(
                source=lhs.source, lower_factor=rhs, extent=_ceildiv(lhs.extent, rhs), scale=lhs.scale
            )
        else:
            self.errors.append("FloorDiv requires padding (out of corpus scope)")
            result = None
        return result

    def _split_floormod_const(self, lhs: IterSplitExpr, base: int, rhs: int) -> IterSplitExpr | None:
        """``(lhs + base) % rhs`` for constant ``rhs``, mirroring ``SplitFloorModConst`` (~line 1981).

        ``rhs == 1`` is always zero (rejected here as the corpus never mods by 1).
        For ``scale != 1`` the divisible sub-cases reduce ``rhs`` / ``base``; a
        non-divisible scale is rejected. With ``scale == 1`` (and the corpus'
        ``base == 0``, evenly divisible extent) the result is ``IterSplit(source,
        lower_factor, rhs, scale)`` -- selecting ``rhs`` values out of the band.
        """
        result = self._floormod_handle_scale(lhs, base, rhs)
        if isinstance(result, _Unhandled):
            result = self._floormod_after_scale(result.lhs, result.base, result.rhs)
        return result

    def _floormod_handle_scale(self, lhs: IterSplitExpr, base: int, rhs: int) -> IterSplitExpr | _Unhandled | None:
        """Resolve the ``rhs == 1`` / ``scale != 1`` cases of ``SplitFloorModConst``.

        Returns ``None`` on the unrepresentable / out-of-scope branches (after
        logging) or an :class:`_Unhandled` carrying the rescaled operands for the
        scale-1 tail in :meth:`_floormod_after_scale`.
        """
        result: IterSplitExpr | _Unhandled | None
        if rhs == 1:
            self.errors.append("FloorMod by 1 (out of corpus scope)")
            result = None
        elif lhs.scale != 1:
            result = self._floormod_scale_ne_one(lhs, base, rhs)
        else:
            result = _Unhandled(lhs=lhs, base=base, rhs=rhs)
        return result

    def _floormod_scale_ne_one(self, lhs: IterSplitExpr, base: int, rhs: int) -> IterSplitExpr | _Unhandled | None:
        """Handle ``scale != 1`` in ``SplitFloorModConst`` (~line 1989).

        ``floormod(x*c1*c2, c1) = 0`` (rejected, the zero result is out of corpus
        scope). The ``rhs`` divisible by ``scale`` sub-cases reduce ``rhs`` /
        ``base`` and continue with scale 1. A non-divisible scale is rejected.
        """
        result: IterSplitExpr | _Unhandled | None
        if rhs != 0 and lhs.scale % rhs == 0 and base % rhs == 0:
            self.errors.append("FloorMod reduces to zero (out of corpus scope)")
            result = None
        elif lhs.scale != 0 and rhs % lhs.scale == 0 and base == 0:
            result = _Unhandled(lhs=_with_scale(lhs, 1), base=0, rhs=rhs // lhs.scale)
        elif lhs.scale != 0 and rhs % lhs.scale == 0 and base % lhs.scale == 0:
            result = _Unhandled(lhs=_with_scale(lhs, 1), base=base // lhs.scale, rhs=rhs // lhs.scale)
        else:
            self.errors.append("Cannot represent as IterMap: FloorMod scale and divisor incompatible")
            result = None
        return result

    def _floormod_after_scale(self, lhs: IterSplitExpr, base: int, rhs: int) -> IterSplitExpr | None:
        """Scale-1 tail of ``SplitFloorModConst`` (~lines 2010-2023), no padding.

        With the corpus' ``left_pad = base % rhs == 0`` and evenly divisible
        ``base + extent``, the result is ``IterSplit(source, lower_factor, rhs,
        scale)``. A non-divisible right edge would require padding and is rejected.
        """
        left_pad = base % rhs
        right_edge = left_pad + lhs.extent
        result: IterSplitExpr | None
        if right_edge % rhs == 0:
            result = IterSplitExpr(source=lhs.source, lower_factor=lhs.lower_factor, extent=rhs, scale=lhs.scale)
        else:
            self.errors.append("FloorMod requires padding (out of corpus scope)")
            result = None
        return result

    def _add_to_lhs(self, lhs: IterSumExpr, rhs: IterSumExpr, sign: int) -> None:
        """Add (``sign>0``) or subtract (``sign<0``) ``rhs`` into ``lhs``, mirroring ``AddToLhs`` (~line 1253).

        Each split arg of ``rhs`` is merged: a split sharing source / lower_factor
        / extent with an existing ``lhs`` arg combines scales; otherwise it is
        appended (negated for ``sign < 0``). ``base`` accumulates likewise.
        """
        for arg in rhs.args:
            self._add_split_to_lhs(lhs, arg, sign)
        lhs.base += rhs.base if sign > 0 else -rhs.base

    def _add_split_to_lhs(self, lhs: IterSumExpr, rhs: IterSplitExpr, sign: int) -> None:
        """Merge one split into ``lhs.args``, mirroring scalar ``AddToLhs`` (~line 1230).

        If an existing arg matches ``rhs`` on source identity, lower_factor and
        extent, their scales combine (added or subtracted) in place; otherwise
        ``rhs`` is appended (with negated scale for ``sign < 0``).
        """
        for i, lvalue in enumerate(lhs.args):
            if lvalue.source is rhs.source and lvalue.lower_factor == rhs.lower_factor and lvalue.extent == rhs.extent:
                merged = _with_scale(rhs, lvalue.scale + rhs.scale if sign > 0 else lvalue.scale - rhs.scale)
                lhs.args[i] = merged
                return
        lhs.args.append(rhs if sign > 0 else _with_scale(rhs, -rhs.scale))

    def _normalize_to_iter_with_offset(self, expr: IterSumExpr) -> IterSumExpr:
        """Normalize a sum to one iter + offset, mirroring ``NormalizeToIterWithOffset`` (~line 746).

        An empty / no-arg sum passes through. Otherwise :meth:`_try_fuse_iters`
        (with early skip) must succeed; failure logs an error and returns the
        unfused sum.
        """
        result: IterSumExpr
        if len(expr.args) < 1:
            result = expr
        else:
            fused = self._try_fuse_iters(expr, allow_early_skip=True)
            if fused is not None:
                result = fused
            else:
                self.errors.append("Could not normalize iterators")
                result = expr
        return result

    def _try_fuse_iters(self, expr: IterSumExpr, allow_early_skip: bool) -> IterSumExpr | None:
        """Fuse a sum into a single :class:`IterSplitExpr` over a fresh mark, mirroring ``TryFuseIters`` (~line 1089).

        First combines splits from the same source (:meth:`_try_combine_split_from_same_source`).
        Then walks the args by ascending scale starting at the base iter
        (:meth:`_find_base_iter`), matching each next arg's scale to the running
        ``expected_scale`` (:meth:`_find_iter_with_exact_scale`) and accumulating
        the flattened / grouped split lists. A successful walk wraps the grouped
        form in a new :class:`IterMark` of the product extent (registered in
        ``sum_fuse_map_`` for reuse). Returns ``None`` when the scales do not chain
        (the iterators are not contiguous and cannot be fused).
        """
        combined = self._try_combine_split_from_same_source(expr)
        if combined is not None:
            expr = combined
        result: IterSumExpr | None
        if combined is not None and len(expr.args) <= 1 and allow_early_skip:
            result = expr
        else:
            walk = self._fuse_walk(expr)
            if walk is None:
                result = None
            else:
                flattened_iters, grouped_iters, base_scale, expected_scale = walk
                result = self._build_fused_sum(expr, flattened_iters, grouped_iters, base_scale, expected_scale)
        return result

    def _fuse_walk(self, expr: IterSumExpr) -> tuple[list[IterSplitExpr], list[IterSplitExpr], int, int] | None:
        """Walk args by chaining scales, the core loop of ``TryFuseIters`` (~lines 1097-1197).

        Returns the flattened and grouped split lists (innermost-first), the base
        scale, and the final ``expected_scale`` (the running product of matched
        extents). Returns ``None`` if no base iter exists or a step finds no arg
        whose scale equals the running ``expected_scale`` (constraints / smaller-
        closest matching are out of corpus scope). Each matched arg's scale is
        divided by ``base_scale`` (always 1 in the corpus).
        """
        visited = [False] * len(expr.args)
        base_index = self._find_base_iter(expr, visited)
        result: tuple[list[IterSplitExpr], list[IterSplitExpr], int, int] | None
        if base_index == -1:
            result = None
        else:
            base_scale = expr.args[base_index].scale
            flattened: list[IterSplitExpr] = []
            grouped: list[IterSplitExpr] = []
            expected_scale = base_scale
            result = self._fuse_chain(expr, visited, base_index, base_scale, flattened, grouped, expected_scale)
        return result

    def _fuse_chain(
        self,
        expr: IterSumExpr,
        visited: list[bool],
        base_index: int,
        base_scale: int,
        flattened: list[IterSplitExpr],
        grouped: list[IterSplitExpr],
        expected_scale: int,
    ) -> tuple[list[IterSplitExpr], list[IterSplitExpr], int, int] | None:
        """Run the scale-matching loop, accumulating ``flattened`` / ``grouped`` (~lines 1111-1197).

        At each step finds the arg whose scale equals ``expected_scale`` (the base
        iter on the first step), records it (rescaled by ``base_scale``), and
        advances ``expected_scale`` to ``extent * matched_scale``. Returns the
        filled lists with the final ``expected_scale``, or ``None`` if a step
        finds no matching arg.
        """
        result: tuple[list[IterSplitExpr], list[IterSplitExpr], int, int] | None = (
            flattened,
            grouped,
            base_scale,
            expected_scale,
        )
        i = 0
        while i < len(expr.args):
            matched_pos = base_index if i == 0 else self._find_iter_with_exact_scale(expr, visited, expected_scale)
            if matched_pos == -1:
                result = None
                break
            matched_scale = _matched_scale_of(expr, matched_pos, base_scale, i)
            visited[matched_pos] = True
            arg = _with_scale(expr.args[matched_pos], expr.args[matched_pos].scale // base_scale)
            flattened.append(arg)
            grouped.append(arg)
            expected_scale = expr.args[matched_pos].extent * matched_scale
            i += 1
        if result is not None:
            result = (flattened, grouped, base_scale, expected_scale)
        return result

    def _build_fused_sum(
        self,
        expr: IterSumExpr,
        flattened_iters: list[IterSplitExpr],
        grouped_iters: list[IterSplitExpr],
        base_scale: int,
        expected_scale: int,
    ) -> IterSumExpr:
        """Wrap the walked splits into one fused split, mirroring ``TryFuseIters`` tail (~lines 1198-1222).

        Builds the flattened and structured (grouped) forms with args reversed to
        outermost-first and zero base. An identical flattened form already in
        ``sum_fuse_map_`` reuses its mark; otherwise a new :class:`IterMark` of
        extent ``expected_scale // base_scale`` is formed and registered. Returns
        ``IterSum([IterSplit(mark, base_scale)], base=expr.base)``.
        """
        flattened_form = IterSumExpr(args=list(reversed(flattened_iters)), base=0)
        structured_form = IterSumExpr(args=list(reversed(grouped_iters)), base=0)
        existing = self._lookup_flattened(flattened_form)
        if existing is not None:
            mark = existing[0]
        else:
            mark = IterMark(source=structured_form, extent=expected_scale // base_scale)
            self._register_flattened(flattened_form, mark)
        result = IterSumExpr(args=[IterSplitExpr(source=mark, scale=base_scale)], base=expr.base)
        return result

    def _try_combine_split_from_same_source(self, expr: IterSumExpr) -> IterSumExpr | None:
        """Combine consecutive splits of the same source, mirroring ``TryCombineSplitFromSameSource`` (~line 987).

        With no two args sharing an iter mark (the corpus' Split / Fuse inputs),
        there is nothing to combine and ``None`` is returned (the no-overlap early
        exit). The full same-source merging loop -- which collapses e.g.
        ``(f//4)*4 + f%4`` back to ``f`` -- is out of corpus scope; when an
        overlap is seen, an error is logged (faithful to TVM's ``ErrorLogger``:
        log and leave the expression unrepresentable) so :func:`detect_iter_map`
        conservatively fails with ``None`` rather than fabricating a result.
        """
        result: IterSumExpr | None = None
        if len(expr.args) > 1:
            sources = [id(arg.source) for arg in expr.args]
            if len(set(sources)) != len(sources):
                self.errors.append("Combining splits from the same source is out of corpus scope")
        return result

    def _find_base_iter(self, expr: IterSumExpr, skip_flag: list[bool]) -> int:
        """Find the arg with the smallest constant scale, mirroring ``FindBaseIter`` (~line 850).

        Reverse scan (smallest scale usually rightmost): track the minimum
        constant ``scale`` among non-skipped args, breaking ties toward a unit
        ``extent`` arg. Returns its index, or -1 if every arg is skipped. The
        symbolic-scale fallback is out of corpus scope.
        """
        base_index = -1
        min_const_scale = 0
        for i in range(len(expr.args) - 1, -1, -1):
            if skip_flag[i]:
                continue
            scale = expr.args[i].scale
            if base_index == -1 or scale < min_const_scale:
                min_const_scale = scale
                base_index = i
            elif scale == min_const_scale and expr.args[i].extent == 1 and expr.args[base_index].extent != 1:
                base_index = i
        return base_index

    def _find_iter_with_exact_scale(self, expr: IterSumExpr, skip_flag: list[bool], expected_scale: int) -> int:
        """Find a non-skipped arg whose scale equals ``expected_scale``, mirroring ``FindIterWithExactScale`` (~line 920).

        Reverse scan; a unit-extent match returns immediately, otherwise the first
        match is held (unit-extent splits get priority since they do not change
        the scale). Returns the matched index or -1. The ``first_possible_unit_extent``
        early-exit refinement is collapsed since the corpus has no unit extents.
        """
        matched_pos = -1
        for j in range(len(expr.args) - 1, -1, -1):
            if skip_flag[j]:
                continue
            if expr.args[j].scale == expected_scale:
                if expr.args[j].extent == 1:
                    matched_pos = j
                    break
                if matched_pos == -1:
                    matched_pos = j
        return matched_pos

    def _lookup_flattened(self, flattened_form: IterSumExpr) -> tuple[IterMark, int] | None:
        """Look up a flattened form in ``sum_fuse_map_`` by structural equality.

        Mirrors the ``sum_fuse_map_.find(flattened_form)`` lookup keyed by
        ``IterSumEqual`` (``iter_affine_map.cc`` ~line 421): two flattened forms
        match when their args agree on source identity, lower_factor, extent and
        scale, and their bases agree. Returns the stored ``(mark, offset)`` or
        ``None``. Offsets are always 0 in the corpus (no predicate-induced base).
        """
        result: tuple[IterMark, int] | None = None
        for stored, mark in self._sum_fuse_map:
            if _iter_sum_equal(stored, flattened_form):
                result = (mark, stored.base)
                break
        return result

    def _register_flattened(self, flattened_form: IterSumExpr, mark: IterMark) -> None:
        """Register a new flattened-form / mark pair, mirroring the ``sum_fuse_map_`` insert.

        Records the mapping consulted by :meth:`_lookup_flattened` so a repeated
        flattened form reuses the same :class:`IterMark` (``sum_fuse_map_`` insert,
        ``iter_affine_map.cc`` ~line 1219).
        """
        self._sum_fuse_map.append((flattened_form, mark))

    def check_mapping(self, bindings: list[IterSumExpr]) -> bool:
        """Check the detected bindings are independent, mirroring ``CheckMapping`` (~line 254).

        Surjective level only: collect every mark's outgoing splits
        (:class:`_IterMarkSplitCollector`) and require each mark's splits to
        normalize (:meth:`_try_normalize_splits`) -- i.e. chain into a
        non-overlapping cover of (a factor of) the mark extent. The bijective
        "all input marks used" condition is out of scope.
        """
        collector = _IterMarkSplitCollector()
        collector.collect(bindings)
        result = True
        for mark in collector.visited:
            if not self._try_normalize_splits(mark, collector.mark2splits[id(mark)]):
                result = False
                break
        return result

    def _try_normalize_splits(self, mark: IterMark, splits: list[IterSplitExpr]) -> bool:
        """Verify ``splits`` chain into a valid cover of ``mark``, mirroring ``TryNormalizeSplits`` (~line 561).

        Surjective, no-padding path: greedily order the splits so each next
        ``lower_factor`` equals the running ``expected_lower_factor`` (starting at
        1), advancing it by ``lower_factor * extent``; an unused gap is bridged via
        :meth:`_search_skip_lower_factor`. Valid iff the covered extent equals
        ``mark.extent`` or is a divisor of it. Returns ``False`` on overlap / gap /
        non-divisor.
        """
        used = [False] * len(splits)
        expected_lower_factor = 1
        ok = True
        for _ in range(len(splits)):
            j = self._next_split_index(splits, used, expected_lower_factor)
            if j == -1:
                ok = False
                break
            used[j] = True
            expected_lower_factor = splits[j].lower_factor * splits[j].extent
        if ok:
            match_full = expected_lower_factor == mark.extent
            ok = match_full or mark.extent % expected_lower_factor == 0
        return ok

    def _next_split_index(self, splits: list[IterSplitExpr], used: list[bool], expected_lower_factor: int) -> int:
        """Pick the next split whose lower_factor matches, mirroring the inner loop of ``TryNormalizeSplits``.

        Returns the index of an unused split whose ``lower_factor`` equals
        ``expected_lower_factor``; if none matches exactly, defers to
        :meth:`_search_skip_lower_factor` (the surjective gap-skip). Returns -1 if
        no usable split remains.
        """
        j = -1
        for k, split in enumerate(splits):
            if not used[k] and split.lower_factor == expected_lower_factor:
                j = k
                break
        if j == -1:
            j = self._search_skip_lower_factor(splits, used, expected_lower_factor)
        return j

    def _search_skip_lower_factor(
        self, splits: list[IterSplitExpr], used: list[bool], expected_lower_factor: int
    ) -> int:
        """Find the unused split with smallest divisible lower_factor, mirroring ``SearchSkipLowerFactor`` (~line 532).

        For the surjective skip: every remaining unused split's ``lower_factor``
        must be divisible by ``expected_lower_factor`` (else -1), and the one with
        the smallest such factor is chosen. Lets ``[y//6, y%2]`` skip ``(y//2)%6``.
        """
        res = -1
        for i, split in enumerate(splits):
            if used[i]:
                continue
            if split.lower_factor % expected_lower_factor != 0:
                res = -1
                break
            if res == -1 or splits[res].lower_factor % split.lower_factor == 0:
                res = i
        return res


@dataclass
class _Unhandled:
    """Carrier for the scale-1 continuation of div/mod split handling.

    Internal sentinel returned by the ``scale != 1`` resolvers when the divisor
    or base has been reduced and the scale-1 tail still needs to run; mirrors the
    fall-through after the ``if (!is_one(lhs->scale))`` block in
    ``SplitFloorDivConst`` / ``SplitFloorModConst``.
    """

    lhs: IterSplitExpr
    base: int
    rhs: int


class _IterMarkSplitCollector:
    """Collects the outgoing splits of every iter mark, mirroring ``IterMarkSplitCollector`` (~line 132).

    Recursively walks the detected sum expressions, recording for each
    :class:`IterMark` (keyed by object identity, like TVM's ``ObjectPtrHash``) the
    list of splits referencing it -- the input to the independence check.
    """

    def __init__(self) -> None:
        """Initialise empty visited-mark and mark-to-splits collections."""
        self.visited: list[IterMark] = []
        self.mark2splits: dict[int, list[IterSplitExpr]] = {}

    def collect(self, indices: list[IterSumExpr]) -> None:
        """Collect splits from each index's args, mirroring ``Collect`` (~line 143)."""
        for sum_expr in indices:
            for split in sum_expr.args:
                self._collect_internal(split.source)
                self.mark2splits.setdefault(id(split.source), []).append(split)

    def _collect_internal(self, mark: IterMark) -> None:
        """Recurse into a mark's nested source sum, mirroring ``CollectInternal`` (~line 152).

        Marks ``mark`` visited (once) and, if its ``source`` is a nested
        :class:`IterSumExpr`, recurses into each of that sum's splits, recording
        them against their own source marks.
        """
        if any(m is mark for m in self.visited):
            return
        self.visited.append(mark)
        if isinstance(mark.source, IterSumExpr):
            for split in mark.source.args:
                self._collect_internal(split.source)
                self.mark2splits.setdefault(id(split.source), []).append(split)


class _IterMapToExprNormalizer:
    """Lowers an iter-map expression back to a plain :class:`Expr`, mirroring ``IterMapToExprNormalizer`` (~line 2068)."""

    def __init__(self, analyzer: Analyzer) -> None:
        """Hold the analyzer used for the extent/full-extent equality checks."""
        self._analyzer = analyzer

    def convert(self, expr: IterSumExpr | IterSplitExpr | Expr) -> Expr:
        """Dispatch lowering by node type, mirroring ``VisitExpr`` (~line 2076)."""
        result: Expr
        if isinstance(expr, IterSplitExpr):
            result = self._convert_split(expr)
        elif isinstance(expr, IterSumExpr):
            result = self._convert_sum(expr)
        else:
            result = expr
        return result

    def _convert_sum(self, expr: IterSumExpr) -> Expr:
        """Lower a sum to ``sum(convert(arg)) + base``, mirroring ``ConvertIterSumExpr`` (~line 2086)."""
        result: Expr = Const(value=0)
        for arg in expr.args:
            result = Add(left=result, right=self._convert_split(arg))
        result = Add(left=result, right=Const(value=expr.base))
        return self._analyzer.simplify(result)

    def _convert_split(self, expr: IterSplitExpr) -> Expr:
        """Lower a split, mirroring ``ConvertIterSplitExpr`` (~line 2095).

        The source is a Var or a nested sum (lowered recursively). Full-extent
        unit-lower-factor splits collapse to ``source * scale``; a split spanning
        the source's full extent (``lower_factor * extent == source.extent``)
        collapses to ``floordiv(source, lower_factor) * scale``; otherwise the
        general ``floordiv(floormod(source, lower_factor*extent), lower_factor) *
        scale`` form is emitted. The result is run through the analyzer so trivial
        scales / divisions fold (matching TVM's canonical simplification).
        """
        source = self._convert_split_source(expr.source)
        scale = expr.scale
        full_extent = expr.lower_factor * expr.extent
        result: Expr
        if expr.extent == expr.source.extent and expr.lower_factor == 1:
            result = Mul(left=source, right=Const(value=scale))
        elif expr.source.extent == full_extent:
            result = Mul(left=FloorDiv(left=source, right=Const(value=expr.lower_factor)), right=Const(value=scale))
        else:
            inner = FloorDiv(
                left=Mod(left=source, right=Const(value=full_extent)), right=Const(value=expr.lower_factor)
            )
            result = Mul(left=inner, right=Const(value=scale))
        return self._analyzer.simplify(result)

    def _convert_split_source(self, mark: IterMark) -> Expr:
        """Lower a mark's source to an :class:`Expr`, mirroring the source branch of ``ConvertIterSplitExpr``.

        A plain :class:`~nkigym.ir.arith.expr.Var` source passes through; a nested
        :class:`IterSumExpr` source is lowered recursively.
        """
        result: Expr
        if isinstance(mark.source, IterSumExpr):
            result = self._convert_sum(mark.source)
        else:
            result = mark.source
        return result


def _matched_scale_of(expr: IterSumExpr, matched_pos: int, base_scale: int, step: int) -> int:
    """Return the matched scale at a fuse step, mirroring ``matched_scale`` in ``TryFuseIters``.

    On the first step (``step == 0``) the matched scale is the base scale;
    afterwards it is the matched arg's own scale (which equals the running
    ``expected_scale`` by construction). Kept as a small helper so the chain loop
    advances ``expected_scale = extent * matched_scale`` exactly as TVM does.
    """
    return base_scale if step == 0 else expr.args[matched_pos].scale


def _with_scale(split: IterSplitExpr, scale: int) -> IterSplitExpr:
    """Return a copy of ``split`` with ``scale`` replaced (a shallow copy-on-write).

    Mirrors TVM's ``CopyOnWrite()->scale = ...``: source / lower_factor / extent
    are preserved (source identity in particular, so independence-collection keying
    is unaffected) and only ``scale`` changes.
    """
    return IterSplitExpr(source=split.source, lower_factor=split.lower_factor, extent=split.extent, scale=scale)


def _is_iter(expr: object) -> bool:
    """Return whether ``expr`` is an iter-map node (``IterSumExpr`` / ``IterSplitExpr``).

    Mirrors TVM's ``expr->IsInstance<IterMapExprNode>()`` test used throughout the
    visitors to decide the iter-vs-plain branches.
    """
    return isinstance(expr, (IterSumExpr, IterSplitExpr))


def _as_expr(expr: IterSumExpr | IterSplitExpr | Expr) -> Expr:
    """Return ``expr`` as a plain :class:`Expr`, asserting it is not an iter-map node.

    Used on the non-iter branches of the visitors, where TVM has already
    established the operand is an ordinary PrimExpr (a constant in the corpus).
    """
    if isinstance(expr, (IterSumExpr, IterSplitExpr)):
        raise ValueError(f"expected plain Expr, got iter-map node {type(expr).__name__}")
    return expr


def _iter_sum_equal(lhs: IterSumExpr, rhs: IterSumExpr) -> bool:
    """Structural equality of two sums, mirroring ``IterSumEqual`` (~line 421).

    Equal iff bases match and args match pairwise on source identity, lower_factor,
    extent and scale (``IterSplitEqual`` with ``check_scale=True``).
    """
    result = len(lhs.args) == len(rhs.args) and lhs.base == rhs.base
    if result:
        for la, ra in zip(lhs.args, rhs.args):
            same = la.source is ra.source and la.lower_factor == ra.lower_factor
            same = same and la.extent == ra.extent and la.scale == ra.scale
            if not same:
                result = False
                break
    return result


def _ceildiv(a: int, b: int) -> int:
    """Ceiling division of positive integers, mirroring ``ceildiv`` used in the FloorDiv split."""
    return -(-a // b)


def detect_iter_map(indices: list[Expr], input_iters: dict[str, tuple[int, int]]) -> list[IterSumExpr] | None:
    """Detect whether ``indices`` form an affine iter map over ``input_iters``.

    Faithful subset of ``DetectIterMap`` (``iter_affine_map.cc`` ~line 1431) at
    the default surjective check level with no predicate and no padding. Each
    index is rewritten to an :class:`IterSumExpr` (:meth:`_IterMapRewriter.rewrite`);
    if every rewrite succeeds and the bindings pass the independence check
    (:meth:`_IterMapRewriter.check_mapping`), the list of normalized sums is
    returned. Returns ``None`` (TVM's empty ``.indices``) if any index is not a
    valid affine iter or the indices are not independent.

    Args:
        indices: The index expressions to detect (over the input iterators).
        input_iters: Map from iterator name to its half-open range ``[lo, hi)``.
            Every ``lo`` must be 0 (the corpus never offsets an input iterator).

    Returns:
        The detected list of :class:`IterSumExpr`, one per index, or ``None``.
    """
    rewriter = _IterMapRewriter(input_iters)
    rewritten: list[IterSumExpr] = []
    for index in indices:
        rewritten.append(rewriter.rewrite(index))
        if rewriter.errors:
            break
    result: list[IterSumExpr] | None
    if rewriter.errors:
        result = None
    elif not rewriter.check_mapping(rewritten):
        result = None
    else:
        result = rewritten
    return result


def normalize_iter_map_to_expr(sum_expr: IterSumExpr) -> Expr:
    """Lower a detected :class:`IterSumExpr` back to a plain :class:`Expr`.

    Mirrors ``NormalizeIterMapToExpr`` (``iter_affine_map.cc`` ~line 2142): runs
    :class:`_IterMapToExprNormalizer` over the iter-map expression to recover the
    ``floordiv`` / ``floormod`` / ``*`` / ``+`` PrimExpr it denotes.
    """
    return _IterMapToExprNormalizer(Analyzer()).convert(sum_expr)


def iter_map_simplify(indices: list[Expr], input_iters: dict[str, tuple[int, int]]) -> list[Expr] | None:
    """Detect then lower ``indices`` back to simplified :class:`Expr`s.

    Mirrors ``IterMapSimplify`` (``iter_affine_map.cc`` ~line 2153): run
    :func:`detect_iter_map`, then :func:`normalize_iter_map_to_expr` each detected
    sum. Returns ``None`` when detection fails (the corpus has no predicate, so
    TVM's predicate-retry branch does not apply).

    Args:
        indices: The index expressions to simplify.
        input_iters: Map from iterator name to its half-open range ``[lo, hi)``.

    Returns:
        The simplified expressions (one per index) or ``None`` if not an iter map.
    """
    detected = detect_iter_map(indices, input_iters)
    result: list[Expr] | None
    if detected is None:
        result = None
    else:
        result = [normalize_iter_map_to_expr(sum_expr) for sum_expr in detected]
    return result


__all__ = [
    "IterMark",
    "IterSplitExpr",
    "IterSumExpr",
    "detect_iter_map",
    "iter_map_simplify",
    "normalize_iter_map_to_expr",
]
