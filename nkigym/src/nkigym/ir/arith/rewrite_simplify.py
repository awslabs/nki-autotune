"""Faithful port of TVM's ``RewriteSimplifier`` (``src/arith/rewrite_simplify.cc``).

Recursively rewrites an :class:`Expr` AST node-by-node, mirroring
``tvm.arith.Analyzer().rewrite_simplify(...)``. Each node's ``_visit_*`` method
recurses into its children first, rebuilds the node, then applies that node's
constant-folding (``const_fold.h``) and pattern rewrite rules exactly as the
corresponding ``RewriteSimplifier::Impl::VisitExpr_`` does in the TVM source.

Only the rules exercised by the oracle test corpus are transcribed; every
transcribed rule is a faithful copy of TVM's rule (line refs in each method's
docstring), never an invented simplification. Correctness is arbitrated by a
TVM oracle in ``test/ir/arith/test_rewrite_simplify.py``.

The public :meth:`RewriteSimplifier.simplify` mirrors
``RewriteSimplifier::operator()`` (``rewrite_simplify.cc`` ~line 2427): it runs
the post-order visitor to a fixpoint, capped at two iterations, returning early
when an iteration leaves the expression unchanged.

This module also ports the inequality-proving machinery: :meth:`bind` mirrors
``Analyzer::Bind(var, Range)`` (``analyzer.cc`` ~line 57) feeding
``ConstIntBoundAnalyzer::Bind`` (``const_int_bound.cc`` ~line 111);
:meth:`const_int_bound` mirrors ``ConstIntBoundAnalyzer::Impl`` (interval
``[min, max]`` propagation, ``const_int_bound.cc``); and :meth:`can_prove`
simplifies the predicate (as ``Analyzer::CanProve`` at ``kDefault`` does,
``analyzer.cc`` ~line 192) and then reproduces the comparison folding that
TVM's LT/LE/EQ visitors perform inside ``simplify`` --
``RewriteSimplifier::Impl::TryCompare(x, 0)`` over ``const_int_bound(a - b)``
(``rewrite_simplify.cc`` ~line 353 / LT visitor ~line 1843) -- relocating that
same ``TryCompare`` folding into ``can_prove`` itself. It is this visitor-level
folding (not a distinct kDefault code path) that lets kDefault prove these
predicates in TVM.
"""

from __future__ import annotations

from collections.abc import Callable

from nkigym.ir.arith.expr import EQ, LE, LT, Add, Const, Expr, FloorDiv, Max, Min, Mod, Mul, Sub, Var

_MAX_ITER = 2
"""Fixpoint iteration cap, mirroring ``RewriteSimplifier::operator()`` ``max_iter``."""

_POS_INF = 2**63 - 1
"""Sentinel for +inf, mirroring ``ConstIntBound::kPosInf`` (``int64`` max)."""

_NEG_INF = -_POS_INF
"""Sentinel for -inf, mirroring ``ConstIntBound::kNegInf`` (``-kPosInf``)."""

_I32_MIN = -(2**31)
"""Lower limit of ``Everything(int32)`` (``const_int_bound.cc`` ~line 724)."""

_I32_MAX = 2**31 - 1
"""Upper limit of ``Everything(int32)`` (``const_int_bound.cc`` ~line 724)."""


def _floordiv(a: int, b: int) -> int:
    """Floor division matching ``arith::floordiv`` (Python ``//`` is already floor)."""
    return a // b


def _floormod(a: int, b: int) -> int:
    """Floor modulo matching ``arith::floormod`` (Python ``%`` is already floor)."""
    return a % b


def _inf_aware_add(x: int, y: int) -> int:
    """Compute ``x + y`` aware of inf, mirroring ``InfAwareAdd`` (``const_int_bound.cc`` ~line 578).

    ``+inf`` dominates a finite or ``+inf`` addend; ``-inf`` dominates a finite
    or ``-inf`` addend; a finite-finite sum is the plain integer sum (our
    operands stay in int32 range, so no overflow saturation is required).
    """
    result: int
    if x == _POS_INF:
        result = _POS_INF
    elif x == _NEG_INF:
        result = _NEG_INF
    elif y == _POS_INF or y == _NEG_INF:
        result = y
    else:
        result = x + y
    return result


def _inf_aware_mul(x: int, y: int) -> int:
    """Compute ``x * y`` aware of inf, mirroring ``InfAwareMul`` (``const_int_bound.cc`` ~line 600).

    A finite-finite product is the plain product; otherwise the sign of the
    (infinite) product selects ``+inf`` or ``-inf``: like signs give ``+inf``,
    unlike signs give ``-inf``. The finite-finite branch relies on int32-bounded
    operands keeping the product below the ``_POS_INF`` sentinel, so a finite
    product never collides with the sentinel (the same assumption the Add helper
    documents).
    """
    x_inf = x == _POS_INF or x == _NEG_INF
    y_inf = y == _POS_INF or y == _NEG_INF
    result: int
    if not x_inf and not y_inf:
        result = x * y
    elif (x > 0 and y > 0) or (x < 0 and y < 0):
        result = _POS_INF
    else:
        result = _NEG_INF
    return result


def _inf_aware_floordiv(x: int, y: int) -> int:
    """Compute ``floordiv(x, y)`` aware of inf, mirroring ``InfAwareFloorDiv`` (~line 625).

    ``y`` is asserted non-zero by the caller. An infinite numerator keeps its
    sign for positive ``y`` and flips it for negative ``y``; otherwise the plain
    floor division is returned.
    """
    if y == 0:
        raise ZeroDivisionError("FloorDiv by zero")
    result: int
    if x == _POS_INF or x == _NEG_INF:
        result = x if y > 0 else -x
    else:
        result = _floordiv(x, y)
    return result


def _binary_op_boundary(a: tuple[int, int], b: tuple[int, int], op: Callable[[int, int], int]) -> tuple[int, int]:
    """Boundary of a monotonic binary op, mirroring ``BinaryOpBoundary`` (~line 525).

    Evaluates ``op`` at all four corner combinations of the two ``[min, max]``
    intervals and returns ``(min, max)`` over those four results.
    """
    v1 = op(a[0], b[0])
    v2 = op(a[1], b[1])
    v3 = op(a[0], b[1])
    v4 = op(a[1], b[0])
    return (min(v1, v2, v3, v4), max(v1, v2, v3, v4))


class RewriteSimplifier:
    """Recursive expression simplifier transcribing TVM's rewrite rules.

    Mirrors ``RewriteSimplifier::Impl`` node-for-node: a per-node ``_visit_*``
    method recurses children, rebuilds the node, runs constant folding, then
    applies pattern rewrites. :meth:`simplify` drives the post-order visitor to
    a fixpoint exactly as ``RewriteSimplifier::operator()`` does.

    The :attr:`_var_bounds` map carries the constant-integer bounds bound via
    :meth:`bind`, consulted by :meth:`const_int_bound` for ``Var`` lookups
    exactly as ``ConstIntBoundAnalyzer::Impl::var_map_`` is in TVM.
    """

    def __init__(self) -> None:
        """Initialise an empty variable-bound map.

        ``_var_bounds`` maps a variable name to its inclusive constant-integer
        bound ``(min, max)``, mirroring ``ConstIntBoundAnalyzer::Impl::var_map_``
        (``const_int_bound.cc`` ~line 505).
        """
        self._var_bounds: dict[str, tuple[int, int]] = {}

    def bind(self, name: str, lo: int, hi: int) -> None:
        """Bind ``name`` to the half-open range ``[lo, hi)``, mirroring ``Analyzer::Bind``.

        Mirrors ``Analyzer::Bind(var, Range)`` (``analyzer.cc`` ~line 57) routing
        to ``ConstIntBoundAnalyzer::Impl::Bind`` (``const_int_bound.cc`` ~line
        111): TVM's ``Range`` is ``min`` + ``extent`` (half-open), and the stored
        inclusive const-int bound is ``min_value = min`` and
        ``max_value = min + extent - 1``. Here ``extent = hi - lo``, so the
        inclusive max is ``hi - 1``.
        """
        self._var_bounds[name] = (lo, hi - 1)

    def simplify(self, expr: Expr) -> Expr:
        """Simplify ``expr`` to a fixpoint, mirroring ``operator()`` (~line 2427).

        Runs the post-order visitor up to ``_MAX_ITER`` times, returning early
        once an iteration produces a structurally identical expression.
        """
        result = expr
        for _ in range(_MAX_ITER):
            new_expr = self._visit(result)
            if new_expr == result:
                break
            result = new_expr
        return result

    def _visit(self, expr: Expr) -> Expr:
        """Dispatch one post-order rewrite pass over ``expr`` by node type.

        Mirrors ``IRMutatorWithAnalyzer::VisitExpr_`` dispatch: each handler
        first recurses into children, then applies that node's rules.
        """
        result: Expr
        if isinstance(expr, Const):
            result = expr
        elif isinstance(expr, Var):
            result = expr
        elif isinstance(expr, Add):
            result = self._visit_add(expr)
        elif isinstance(expr, Sub):
            result = self._visit_sub(expr)
        elif isinstance(expr, Mul):
            result = self._visit_mul(expr)
        elif isinstance(expr, FloorDiv):
            result = self._visit_floordiv(expr)
        elif isinstance(expr, Mod):
            result = self._visit_mod(expr)
        elif isinstance(expr, Min):
            result = Min(left=self._visit(expr.left), right=self._visit(expr.right))
        elif isinstance(expr, Max):
            result = Max(left=self._visit(expr.left), right=self._visit(expr.right))
        elif isinstance(expr, LT):
            result = LT(left=self._visit(expr.left), right=self._visit(expr.right))
        elif isinstance(expr, LE):
            result = LE(left=self._visit(expr.left), right=self._visit(expr.right))
        elif isinstance(expr, EQ):
            result = EQ(left=self._visit(expr.left), right=self._visit(expr.right))
        else:
            raise TypeError(f"RewriteSimplifier: unknown Expr node {type(expr).__name__}")
        return result

    def _visit_add(self, op: Add) -> Expr:
        """Rewrite an ``Add`` node, mirroring ``VisitExpr_(AddNode*)`` (~line 415).

        Recurses children, then applies ``TryConstFold<Add>`` (``const_fold.h``
        ~line 132): folds two constants, and drops a zero operand
        (``0 + b -> b``, ``a + 0 -> a``).

        Then the constant-folding index rule at ~line 474
        ``(x + c1) + c2 -> x + (c1 + c2)`` merges a trailing constant addend
        into a constant already on the right of the left subtree, folding the
        two constants into one (TVM's RHS ``x + (c1 + c2)`` is itself
        const-folded since ``c1`` and ``c2`` are both ``IntImm``). This is the
        rule that collapses ``(x + 2) + 3 -> x + 5`` and Split-style bindings
        ``Sum(var*factor) + c1 + c2``. No further Add pattern rule in the
        corpus fires.
        """
        a = self._visit(op.left)
        b = self._visit(op.right)
        result: Expr
        if isinstance(a, Const) and isinstance(b, Const):
            result = Const(value=a.value + b.value)
        elif isinstance(a, Const) and a.value == 0:
            result = b
        elif isinstance(b, Const) and b.value == 0:
            result = a
        elif isinstance(a, Add) and isinstance(a.right, Const) and isinstance(b, Const):
            result = Add(left=a.left, right=Const(value=a.right.value + b.value))
        else:
            result = Add(left=a, right=b)
        return result

    def _visit_sub(self, op: Sub) -> Expr:
        """Rewrite a ``Sub`` node, mirroring ``VisitExpr_(SubNode*)`` (~line 566).

        Recurses children, then applies ``TryConstFold<Sub>`` (``const_fold.h``
        ~line 156): folds two constants and drops a zero subtrahend
        (``a - 0 -> a``). No further Sub pattern rule in the corpus fires.
        """
        a = self._visit(op.left)
        b = self._visit(op.right)
        result: Expr
        if isinstance(a, Const) and isinstance(b, Const):
            result = Const(value=a.value - b.value)
        elif isinstance(b, Const) and b.value == 0:
            result = a
        else:
            result = Sub(left=a, right=b)
        return result

    def _visit_mul(self, op: Mul) -> Expr:
        """Rewrite a ``Mul`` node, mirroring ``VisitExpr_(MulNode*)`` (~line 755).

        Recurses children, then applies ``TryConstFold<Mul>`` (``const_fold.h``
        ~line 182): folds two constants; ``1 * b -> b`` and ``a * 1 -> a``;
        ``0 * b -> 0`` and ``a * 0 -> 0`` (returning the zero operand). No
        further Mul pattern rule in the corpus fires.
        """
        a = self._visit(op.left)
        b = self._visit(op.right)
        result: Expr
        if isinstance(a, Const) and isinstance(b, Const):
            result = Const(value=a.value * b.value)
        elif isinstance(a, Const) and a.value == 1:
            result = b
        elif isinstance(a, Const) and a.value == 0:
            result = a
        elif isinstance(b, Const) and b.value == 1:
            result = a
        elif isinstance(b, Const) and b.value == 0:
            result = b
        else:
            result = Mul(left=a, right=b)
        return result

    def _visit_floordiv(self, op: FloorDiv) -> Expr:
        """Rewrite a ``FloorDiv`` node, mirroring ``VisitExpr_(FloorDivNode*)`` (~line 1037).

        Recurses children, then applies ``TryConstFold<FloorDiv>``
        (``const_fold.h`` ~line 274): folds two constants (floor semantics),
        ``0 // b -> 0`` and ``a // 1 -> a``.

        Then the index rule at ~line 1088 ``floordiv(x*c1, c2)``: with constant
        divisor ``c2 != 0`` it forms the residue
        ``floordiv(x*floormod(c1,c2) + floormod(y,c2), c2)`` (here ``y = 0``),
        and when that residue simplifies to a constant (TVM's
        ``const_int_bound`` with ``min == max``) rewrites to
        ``x*floordiv(c1,c2) + (y_div + residue)``, with ``y_div = 0`` since
        ``floordiv(0,c2) == 0``.
        """
        a = self._visit(op.left)
        b = self._visit(op.right)
        result = self._floordiv_const_fold(a, b)
        if result is None:
            result = self._floordiv_mul_rule(a, b)
        if result is None:
            result = FloorDiv(left=a, right=b)
        return result

    def _floordiv_const_fold(self, a: Expr, b: Expr) -> Expr | None:
        """Constant fold for ``FloorDiv``, mirroring ``TryConstFold<FloorDiv>`` (~line 274).

        Returns the folded expression, or ``None`` when no fold applies.
        """
        result: Expr | None = None
        if isinstance(a, Const) and isinstance(b, Const):
            if b.value == 0:
                raise ZeroDivisionError("FloorDiv by zero")
            result = Const(value=_floordiv(a.value, b.value))
        elif isinstance(a, Const) and a.value == 0:
            result = a
        elif isinstance(b, Const) and b.value == 1:
            result = a
        return result

    def _floordiv_mul_rule(self, a: Expr, b: Expr) -> Expr | None:
        """Index rule ``floordiv(x*c1, c2)`` from ``VisitExpr_(FloorDivNode*)`` (~line 1088).

        Matches ``a == x * c1`` with constant ``c1`` and constant divisor
        ``b == c2 != 0``. Forms the residue
        ``floordiv(x*floormod(c1,c2) + floormod(0,c2), c2)``; when it simplifies
        to a constant, returns ``x*floordiv(c1,c2) + (0 + residue)`` (faithful to
        TVM with ``y = 0`` so ``y_div = 0``). Returns ``None`` if the pattern or
        the constant-residue condition does not hold.
        """
        match = _match_mul_const(a)
        result: Expr | None = None
        if match is not None and isinstance(b, Const) and b.value != 0:
            x, c1 = match
            c2 = b.value
            residue_expr = FloorDiv(
                left=Add(left=Mul(left=x, right=Const(value=_floormod(c1, c2))), right=Const(value=_floormod(0, c2))),
                right=Const(value=c2),
            )
            residue = self.simplify(residue_expr)
            if isinstance(residue, Const):
                quotient = Mul(left=x, right=Const(value=_floordiv(c1, c2)))
                result = Add(left=quotient, right=Add(left=Const(value=0), right=residue))
        return result

    def _visit_mod(self, op: Mod) -> Expr:
        """Rewrite a ``Mod`` (TVM ``FloorMod``) node, mirroring ``VisitExpr_(FloorModNode*)`` (~line 1189).

        Recurses children, then applies ``TryConstFold<FloorMod>``
        (``const_fold.h`` ~line 309): folds two constants (floor semantics) and
        ``a % 1 -> 0``.

        Then the Mul-numerator coefficient-reduction rule at ~line 1244
        ``floormod(x*c1, c2) -> floormod(x*floormod(c1,c2), c2)`` when
        ``c2 != 0`` (applied before the Add-numerator rule, matching TVM's rule
        order), followed by the index rule at ~line 1247
        ``floormod(x*c1 + y, c2) -> floormod(x, floordiv(c2,c1))*c1 + y`` when
        ``c1 > 0``, ``c2 > 0``, ``c2 % c1 == 0`` and ``floordiv(y, c1) == 0``
        (TVM's ``CanProveEqual(floordiv(y, c1), 0)``).
        """
        a = self._visit(op.left)
        b = self._visit(op.right)
        result = self._mod_const_fold(a, b)
        if result is None:
            result = self._mod_mul_rule(a, b)
        if result is None:
            result = self._mod_mul_add_rule(a, b)
        if result is None:
            result = Mod(left=a, right=b)
        return result

    def _mod_mul_rule(self, a: Expr, b: Expr) -> Expr | None:
        """Mul-numerator rule ``floormod(x*c1, c2)`` from ``VisitExpr_(FloorModNode*)`` (~line 1244).

        Matches ``a == x*c1`` with constant ``c1`` and constant divisor
        ``b == c2 != 0``, rewriting to ``floormod(x*floormod(c1,c2), c2)``. When
        ``floormod(c1,c2) == 0`` the rebuilt numerator ``x*0`` const-folds to
        ``0`` and the whole node to ``Const(0)`` on the next fixpoint pass
        (e.g. ``(4*x) % 4 -> 0``, ``(x*512) % 256 -> 0``); a residual coefficient
        is left for further folding (matching TVM's ``TVM_TRY_REWRITE_IF``).
        Returns ``None`` if the pattern or the divisor condition does not hold.
        """
        match = _match_mul_const(a)
        result: Expr | None = None
        if match is not None and isinstance(b, Const) and b.value != 0:
            x, c1 = match
            c2 = b.value
            reduced = self._visit_mul(Mul(left=x, right=Const(value=_floormod(c1, c2))))
            result = Mod(left=reduced, right=Const(value=c2))
        return result

    def _mod_const_fold(self, a: Expr, b: Expr) -> Expr | None:
        """Constant fold for ``Mod``, mirroring ``TryConstFold<FloorMod>`` (~line 309).

        Returns the folded expression, or ``None`` when no fold applies.
        """
        result: Expr | None = None
        if isinstance(a, Const) and isinstance(b, Const):
            if b.value == 0:
                raise ZeroDivisionError("Mod by zero")
            result = Const(value=_floormod(a.value, b.value))
        elif isinstance(a, Const) and a.value == 0:
            result = a
        elif isinstance(b, Const) and b.value == 1:
            result = Const(value=0)
        return result

    def _mod_mul_add_rule(self, a: Expr, b: Expr) -> Expr | None:
        """Index rule ``floormod(x*c1 + y, c2)`` from ``VisitExpr_(FloorModNode*)`` (~line 1247).

        Matches ``a == x*c1 + y`` with constant ``c1`` and constant divisor
        ``b == c2``. Rewrites to ``floormod(x, floordiv(c2,c1))*c1 + y`` when
        ``c1 > 0``, ``c2 > 0``, ``c2 % c1 == 0`` and ``floordiv(y, c1)``
        simplifies to ``0``. Returns ``None`` if the pattern or the conditions do
        not hold.
        """
        match = _match_mul_const_plus(a)
        result: Expr | None = None
        if match is not None and isinstance(b, Const):
            x, c1, y = match
            c2 = b.value
            y_div = self.simplify(FloorDiv(left=y, right=Const(value=c1)))
            condition = c1 > 0 and c2 > 0 and c2 % c1 == 0 and y_div == Const(value=0)
            if condition:
                inner = Mod(left=x, right=Const(value=_floordiv(c2, c1)))
                result = Add(left=Mul(left=inner, right=Const(value=c1)), right=y)
        return result

    def const_int_bound(self, expr: Expr) -> tuple[int | None, int | None]:
        """Constant-integer bound ``(min, max)`` of ``expr``, mirroring ``ConstIntBoundAnalyzer``.

        Faithful port of ``ConstIntBoundAnalyzer::Impl`` (``const_int_bound.cc``):
        propagates an inclusive interval per node type. A bound equal to the
        ``+inf`` / ``-inf`` sentinel is reported as ``None`` (truly unbounded);
        an unbound ``Var`` keeps TVM's finite ``Everything(int32)`` limits.
        """
        lo, hi = self._bound(expr)
        min_out = None if lo == _NEG_INF else lo
        max_out = None if hi == _POS_INF else hi
        return (min_out, max_out)

    def _bound(self, expr: Expr) -> tuple[int, int]:
        """Recursive inclusive ``[min, max]`` bound over the sentinel-inf integer domain.

        Internal worker for :meth:`const_int_bound`. Each branch mirrors the
        matching ``ConstIntBoundAnalyzer::Impl::VisitExpr_`` in
        ``const_int_bound.cc``; values use ``_POS_INF`` / ``_NEG_INF`` sentinels
        so the inf-aware helpers compose exactly as TVM's do.
        """
        result: tuple[int, int]
        if isinstance(expr, Const):
            result = (expr.value, expr.value)
        elif isinstance(expr, Var):
            result = self._var_bounds.get(expr.name, (_I32_MIN, _I32_MAX))
        elif isinstance(expr, Add):
            result = self._bound_add(expr)
        elif isinstance(expr, Sub):
            result = self._bound_sub(expr)
        elif isinstance(expr, Mul):
            result = _binary_op_boundary(self._bound(expr.left), self._bound(expr.right), _inf_aware_mul)
        elif isinstance(expr, FloorDiv):
            result = self._bound_floordiv(expr)
        elif isinstance(expr, Mod):
            result = self._bound_mod(expr)
        elif isinstance(expr, Min):
            result = self._bound_min(expr)
        elif isinstance(expr, Max):
            result = self._bound_max(expr)
        else:
            result = (0, 1)
        return result

    def _bound_add(self, op: Add) -> tuple[int, int]:
        """Bound of ``Add``, mirroring ``VisitExpr_(AddNode*)`` (``const_int_bound.cc`` ~line 236)."""
        a = self._bound(op.left)
        b = self._bound(op.right)
        return (_inf_aware_add(a[0], b[0]), _inf_aware_add(a[1], b[1]))

    def _bound_sub(self, op: Sub) -> tuple[int, int]:
        """Bound of ``Sub``, mirroring ``VisitExpr_(SubNode*)`` (``const_int_bound.cc`` ~line 246)."""
        a = self._bound(op.left)
        b = self._bound(op.right)
        return (_inf_aware_add(a[0], -b[1]), _inf_aware_add(a[1], -b[0]))

    def _bound_floordiv(self, op: FloorDiv) -> tuple[int, int]:
        """Bound of ``FloorDiv``, mirroring ``VisitExpr_(FloorDivNode*)`` (~line 318).

        Routes through ``HandleDivision`` with ``InfAwareFloorDiv``; the divisor
        is passed through ``AssumeNoZeroDivisor`` (a zero lower-bound becomes 1,
        since a valid program never divides by zero).
        """
        a = self._bound(op.left)
        b = _assume_no_zero_divisor(self._bound(op.right))
        return _handle_division(a, b, _inf_aware_floordiv)

    def _bound_mod(self, op: Mod) -> tuple[int, int]:
        """Bound of ``Mod`` (TVM ``FloorMod``), mirroring ``VisitExpr_(FloorModNode*)`` (~line 324).

        For a positive divisor with non-negative numerator: if the numerator is
        already below the divisor the numerator bound passes through, otherwise
        the result lies in ``[0, min(a_max, b_max - 1)]``. A negative-reachable
        numerator under a positive divisor gives ``[0, b_max - 1]``. The general
        modular-set tightening in TVM is not modelled (no ``ModularSet`` here);
        these affine-corpus cases never reach it. TVM's
        ``Intersect(..., Everything(int32))`` on the negative-divisor branch
        (``const_int_bound.cc`` ~lines 385-387) is likewise not modelled, as that
        branch is unreachable for the positive-constant divisors of the affine
        corpus.
        """
        a = self._bound(op.left)
        b = _assume_no_zero_divisor(self._bound(op.right))
        result: tuple[int, int]
        if b[0] > 0:
            b_max_cap = _inf_aware_add(b[1], -1)
            if a[0] >= 0:
                result = a if a[1] < b[0] else (0, min(a[1], b_max_cap))
            else:
                result = (0, b_max_cap)
        else:
            b_min_cap = _inf_aware_add(b[0], 1)
            b_max_cap = _inf_aware_add(b[1], -1)
            result = (min(0, b_min_cap), max(0, b_max_cap))
        return result

    def _bound_min(self, op: Min) -> tuple[int, int]:
        """Bound of ``Min``, mirroring ``VisitExpr_(MinNode*)`` (``const_int_bound.cc`` ~line 391)."""
        a = self._bound(op.left)
        b = self._bound(op.right)
        return (min(a[0], b[0]), min(a[1], b[1]))

    def _bound_max(self, op: Max) -> tuple[int, int]:
        """Bound of ``Max``, mirroring ``VisitExpr_(MaxNode*)`` (``const_int_bound.cc`` ~line 400)."""
        a = self._bound(op.left)
        b = self._bound(op.right)
        return (max(a[0], b[0]), max(a[1], b[1]))

    def can_prove(self, pred: Expr) -> bool:
        """Decide whether ``pred`` is provably true.

        ``Analyzer::CanProve`` at ``ProofStrength::kDefault`` (``analyzer.cc``
        ~line 192) simplifies the predicate; the proving power for index
        comparisons actually comes from the ``TryCompare``-over-``const_int_bound``
        folding inside the LT/LE/EQ visitors of ``simplify``
        (``RewriteSimplifier::Impl``, ``rewrite_simplify.cc`` LT/LE/EQ visitors).
        Here that same folding is relocated into ``can_prove``: first
        :meth:`simplify` the predicate and, if it folds to a constant truth,
        return it; otherwise reproduce ``TryCompare(simplify(a - b), 0)`` over
        ``const_int_bound`` (~line 353): ``LT`` holds when ``max(a - b) < 0``;
        ``LE`` when ``max(a - b) <= 0``; ``EQ`` when
        ``min(a - b) == max(a - b) == 0``. This reproduces kDefault's result on
        these predicates without being the literal kDefault code path. Unprovable
        predicates and non-comparison expressions yield ``False``.
        """
        simplified = self.simplify(pred)
        result: bool
        if isinstance(simplified, Const):
            result = simplified.value != 0
        elif isinstance(simplified, (LT, LE, EQ)):
            diff = self.simplify(Sub(left=simplified.left, right=simplified.right))
            lo, hi = self.const_int_bound(diff)
            result = _bound_proves(simplified, lo, hi)
        else:
            result = False
        return result


def _assume_no_zero_divisor(divisor: tuple[int, int]) -> tuple[int, int]:
    """Tighten a divisor bound assuming no divide-by-zero, mirroring ``AssumeNoZeroDivisor`` (~line 220).

    A valid program never divides by zero, so a divisor whose lower bound is 0
    is raised to 1; a divisor pinned to the constant 0 is rejected loudly.
    """
    if divisor[0] == 0 and divisor[1] == 0:
        raise ZeroDivisionError("divide by zero")
    lo = 1 if divisor[0] == 0 else divisor[0]
    return (lo, divisor[1])


def _handle_division(a: tuple[int, int], b: tuple[int, int], op: Callable[[int, int], int]) -> tuple[int, int]:
    """Division bound, mirroring ``HandleDivision`` (``const_int_bound.cc`` ~line 546).

    When the divisor interval straddles zero the range is split into its
    negative and positive parts (each narrowed away from zero) and the two
    corner boundaries unioned; otherwise the plain four-corner boundary is used.
    """
    result: tuple[int, int]
    if b[0] <= 0 <= b[1]:
        b_neg = (b[0], -1) if b[0] < 0 else (_I32_MIN, _I32_MAX)
        b_pos = (1, b[1]) if b[1] > 0 else (_I32_MIN, _I32_MAX)
        e_neg = _binary_op_boundary(a, b_neg, op)
        e_pos = _binary_op_boundary(a, b_pos, op)
        result = (min(e_neg[0], e_pos[0]), max(e_neg[1], e_pos[1]))
    else:
        result = _binary_op_boundary(a, b, op)
    return result


def _bound_proves(pred: LT | LE | EQ, lo: int | None, hi: int | None) -> bool:
    """Fold a comparison from the bound of ``left - right``, mirroring ``TryCompare(x, 0)``.

    Returns the truth of ``pred`` decided purely from the bound ``(lo, hi)`` of
    the simplified difference ``left - right`` (``rewrite_simplify.cc`` ~line 353
    via the LT/LE/EQ visitors): ``LT`` when ``hi < 0``; ``LE`` when ``hi <= 0``;
    ``EQ`` when ``lo == hi == 0``. A ``None`` (unbounded) endpoint never proves
    the predicate.
    """
    result: bool
    if isinstance(pred, LT):
        result = hi is not None and hi < 0
    elif isinstance(pred, LE):
        result = hi is not None and hi <= 0
    else:
        result = lo == 0 and hi == 0
    return result


def _match_mul_const(expr: Expr) -> tuple[Expr, int] | None:
    """Match ``x * c`` with a constant operand, returning ``(x, c)`` or ``None``.

    Mirrors TVM matching ``x * c1`` where ``c1`` is an ``IntImm``; either operand
    may carry the constant.
    """
    result: tuple[Expr, int] | None = None
    if isinstance(expr, Mul):
        if isinstance(expr.right, Const):
            result = (expr.left, expr.right.value)
        elif isinstance(expr.left, Const):
            result = (expr.right, expr.left.value)
    return result


def _match_mul_const_plus(expr: Expr) -> tuple[Expr, int, Expr] | None:
    """Match ``x * c + y``, returning ``(x, c, y)`` or ``None``.

    Mirrors TVM matching ``x * c1 + y`` where ``c1`` is an ``IntImm`` and ``y``
    is the remaining addend.
    """
    result: tuple[Expr, int, Expr] | None = None
    if isinstance(expr, Add):
        left_match = _match_mul_const(expr.left)
        if left_match is not None:
            x, c1 = left_match
            result = (x, c1, expr.right)
    return result


__all__ = ["RewriteSimplifier"]
