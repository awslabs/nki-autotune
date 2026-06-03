"""Public ``arith`` facade mirroring TVM's ``arith::Analyzer`` (``src/arith/analyzer.cc``).

This is the entry point the IR transforms (Split / Fuse / Reorder / compute_at)
call to simplify bindings and discharge legality predicates. It composes the
sub-analyzers TVM's ``Analyzer`` owns; here that surface is a single
:class:`~nkigym.ir.arith.rewrite_simplify.RewriteSimplifier` (which already
bundles the rewrite rules and the ``ConstIntBoundAnalyzer`` machinery), exposed
through the same four operations TVM's ``Analyzer`` exposes:
``Bind`` / ``Simplify`` / ``CanProve`` / ``CanProveEqual``.
"""

from __future__ import annotations

from nkigym.ir.arith.expr import Const, Expr, Sub, affine_terms
from nkigym.ir.arith.rewrite_simplify import RewriteSimplifier


class Analyzer:
    """Facade over the rewrite simplifier, mirroring ``arith::Analyzer``.

    Mirrors ``Analyzer::Analyzer()`` (``analyzer.cc`` ~line 37), which holds a
    family of sub-analyzers/simplifiers. Our single backing simplifier is a
    :class:`RewriteSimplifier`, which already carries both the rewrite rules and
    the const-int-bound state (the only sub-analyzers the affine corpus needs).
    Binds registered through :meth:`bind` persist on that backing simplifier for
    every subsequent :meth:`simplify`, :meth:`can_prove` and
    :meth:`can_prove_equal` query, exactly as TVM's ``Bind`` updates the
    persistent sub-analyzer state.
    """

    def __init__(self) -> None:
        """Construct the facade, mirroring ``Analyzer::Analyzer()`` (~line 37).

        Holds one :class:`RewriteSimplifier` standing in for TVM's family of
        sub-analyzers.
        """
        self._simplifier = RewriteSimplifier()

    def bind(self, name: str, lo: int, hi: int) -> None:
        """Bind ``name`` to the half-open range ``[lo, hi)``, mirroring ``Analyzer::Bind``.

        Mirrors ``Analyzer::Bind(var, Range)`` (``analyzer.cc`` ~line 57), which
        routes a variable's ``Range`` into the persistent sub-analyzer state. Our
        half-open ``[lo, hi)`` matches TVM's ``min`` + ``extent`` ``Range``; the
        bind is forwarded to the backing simplifier's
        :meth:`RewriteSimplifier.bind` and persists for subsequent queries.
        """
        self._simplifier.bind(name, lo, hi)

    def simplify(self, expr: Expr, steps: int = 2) -> Expr:
        """Simplify ``expr``, mirroring ``Analyzer::Simplify`` (``analyzer.cc`` ~line 236).

        TVM's ``Simplify`` re-runs its simplifiers ``steps`` times. Our backing
        :meth:`RewriteSimplifier.simplify` already drives its own rewrite fixpoint
        (capped internally), so ``steps`` is accepted only to mirror TVM's
        signature and is otherwise inert -- it is not used to add any iteration
        behaviour the backing simplifier does not itself provide.
        """
        del steps
        return self._simplifier.simplify(expr)

    def const_int_bound(self, expr: Expr) -> tuple[int | None, int | None]:
        """Constant-integer bound ``(min, max)`` of ``expr``, mirroring ``Analyzer::const_int_bound``.

        Mirrors the public ``Analyzer::const_int_bound`` accessor (``analyzer.cc``),
        which routes onto the ``ConstIntBoundAnalyzer`` sub-analyzer. Our backing
        :class:`RewriteSimplifier` owns that machinery, so this forwards to its
        :meth:`RewriteSimplifier.const_int_bound`. Endpoints reachable only as
        ``+inf`` / ``-inf`` are reported as ``None``. This is the interval
        :class:`~nkigym.ir.arith.int_set.IntSet` propagates through an affine
        expression in :meth:`IntSet.eval`; over the const-bounded affine corpus
        that propagation coincides with TVM's ``IntervalSetEvaluator``.
        """
        return self._simplifier.const_int_bound(expr)

    def can_prove(self, pred: Expr) -> bool:
        """Decide whether ``pred`` is provably true, mirroring ``Analyzer::CanProve`` (~line 192).

        TVM's ``CanProve`` at ``ProofStrength::kDefault`` simplifies the predicate
        and checks whether it folded to a constant truth; the index-comparison
        proving power comes from the ``TryCompare``-over-``const_int_bound``
        folding inside the simplifier's comparison visitors. Both are carried by
        the backing :meth:`RewriteSimplifier.can_prove`, to which this forwards.
        """
        return self._simplifier.can_prove(pred)

    def can_prove_equal(self, lhs: Expr, rhs: Expr) -> bool:
        """Decide whether ``lhs`` and ``rhs`` are provably equal, mirroring ``CanProveEqual`` (~line 164).

        TVM's ``CanProveEqual`` proves equality via ``CanProve(lhs - rhs == 0)``,
        i.e. that ``Simplify(lhs - rhs)`` folds to ``0``. Our backing
        :class:`RewriteSimplifier` does not affine-canonicalize ``Sub`` (its
        ``Sub`` visitor only const-folds and drops a zero subtrahend), so a purely
        commutative difference such as ``(x + y) - (y + x)`` does not fold to
        ``Const(0)`` through :meth:`simplify` alone.

        The proof therefore takes the affine route, which is sound and exact for
        the affine corpus these transforms emit: ``lhs`` and ``rhs`` are provably
        equal iff their :func:`~nkigym.ir.arith.expr.affine_terms` coefficient
        dicts are equal (equal affine-term dicts denote the same affine function,
        with non-affine subterms carried opaquely as identical keys). TVM's
        ``Simplify(lhs - rhs) == 0`` route is preserved as a fast path for the
        cases the simplifier does fold (e.g. two integer constants).
        """
        diff = self.simplify(Sub(left=lhs, right=rhs))
        result = diff == Const(value=0) or affine_terms(lhs) == affine_terms(rhs)
        return result


__all__ = ["Analyzer"]
