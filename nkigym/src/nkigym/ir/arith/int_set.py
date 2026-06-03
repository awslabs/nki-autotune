"""Minimal integer-interval set, mirroring TVM ``src/arith/int_set.cc``.

A focused skeleton over the const-bounded affine corpus the IR transforms emit:
just the closed-interval :class:`IntSet` plus :meth:`IntSet.interval`,
:meth:`IntSet.union`, :meth:`IntSet.intersect`, and :meth:`IntSet.eval`. The
region-solve machinery of ``int_set.cc`` (``EstimateRegionLowerBound`` and the
symbolic ``IterMapRewriter`` paths) is intentionally not ported here; a later
spec covers heavy region use.

Mapping to ``int_set.cc``:

* :class:`IntSet` mirrors ``IntervalSetNode`` -- a closed interval
  ``[min_value, max_value]`` (``int_set.cc`` ~line 56). Endpoints are
  :class:`~nkigym.ir.arith.expr.Const` over the const-bounded corpus.
* :meth:`IntSet.interval` mirrors ``IntSet::Interval`` (~line 865).
* :meth:`IntSet.union` mirrors ``Union`` (~line 84):
  ``[min(amin, bmin), max(amax, bmax)]``.
* :meth:`IntSet.intersect` mirrors ``Intersect`` (~line 72):
  ``[max(amin, bmin), min(amax, bmax)]``.
* :meth:`IntSet.eval` mirrors ``IntervalSetEvaluator::Eval`` /
  ``Analyzer::int_set`` (~line 401, ~line 1014): propagate an interval through
  an affine expression under a var -> domain map. Over the const-bounded affine
  corpus this propagation is identical to the inclusive ``[min, max]`` interval
  arithmetic surfaced by
  :meth:`~nkigym.ir.arith.analyzer.Analyzer.const_int_bound`
  (an oracle-verified port of ``ConstIntBoundAnalyzer::Impl``): both run
  the same per-node rules -- ``Const`` -> point, ``Var`` -> its bound,
  ``Add`` -> endpoint add, ``Sub`` -> endpoint sub with swap, and
  ``Mul``-by-const -> sign-aware scale (compare ``Combine<Add>`` ~line 145,
  ``Combine<Sub>`` ~line 160, ``Combine<Mul>`` ~line 175). :meth:`eval`
  therefore delegates to the analyzer's bound rather than duplicate that
  per-node walk; the IntSet API surface above is the TVM-faithful entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Const, Expr


@dataclass(frozen=True, kw_only=True)
class IntSet:
    """A closed integer interval ``[min_value, max_value]`` (``IntervalSetNode``).

    Mirrors ``arith::IntervalSetNode`` (``int_set.cc`` ~line 56). Over the
    const-bounded affine corpus both endpoints are
    :class:`~nkigym.ir.arith.expr.Const`. ``min_value > max_value`` denotes the
    empty set, exactly as ``IntervalSet::Empty`` (``interval_set.h``) does.
    """

    min_value: Expr
    max_value: Expr

    @staticmethod
    def interval(min_value: Expr, max_value: Expr) -> "IntSet":
        """Construct the closed interval ``[min_value, max_value]``.

        Mirrors ``IntSet::Interval(min, max)`` (``int_set.cc`` ~line 865).
        """
        return IntSet(min_value=min_value, max_value=max_value)

    @staticmethod
    def union(a: "IntSet", b: "IntSet") -> "IntSet":
        """Hull union ``[min(amin, bmin), max(amax, bmax)]``.

        Mirrors ``Union(analyzer, a, b)`` (``int_set.cc`` ~line 84): the smallest
        interval covering both operands. Over the const-bounded corpus the
        endpoint ``min`` / ``max`` fold to integers.
        """
        lo = min(_as_int(a.min_value), _as_int(b.min_value))
        hi = max(_as_int(a.max_value), _as_int(b.max_value))
        return IntSet(min_value=Const(value=lo), max_value=Const(value=hi))

    @staticmethod
    def intersect(a: "IntSet", b: "IntSet") -> "IntSet":
        """Intersection ``[max(amin, bmin), min(amax, bmax)]``.

        Mirrors ``Intersect(analyzer, a, b)`` (``int_set.cc`` ~line 72). A result
        whose ``min`` exceeds its ``max`` denotes the empty set (``IntervalSet``
        keeps the raw endpoints; TVM normalises to ``Empty`` only when it can
        prove ``max < min``).
        """
        lo = max(_as_int(a.min_value), _as_int(b.min_value))
        hi = min(_as_int(a.max_value), _as_int(b.max_value))
        return IntSet(min_value=Const(value=lo), max_value=Const(value=hi))

    @staticmethod
    def eval(expr: Expr, analyzer: Analyzer) -> "IntSet":
        """Propagate an interval through ``expr`` under ``analyzer``'s var bounds.

        Mirrors ``IntervalSetEvaluator::Eval`` / ``Analyzer::int_set``
        (``int_set.cc`` ~line 401, ~line 1014). The variable domain is the set of
        bounds registered on ``analyzer`` via
        :meth:`~nkigym.ir.arith.analyzer.Analyzer.bind`. Over the const-bounded
        affine corpus the per-node interval propagation is identical to the
        analyzer's :meth:`~nkigym.ir.arith.analyzer.Analyzer.const_int_bound`
        (same ``Add`` / ``Sub`` / ``Mul``-by-const endpoint rules as TVM's
        ``Combine`` specialisations), so this delegates there. An unbounded
        endpoint (``None`` from ``const_int_bound``) is rejected loudly: the
        const-bounded corpus this skeleton serves never produces one.
        """
        lo, hi = analyzer.const_int_bound(expr)
        if lo is None or hi is None:
            raise ValueError(f"IntSet.eval got an unbounded endpoint for {expr!r}: ({lo}, {hi})")
        return IntSet(min_value=Const(value=lo), max_value=Const(value=hi))


def _as_int(endpoint: Expr) -> int:
    """Return the integer value of a :class:`Const` endpoint.

    Union / Intersect over this skeleton's const-bounded corpus operate on
    integer endpoints; a non-``Const`` endpoint is rejected loudly rather than
    silently mishandled.
    """
    if not isinstance(endpoint, Const):
        raise TypeError(f"IntSet endpoint is not a Const: {endpoint!r}")
    return endpoint.value


__all__ = ["IntSet"]
