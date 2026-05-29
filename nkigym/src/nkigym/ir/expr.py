"""Affine integer Expression AST for iter_values and BufferRegion ranges.

Supports affine combinations of integer-typed Vars: Const, Var, Add,
Mul, FloorDiv, Mod. Sufficient for every binding and region range our
canonical builder and transforms emit.

Non-affine inputs (Var * Var, Mod / FloorDiv with non-Const divisor)
raise :class:`NonAffineError` from :func:`to_affine`.
"""

from __future__ import annotations

from dataclasses import dataclass


class NonAffineError(ValueError):
    """Raised when ``to_affine`` encounters a pattern that is not affine in Vars."""


@dataclass(frozen=True, kw_only=True)
class Const:
    """Integer literal."""

    value: int


@dataclass(frozen=True, kw_only=True)
class Var:
    """Symbolic variable identified by ``name``."""

    name: str


@dataclass(frozen=True, kw_only=True)
class Add:
    """Binary addition."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class Mul:
    """Binary multiplication. At most one operand may contain a Var (affinity)."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class FloorDiv:
    """Floor division. ``right`` must reduce to a non-zero ``Const`` for affinity."""

    left: "Expr"
    right: "Expr"


@dataclass(frozen=True, kw_only=True)
class Mod:
    """Modulo. ``right`` must reduce to a non-zero ``Const`` for affinity."""

    left: "Expr"
    right: "Expr"


Expr = Const | Var | Add | Mul | FloorDiv | Mod


def to_affine(expr: Expr) -> dict[str | None, int]:
    """Collapse ``expr`` to canonical affine form ``c0 + c1*v1 + c2*v2 + ...``.

    The returned dict maps variable names to integer coefficients; the
    constant term lives under key ``None``. Zero coefficients are
    pruned. Raises :class:`NonAffineError` on patterns we don't
    support (Var * Var, FloorDiv / Mod with a non-constant divisor).
    """
    coeffs = _accumulate(expr)
    return {k: v for k, v in coeffs.items() if v != 0}


def _accumulate(expr: Expr) -> dict[str | None, int]:
    """Recurse into ``expr`` and return its raw affine coefficients.

    Internal helper for :func:`to_affine` that does not prune zeroes
    so callers can detect ``0`` outputs (e.g. for ``from_affine``
    round-trip). Raises :class:`NonAffineError` on non-affine patterns.
    """
    if isinstance(expr, Const):
        return {None: expr.value}
    if isinstance(expr, Var):
        return {expr.name: 1}
    if isinstance(expr, Add):
        return _add(_accumulate(expr.left), _accumulate(expr.right))
    if isinstance(expr, Mul):
        return _mul(_accumulate(expr.left), _accumulate(expr.right))
    if isinstance(expr, FloorDiv):
        right = _accumulate(expr.right)
        if set(right) - {None}:
            raise NonAffineError(f"FloorDiv divisor is not constant: {expr.right}")
        divisor = right.get(None, 0)
        if divisor == 0:
            raise NonAffineError("FloorDiv by zero")
        left = _accumulate(expr.left)
        for var, coeff in left.items():
            if coeff % divisor != 0:
                raise NonAffineError(f"FloorDiv coefficient {coeff} of {var} not divisible by {divisor}")
        return {var: coeff // divisor for var, coeff in left.items()}
    if isinstance(expr, Mod):
        right = _accumulate(expr.right)
        if set(right) - {None}:
            raise NonAffineError(f"Mod divisor is not constant: {expr.right}")
        divisor = right.get(None, 0)
        if divisor == 0:
            raise NonAffineError("Mod by zero")
        left = _accumulate(expr.left)
        if set(left) - {None}:
            raise NonAffineError(f"Mod left side is not constant: {expr.left}")
        return {None: left.get(None, 0) % divisor}
    raise TypeError(f"Unknown Expr node {type(expr).__name__}")


def _add(a: dict[str | None, int], b: dict[str | None, int]) -> dict[str | None, int]:
    """Coefficient-wise sum of two affine coefficient maps."""
    out = dict(a)
    for var, coeff in b.items():
        out[var] = out.get(var, 0) + coeff
    return out


def _mul(a: dict[str | None, int], b: dict[str | None, int]) -> dict[str | None, int]:
    """Coefficient-wise product. At most one operand may contain a non-None key."""
    a_vars = set(a) - {None}
    b_vars = set(b) - {None}
    if a_vars and b_vars:
        raise NonAffineError(f"Var * Var: {sorted(a_vars)} times {sorted(b_vars)}")
    if not a_vars:
        scale = a.get(None, 0)
        return {var: coeff * scale for var, coeff in b.items()}
    scale = b.get(None, 0)
    return {var: coeff * scale for var, coeff in a.items()}


def from_affine(coeffs: dict[str | None, int]) -> Expr:
    """Inverse of :func:`to_affine`. Returns a canonical-form Expr.

    Variables are emitted in sorted name order so equal coefficient
    maps produce structurally equal Exprs.
    """
    terms: list[Expr] = []
    var_names = sorted(name for name in coeffs if name is not None)
    for name in var_names:
        coeff = coeffs[name]
        if coeff == 0:
            continue
        if coeff == 1:
            terms.append(Var(name=name))
        else:
            terms.append(Mul(left=Var(name=name), right=Const(value=coeff)))
    constant = coeffs.get(None, 0)
    if constant != 0 or not terms:
        terms.append(Const(value=constant))
    result = terms[0]
    for term in terms[1:]:
        result = Add(left=result, right=term)
    return result


def substitute(expr: Expr, subs: dict[str, Expr]) -> Expr:
    """Replace each ``Var(name)`` in ``expr`` by ``subs[name]`` recursively.

    Variables not present in ``subs`` are left unchanged. The returned
    expression is not normalised; pipe through ``from_affine(to_affine(...))``
    if a canonical form is needed.
    """
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Var):
        return subs.get(expr.name, expr)
    if isinstance(expr, Add):
        return Add(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    if isinstance(expr, Mul):
        return Mul(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    if isinstance(expr, FloorDiv):
        return FloorDiv(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    if isinstance(expr, Mod):
        return Mod(left=substitute(expr.left, subs), right=substitute(expr.right, subs))
    raise TypeError(f"Unknown Expr node {type(expr).__name__}")


def format_expr(expr: Expr) -> str:
    """Pretty-print ``expr`` as Python source.

    Normalises through :func:`to_affine` / :func:`from_affine` first
    so bindings render in a deterministic, sorted-name canonical form.
    Variables come first in sorted name order; the constant term
    trails (omitted if zero or no other terms exist).
    """
    canonical = from_affine(to_affine(expr))
    return _format_raw(canonical)


def _format_raw(expr: Expr) -> str:
    """Format an Expr without prior normalisation. Internal helper."""
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Add):
        return f"{_format_raw(expr.left)} + {_format_raw(expr.right)}"
    if isinstance(expr, Mul):
        return f"{_format_raw(expr.left)} * {_format_raw(expr.right)}"
    if isinstance(expr, FloorDiv):
        return f"{_format_raw(expr.left)} // {_format_raw(expr.right)}"
    if isinstance(expr, Mod):
        return f"{_format_raw(expr.left)} % {_format_raw(expr.right)}"
    raise TypeError(f"Unknown Expr node {type(expr).__name__}")


__all__ = [
    "Add",
    "Const",
    "Expr",
    "FloorDiv",
    "Mod",
    "Mul",
    "NonAffineError",
    "Var",
    "format_expr",
    "from_affine",
    "substitute",
    "to_affine",
]
