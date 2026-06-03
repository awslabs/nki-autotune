"""Structural comparison of rendered transform output vs a hand kernel.

The renderer and the hand-written ``kernel_transforms.py`` ladder describe
the *same* NKI program but in two different surface styles. A naive
string / black-format comparison fails on purely cosmetic, semantically
empty skews, so :func:`assert_matches_hand` compares the two as canonical
Python ASTs after a small set of structure-preserving rewrites.

What is normalized (and why each is non-semantic):

* **Function / accumulator names** — the renderer names the kernel
  ``nki_f_<fn>`` and the matmul accumulator ``psum_prod``; the hand
  kernels use ``kernel_N`` and ``psum_acc``. Renamed to fixed
  placeholders ``KFN`` / ``PACC``.
* **Shape ``assert`` statements** — the renderer emits
  ``assert lhs_T.shape == (...)`` guards the hand kernels omit. Dropped.
* **Buffer-declaration placement** — buffers are SSA allocations; the
  renderer declares each on its LCA block (so a sunk buffer's
  ``nl.ndarray`` appears mid-body) while the hand kernels list every
  declaration at the top. All ``nl.ndarray(...) = ...`` assignments are
  hoisted to the top of the function body and ordered by target name, so
  placement no longer matters.
* **``nisa.*`` argument style** — the renderer always passes operands by
  keyword in ``OPERAND_AXES`` order (``src=...`` before ``dst=...``); the
  hand kernels write ``dst`` first and pass ``memset`` / ``tensor_copy``
  operands positionally. Positional operands are lifted to keywords via
  :data:`_POSITIONAL_SLOTS` and every call's keyword arguments are sorted
  by name, so operand order / positional-vs-keyword no longer matters.
* **Affine slice arithmetic** — the renderer prints ``0:0 + 2048`` and
  bare ``i_d0_0 * 128`` where the hand kernels write ``0:2048`` and
  ``(i_d0_0) * 128``; Python's parser already discards the redundant
  parens, and constant folding (``0 + 2048`` → ``2048``) plus
  affine-canonical ordering reconcile the rest.

None of these rewrites can equate two genuinely different programs: any
difference in op name, operand tensor, slice bounds, loop variable, loop
extent, or nesting structure survives into the canonical AST. A guard
test (``test_oracle_rejects_different_kernel``) pins this property.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable

_NAME_RENAMES: dict[str, str] = {"psum_prod": "PACC", "psum_acc": "PACC"}

_POSITIONAL_SLOTS: dict[str, tuple[str, ...]] = {"memset": ("dst",), "tensor_copy": ("dst", "src")}


class _Canonicalize(ast.NodeTransformer):
    """Rewrite a kernel function's AST into a placement / order canonical form."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename the function, drop asserts, hoist ``nl.ndarray`` declarations."""
        node.name = "KFN"
        kept: list[ast.stmt] = [stmt for stmt in node.body if not isinstance(stmt, ast.Assert)]
        decls = [stmt for stmt in kept if _is_ndarray_decl(stmt)]
        body = [stmt for stmt in kept if not _is_ndarray_decl(stmt)]
        decls.sort(key=_decl_target_name)
        node.body = [self.visit(stmt) for stmt in (decls + body)]
        node.decorator_list = [self.visit(dec) for dec in node.decorator_list]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Apply the fixed buffer-name renames."""
        node.id = _NAME_RENAMES.get(node.id, node.id)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Canonicalize ``nisa.*`` calls: lift positionals to keywords, sort keywords."""
        self.generic_visit(node)
        if _is_nisa_call(node):
            _normalize_nisa_call(node)
        return node


def _is_nisa_call(node: ast.Call) -> bool:
    """True when ``node`` is an ``nisa.<op>(...)`` attribute call."""
    func = node.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "nisa"


def _normalize_nisa_call(node: ast.Call) -> None:
    """Lift positional operands to keywords and sort all keywords by name (in place)."""
    assert isinstance(node.func, ast.Attribute)
    op_name = node.func.attr
    slots = _POSITIONAL_SLOTS.get(op_name, ())
    assert len(node.args) <= len(slots), f"nisa.{op_name}: {len(node.args)} positional args, no slot map"
    lifted = [ast.keyword(arg=slots[i], value=arg) for i, arg in enumerate(node.args)]
    keywords = lifted + list(node.keywords)
    keywords.sort(key=lambda kw: kw.arg or "")
    node.args = []
    node.keywords = keywords


def _is_ndarray_decl(stmt: ast.stmt) -> bool:
    """True when ``stmt`` is ``<name> = nl.ndarray(...)``."""
    if not isinstance(stmt, ast.Assign):
        return False
    value = stmt.value
    is_ndarray = (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "nl"
        and value.func.attr == "ndarray"
    )
    return is_ndarray and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)


def _decl_target_name(stmt: ast.stmt) -> str:
    """Return the assignment target name of an ``nl.ndarray`` declaration."""
    assert isinstance(stmt, ast.Assign)
    target = stmt.targets[0]
    assert isinstance(target, ast.Name)
    return target.id


def _normalize(src: str) -> str:
    """Parse ``src`` and return the canonical AST dump of its kernel function.

    Only the (single) ``FunctionDef`` is compared: a rendered module
    carries top-level ``import`` statements that ``inspect.getsource`` of a
    hand kernel omits, and those imports are not part of the kernel body.
    """
    module = ast.parse(src)
    fn = _single_function_def(module)
    canonical = _Canonicalize().visit(fn)
    folded = _ConstantFold().visit(canonical)
    ast.fix_missing_locations(folded)
    return ast.dump(folded, annotate_fields=True)


def _single_function_def(module: ast.Module) -> ast.FunctionDef:
    """Return the lone ``FunctionDef`` in ``module``; fail loudly otherwise."""
    fns = [stmt for stmt in module.body if isinstance(stmt, ast.FunctionDef)]
    assert len(fns) == 1, f"expected exactly one function def; got {len(fns)}"
    return fns[0]


class _ConstantFold(ast.NodeTransformer):
    """Canonicalize integer ``BinOp`` slice arithmetic to one affine spelling.

    Two value-preserving rewrites, so a factored hand-written offset and the
    renderer's flattened one agree:

    * Constant folding — ``0 + 2048`` and ``2048`` collapse to the same node.
    * Distribution over a constant factor — ``(i * 4 + j) * 128`` becomes
      ``i * 512 + j * 128`` (and the symmetric ``128 * (i * 4 + j)``), matching
      the renderer, whose offsets pass through ``from_affine`` and so always
      print fully distributed. Only multiplication by an integer ``Constant``
      distributes (a non-affine ``Var * Var`` product is left untouched), so no
      two semantically different offsets are equated.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        """Fold constant operands, else canonicalize a constant-scaled BinOp."""
        self.generic_visit(node)
        left, right = node.left, node.right
        out: ast.expr = node
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            if isinstance(left.value, int) and isinstance(right.value, int):
                out = ast.Constant(value=_apply_binop(node.op, left.value, right.value))
        elif isinstance(node.op, ast.Mult):
            out = self._scale(node, left, right)
        return out

    def _scale(self, node: ast.BinOp, left: ast.expr, right: ast.expr) -> ast.expr:
        """Canonicalize ``inner * Const`` to the renderer's fully-distributed form.

        With ``inner`` the non-constant operand and ``k`` the integer
        ``Constant`` factor:

        * ``(a + b) * k`` → ``a*k + b*k`` (distribute; ``Sub`` likewise),
        * ``(x * c) * k`` → ``x * (c*k)`` (reassociate a nested constant
          product).

        Together these flatten any nested affine times a constant to the single
        ``from_affine`` spelling the renderer prints. Each rewrite re-enters
        ``visit`` so the new sub-product / constant pair folds in turn. A
        product with no integer-``Constant`` factor (e.g. ``Var * Var``) is left
        untouched, so value-distinct offsets never collapse together.
        """
        inner, k = _scaled_operands(left, right)
        out: ast.expr = node
        if inner is not None and k is not None and isinstance(inner.op, (ast.Add, ast.Sub)):
            out = self.visit(
                ast.BinOp(
                    left=ast.BinOp(left=inner.left, op=ast.Mult(), right=k),
                    op=inner.op,
                    right=ast.BinOp(left=inner.right, op=ast.Mult(), right=k),
                )
            )
        elif inner is not None and k is not None and isinstance(inner.op, ast.Mult) and _is_int_const(inner.right):
            out = self.visit(
                ast.BinOp(left=inner.left, op=ast.Mult(), right=ast.BinOp(left=inner.right, op=ast.Mult(), right=k))
            )
        return out


def _scaled_operands(left: ast.expr, right: ast.expr) -> tuple[ast.BinOp | None, ast.expr | None]:
    """Split a ``BinOp * int-Constant`` product into ``(inner_binop, factor)``.

    Returns the non-constant ``BinOp`` operand and the integer-``Constant``
    factor (either operand order), or ``(None, None)`` when the product is not a
    constant scaling of a ``BinOp``.
    """
    inner: ast.BinOp | None = None
    factor: ast.expr | None = None
    if isinstance(left, ast.BinOp) and _is_int_const(right):
        inner, factor = left, right
    elif isinstance(right, ast.BinOp) and _is_int_const(left):
        inner, factor = right, left
    return inner, factor


def _is_int_const(node: ast.expr) -> bool:
    """True when ``node`` is an integer ``ast.Constant``."""
    return isinstance(node, ast.Constant) and isinstance(node.value, int)


def _apply_binop(op: ast.operator, left: int, right: int) -> int:
    """Evaluate one integer binary operator used in slice arithmetic."""
    if isinstance(op, ast.Add):
        result = left + right
    elif isinstance(op, ast.Sub):
        result = left - right
    elif isinstance(op, ast.Mult):
        result = left * right
    else:
        raise TypeError(f"unsupported constant-fold operator {type(op).__name__}")
    return result


def assert_matches_hand(rendered_src: str, hand_fn: Callable[..., object]) -> None:
    """Assert ``rendered_src`` equals ``hand_fn``'s source after AST canonicalization."""
    hand_src = inspect.getsource(hand_fn)
    got = _normalize(rendered_src)
    want = _normalize(hand_src)
    assert got == want, f"rendered != hand kernel\n--- got ---\n{got}\n--- want ---\n{want}"
