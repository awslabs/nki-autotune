"""Enumerate ``KernelIR`` variants reachable by applying subsets of IR rewrites.

Each rewrite (see :class:`IRRewrite`) exposes ``analyze`` + ``apply``.
This module composes those primitives into a variant-space iterator:
for a list of ``K`` rewrites, every subset of ``{0, ..., K-1}`` is
applied in list order to produce one variant IR.

Each rewrite is applied **all-or-nothing** — every match its
``analyze`` returns, or none. For finer-grained match curation, call
``analyze`` / ``apply`` on the rewrite directly.

Rewrites in a subset run in the order they appear in the input list —
each ``analyze`` runs on the IR produced by the prior rewrites, so
later rewrites see the effect of earlier ones. Variants whose IR
``repr`` matches an earlier-yielded variant are deduplicated (so a
rewrite whose pattern is absent collapses into the ``"base"`` entry
rather than producing a redundant variant).
"""

from collections.abc import Iterator, Sequence

from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.rewrites.base import IRRewrite


def enumerate_rewrite_combinations(ir: KernelIR, rewrites: Sequence[IRRewrite]) -> Iterator[tuple[str, KernelIR]]:
    """Yield ``(name, variant_ir)`` for every subset of ``rewrites``.

    Args:
        ir: Base IR — yielded as ``("base", ir)`` for the empty subset.
        rewrites: Ordered list of IR rewrites. Each contributes one
            bit to the subset mask; rewrites in a subset are applied
            in this order.

    Yields:
        ``(name, variant_ir)`` pairs. ``name`` is ``"base"`` for the
        empty subset, otherwise the ``+``-join of the applied
        rewrites' class names (e.g. ``"LoadTranspose"`` or
        ``"LoadTranspose+OnlineFusion"``). Up to
        ``2 ** len(rewrites)`` pairs; duplicates (matched by
        ``repr(ir)``) are skipped.
    """
    seen: set[str] = set()
    for mask in range(1 << len(rewrites)):
        cur = ir
        labels: list[str] = []
        for k, rewrite in enumerate(rewrites):
            if mask & (1 << k):
                matches = rewrite.analyze(cur)
                cur = rewrite.apply(cur, matches)
                labels.append(type(rewrite).__name__)
        name = "+".join(labels) if labels else "base"
        key = repr(cur)
        if key in seen:
            continue
        seen.add(key)
        yield name, cur
