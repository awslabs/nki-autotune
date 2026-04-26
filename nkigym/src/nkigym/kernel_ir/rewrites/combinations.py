"""Enumerate ``KernelIR`` variants reachable by applying subsets of IR rewrites.

Each rewrite (see :class:`IRRewrite`) exposes ``analyze`` + ``apply``.
This module composes those primitives into a variant-space iterator:
every rewrite's ``analyze`` runs once on the base IR, the results are
flattened into a single ``(rewrite, match)`` list, and every subset of
that list is applied to produce one variant IR.

For ``N`` total matches across all rewrites the enumerator yields up to
``2 ** N`` variants (deduped by ``repr(ir)``). This assumes rewrites
commute — matches discovered on the base IR remain valid after other
rewrites fire. If that invariant ever breaks, switch to re-``analyze``
after each ``apply``.

When a subset includes multiple matches from the same rewrite, that
rewrite's ``apply`` is called once with all selected matches; rewrites
without selected matches are skipped. Rewrites run in the order they
appear in the input list.
"""

from collections.abc import Iterator, Sequence

from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.rewrites.base import IRRewrite


def enumerate_rewrite_combinations(ir: KernelIR, rewrites: Sequence[IRRewrite]) -> Iterator[tuple[str, KernelIR]]:
    """Yield ``(name, variant_ir)`` for every subset of matches across ``rewrites``.

    Args:
        ir: Base IR — yielded as ``("base", ir)`` for the empty subset.
        rewrites: Ordered list of IR rewrites. Each rewrite's
            ``analyze`` is called once on ``ir`` up front; matches from
            all rewrites are flattened into a single bit vector.

    Yields:
        ``(name, variant_ir)`` pairs. ``name`` is ``"base"`` for the
        empty subset, otherwise the ``+``-join of
        ``<RewriteName>[<match_index>]`` tokens for each selected
        match (match index is local to its rewrite's match list, e.g.
        ``"LoadTranspose[0]+LoadTranspose[1]"``). Up to ``2 ** N``
        pairs for ``N`` total matches; duplicates (matched by
        ``repr(ir)``) are skipped.
    """
    per_rewrite_matches: list[list] = [list(r.analyze(ir)) for r in rewrites]
    flat: list[tuple[int, int]] = []
    for r_idx, matches in enumerate(per_rewrite_matches):
        for m_idx in range(len(matches)):
            flat.append((r_idx, m_idx))

    seen: set[str] = set()
    for mask in range(1 << len(flat)):
        selected_per_rewrite: list[list] = [[] for _ in rewrites]
        labels: list[str] = []
        for bit, (r_idx, m_idx) in enumerate(flat):
            if mask & (1 << bit):
                selected_per_rewrite[r_idx].append(per_rewrite_matches[r_idx][m_idx])
                labels.append(f"{type(rewrites[r_idx]).__name__}[{m_idx}]")
        cur = ir
        for r_idx, rewrite in enumerate(rewrites):
            picks = selected_per_rewrite[r_idx]
            if picks:
                cur = rewrite.apply(cur, picks)
        name = "+".join(labels) if labels else "base"
        key = repr(cur)
        if key in seen:
            continue
        seen.add(key)
        yield name, cur
