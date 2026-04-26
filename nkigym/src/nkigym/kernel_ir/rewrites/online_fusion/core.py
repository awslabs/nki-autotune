"""Core rewrite class + recipe registry for online fusion.

Per-workload recipes (pattern + ``build_after`` callback) register
themselves at import time via :func:`register_recipe`. The rewrite
class walks the registry and dispatches — neither the class nor this
module mentions any specific workload.
"""

from collections.abc import Callable
from dataclasses import dataclass

from nkigym.kernel_ir.build import _mint_psum_buffers
from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.rewrites.base import IRRewrite


@dataclass(frozen=True)
class OnlineFusionMatch:
    """One pattern-matched online-fusion site.

    Attributes:
        recipe_name: Key in the recipe registry naming the workload.
        op_indices: Indices of the matched ops in ``ir.ops``, in the
            pattern's declared order.
        context: Recipe-specific extras the matcher stashed for
            ``build_after`` to consume (e.g. the dim id assigned to
            the reducer's blocking axis). Opaque to the rewrite core.
    """

    recipe_name: str
    op_indices: tuple[int, ...]
    context: dict


@dataclass
class RewriteOutput:
    """What a recipe's ``build_after`` returns.

    Attributes:
        new_ops: Replacement ops, in execution order. Spliced in at
            the position of the first matched op; matched ops are
            dropped.
        new_buffers: Physical buffers the new ops reference that
            aren't already in ``ir.physical_buffers``.
        scratch_knobs: ``(scope, num_buffers, emission_depth)`` for
            each new buffer. Canonical INNER / ``NumBuffers()`` /
            ``0`` unless the recipe needs otherwise.
        retired_buffers: Intermediate SBUF buffers the old ops
            produced that no new op references — their knob entries
            get dropped too.
    """

    new_ops: list[Op]
    new_buffers: dict[str, PhysicalBuffer]
    scratch_knobs: dict[str, tuple[BufferScope, NumBuffers, int]]
    retired_buffers: tuple[str, ...]


@dataclass
class Recipe:
    """Pattern spec + build-after callback for one online-fusion workload.

    Attributes:
        name: Workload key — used in
            :attr:`OnlineFusionMatch.recipe_name` and as the registry
            key.
        find_matches: Walks ``ir.ops``, returns every site matching
            this recipe. Stateless; no IR mutation.
        build_after: Given ``ir`` and a specific match, constructs
            the replacement ops + scratch buffers + knobs. Does not
            mutate ``ir`` — returns a :class:`RewriteOutput` the
            rewrite core splices in.
    """

    name: str
    find_matches: Callable[[KernelIR], list[OnlineFusionMatch]]
    build_after: Callable[[KernelIR, OnlineFusionMatch], RewriteOutput]


_RECIPES: dict[str, Recipe] = {}
"""Module-level registry. Recipes call :func:`register_recipe` at
import time to add themselves; :class:`OnlineFusion` dispatches
through this dict."""


def register_recipe(recipe: Recipe) -> None:
    """Add ``recipe`` to the registry.

    Idempotent: re-registering the same name replaces the existing
    entry. Call at module import time from each recipe module.
    """
    _RECIPES[recipe.name] = recipe


def get_recipe(name: str) -> Recipe:
    """Look up a registered recipe by name."""
    return _RECIPES[name]


class OnlineFusion(IRRewrite[OnlineFusionMatch]):
    """Recipe-dispatched online-fusion rewrite.

    ``analyze(ir)`` asks every registered recipe for its matches.
    ``apply(ir, matches)`` splices each match's replacement ops in
    place, dropping the original chain and retiring intermediate
    buffers.
    """

    def analyze(self, ir: KernelIR) -> list[OnlineFusionMatch]:
        """Return every (recipe, site) pair in ``ir``."""
        matches: list[OnlineFusionMatch] = []
        for recipe in _RECIPES.values():
            matches.extend(recipe.find_matches(ir))
        return matches

    def apply(self, ir: KernelIR, matches: list[OnlineFusionMatch]) -> KernelIR:
        """Splice each match's replacement ops into ``ir`` and return a new IR."""
        if not matches:
            return ir

        ops = list(ir.ops)
        physical_buffers = dict(ir.physical_buffers)
        buffer_scopes = dict(ir.buffer_scopes)
        num_buffers = dict(ir.num_buffers)
        emission_depth = dict(ir.emission_depth)

        drop_indices: set[int] = set()
        """Sort matches by first op index so splicing preserves remaining
        indices (we rebuild ops at the end anyway, but stable ordering
        keeps the final op list in source order)."""
        for match in sorted(matches, key=lambda m: m.op_indices[0]):
            recipe = _RECIPES[match.recipe_name]
            output = recipe.build_after(ir, match)
            for i, op in enumerate(output.new_ops):
                """Inject the new ops at the first matched op's slot;
                later ops shift up. Easier: emit at the same index list-
                position slot, then drop retired indices at the end."""
                ops.insert(match.op_indices[0] + i, op)
                """Adjust drop_indices — any index ≥ insertion point
                shifts by one per inserted op."""
                shifted: set[int] = set()
                for idx in drop_indices:
                    shifted.add(idx + 1 if idx >= match.op_indices[0] + i else idx)
                drop_indices = shifted
            drop_indices.update(idx + len(output.new_ops) for idx in match.op_indices)
            for name, buf in output.new_buffers.items():
                physical_buffers[name] = buf
            for name, (scope, nb, depth) in output.scratch_knobs.items():
                buffer_scopes[name] = scope
                num_buffers[name] = nb
                emission_depth[name] = depth
            for name in output.retired_buffers:
                physical_buffers.pop(name, None)
                buffer_scopes.pop(name, None)
                num_buffers.pop(name, None)
                emission_depth.pop(name, None)

        ops = [op for i, op in enumerate(ops) if i not in drop_indices]
        """Drop stale ``psum_<stem>`` buffers whose vanilla matmul got
        rewritten away. ``_mint_psum_buffers`` is additive — it won't
        remove a psum buffer whose SBUF drain target no longer has an
        NKIMatmul producer. Walk the current ops and keep only PSUM
        entries still backed by a live matmul."""
        live_psum_names: set[str] = set()
        for op in ops:
            if op.kind == "NKIMatmul" and op.outputs and op.outputs[0].startswith("sbuf_"):
                live_psum_names.add("psum_" + op.outputs[0][len("sbuf_") :])
        stale_psum = [
            name for name, buf in list(physical_buffers.items()) if buf.loc == "psum" and name not in live_psum_names
        ]
        for name in stale_psum:
            physical_buffers.pop(name, None)
            buffer_scopes.pop(name, None)
            num_buffers.pop(name, None)
            emission_depth.pop(name, None)
        """Remint PSUM buffers for any new matmul-style ops.
        ``_mint_psum_buffers`` is idempotent — existing psum_* entries
        from vanilla matmuls skip."""
        _mint_psum_buffers(ops, physical_buffers, ir.dimensions)
        """PSUM buffers need canonical knobs too — mint defaults for any
        new psum_* entries. INNER / ``NumBuffers()`` / depth 0 matches
        what ``build_ir`` sets."""
        for name, buf in physical_buffers.items():
            if buf.loc != "hbm" and name not in buffer_scopes:
                buffer_scopes[name] = BufferScope.INNER
                num_buffers[name] = NumBuffers()
                emission_depth[name] = 0

        edges = _derive_edges(ops)
        return KernelIR(
            func_name=ir.func_name,
            param_names=ir.param_names,
            return_name=ir.return_name,
            dimensions=ir.dimensions,
            logical_tensors=ir.logical_tensors,
            physical_buffers=physical_buffers,
            ops=ops,
            edges=edges,
            dim_order=ir.dim_order,
            ltiles_per_block=ir.ltiles_per_block,
            buffer_scopes=buffer_scopes,
            num_buffers=num_buffers,
            emission_depth=emission_depth,
        )


def _derive_edges(ops: list[Op]) -> list[tuple[int, int]]:
    """Forward-only producer→consumer edges over ``ops``."""
    producer_of: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        for tname in op.inputs.values():
            src = producer_of.get(tname)
            if src is not None and src != i:
                edges.append((src, i))
        for out in op.outputs:
            producer_of[out] = i
    return edges


def sole_consumer(ops: list[Op], tensor: str) -> int | None:
    """Return the index of the sole op reading ``tensor``, or ``None``.

    Public utility — recipes use this when walking the IR for
    dependency chains with a single consumer at each step.
    """
    consumers = [i for i, op in enumerate(ops) if tensor in op.inputs.values()]
    if len(consumers) != 1:
        return None
    return consumers[0]


def has_other_consumers(ops: list[Op], tensor: str, allowed: set[int]) -> bool:
    """True iff any op outside ``allowed`` reads ``tensor``.

    Public utility — recipes use this to verify intermediate SBUFs
    can be retired safely (no reads outside the matched chain).
    """
    for i, op in enumerate(ops):
        if i in allowed:
            continue
        if tensor in op.inputs.values():
            return True
    return False
