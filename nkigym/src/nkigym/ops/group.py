"""NKIGroup: composite container for a shared loop nest.

A ``NKIGroup`` subclass represents a fusion group as a first-class
op-graph node. Its ``CHILDREN`` attribute lists the ordered child
``NKIOp`` classes that share the composite's loop nest; the
rendering path walks them in order to emit ISA calls. Every other
composite (online fusion chain, fused DMA transpose, ...)
inherits from ``NKIGroup`` and adds its own semantics on top.

The state that used to live on ``KernelIR.fusion_groups`` — the
group's chosen ``dim_order``, per-buffer ``buffer_degrees`` and
per-tensor ``tensor_placements`` — is carried per-composite on
dynamically-built subclasses via ``make_group_class``. The
sampler rebuilds the subclass when it picks a new state;
downstream passes read these attrs without indexing into a
separate ``fusion_groups`` list.
"""

from typing import Any, ClassVar

from nkigym.ops.base import NKIOp


class NKIGroup(NKIOp):
    """Composite that contains an ordered list of children sharing a loop nest.

    Placeholder class attributes below — ``make_group_class``
    produces concrete subclasses per composite instance. The base
    class is never itself instantiated into a built ``OpGraph``.

    State attributes carried on each dynamically-built subclass:

    * ``CHILDREN``: child ``NKIOp`` classes in emission order.
    * ``CHILDREN_TENSORS``: per-child ``(inputs_map, output_names)``
      tuples mirroring ``OpGraph.op_tensors`` entries.
    * ``CHILDREN_KWARGS``: per-child ``{kwarg: literal}`` maps
      mirroring ``OpGraph.op_all_kwargs`` entries.
    * ``DIM_ORDER``: outer-to-inner loop order for this group.
    * ``BUFFER_DEGREES``: per-``(buffer_kind, tensor, dim)`` degree.
    * ``TENSOR_PLACEMENTS``: per-``(buffer_kind, tensor, dim)`` tier.
    """

    NAME: ClassVar[str] = "group"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset()
    CHILDREN: ClassVar[tuple[type[NKIOp], ...]] = ()
    CHILDREN_TENSORS: ClassVar[tuple[tuple[dict[str, str], tuple[str, ...]], ...]] = ()
    CHILDREN_KWARGS: ClassVar[tuple[dict[str, str], ...]] = ()
    DIM_ORDER: ClassVar[tuple[str, ...]] = ()
    BUFFER_DEGREES: ClassVar[dict[tuple[str, str, str], int]] = {}
    TENSOR_PLACEMENTS: ClassVar[dict[tuple[str, str, str], str]] = {}

    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation is not provided — composites are codegen artifacts.

        Matches the ``NKIOnlineFusionChain`` convention: the numpy
        golden path runs the pre-rewrite math function, so the
        composite's ``__call__`` is only reached if something
        accidentally tries to execute the rewritten graph in Python.
        Raising keeps that failure explicit.
        """
        raise NotImplementedError(
            f"{self.NAME} is a codegen-only composite; run the pre-rewrite math function for a numpy reference."
        )

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Composites bypass the standard ISA formatter — render path handles children directly."""
        raise NotImplementedError(f"{cls.NAME} renders through codegen composite path, not format_isa_call.")


def make_group_class(
    *,
    label: str,
    children: tuple[type[NKIOp], ...],
    children_tensors: tuple[tuple[dict[str, str], tuple[str, ...]], ...],
    children_kwargs: tuple[dict[str, str], ...],
    operand_axes: dict[str, tuple[str, ...]],
    output_axes: dict[str, tuple[str, ...]],
    blocking_axes: frozenset[str],
    tile_limits: dict[str, int],
    input_locs: dict[str, str],
    float32_kwargs: frozenset[str],
    isa_loc: str = "sbuf",
    psum_dtype: str | None = None,
    dim_order: tuple[str, ...] = (),
    buffer_degrees: dict[tuple[str, str, str], int] | None = None,
    tensor_placements: dict[tuple[str, str, str], str] | None = None,
    base: type["NKIGroup"] | None = None,
) -> type["NKIGroup"]:
    """Return a fresh subclass of ``NKIGroup`` (or ``base``) for one composite instance.

    The returned subclass carries concrete values on all the
    ``ClassVar`` state attrs. Rebuild it any time the sampler
    chooses new ``dim_order`` / ``buffer_degrees`` /
    ``tensor_placements`` for the same composite shape.

    Args:
        label: Suffix for the subclass ``__name__``; for debugging.
        children: Ordered child ``NKIOp`` classes.
        children_tensors: Per-child ``(inputs, outputs)`` mirroring
            ``OpGraph.op_tensors`` entries. Output tuples are
            immutable to keep the subclass hashable-safe.
        children_kwargs: Per-child kwargs mirroring
            ``OpGraph.op_all_kwargs``.
        operand_axes: Composite-level external input axes.
        output_axes: Composite-level external output axes.
        blocking_axes: Union of children's blocking axes.
        tile_limits: Composite-level tile limits (union of
            children's entries; children sharing an axis must
            agree).
        input_locs: Per-role memory requirement for external
            inputs.
        float32_kwargs: Kwarg names that must be promoted to
            float32 (union of children's entries).
        isa_loc: ``"sbuf"`` / ``"psum"`` / ``"hbm"`` — where the
            composite's external output lands.
        psum_dtype: PSUM dtype override if ``isa_loc == "psum"``.
        dim_order: Sampler-chosen outer-to-inner loop order.
        buffer_degrees: Sampler-chosen per-buffer degrees.
        tensor_placements: Sampler-chosen per-buffer tiers.
        base: ``NKIGroup`` subclass to extend. Defaults to
            ``NKIGroup``. Pass a specialized subclass (e.g.
            ``NKIOnlineFusionChain``) to retain its extra specs.

    Returns:
        Dynamically-built ``NKIGroup`` subclass.
    """
    parent = base if base is not None else NKIGroup
    attrs: dict[str, Any] = {
        "NAME": parent.NAME if parent.NAME else "group",
        "OPERAND_AXES": dict(operand_axes),
        "OUTPUT_AXES": dict(output_axes),
        "BLOCKING_AXES": blocking_axes,
        "TILE_LIMITS": dict(tile_limits),
        "ISA_LOC": isa_loc,
        "PSUM_DTYPE": psum_dtype,
        "INPUT_LOCS": dict(input_locs),
        "FLOAT32_KWARGS": float32_kwargs,
        "CHILDREN": tuple(children),
        "CHILDREN_TENSORS": tuple((dict(ins), tuple(outs)) for ins, outs in children_tensors),
        "CHILDREN_KWARGS": tuple(dict(kw) for kw in children_kwargs),
        "DIM_ORDER": tuple(dim_order),
        "BUFFER_DEGREES": dict(buffer_degrees) if buffer_degrees is not None else {},
        "TENSOR_PLACEMENTS": dict(tensor_placements) if tensor_placements is not None else {},
    }
    return type(f"NKIGroup_{label}", (parent,), attrs)
