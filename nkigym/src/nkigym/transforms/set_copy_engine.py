"""Select a copy execution engine or its DMA descriptor generator."""

from dataclasses import dataclass, replace
from typing import Literal

from nkigym.ir import Const, KernelIR
from nkigym.ir.buffer_placement import layout_satisfies_alignment
from nkigym.ir.dependency_rebind import rebind_unchanged_dependency
from nkigym.ir.tree import ISANode
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.strided_copy import NKIStridedCopy
from nkigym.ops.strided_tensor_copy import NKIStridedTensorCopy
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.ops.vector_transpose import NKIVectorTranspose
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import (
    finalize_rewrite,
    is_canonical_block,
    owning_block,
    replace_buffer,
    single_leaf,
)
from nkigym.transforms.helper.transpose_pattern import TransposeChain, match_transpose_chain

CopyEngine = Literal["vector", "scalar", "dma", "tensor", "hwdge"]
_ENGINES: tuple[CopyEngine, ...] = ("vector", "scalar", "dma", "tensor", "hwdge")
_FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32"})


@dataclass(frozen=True)
class SetCopyEngineOption(TransformOption):
    """Identify one copy and its requested execution or descriptor engine."""

    isa_nid: int
    engine: CopyEngine


class SetCopyEngine(Transform[SetCopyEngineOption]):
    """Choose a compatible engine without changing copied or transposed values."""

    def analyze(self, ir: KernelIR) -> list[SetCopyEngineOption]:
        """Offer copy, transpose, and DMA descriptor-engine choices."""
        options: list[SetCopyEngineOption] = []
        for nid in ir.tree.leaves():
            for engine in _ENGINES:
                option = SetCopyEngineOption(nid, engine)
                if (
                    _eligible(ir, option)
                    or _dma_descriptor_engine(ir, option) is not None
                    or _strided_copy(ir, option) is not None
                    or _sbuf_transpose(ir, option) is not None
                    or _transpose_match(ir, option)[0] is not None
                ):
                    options.append(option)
        return options

    def apply(self, ir: KernelIR, option: SetCopyEngineOption) -> KernelIR:
        """Recheck and change one engine with its required storage representation."""
        transpose, reverse = _transpose_match(ir, option)
        strided = _strided_copy(ir, option)
        descriptor = _dma_descriptor_engine(ir, option)
        sbuf_transpose = _sbuf_transpose(ir, option)
        if (
            transpose is None
            and strided is None
            and descriptor is None
            and sbuf_transpose is None
            and not _eligible(ir, option)
        ):
            raise TransformLegalityError(f"illegal SetCopyEngine option: {option}")
        result = copy_for_rewrite(ir)
        if transpose is None:
            node = result.tree.isa(option.isa_nid)
            replacement = descriptor if descriptor is not None else strided if strided is not None else sbuf_transpose
            result.tree.graph.nodes[option.isa_nid]["data"] = (
                replacement
                if replacement is not None
                else replace(node, kwargs={**node.kwargs, "engine": option.engine})
            )
            result.dependency = rebind_unchanged_dependency(ir.dependency, result.tree)
        else:
            match, _reverse = _transpose_match(result, option)
            if match is None:
                raise AssertionError("transpose engine choice disappeared after copying")
            _apply_transpose(result, match, reverse)
            finalize_rewrite(result)
        return result


def _sbuf_transpose(ir: KernelIR, option: SetCopyEngineOption) -> ISANode | None:
    """Choose one direct SBUF transpose encoding while retaining its tile and schedule."""
    expected = NKIDMATranspose if option.engine == "vector" else NKIVectorTranspose if option.engine == "dma" else None
    if expected is None or option.isa_nid not in ir.tree.graph:
        return None
    node = ir.tree.data(option.isa_nid)
    if not isinstance(node, ISANode) or node.op_cls is not expected or node.access_patterns:
        return None
    if set(node.kwargs) - {"engine", "name", "no_reorder", "program_ownership"}:
        return None
    source_slot = "src" if expected is NKIDMATranspose else "data"
    if set(node.operand_bindings) != {source_slot, "dst"}:
        return None
    source_region, output_region = (node.operand_bindings[slot] for slot in (source_slot, "dst"))
    source, output = (ir.buffer(region.tensor) for region in (source_region, output_region))
    if (
        source.name == output.name
        or source.location != "sbuf"
        or output.location != "sbuf"
        or source.dtype != output.dtype
        or source.physical_dtype() != "float32"
        or output.physical_dtype() != "float32"
    ):
        return None
    sizes = tuple(tuple(size for _, size in region.ranges) for region in (source_region, output_region))
    if (
        any(len(shape) != 2 for shape in sizes)
        or sizes[0] != sizes[1][::-1]
        or any(not isinstance(size, Const) or not 1 <= size.value <= 32 for shape in sizes for size in shape)
    ):
        return None
    if option.engine == "dma" and not layout_satisfies_alignment(output, 32):
        return None
    kwargs = {key: value for key, value in node.kwargs.items() if key != "engine"}
    if option.engine == "vector":
        kwargs["engine"] = "vector"
    return replace(
        node,
        op_cls=NKIVectorTranspose if option.engine == "vector" else NKIDMATranspose,
        operand_bindings={("data" if option.engine == "vector" else "src"): source_region, "dst": output_region},
        kwargs=kwargs,
    )


def _dma_descriptor_engine(ir: KernelIR, option: SetCopyEngineOption) -> ISANode | None:
    """Choose descriptors without pinning a sequencer; hardware requires matching dtypes."""
    if option.isa_nid not in ir.tree.graph or option.engine not in {"dma", "hwdge"}:
        return None
    node = ir.tree.data(option.isa_nid)
    if not isinstance(node, ISANode) or node.op_cls.NAME != "dma_copy" or node.op_cls.INDIRECT_DMA_MODE is not None:
        return None
    if node.kwargs.get("engine", "unknown") != "unknown":
        return None
    mode = "hwdge" if option.engine == "hwdge" else "unknown"
    if node.kwargs.get("dge_mode", "unknown") == mode:
        return None
    if option.engine == "hwdge":
        buffers = ir.all_buffers()
        dtypes = {
            slot: getattr(node.op_cls, "REINTERPRET_INPUT_DTYPES", {}).get(
                slot, buffers[region.tensor].physical_dtype()
            )
            for slot, region in node.operand_bindings.items()
            if slot in {"src", "dst"}
        }
        if len(dtypes) != 2 or dtypes.get("src") != dtypes.get("dst"):
            return None
    kwargs = {key: value for key, value in node.kwargs.items() if key != "dge_mode"}
    if option.engine != "dma":
        kwargs["dge_mode"] = mode
    return replace(node, kwargs=kwargs)


def _strided_copy(ir: KernelIR, option: SetCopyEngineOption) -> ISANode | None:
    """Select one equivalent SBUF strided-copy encoding without changing views."""
    if option.isa_nid not in ir.tree.graph or option.engine not in {"dma", "vector"}:
        return None
    node = ir.tree.data(option.isa_nid)
    expected = NKIStridedCopy if option.engine == "vector" else NKIStridedTensorCopy
    if not isinstance(node, ISANode) or node.op_cls is not expected or set(node.operand_bindings) != {"src", "dst"}:
        return None
    source, output = (ir.buffer(node.operand_bindings[slot].tensor) for slot in ("src", "dst"))
    if (
        source.location != "sbuf"
        or output.location != "sbuf"
        or source.physical_dtype() not in NKIStridedTensorCopy.INPUT_STORAGE_DTYPES["src"]
        or source.dtype != output.dtype
        or source.physical_dtype() != output.physical_dtype()
    ):
        return None
    kwargs = {key: value for key, value in node.kwargs.items() if key not in {"engine", "dge_mode"}}
    if option.engine == "vector":
        kwargs["engine"] = "vector"
    return replace(node, op_cls=NKIStridedTensorCopy if option.engine == "vector" else NKIStridedCopy, kwargs=kwargs)


def _native_copy_engine(node: ISANode) -> str | None:
    """Resolve the actual copy encoding, including explicit float casts."""
    name, parameters = node.op_cls.NAME, dict(node.kwargs)
    encoder = getattr(node.op_cls, "native_parameters", None)
    if encoder is not None:
        name, parameters = encoder(parameters, frozenset(node.operand_bindings))
    engine = parameters.get("engine", "unknown")
    return engine if name == "tensor_copy" and isinstance(engine, str) else None


def _eligible(ir: KernelIR, option: SetCopyEngineOption) -> bool:
    """Require unchanged floating values, legal memory access, and an honored keyword."""
    if option.engine not in {"vector", "scalar"} or option.isa_nid not in ir.tree.graph:
        return False
    node = ir.tree.data(option.isa_nid)
    if not isinstance(node, ISANode) or getattr(node.op_cls, "REINTERPRET_INPUT_DTYPES", {}):
        return False
    if option.engine not in getattr(node.op_cls, "COPY_ENGINES", ()):
        return False
    current = _native_copy_engine(node)
    if current is None or current == option.engine:
        return False
    buffers = [ir.buffer(region.tensor) for region in node.operand_bindings.values()]
    if not buffers or any(
        buffer.location not in {"sbuf", "psum"} or buffer.physical_dtype() not in _FLOAT_DTYPES for buffer in buffers
    ):
        return False
    updated = replace(node, kwargs={**node.kwargs, "engine": option.engine})
    return _native_copy_engine(updated) == option.engine


def _transpose_match(ir: KernelIR, option: SetCopyEngineOption) -> tuple[TransposeChain | None, bool]:
    """Resolve one effective Tensor Engine or DMA transpose choice."""
    if option.engine not in {"dma", "tensor"} or option.isa_nid not in ir.tree.graph:
        return None, False
    node = ir.tree.data(option.isa_nid)
    expected = NKITranspose if option.engine == "dma" else NKIDMATranspose
    if not isinstance(node, ISANode) or node.op_cls is not expected:
        return None, False
    block = owning_block(ir.tree, option.isa_nid)
    match, reverse = _match_transpose_block(ir, block)
    return (match, reverse) if reverse == (option.engine == "tensor") else (None, False)


def _match_transpose_block(ir: KernelIR, transpose_nid: int) -> tuple[TransposeChain | None, bool]:
    """Return the logical transpose named by ``option``."""
    result: TransposeChain | None = None
    reverse = False
    root_children = ir.tree.children(ir.tree.root)
    if transpose_nid in root_children:
        index = root_children.index(transpose_nid)
        if index + 1 < len(root_children):
            result = match_transpose_chain(ir, transpose_nid, root_children[index + 1], adjacent=True)
            if result is not None:
                source = ir.buffer(result.source)
                candidate = replace(ir.buffer(result.psum), location="sbuf")
                alignment = NKIDMATranspose.OUTPUT_TILE_ALIGNMENT_BYTES["dst"]
                valid = NKIDMATranspose.accepts_input_storage_dtypes({"src": source.physical_dtype()})
                if not valid or not layout_satisfies_alignment(candidate, alignment):
                    result = None
            else:
                result = _match_dma_transpose_chain(ir, transpose_nid, root_children[index + 1])
                reverse = result is not None
    if result is not None and set(ir.tree.isa(result.transpose_leaf).kwargs) - {
        "name",
        "no_reorder",
        "program_ownership",
    }:
        result = None
    return result, reverse


def _match_dma_transpose_chain(ir: KernelIR, transpose_block: int, drain_block: int) -> TransposeChain | None:
    """Return one isolated DMA-transpose and tensor-copy chain."""
    result: TransposeChain | None = None
    if is_canonical_block(ir, transpose_block) and is_canonical_block(ir, drain_block):
        transpose_leaf = single_leaf(ir.tree, transpose_block)
        drain_leaf = single_leaf(ir.tree, drain_block)
        if transpose_leaf is not None and drain_leaf is not None:
            transpose = ir.tree.isa(transpose_leaf)
            drain = ir.tree.isa(drain_leaf)
            if transpose.op_cls is NKIDMATranspose and drain.op_cls is NKITensorCopy:
                source = transpose.operand_bindings["src"].tensor
                intermediate = transpose.operand_bindings["dst"].tensor
                output = drain.operand_bindings["dst"].tensor
                source_buffer = ir.buffer(source)
                intermediate_buffer = ir.buffer(intermediate)
                output_buffer = ir.buffer(output)
                axes = ir.tree.block(transpose_block).axis_map
                drain_axes = ir.tree.block(drain_block).axis_map
                valid = (
                    drain.operand_bindings["src"].tensor == intermediate
                    and source_buffer.shape[::-1] == intermediate_buffer.shape == output_buffer.shape
                    and source_buffer.location == intermediate_buffer.location == output_buffer.location == "sbuf"
                    and source_buffer.dtype == intermediate_buffer.dtype == output_buffer.dtype
                    and NKITranspose.accepts_input_storage_dtypes({"data": source_buffer.physical_dtype()})
                    and source_buffer.physical_dtype() == intermediate_buffer.physical_dtype()
                    and drain_axes.get("P") == axes.get("F")
                    and drain_axes.get("F") == axes.get("P")
                    and set(ir.dependency.touches_by_tensor.get(intermediate, ())) == {transpose_leaf, drain_leaf}
                )
                if valid and isinstance(axes.get("P"), str) and isinstance(axes.get("F"), str):
                    result = TransposeChain(
                        transpose_block=transpose_block,
                        drain_block=drain_block,
                        transpose_leaf=transpose_leaf,
                        drain_leaf=drain_leaf,
                        source=source,
                        psum=intermediate,
                        output=output,
                        source_axes=(axes["P"], axes["F"]),
                    )
    return result


def _apply_transpose(ir: KernelIR, match: TransposeChain, reverse: bool) -> None:
    """Execute one concrete transpose in SBUF while retaining its drain."""
    transpose = ir.tree.isa(match.transpose_leaf)
    source_slot = "src" if reverse else "data"
    source_region = transpose.operand_bindings[source_slot]
    output_region = transpose.operand_bindings["dst"]
    replace_buffer(ir, replace(ir.buffer(match.psum), location="psum" if reverse else "sbuf"))
    ir.tree.graph.nodes[match.transpose_leaf]["data"] = ISANode(
        op_cls=NKITranspose if reverse else NKIDMATranspose,
        operand_bindings={("data" if reverse else "src"): source_region, "dst": output_region},
        kwargs=dict(transpose.kwargs),
    )


__all__ = ["SetCopyEngine", "SetCopyEngineOption"]
