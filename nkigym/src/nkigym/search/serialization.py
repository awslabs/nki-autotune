"""Compact process-transfer serialization for immutable rollout states."""

from __future__ import annotations

import os
import pickle
import zlib
from collections.abc import Callable
from dataclasses import replace
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ParamSpec, Protocol, TypeVar, cast
from weakref import WeakKeyDictionary, finalize

from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.search.state_facts import OperationFacts, compute_operation_facts

_KernelFields = tuple[str, list[str], tuple[str, ...], KernelTree, Dependency, dict[str, Buffer]]
_KernelFactory = Callable[[str, list[str], tuple[str, ...], KernelTree, Dependency, dict[str, Buffer]], object]
_SERIALIZED_KERNELS: WeakKeyDictionary[KernelTree, str] = WeakKeyDictionary()
_SERIALIZED_FACTS: WeakKeyDictionary[KernelTree, OperationFacts] = WeakKeyDictionary()
_CANONICAL_TREES: WeakKeyDictionary[KernelTree, bool] = WeakKeyDictionary()
_INHERITED_ANALYSES: WeakKeyDictionary[KernelTree, dict[str, str]] = WeakKeyDictionary()
_AnalysisValue = TypeVar("_AnalysisValue")
_AnalysisParams = ParamSpec("_AnalysisParams")


class KernelEnvelope(Protocol):
    """Fields required to reconstruct a kernel envelope."""

    func_name: str
    param_names: list[str]
    return_names: tuple[str, ...]
    tree: KernelTree
    dependency: Dependency
    param_buffers: dict[str, Buffer]

    def all_buffers(self) -> dict[str, Buffer]:
        """Return every parameter and local buffer."""
        ...


def _canonicalize_kernel_values(ir: KernelEnvelope) -> None:
    """Share equal immutable tree and dependency values before serialization."""
    regions: dict[BufferRegion, BufferRegion] = {}
    regions_by_id: dict[int, BufferRegion] = {}
    iter_vars: dict[IterVar, IterVar] = {}
    iter_vars_by_id: dict[int, IterVar] = {}
    loops: dict[ForNode, ForNode] = {}
    loops_by_id: dict[int, ForNode] = {}
    extents: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}
    extents_by_id: dict[int, dict[str, int]] = {}

    def region(value: BufferRegion) -> BufferRegion:
        """Return one shared region value."""
        result = regions_by_id.get(id(value))
        if result is None:
            result = regions.setdefault(value, value)
            regions_by_id[id(value)] = result
        return result

    def iter_var(value: IterVar) -> IterVar:
        """Return one shared iteration-variable value."""
        result = iter_vars_by_id.get(id(value))
        if result is None:
            result = iter_vars.setdefault(value, value)
            iter_vars_by_id[id(value)] = result
        return result

    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ForNode):
            shared_loop = loops_by_id.get(id(data))
            if shared_loop is None:
                shared_loop = loops.setdefault(data, data)
                loops_by_id[id(data)] = shared_loop
            if shared_loop is not data:
                ir.tree.graph.nodes[nid]["data"] = shared_loop
        elif isinstance(data, BlockNode):
            shared_iter_vars = tuple(iter_var(value) for value in data.iter_vars)
            shared_reads = tuple(region(value) for value in data.reads)
            shared_writes = tuple(region(value) for value in data.writes)
            changed = any(
                shared is not original
                for shared, original in (
                    *zip(shared_iter_vars, data.iter_vars, strict=True),
                    *zip(shared_reads, data.reads, strict=True),
                    *zip(shared_writes, data.writes, strict=True),
                )
            )
            if changed:
                ir.tree.graph.nodes[nid]["data"] = replace(
                    data, iter_vars=shared_iter_vars, reads=shared_reads, writes=shared_writes
                )
        elif isinstance(data, ISANode):
            bindings = {slot: region(value) for slot, value in data.operand_bindings.items()}
            if any(bindings[slot] is not value for slot, value in data.operand_bindings.items()):
                ir.tree.graph.nodes[nid]["data"] = replace(data, operand_bindings=bindings)
    for nid in ir.dependency.graph.nodes:
        info = ir.dependency.graph.nodes[nid]["info"]
        shared_extents = extents_by_id.get(id(info.extents))
        if shared_extents is None:
            extent_key = tuple(sorted(info.extents.items()))
            shared_extents = extents.setdefault(extent_key, info.extents)
            extents_by_id[id(info.extents)] = shared_extents
        shared_reads = tuple(region(value) for value in info.read_regions)
        shared_writes = tuple(region(value) for value in info.write_regions)
        changed = shared_extents is not info.extents or any(
            shared is not original
            for shared, original in (
                *zip(shared_reads, info.read_regions, strict=True),
                *zip(shared_writes, info.write_regions, strict=True),
            )
        )
        if changed:
            ir.dependency.graph.nodes[nid]["info"] = replace(
                info, read_regions=shared_reads, write_regions=shared_writes, extents=shared_extents
            )


def _restore_kernel_ir(
    factory: _KernelFactory, snapshot_path: str, facts: OperationFacts, inherited: dict[str, str]
) -> object:
    """Restore one independent kernel envelope from its shared snapshot."""
    fields = cast(_KernelFields, pickle.loads(zlib.decompress(Path(snapshot_path).read_bytes())))
    _CANONICAL_TREES[fields[3]] = True
    result = factory(*fields)
    setattr(result, "_operation_facts", facts)
    setattr(result, "_analysis_snapshot_path", snapshot_path)
    setattr(result, "_inherited_analysis_paths", inherited)
    return result


def _analysis_path(snapshot_path: str, name: str) -> Path:
    """Return the sidecar path for one analysis result."""
    return Path(f"{snapshot_path}.analysis-{name}.pkl")


def _cleanup_snapshot(snapshot_path: str) -> None:
    """Delete one serialized state and every analysis result it owns."""
    path = Path(snapshot_path)
    path.unlink(missing_ok=True)
    for sidecar in path.parent.glob(f"{path.name}.analysis-*.pkl"):
        sidecar.unlink(missing_ok=True)


def _load_analysis_result(ir: KernelEnvelope, name: str) -> tuple[object, ...] | None:
    """Load an inherited analysis result without recomputing it."""
    inherited = cast(dict[str, str], getattr(ir, "_inherited_analysis_paths", {}))
    path = inherited.get(name)
    return None if path is None else cast(tuple[object, ...], pickle.loads(Path(path).read_bytes()))


def _store_analysis_result(ir: KernelEnvelope, name: str, values: tuple[object, ...]) -> None:
    """Atomically store one worker analysis for a possible metadata rewrite."""
    snapshot_path = cast(str | None, getattr(ir, "_analysis_snapshot_path", None))
    if snapshot_path is None:
        return
    path = _analysis_path(snapshot_path, name)
    with NamedTemporaryFile(prefix=f"{path.name}.", dir=path.parent, delete=False) as output:
        output.write(pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL))
        temporary = output.name
    os.replace(temporary, path)


def inherited_analysis(
    name: str,
) -> Callable[[Callable[_AnalysisParams, list[_AnalysisValue]]], Callable[_AnalysisParams, list[_AnalysisValue]]]:
    """Cache an analysis result inherited only by explicitly safe rewrites."""

    def decorate(
        analyze: Callable[_AnalysisParams, list[_AnalysisValue]],
    ) -> Callable[_AnalysisParams, list[_AnalysisValue]]:
        """Wrap one transform analysis with worker-side sidecar reuse."""

        @wraps(analyze)
        def wrapped(*args: _AnalysisParams.args, **kwargs: _AnalysisParams.kwargs) -> list[_AnalysisValue]:
            """Load an inherited result or run and store the analysis."""
            positional = cast(tuple[object, ...], args)
            keywords = cast(dict[str, object], kwargs)
            ir = cast(KernelEnvelope, positional[1] if len(positional) > 1 else keywords["ir"])
            cached = _load_analysis_result(ir, name)
            values = tuple(analyze(*args, **kwargs)) if cached is None else cast(tuple[_AnalysisValue, ...], cached)
            _store_analysis_result(ir, name, cast(tuple[object, ...], values))
            return list(values)

        return wrapped

    return decorate


def inherit_analysis_result(source: KernelEnvelope, target: KernelEnvelope, name: str) -> None:
    """Hard-link one result into a metadata-only successor state's lifetime."""
    snapshot_path = _SERIALIZED_KERNELS.get(source.tree)
    if snapshot_path is None or not (source_path := _analysis_path(snapshot_path, name)).is_file():
        return
    with NamedTemporaryFile(prefix="nkigym-analysis-", suffix=".pkl", dir=source_path.parent, delete=False) as output:
        target_path = Path(output.name)
    target_path.unlink()
    os.link(source_path, target_path)
    inherited = dict(_INHERITED_ANALYSES.get(target.tree, {}))
    inherited[name] = str(target_path)
    _INHERITED_ANALYSES[target.tree] = inherited
    finalize(target.tree, target_path.unlink, missing_ok=True)


def inherit_canonical_values(source: KernelTree, target: KernelTree) -> None:
    """Mark a rewrite clone as sharing its source's canonical immutable values."""
    if source in _CANONICAL_TREES:
        _CANONICAL_TREES[target] = True


def reduce_kernel_ir(ir: KernelEnvelope, protocol: int) -> tuple[Callable[..., object], tuple[object, ...]]:
    """Return one cached compressed reducer for repeated process-pool submissions."""
    snapshot_path = _SERIALIZED_KERNELS.get(ir.tree)
    facts = _SERIALIZED_FACTS.get(ir.tree)
    if snapshot_path is None:
        if ir.tree not in _CANONICAL_TREES:
            _canonicalize_kernel_values(ir)
            _CANONICAL_TREES[ir.tree] = True
        fields: _KernelFields = (
            ir.func_name,
            ir.param_names,
            ir.return_names,
            ir.tree,
            ir.dependency,
            ir.param_buffers,
        )
        facts = compute_operation_facts(ir.tree)
        payload = zlib.compress(pickle.dumps(fields, protocol=protocol), level=1)
        directory = "/dev/shm" if Path("/dev/shm").is_dir() else None
        with NamedTemporaryFile(prefix="nkigym-ir-", suffix=".pklz", dir=directory, delete=False) as snapshot:
            snapshot.write(payload)
            snapshot_path = snapshot.name
        _SERIALIZED_KERNELS[ir.tree] = snapshot_path
        _SERIALIZED_FACTS[ir.tree] = facts
        finalize(ir.tree, _cleanup_snapshot, snapshot_path)
    if facts is None:
        raise RuntimeError("serialized kernel facts are missing")
    factory = cast(_KernelFactory, type(ir))
    return _restore_kernel_ir, (factory, snapshot_path, facts, dict(_INHERITED_ANALYSES.get(ir.tree, {})))
