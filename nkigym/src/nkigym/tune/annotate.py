"""Unified ``Annotate`` atom — keyed-dispatch consolidation of v1's
``MultiBuffer`` + ``SoftwarePipeline`` atoms.

Spec: ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` 4.7.

Registered keys (add new keys via ``register_annotation_key``):

- ``buffer_degree`` — target: SBlock. Value: ``dict[tensor_name, int]``
  where each degree >= 1 and each tensor is not a param.
- ``software_pipeline_depth`` — target: ForNode. Value: ``int`` >= 1.

Consumption happens in ``nkigym.codegen.lowering.inject_annotations``
sub-passes (Task 18).
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from nkigym.codegen.ir import ForNode, KernelModule, SBlock, replace_at_path, resolve_node
from nkigym.tune import AtomLegalityError

_Validator = Callable[[Any, "ForNode | SBlock", KernelModule], bool]

_KEY_VALIDATORS: dict[str, _Validator] = {}


def register_annotation_key(key: str, validator: _Validator) -> None:
    """Register a key + its legality validator. Idempotent."""
    _KEY_VALIDATORS[key] = validator


def _validate_buffer_degree(value: Any, target: "ForNode | SBlock", module: KernelModule) -> bool:
    """``buffer_degree`` requires: target is SBlock; value is ``dict[str, int]``;
    every named tensor exists and is not a param; each degree >= 1.
    """
    result = True
    if not isinstance(target, SBlock):
        result = False
    elif not isinstance(value, dict):
        result = False
    else:
        for tname, degree in value.items():
            if not isinstance(tname, str):
                result = False
                break
            if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
                result = False
                break
            if tname not in module.tensors:
                result = False
                break
            if module.tensors[tname].origin == "param":
                result = False
                break
    return result


def _validate_software_pipeline_depth(value: Any, target: "ForNode | SBlock", module: KernelModule) -> bool:
    """``software_pipeline_depth`` requires: target is ForNode; value is int >= 1."""
    _ = module
    result = True
    if not isinstance(target, ForNode):
        result = False
    elif not isinstance(value, int) or isinstance(value, bool):
        """Reject bool — ``True == 1`` evaluates int-ish but is semantically wrong."""
        result = False
    elif value < 1:
        result = False
    return result


register_annotation_key("buffer_degree", _validate_buffer_degree)
register_annotation_key("software_pipeline_depth", _validate_software_pipeline_depth)


@dataclass(frozen=True)
class Annotate:
    """Attach a keyed annotation to a ForNode or SBlock.

    Attributes:
        target_path: Path to the target in ``module.body``.
        key: Registered annotation key.
        value: Serializable value matching the key's contract.
    """

    target_path: tuple[int, ...]
    key: str
    value: Any

    def is_legal(self, module: KernelModule) -> bool:
        """Dispatch to the key's registered validator.

        Unknown keys are illegal.
        """
        target = resolve_node(module.body, self.target_path)
        result = False
        if target is not None:
            validator = _KEY_VALIDATORS.get(self.key)
            if validator is not None:
                result = validator(self.value, target, module)
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Attach annotation to target; return a new module with replaced target.

        Raises:
            AtomLegalityError: when ``is_legal`` returns False.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"Annotate.apply: illegal {self!r}")
        target = resolve_node(module.body, self.target_path)
        assert target is not None
        new_annotations = dict(target.annotations)
        new_annotations[self.key] = self.value
        new_target: ForNode | SBlock
        if isinstance(target, SBlock):
            new_target = SBlock(
                iter_vars=list(target.iter_vars),
                reads=dict(target.reads),
                writes=dict(target.writes),
                reads_writes=dict(target.reads_writes),
                body=list(target.body),
                annotations=new_annotations,
            )
        else:
            new_target = ForNode(
                iter_var=target.iter_var, children=list(target.children), name=target.name, annotations=new_annotations
            )
        new_body = replace_at_path(module.body, self.target_path, new_target)
        return replace(module, body=new_body)


def enumerate_annotate_atoms(module: KernelModule) -> list[Annotate]:
    """Emit every legal Annotate instance across registered keys.

    Atoms for ``buffer_degree`` enumerate ``degree in {2, 3, 4}`` on each
    non-param alloc SBlock's owned tensor. Atoms for
    ``software_pipeline_depth`` enumerate ``depth in {2, 3}`` on each
    ForNode.
    """
    atoms: list[Annotate] = []
    atoms.extend(_enumerate_buffer_degree(module))
    atoms.extend(_enumerate_software_pipeline_depth(module))
    return atoms


def _enumerate_buffer_degree(module: KernelModule) -> list[Annotate]:
    """For every alloc SBlock at the forest root, emit ``buffer_degree`` atoms."""
    atoms: list[Annotate] = []
    for i, root in enumerate(module.body):
        if not isinstance(root, SBlock):
            continue
        alloc_call = next((c for c in root.body if c.op_cls.__name__ == "NKIAlloc"), None)
        if alloc_call is None:
            continue
        tname = alloc_call.kwargs.get("tensor_name")
        if tname is None or tname not in module.tensors:
            continue
        if module.tensors[tname].origin == "param":
            continue
        for degree in (2, 3, 4):
            atom = Annotate(target_path=(i,), key="buffer_degree", value={tname: degree})
            if atom.is_legal(module):
                atoms.append(atom)
    return atoms


def _enumerate_software_pipeline_depth(module: KernelModule) -> list[Annotate]:
    """For every ForNode, emit ``software_pipeline_depth`` atoms for depths 2 and 3."""
    atoms: list[Annotate] = []

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        if isinstance(node, ForNode):
            for depth in (2, 3):
                atom = Annotate(target_path=path, key="software_pipeline_depth", value=depth)
                if atom.is_legal(module):
                    atoms.append(atom)
            for i, child in enumerate(node.children):
                walk(child, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms
