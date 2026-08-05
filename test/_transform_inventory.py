"""Test-owned source inventory for public transform implementations."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransformMetric:
    """Source location and size for one public transform class."""

    name: str
    module: str
    class_lines: int
    module_lines: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "module": self.module,
            "class_lines": self.class_lines,
            "module_lines": self.module_lines,
        }


@dataclass(frozen=True)
class _ClassRecord:
    """Parsed class metadata used to resolve transform inheritance."""

    metric: TransformMetric
    bases: tuple[str, ...]
    node: ast.ClassDef


@dataclass(frozen=True)
class TransformAPIInspection:
    """API-contract result for one public transform."""

    name: str
    module: str
    violations: tuple[str, ...]


def _base_name(expression: ast.expr) -> str:
    """Return the terminal name of one class base expression."""
    base = expression.value if isinstance(expression, ast.Subscript) else expression
    if isinstance(base, ast.Name):
        name = base.id
    elif isinstance(base, ast.Attribute):
        name = base.attr
    else:
        name = ""
    return name


def _class_records(transforms_directory: Path) -> tuple[_ClassRecord, ...]:
    """Parse public transform modules without importing candidate code."""
    records: list[_ClassRecord] = []
    for path in sorted(transforms_directory.glob("*.py")):
        if path.name.startswith("_") or path.name == "base.py":
            continue
        source = path.read_text(encoding="utf-8")
        module_lines = len(source.splitlines())
        module = ast.parse(source, filename=str(path))
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                if node.end_lineno is None:
                    raise ValueError(f"AST has no ending line for class {node.name} in {path}")
                metric = TransformMetric(
                    name=node.name,
                    module=path.name,
                    class_lines=node.end_lineno - node.lineno + 1,
                    module_lines=module_lines,
                )
                records.append(
                    _ClassRecord(metric=metric, bases=tuple(_base_name(base) for base in node.bases), node=node)
                )
    return tuple(records)


def _public_transform_records(worktree: Path) -> tuple[_ClassRecord, ...]:
    """Return every public concrete Transform subclass in the worktree."""
    transforms_directory = worktree / "nkigym/src/nkigym/transforms"
    if not transforms_directory.is_dir():
        raise ValueError(f"transform directory does not exist: {transforms_directory}")
    records = _class_records(transforms_directory)
    transform_names = {"Transform"}
    changed = True
    while changed:
        previous_count = len(transform_names)
        transform_names.update(
            record.metric.name for record in records if any(base in transform_names for base in record.bases)
        )
        changed = len(transform_names) != previous_count
    records = tuple(
        record for record in records if record.metric.name in transform_names and not record.metric.name.startswith("_")
    )
    return records


def inspect_transforms(worktree: Path) -> tuple[TransformMetric, ...]:
    """Return source-size metrics for every public transform."""
    metrics = tuple(record.metric for record in _public_transform_records(worktree))
    return metrics


def _method_definitions(node: ast.ClassDef, name: str) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    """Return direct definitions of one public API method."""
    definitions = tuple(
        statement
        for statement in node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == name
    )
    return definitions


def _terminal_name(expression: ast.expr | None) -> str:
    """Return the terminal identifier from a simple expression."""
    name = ""
    if isinstance(expression, ast.Name):
        name = expression.id
    elif isinstance(expression, ast.Attribute):
        name = expression.attr
    return name


def _annotation_text(annotation: ast.expr | None) -> str:
    """Render an annotation for a stable diagnostic."""
    text = "<missing>" if annotation is None else ast.unparse(annotation)
    return text


def _list_item_annotation(expression: ast.expr | None) -> ast.expr | None:
    """Return the item annotation from ``list[T]``."""
    item: ast.expr | None = None
    if isinstance(expression, ast.Subscript) and _terminal_name(expression.value) == "list":
        item = expression.slice
    return item


def _signature_violations(
    transform_name: str, method: ast.FunctionDef | ast.AsyncFunctionDef, expected_parameters: tuple[str, ...]
) -> list[str]:
    """Validate the common positional method shape."""
    violations: list[str] = []
    arguments = method.args
    actual_parameters = tuple(argument.arg for argument in (*arguments.posonlyargs, *arguments.args))
    if isinstance(method, ast.AsyncFunctionDef):
        violations.append(f"{transform_name}.{method.name} must be synchronous")
    if arguments.posonlyargs:
        violations.append(f"{transform_name}.{method.name} must not use positional-only parameters")
    if actual_parameters != expected_parameters:
        violations.append(
            f"{transform_name}.{method.name} parameters must be {expected_parameters}; got {actual_parameters}"
        )
    if arguments.vararg is not None or arguments.kwarg is not None or arguments.kwonlyargs:
        violations.append(f"{transform_name}.{method.name} must not use variadic or keyword-only parameters")
    if arguments.defaults or any(default is not None for default in arguments.kw_defaults):
        violations.append(f"{transform_name}.{method.name} must not define parameter defaults")
    decorator_names = {_terminal_name(decorator) for decorator in method.decorator_list}
    if decorator_names & {"classmethod", "staticmethod"}:
        violations.append(f"{transform_name}.{method.name} must be an instance method")
    return violations


def _method_contract(
    record: _ClassRecord, method_name: str
) -> tuple[list[str], ast.FunctionDef | ast.AsyncFunctionDef | None]:
    """Require one direct definition with the standard signature."""
    definitions = _method_definitions(record.node, method_name)
    violations: list[str] = []
    method: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    expected = ("self", "ir") if method_name == "analyze" else ("self", "ir", "option")
    if len(definitions) != 1:
        violations.append(
            f"{record.metric.name} in {record.metric.module} must define exactly one {method_name}; "
            f"found {len(definitions)}"
        )
    else:
        method = definitions[0]
        violations.extend(_signature_violations(record.metric.name, method, expected))
        ir_index = expected.index("ir")
        positional = (*method.args.posonlyargs, *method.args.args)
        if len(positional) > ir_index and _terminal_name(positional[ir_index].annotation) != "KernelIR":
            violations.append(
                f"{record.metric.name}.{method_name} ir must be annotated KernelIR; "
                f"got {_annotation_text(positional[ir_index].annotation)}"
            )
    return violations, method


def _return_contract(
    record: _ClassRecord,
    analyze: ast.FunctionDef | ast.AsyncFunctionDef | None,
    apply: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> tuple[str, ...]:
    """Validate option and return annotations shared by both methods."""
    violations: list[str] = []
    analyze_option: ast.expr | None = None
    if analyze is not None:
        analyze_option = _list_item_annotation(analyze.returns)
        if analyze_option is None:
            violations.append(
                f"{record.metric.name}.analyze must return list[TransformOption subtype]; "
                f"got {_annotation_text(analyze.returns)}"
            )
    if apply is not None:
        positional = (*apply.args.posonlyargs, *apply.args.args)
        option_annotation = positional[2].annotation if len(positional) > 2 else None
        if option_annotation is None:
            violations.append(f"{record.metric.name}.apply option must have a type annotation")
        elif analyze_option is not None and ast.dump(option_annotation) != ast.dump(analyze_option):
            violations.append(
                f"{record.metric.name} option type differs between analyze and apply: "
                f"{_annotation_text(analyze_option)} != {_annotation_text(option_annotation)}"
            )
        if _terminal_name(apply.returns) != "KernelIR":
            violations.append(f"{record.metric.name}.apply must return KernelIR; got {_annotation_text(apply.returns)}")
    return tuple(violations)


def inspect_transform_api(worktree: Path) -> tuple[TransformAPIInspection, ...]:
    """Inspect every public transform for the uniform analyze/apply contract."""
    inspections: list[TransformAPIInspection] = []
    for record in _public_transform_records(worktree):
        analyze_violations, analyze = _method_contract(record, "analyze")
        apply_violations, apply = _method_contract(record, "apply")
        violations = (*analyze_violations, *apply_violations, *_return_contract(record, analyze, apply))
        inspections.append(
            TransformAPIInspection(name=record.metric.name, module=record.metric.module, violations=violations)
        )
    return tuple(inspections)


__all__ = ["TransformAPIInspection", "TransformMetric", "inspect_transform_api", "inspect_transforms"]
