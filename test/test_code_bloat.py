"""Deterministic source-size and public-transform API limits."""

from __future__ import annotations

from pathlib import Path

from _transform_inventory import inspect_transform_api, inspect_transforms

MAX_PUBLIC_TRANSFORMS = 25
TRANSFORM_FILE_LINE_LIMIT = 1000
MAX_IR_IMPLEMENTATION_LINES = 5000
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _transform_file_violations() -> tuple[list[str], int, int]:
    """Require one public transform per bounded source file."""
    metrics = inspect_transforms(REPOSITORY_ROOT)
    violations: list[str] = []
    if len(metrics) > MAX_PUBLIC_TRANSFORMS:
        violations.append(f"public transform count is {len(metrics)}, limit {MAX_PUBLIC_TRANSFORMS}")

    by_module: dict[str, list[str]] = {}
    for metric in metrics:
        by_module.setdefault(metric.module, []).append(metric.name)

    transforms_directory = REPOSITORY_ROOT / "nkigym/src/nkigym/transforms"
    source_files = {
        path.name: path for path in transforms_directory.glob("*.py") if path.name not in {"__init__.py", "base.py"}
    }
    line_counts: list[int] = []
    for module in sorted(set(source_files) | set(by_module)):
        transform_names = by_module.get(module, [])
        if len(transform_names) != 1:
            names = ", ".join(transform_names) or "none"
            violations.append(
                f"{module} defines {len(transform_names)} public transforms ({names}); exactly one is required"
            )
        path = source_files.get(module)
        if path is None:
            violations.append(f"{module} does not exist directly under {transforms_directory}")
            continue
        lines = len(path.read_text(encoding="utf-8").splitlines())
        line_counts.append(lines)
        if lines >= TRANSFORM_FILE_LINE_LIMIT:
            violations.append(
                f"{module} has {lines} lines; each transform file must have fewer than " f"{TRANSFORM_FILE_LINE_LIMIT}"
            )
    largest_file = max(line_counts, default=0)
    return violations, len(metrics), largest_file


def _transform_api_violations() -> list[str]:
    """Return all public analyze/apply contract violations."""
    inspections = inspect_transform_api(REPOSITORY_ROOT)
    violations = [violation for inspection in inspections for violation in inspection.violations]
    return violations


def _ir_size_violation() -> tuple[str | None, int]:
    """Return the total IR implementation size and any limit violation."""
    ir_directory = REPOSITORY_ROOT / "nkigym/src/nkigym/ir"
    paths = sorted(ir_directory.rglob("*.py"))
    total_lines = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in paths)
    violation = None
    if total_lines > MAX_IR_IMPLEMENTATION_LINES:
        violation = f"IR implementation has {total_lines} lines, limit {MAX_IR_IMPLEMENTATION_LINES}"
    return violation, total_lines


def test_source_size_and_public_transform_contracts() -> None:
    """Source growth stays bounded and every public transform uses analyze/apply."""
    file_violations, transform_count, largest_transform_file = _transform_file_violations()
    api_violations = _transform_api_violations()
    ir_violation, ir_lines = _ir_size_violation()
    violations = [*file_violations, *api_violations]
    if ir_violation is not None:
        violations.append(ir_violation)
    print(
        f"public_transforms={transform_count} largest_transform_file={largest_transform_file} " f"ir_lines={ir_lines}",
        flush=True,
    )
    assert not violations, "\n".join(violations)
