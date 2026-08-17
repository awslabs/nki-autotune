"""Enforce this repository structure and its source limits.

nkigym/src/nkigym/
|-- __init__.py
|-- ir/**/*.py                       at most 3,000 code lines total
|-- codegen/**/*.py                  at most 2,000 code lines total
|-- profile/**/*.py                  at most 2,000 code lines total
|-- search/**/*.py                   at most 2,000 code lines total
|-- synthesis/**/*.py                at most 1,000 code lines total
|-- ops/
|   |-- __init__.py                  at most 100 code lines
|   |-- base.py                      at most 500 code lines
|   `-- <operation>.py               one NKIOp subclass; at most 100 code lines
`-- transforms/                      at most 25 public transforms
    |-- __init__.py
    |-- base.py
    |-- <transform>.py               exactly one public transform per file
    |-- <documentation>.md
    `-- helper/
        |-- __init__.py
        `-- <helper>.py              no public transforms

kernel_library/
|-- __init__.py
`-- nakb_<workload>.py               one direct Python module per workload

Only the files shown above are allowed under ops and transforms. Every transform
Python file must have fewer than 1,000 code lines, and all helper Python files
together must have fewer than 1,000 code lines. Blank lines, comments, and
documentation strings do not count. Public transforms must directly define
typed, synchronous analyze and apply methods. Formatter-control comments are
forbidden because they permit multiple statements to be hidden on one line.
Only the package initializer and direct workload Python modules are allowed
under kernel_library; helper files and subdirectories are forbidden. No Python
source files are allowed outside the documented nkigym implementation
directories.

Required package initializers: nkigym, codegen, ir, ir/arith, ops, profile,
search, synthesis, transforms, and transforms/helper.

Allowed repository imports:

nkigym        -> nkigym
kernel_library -> kernel_library, nkigym
test           -> kernel_library, nkigym

The top-level developer package must not exist.

Exact transform schedules and reproduction traces are allowed only in
kernel_library. Search owns generic iterative refinement infrastructure and
consumes runtime legal actions without importing or constructing concrete
transform options.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

from _transform_inventory import inspect_transform_api, inspect_transforms

MAX_PUBLIC_TRANSFORMS = 25
TRANSFORM_FILE_LINE_LIMIT = 1000
TRANSFORM_HELPER_LINE_LIMIT = 1000
MAX_IR_IMPLEMENTATION_LINES = 3000
MAX_CODEGEN_IMPLEMENTATION_LINES = 2000
MAX_PROFILE_IMPLEMENTATION_LINES = 2000
MAX_SEARCH_IMPLEMENTATION_LINES = 2000
MAX_SYNTHESIS_IMPLEMENTATION_LINES = 1000
OP_FILE_LINE_LIMIT = 100
OP_BASE_FILE_LINE_LIMIT = 500
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_IMPORT_ROOTS = {"developer", "kernel_library", "nkigym"}
FORMATTER_TARGETS = ("nkigym/src", "kernel_library", "test")
NKIGYM_IMPLEMENTATION_DIRECTORIES = frozenset({"codegen", "ir", "ops", "profile", "search", "synthesis", "transforms"})
REQUIRED_PACKAGE_INITIALIZERS = frozenset(
    {
        "nkigym/src/nkigym/__init__.py",
        "nkigym/src/nkigym/codegen/__init__.py",
        "nkigym/src/nkigym/ir/__init__.py",
        "nkigym/src/nkigym/ir/arith/__init__.py",
        "nkigym/src/nkigym/ops/__init__.py",
        "nkigym/src/nkigym/profile/__init__.py",
        "nkigym/src/nkigym/search/__init__.py",
        "nkigym/src/nkigym/synthesis/__init__.py",
        "nkigym/src/nkigym/transforms/__init__.py",
        "nkigym/src/nkigym/transforms/helper/__init__.py",
    }
)
TRANSFORM_INFRASTRUCTURE_FILES = frozenset({"__init__.py", "base.py"})
TRANSFORM_HELPER_DIRECTORY = "helper"
FORMATTER_CONTROL_PATTERN = re.compile(
    r"#\s*(?:fmt\s*:\s*(?:off|on|skip)|yapf\s*:\s*(?:disable|enable))(?:\s|;|$)", re.IGNORECASE
)
SourcePosition = tuple[int, int]
SourceSpan = tuple[SourcePosition, SourcePosition]


def _character_column(source_lines: list[str], line: int, byte_column: int) -> int:
    """Convert one AST UTF-8 byte offset to a tokenizer character offset."""
    prefix = source_lines[line - 1].encode("utf-8")[:byte_column]
    column = len(prefix.decode("utf-8"))
    return column


def _documentation_spans(module: ast.Module, source_lines: list[str]) -> tuple[SourceSpan, ...]:
    """Return source spans occupied by docstrings and string-block comments."""
    spans: list[SourceSpan] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if node.end_lineno is None or node.end_col_offset is None:
                raise ValueError("documentation expression has no ending source position")
            start = (node.lineno, _character_column(source_lines, node.lineno, node.col_offset))
            end = (node.end_lineno, _character_column(source_lines, node.end_lineno, node.end_col_offset))
            spans.append((start, end))
    return tuple(spans)


def _python_code_lines(path: Path) -> int:
    """Count physical lines containing Python code, excluding comments and documentation."""
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    documentation_spans = _documentation_spans(module, source.splitlines(keepends=True))
    ignored_token_types = {
        tokenize.COMMENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }
    code_lines: set[int] = set()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        is_documentation = any(
            span_start <= token.start and token.end <= span_end for span_start, span_end in documentation_spans
        )
        is_separator = token.type == tokenize.OP and token.string == ";"
        if token.type not in ignored_token_types and not is_documentation and not is_separator:
            code_lines.update(range(token.start[0], token.end[0] + 1))
    return len(code_lines)


def _formatter_control_comments(path: Path) -> tuple[tuple[int, str], ...]:
    """Return formatter-control comments and their source lines."""
    source = path.read_text(encoding="utf-8")
    comments = tuple(
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and FORMATTER_CONTROL_PATTERN.match(token.string.strip())
    )
    return comments


def _formatter_control_violations(directory: Path) -> list[str]:
    """Reject formatter escapes that can conceal code from physical line limits."""
    violations: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        relative_path = path.relative_to(REPOSITORY_ROOT)
        for line, comment in _formatter_control_comments(path):
            violations.append(f"{relative_path}:{line} uses forbidden formatter control comment {comment!r}")
    return violations


def _repository_format_violations() -> list[str]:
    """Run Black and isort checks before inspecting repository structure."""
    commands = (("black", "--check"), ("isort", "--check-only"))
    violations: list[str] = []
    for module, check_flag in commands:
        completed = subprocess.run(
            (sys.executable, "-m", module, check_flag, *FORMATTER_TARGETS), cwd=REPOSITORY_ROOT, check=False
        )
        if completed.returncode != 0:
            violations.append(f"{module} formatting check exited with status {completed.returncode}")
    return violations


def _repository_files(directory: Path) -> tuple[Path, ...]:
    """Return tracked and non-ignored files below a repository directory."""
    relative_directory = directory.relative_to(REPOSITORY_ROOT)
    completed = subprocess.run(
        ("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", relative_directory.as_posix()),
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    files: list[Path] = []
    for encoded_path in completed.stdout.split(b"\0"):
        if encoded_path:
            absolute_path = REPOSITORY_ROOT / encoded_path.decode()
            if absolute_path.is_file():
                files.append(absolute_path.relative_to(directory))
    return tuple(sorted(files))


def _package_initializer_violations() -> list[str]:
    """Require every intentional Python package to have an initializer."""
    violations: list[str] = []
    for relative_path in sorted(REQUIRED_PACKAGE_INITIALIZERS):
        if not (REPOSITORY_ROOT / relative_path).is_file():
            violations.append(f"{relative_path} is required")
    return violations


def _nkigym_source_layout_violations() -> list[str]:
    """Reject Python source outside the documented implementation directories."""
    source_directory = REPOSITORY_ROOT / "nkigym/src/nkigym"
    violations: list[str] = []
    for relative_path in _repository_files(source_directory):
        is_root_initializer = relative_path == Path("__init__.py")
        is_implementation_file = (
            len(relative_path.parts) > 1 and relative_path.parts[0] in NKIGYM_IMPLEMENTATION_DIRECTORIES
        )
        if relative_path.suffix == ".py" and not is_root_initializer and not is_implementation_file:
            violations.append(
                f"nkigym/src/nkigym/{relative_path.as_posix()} is not allowed; keep Python source under a "
                "documented implementation directory"
            )
    return violations


def _transform_layout_violations(files: tuple[Path, ...]) -> list[str]:
    """Require the exact root-transform and helper-package layout."""
    violations: list[str] = []
    relative_names = {path.as_posix() for path in files}
    required_files = {"base.py"}
    for required_file in sorted(required_files - relative_names):
        violations.append(f"transforms/{required_file} is required")
    for path in files:
        parts = path.parts
        root_python = len(parts) == 1 and path.suffix == ".py"
        root_markdown = len(parts) == 1 and path.suffix == ".md"
        helper_python = len(parts) == 2 and parts[0] == TRANSFORM_HELPER_DIRECTORY and path.suffix == ".py"
        if not root_python and not root_markdown and not helper_python:
            violations.append(
                f"transforms/{path.as_posix()} is not allowed; use a root transform module, root Markdown, "
                "or helper/*.py"
            )
    return violations


def _transform_structure_violations() -> tuple[list[str], int, int, int]:
    """Enforce transform count, file ownership, and source-size limits."""
    transforms_directory = REPOSITORY_ROOT / "nkigym/src/nkigym/transforms"
    repository_files = _repository_files(transforms_directory)
    metrics = inspect_transforms(REPOSITORY_ROOT)
    violations = _transform_layout_violations(repository_files)
    if len(metrics) > MAX_PUBLIC_TRANSFORMS:
        violations.append(f"public transform count is {len(metrics)}, limit {MAX_PUBLIC_TRANSFORMS}")

    by_module: dict[str, list[str]] = {}
    for metric in metrics:
        by_module.setdefault(metric.module, []).append(metric.name)

    source_files = {path.as_posix(): transforms_directory / path for path in repository_files if path.suffix == ".py"}
    line_counts: list[int] = []
    helper_lines = 0
    for module in sorted(set(source_files) | set(by_module)):
        transform_names = by_module.get(module, [])
        path = source_files.get(module)
        if path is None:
            violations.append(f"transforms/{module} defines a transform but is not a repository source file")
            continue
        code_lines = _python_code_lines(path)
        line_counts.append(code_lines)
        if code_lines >= TRANSFORM_FILE_LINE_LIMIT:
            violations.append(
                f"{module} has {code_lines} code lines; each Python file under transforms must have fewer than "
                f"{TRANSFORM_FILE_LINE_LIMIT}"
            )
        if module.startswith(f"{TRANSFORM_HELPER_DIRECTORY}/"):
            helper_lines += code_lines
            if transform_names:
                names = ", ".join(transform_names)
                violations.append(f"{module} defines transform classes ({names}); helper modules must define none")
        elif module not in TRANSFORM_INFRASTRUCTURE_FILES and "/" not in module:
            if len(transform_names) != 1:
                names = ", ".join(transform_names) or "none"
                violations.append(
                    f"{module} defines {len(transform_names)} public transforms ({names}); exactly one is required"
                )
    if helper_lines >= TRANSFORM_HELPER_LINE_LIMIT:
        violations.append(
            f"transform helper modules have {helper_lines} total code lines; helpers must have fewer than "
            f"{TRANSFORM_HELPER_LINE_LIMIT}"
        )
    largest_file = max(line_counts, default=0)
    return violations, len(metrics), largest_file, helper_lines


def _base_name(expression: ast.expr) -> str:
    """Return the terminal name of a class base expression."""
    if isinstance(expression, ast.Name):
        name = expression.id
    elif isinstance(expression, ast.Attribute):
        name = expression.attr
    else:
        name = ""
    return name


def _operation_class_names(path: Path) -> tuple[str, ...]:
    """Return direct NKIOp subclasses defined in one source file."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = tuple(
        node.name
        for node in module.body
        if isinstance(node, ast.ClassDef) and any(_base_name(base) == "NKIOp" for base in node.bases)
    )
    return names


def _operation_structure_violations() -> tuple[list[str], int, int, int]:
    """Enforce the flat one-operation-per-file package layout."""
    operations_directory = REPOSITORY_ROOT / "nkigym/src/nkigym/ops"
    repository_files = _repository_files(operations_directory)
    relative_names = {path.as_posix() for path in repository_files}
    violations: list[str] = []
    for required_file in sorted({"base.py"} - relative_names):
        violations.append(f"ops/{required_file} is required")

    operation_count = 0
    operation_file_lines: list[int] = []
    base_lines = 0
    for relative_path in repository_files:
        direct_python = len(relative_path.parts) == 1 and relative_path.suffix == ".py"
        if not direct_python:
            violations.append(f"ops/{relative_path.as_posix()} is not allowed; use one direct Python module per op")
            continue
        path = operations_directory / relative_path
        code_lines = _python_code_lines(path)
        if relative_path.name == "base.py":
            base_lines = code_lines
            if code_lines > OP_BASE_FILE_LINE_LIMIT:
                violations.append(f"ops/base.py has {code_lines} code lines, limit {OP_BASE_FILE_LINE_LIMIT}")
            continue
        operation_file_lines.append(code_lines)
        if code_lines > OP_FILE_LINE_LIMIT:
            violations.append(f"ops/{relative_path.name} has {code_lines} code lines, limit {OP_FILE_LINE_LIMIT}")
        if relative_path.name != "__init__.py":
            operation_names = _operation_class_names(path)
            operation_count += len(operation_names)
            if len(operation_names) != 1:
                names = ", ".join(operation_names) or "none"
                violations.append(
                    f"ops/{relative_path.name} defines {len(operation_names)} NKIOp subclasses ({names}); "
                    "exactly one is required"
                )
    largest_operation_file = max(operation_file_lines, default=0)
    return violations, operation_count, largest_operation_file, base_lines


def _kernel_library_structure_violations() -> list[str]:
    """Allow only the initializer and direct NAKB workload modules."""
    kernel_library_directory = REPOSITORY_ROOT / "kernel_library"
    violations: list[str] = []
    for relative_path in _repository_files(kernel_library_directory):
        is_initializer = relative_path == Path("__init__.py")
        is_workload_module = (
            len(relative_path.parts) == 1 and relative_path.suffix == ".py" and relative_path.name.startswith("nakb_")
        )
        if not is_initializer and not is_workload_module:
            violations.append(
                f"kernel_library/{relative_path.as_posix()} is not allowed; keep only __init__.py and direct "
                "nakb_*.py workload modules"
            )
    return violations


def _transform_api_violations() -> list[str]:
    """Return all public analyze/apply contract violations."""
    inspections = inspect_transform_api(REPOSITORY_ROOT)
    violations = [violation for inspection in inspections for violation in inspection.violations]
    return violations


def _source_size_violation(directory: Path, label: str, limit: int) -> tuple[str | None, int]:
    """Return one source subtree's total Python code lines and limit violation."""
    paths = sorted(directory.rglob("*.py"))
    total_lines = sum(_python_code_lines(path) for path in paths)
    violation = None
    if total_lines > limit:
        violation = f"{label} has {total_lines} code lines, limit {limit}"
    return violation, total_lines


def _import_roots(path: Path) -> tuple[tuple[str, int], ...]:
    """Return absolute import roots and source lines from one Python file."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend((alias.name.partition(".")[0], node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imports.append((node.module.partition(".")[0], node.lineno))
    return tuple(imports)


def _boundary_violations(directory: Path, allowed_repository_roots: set[str]) -> list[str]:
    """Return repository-package imports forbidden below one directory."""
    violations: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        for root, line in _import_roots(path):
            if root in REPOSITORY_IMPORT_ROOTS and root not in allowed_repository_roots:
                relative = path.relative_to(REPOSITORY_ROOT)
                violations.append(f"{relative}:{line} must not import repository package {root}")
    return violations


def _dependency_violations() -> list[str]:
    """Enforce the one-way repository dependency graph."""
    violations = [
        *_boundary_violations(REPOSITORY_ROOT / "nkigym/src/nkigym", {"nkigym"}),
        *_boundary_violations(REPOSITORY_ROOT / "kernel_library", {"kernel_library", "nkigym"}),
        *_boundary_violations(REPOSITORY_ROOT / "test", {"kernel_library", "nkigym"}),
    ]
    if (REPOSITORY_ROOT / "developer").exists():
        violations.append("legacy top-level developer package must remain removed")
    return violations


def _search_schedule_violations() -> list[str]:
    """Reject concrete transform ownership from generic search infrastructure."""
    search_directory = REPOSITORY_ROOT / "nkigym/src/nkigym/search"
    violations: list[str] = []
    for path in sorted(search_directory.rglob("*.py")):
        relative = path.relative_to(REPOSITORY_ROOT)
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                if node.module.startswith("nkigym.transforms"):
                    imports_generic_api = node.module == "nkigym.transforms" and all(
                        alias.name in {"Transform", "TransformOption", "public_transforms"} for alias in node.names
                    )
                    if not imports_generic_api:
                        violations.append(
                            f"{relative}:{node.lineno} imports concrete transform APIs from {node.module}"
                        )
            elif isinstance(node, ast.Import):
                forbidden = sorted(alias.name for alias in node.names if alias.name.startswith("nkigym.transforms"))
                if forbidden:
                    violations.append(f"{relative}:{node.lineno} imports concrete transform APIs {forbidden}")
            elif isinstance(node, ast.Call) and _base_name(node.func).endswith("Option"):
                violations.append(
                    f"{relative}:{node.lineno} constructs a transform option; runtime legal actions own payloads"
                )
    return violations


def test_repository_structure() -> None:
    """Repository structure, source growth, dependencies, and APIs remain valid."""
    format_violations = _repository_format_violations()
    assert not format_violations, "\n".join(format_violations)
    structure_violations, transform_count, largest_transform_file, helper_lines = _transform_structure_violations()
    operation_violations, operation_count, largest_operation_file, operation_base_lines = (
        _operation_structure_violations()
    )
    api_violations = _transform_api_violations()
    source_root = REPOSITORY_ROOT / "nkigym/src/nkigym"
    ir_violation, ir_lines = _source_size_violation(
        source_root / "ir", "IR implementation", MAX_IR_IMPLEMENTATION_LINES
    )
    codegen_violation, codegen_lines = _source_size_violation(
        source_root / "codegen", "codegen implementation", MAX_CODEGEN_IMPLEMENTATION_LINES
    )
    profile_violation, profile_lines = _source_size_violation(
        source_root / "profile", "profile implementation", MAX_PROFILE_IMPLEMENTATION_LINES
    )
    search_violation, search_lines = _source_size_violation(
        source_root / "search", "search implementation", MAX_SEARCH_IMPLEMENTATION_LINES
    )
    synthesis_violation, synthesis_lines = _source_size_violation(
        source_root / "synthesis", "synthesis implementation", MAX_SYNTHESIS_IMPLEMENTATION_LINES
    )
    violations = [
        *_package_initializer_violations(),
        *_nkigym_source_layout_violations(),
        *structure_violations,
        *operation_violations,
        *api_violations,
        *_kernel_library_structure_violations(),
        *_formatter_control_violations(source_root),
        *_dependency_violations(),
        *_search_schedule_violations(),
    ]
    if ir_violation is not None:
        violations.append(ir_violation)
    if codegen_violation is not None:
        violations.append(codegen_violation)
    if profile_violation is not None:
        violations.append(profile_violation)
    if search_violation is not None:
        violations.append(search_violation)
    if synthesis_violation is not None:
        violations.append(synthesis_violation)
    print(
        f"public_transforms={transform_count} largest_transform_file={largest_transform_file} " f"ir_lines={ir_lines}",
        f"codegen_lines={codegen_lines} profile_lines={profile_lines}",
        f"search_lines={search_lines} synthesis_lines={synthesis_lines} public_ops={operation_count}",
        f"largest_op_file={largest_operation_file} op_base_lines={operation_base_lines}",
        f"transform_helper_lines={helper_lines}",
        flush=True,
    )
    assert not violations, "\n".join(violations)


def test_python_code_lines_exclude_comments_and_documentation(tmp_path: Path) -> None:
    """Only executable and data-bearing source lines count toward limits."""
    source_path = tmp_path / "line_count_sample.py"
    source_path.write_text(
        '''"""Module documentation.
More documentation.
"""

# A comment does not count.
VALUE = (
    1
)

def read_value() -> int:
    """Function documentation."""
    """A standalone string block does not count."""
    return VALUE  # An inline comment adds no line.

"""Another string block."""; TEXT = """Assigned string data
still counts."""
''',
        encoding="utf-8",
    )
    assert _python_code_lines(source_path) == 7


def test_formatter_control_comments_ignore_string_contents(tmp_path: Path) -> None:
    """Only actual formatter-control comments are rejected."""
    source_path = tmp_path / "formatter_control_sample.py"
    source_path.write_text(
        '''TEXT = "# fmt: off"
DOCUMENTATION = """# yapf: disable"""
# fmt: skip
VALUE = 1  # yapf: disable
''',
        encoding="utf-8",
    )
    assert _formatter_control_comments(source_path) == ((3, "# fmt: skip"), (4, "# yapf: disable"))
