"""Shared utilities for the lowering pipeline.

Holds the indentation-tracking :class:`_Writer`. Lifted out of
``emit_source`` so that :mod:`nkigym.codegen.lowering.emit_ops` and
:mod:`nkigym.codegen.lowering.inject_software_pipeline` can depend on
this helper without forming an import cycle with ``emit_source``
(which in turn depends on the registered body emitters and pipeline
machinery for dispatch).
"""


class _Writer:
    """Line-based writer with indentation tracking."""

    def __init__(self) -> None:
        """Initialize an empty writer at indent depth 0."""
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        """Open a nested block — subsequent ``line`` calls indent one level deeper."""
        self._depth += 1

    def dedent(self) -> None:
        """Close a nested block."""
        self._depth -= 1

    def line(self, text: str = "") -> None:
        """Append a source line at the current indent."""
        self._lines.append(("    " * self._depth + text) if text else "")

    def getvalue(self) -> str:
        """Return the accumulated source with a trailing newline."""
        return "\n".join(self._lines) + "\n"
