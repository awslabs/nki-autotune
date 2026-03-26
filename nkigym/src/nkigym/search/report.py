"""Structured JSON report for search progress and benchmark results.

Manages a live-overwritten JSON file that tracks search exploration,
compilation status, and per-variant benchmark results.
"""

import copy
import json
from pathlib import Path
from typing import Any

_EMPTY_REPORT: dict[str, Any] = {
    "search": {"unique_schedules": 0, "qualifying_schedules": 0, "total_visited": 0},
    "compilation": {"succeeded": 0, "failed": 0},
    "variants": [],
}


class SearchReport:
    """Manages a live JSON report file for search and benchmark progress.

    Writes to disk are deferred during batch operations and flushed
    explicitly via :meth:`flush`, or automatically on :meth:`sort_variants`
    and :meth:`set_compilation`.

    Attributes:
        path: Path to the JSON report file.
    """

    def __init__(self, path: Path) -> None:
        """Initialize an empty report and write it to disk.

        Args:
            path: File path for the JSON report.
        """
        self.path = path
        self._data: dict[str, Any] = copy.deepcopy(_EMPTY_REPORT)
        self._variant_index: dict[str, int] = {}
        self._dirty = False
        self._write()

    def update_search(self, unique_schedules: int, qualifying_schedules: int, total_visited: int) -> None:
        """Update the search section of the report.

        Args:
            unique_schedules: Total unique schedules discovered.
            qualifying_schedules: Number of qualifying variants saved.
            total_visited: Total schedules visited (root + expansions).
        """
        self._data["search"] = {
            "unique_schedules": unique_schedules,
            "qualifying_schedules": qualifying_schedules,
            "total_visited": total_visited,
        }
        self._dirty = True

    def add_variant(self, nki_path: str, depth: int, variant_idx: int) -> None:
        """Append a new pending variant entry.

        Args:
            nki_path: Path to the NKI source file.
            depth: Transform depth of this variant.
            variant_idx: Index within its depth bucket.
        """
        entry = {
            "nki_path": nki_path,
            "depth": depth,
            "variant_idx": variant_idx,
            "status": "pending",
            "min_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p99_ms": None,
            "mac_count": None,
            "mfu": None,
            "correct": None,
            "error": None,
        }
        self._variant_index[nki_path] = len(self._data["variants"])
        self._data["variants"].append(entry)
        self._dirty = True

    def update_variant(self, nki_path: str, **fields: Any) -> None:
        """Update fields on an existing variant entry.

        Args:
            nki_path: Path identifying the variant.
            **fields: Key-value pairs to update on the variant dict.
        """
        idx = self._variant_index.get(nki_path)
        if idx is not None:
            self._data["variants"][idx].update(fields)
            self._dirty = True

    def set_compilation(self, succeeded: int, failed: int) -> None:
        """Update the compilation section and flush to disk.

        Args:
            succeeded: Number of successfully compiled variants.
            failed: Number of compilation failures.
        """
        self._data["compilation"] = {"succeeded": succeeded, "failed": failed}
        self._write()

    def sort_variants(self) -> None:
        """Sort variants by min_ms ascending, with errors/pending at the end."""
        variants = self._data["variants"]
        variants.sort(key=_variant_sort_key)
        self._variant_index = {v["nki_path"]: i for i, v in enumerate(variants)}
        self._write()

    def flush(self) -> None:
        """Write pending changes to disk if dirty."""
        if self._dirty:
            self._write()

    def _write(self) -> None:
        """Atomically write the report to disk."""
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2) + "\n")
        tmp_path.rename(self.path)
        self._dirty = False


def _variant_sort_key(v: dict[str, Any]) -> tuple[int, float]:
    """Sort key: benchmarked variants by min_ms first, then errors/pending."""
    has_time = v.get("min_ms") is not None and v["min_ms"] > 0 and not v.get("error")
    rank = 0 if has_time else 1
    latency = v["min_ms"] if has_time else 0.0
    return (rank, latency)
