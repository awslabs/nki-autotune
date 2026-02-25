"""Structured JSON report for search progress and benchmark results.

Manages a live-overwritten JSON file that tracks search exploration,
compilation status, and per-variant benchmark results.
"""

import json
from pathlib import Path
from typing import Any

_EMPTY_REPORT: dict[str, Any] = {
    "search": {
        "root_stmts": 0,
        "root_opportunities": 0,
        "unique_programs": 0,
        "qualifying_programs": 0,
        "min_depth": 0,
        "total_programs_visited": 0,
        "duplicates": 0,
        "dedup_pct": 0.0,
        "depth_distribution": {},
    },
    "compilation": {"succeeded": 0, "failed": 0},
    "variants": [],
}


class SearchReport:
    """Manages a live JSON report file for search and benchmark progress.

    Attributes:
        path: Path to the JSON report file.
    """

    def __init__(self, path: Path) -> None:
        """Initialize an empty report and write it to disk.

        Args:
            path: File path for the JSON report.
        """
        self.path = path
        self._data: dict[str, Any] = json.loads(json.dumps(_EMPTY_REPORT))
        self._variant_index: dict[str, int] = {}
        self._write()

    def update_search(
        self,
        root_stmts: int,
        root_opportunities: int,
        unique_programs: int,
        qualifying_programs: int,
        min_depth: int,
        total_programs_visited: int,
        depth_distribution: dict[int, int],
    ) -> None:
        """Update the search section of the report.

        Args:
            root_stmts: Number of statements in the root program.
            root_opportunities: Number of transform opportunities at root.
            unique_programs: Total unique programs discovered.
            qualifying_programs: Number of qualifying variants saved.
            min_depth: Minimum depth threshold.
            total_programs_visited: Total programs visited (root + expansions).
            depth_distribution: Map from depth to node count.
        """
        duplicates = max(0, total_programs_visited - unique_programs)
        dedup_pct = (duplicates / total_programs_visited * 100) if total_programs_visited > 0 else 0.0
        self._data["search"] = {
            "root_stmts": root_stmts,
            "root_opportunities": root_opportunities,
            "unique_programs": unique_programs,
            "qualifying_programs": qualifying_programs,
            "min_depth": min_depth,
            "total_programs_visited": total_programs_visited,
            "duplicates": duplicates,
            "dedup_pct": round(dedup_pct, 1),
            "depth_distribution": {str(k): v for k, v in sorted(depth_distribution.items())},
        }
        self._write()

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
            "mac_count": None,
            "mfu": None,
            "correct": None,
            "error": None,
        }
        self._variant_index[nki_path] = len(self._data["variants"])
        self._data["variants"].append(entry)
        self._write()

    def update_variant(self, nki_path: str, **fields: Any) -> None:
        """Update fields on an existing variant entry.

        Args:
            nki_path: Path identifying the variant.
            **fields: Key-value pairs to update on the variant dict.
        """
        idx = self._variant_index.get(nki_path)
        if idx is not None:
            self._data["variants"][idx].update(fields)
            self._write()

    def set_compilation(self, succeeded: int, failed: int) -> None:
        """Update the compilation section.

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

    def _write(self) -> None:
        """Atomically write the report to disk."""
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2) + "\n")
        tmp_path.rename(self.path)


def _variant_sort_key(v: dict[str, Any]) -> tuple[int, float]:
    """Sort key: benchmarked variants by min_ms first, then errors/pending."""
    has_time = v.get("min_ms") is not None and v["min_ms"] > 0 and not v.get("error")
    rank = 0 if has_time else 1
    latency = v["min_ms"] if has_time else 0.0
    return (rank, latency)
