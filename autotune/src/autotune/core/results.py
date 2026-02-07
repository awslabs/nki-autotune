# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any


class ResultEntry:
    """Wraps a single result dictionary from perf_metrics.json.

    Provides typed access to common fields and dict-like access for all others.

    Attributes:
        data: The raw result dictionary.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize a ResultEntry from a result dictionary.

        Args:
            data: A single result dict from the 'results' list in perf_metrics.json.
        """
        self.data = data

    @property
    def has_error(self) -> bool:
        """Whether this result encountered an error."""
        return "error" in self.data

    @property
    def min_ms(self) -> float:
        """Minimum execution time in milliseconds.

        Raises:
            KeyError: If 'min_ms' is not present in the result data.
        """
        return self.data["min_ms"]

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to result fields.

        Args:
            key: The field name to access.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key is not found.
        """
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get with default value.

        Args:
            key: The field name to access.
            default: Value to return if key is not found.

        Returns:
            The value associated with the key, or the default.
        """
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists in the result data.

        Args:
            key: The field name to check.

        Returns:
            True if the key exists in the data.
        """
        return key in self.data

    def __repr__(self) -> str:
        """Return a string representation of the ResultEntry."""
        if self.has_error:
            error_preview = self.data["error"].split("\n")[0][:60]
            return f"ResultEntry(error={error_preview!r})"
        min_ms = self.data.get("min_ms", "N/A")
        index = self.data.get("index", "?")
        return f"ResultEntry(index={index}, min_ms={min_ms})"


class WorkloadResults:
    """Groups result entries for a single workload (same kernel + same input shapes).

    One perf_metrics.json file corresponds to one WorkloadResults instance.
    Entries are sorted by min_ms latency (best first).

    Attributes:
        workload_name: The canonical workload name (e.g., 'nki_matmul/128x512_512x1024').
        entries: List of ResultEntry sorted by min_ms (best first).
        metadata: The metadata dict from perf_metrics.json.
    """

    def __init__(self, workload_name: str, json_data: dict[str, Any]) -> None:
        """Initialize WorkloadResults from a parsed perf_metrics.json.

        Args:
            workload_name: Canonical name for this workload.
            json_data: Parsed contents of a perf_metrics.json file.
        """
        self.workload_name = workload_name
        self.metadata = json_data.get("metadata", {})
        results = json_data.get("results", [])
        self.entries = sorted([ResultEntry(r) for r in results], key=lambda e: e.data.get("min_ms", float("inf")))

    @property
    def best(self) -> ResultEntry:
        """Return the entry with the lowest min_ms.

        Raises:
            IndexError: If there are no entries.
        """
        if not self.entries:
            raise IndexError(f"No entries in workload '{self.workload_name}'")
        return self.entries[0]

    def __len__(self) -> int:
        """Return the number of result entries."""
        return len(self.entries)

    def __repr__(self) -> str:
        """Return a string representation of WorkloadResults."""
        return f"WorkloadResults({self.workload_name!r}, {len(self.entries)} entries)"


class BenchmarkResults:
    """Top-level container for loading and querying benchmark results from cache.

    Walks a cache directory tree, loads all perf_metrics.json files, and provides
    methods for summarizing, filtering, and comparing results.

    Attributes:
        workloads: Dictionary mapping workload names to WorkloadResults.
    """

    def __init__(self, workloads: dict[str, WorkloadResults]) -> None:
        """Initialize BenchmarkResults with a dict of workloads.

        Args:
            workloads: Mapping of workload names to WorkloadResults instances.
        """
        self.workloads = workloads

    @classmethod
    def load(cls, cache_dir: str, kernel_name: str | None = None) -> BenchmarkResults:
        """Load benchmark results from a cache directory.

        Walks the cache tree to find all perf_metrics.json files. Each file
        corresponds to one workload (kernel + input shapes combination).

        Args:
            cache_dir: Root directory of the benchmark cache.
            kernel_name: If provided, only load results for this kernel name.

        Returns:
            BenchmarkResults containing all discovered workloads.

        Raises:
            FileNotFoundError: If cache_dir does not exist.
        """
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

        workloads: dict[str, WorkloadResults] = {}

        for dirpath, _, filenames in os.walk(cache_dir):
            if "perf_metrics.json" not in filenames:
                continue

            rel_path = os.path.relpath(dirpath, cache_dir)
            wl_name = rel_path.replace(os.sep, "/")

            if kernel_name is not None:
                wl_kernel = wl_name.split("/")[0] if "/" in wl_name else wl_name
                if wl_kernel != kernel_name:
                    continue

            json_path = os.path.join(dirpath, "perf_metrics.json")
            with open(json_path, "r") as f:
                json_data = json.load(f)

            workloads[wl_name] = WorkloadResults(wl_name, json_data)

        return cls(workloads)

    def summary(self, metric: str = "min_ms", columns: list[str] | None = None, top_k: int = 1) -> None:
        """Print a tabulate-formatted summary table to stdout.

        For each workload, shows the top_k best entries sorted by the given metric.

        Args:
            metric: Metric to sort by (default: 'min_ms').
            columns: List of column names to display. If None, uses a default set.
            top_k: Number of top entries to show per workload.
        """
        from tabulate import tabulate

        if columns is None:
            columns = ["index", "min_ms", "mean_ms", "mfu_estimated_percent", "correctness_result"]

        headers = ["workload"] + columns
        rows: list[list[Any]] = []

        for wl_name in sorted(self.workloads.keys()):
            wl = self.workloads[wl_name]
            sorted_entries = sorted(
                [e for e in wl.entries if not e.has_error], key=lambda e: e.data.get(metric, float("inf"))
            )
            for entry in sorted_entries[:top_k]:
                row = [wl_name]
                for col in columns:
                    val = entry.get(col)
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    row.append(val)
                rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))

    def best_configs(self, metric: str = "min_ms") -> dict[str, ResultEntry]:
        """Return the best entry per workload for the given metric.

        Args:
            metric: Metric to minimize (default: 'min_ms').

        Returns:
            Dictionary mapping workload names to the best ResultEntry.
        """
        results: dict[str, ResultEntry] = {}
        for wl_name, wl in self.workloads.items():
            valid_entries = [e for e in wl.entries if not e.has_error and metric in e]
            if valid_entries:
                best = min(valid_entries, key=lambda e: e[metric])
                results[wl_name] = best
        return results

    def filter(self, predicate: Callable[[ResultEntry], bool]) -> BenchmarkResults:
        """Return a new BenchmarkResults containing only entries matching predicate.

        Args:
            predicate: Function taking a ResultEntry and returning True to keep it.

        Returns:
            New BenchmarkResults with filtered entries.
        """
        filtered_workloads: dict[str, WorkloadResults] = {}
        for wl_name, wl in self.workloads.items():
            matching = [e for e in wl.entries if predicate(e)]
            if matching:
                json_data = {"metadata": wl.metadata, "results": [e.data for e in matching]}
                filtered_workloads[wl_name] = WorkloadResults(wl_name, json_data)
        return BenchmarkResults(filtered_workloads)

    def compare(self, metric: str, kernel_names: list[str]) -> None:
        """Print a side-by-side comparison table across kernels for a given metric.

        Groups workloads by input shapes (the part after the kernel name in the
        workload path), then shows the best metric value for each kernel.

        Args:
            metric: Metric to compare (e.g., 'min_ms', 'mfu_estimated_percent').
            kernel_names: List of kernel names to compare.
        """
        from tabulate import tabulate

        shape_map: dict[str, dict[str, float | None]] = {}

        for wl_name, wl in self.workloads.items():
            parts = wl_name.split("/", 1)
            if len(parts) != 2:
                continue
            kernel, shapes = parts

            if kernel not in kernel_names:
                continue

            valid = [e for e in wl.entries if not e.has_error and metric in e]
            best_val = min((e[metric] for e in valid), default=None)

            if shapes not in shape_map:
                shape_map[shapes] = {}
            shape_map[shapes][kernel] = best_val

        headers = ["input_shapes"] + kernel_names
        rows: list[list[Any]] = []
        for shapes in sorted(shape_map.keys()):
            row: list[Any] = [shapes]
            for kernel in kernel_names:
                val = shape_map[shapes].get(kernel)
                if val is not None and isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))

    def __len__(self) -> int:
        """Return the total number of workloads."""
        return len(self.workloads)

    def __repr__(self) -> str:
        """Return a string representation of BenchmarkResults."""
        total_entries = sum(len(wl) for wl in self.workloads.values())
        return f"BenchmarkResults({len(self.workloads)} workloads, {total_entries} total entries)"
