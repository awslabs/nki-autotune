"""Backend for the ``tune-kernel`` skill — programmatic cache management.

Two entry points the skill calls:

* :func:`dump_baseline` — run the nkipy compiler baseline once, write
  ``<cache_root>/nkipy_baseline/{nkipy_baseline.py, baseline.json}`` and
  seed ``<cache_root>/summary.json``.
* :func:`submit_batch` — render + ship a list of IRs in one SSH round-trip,
  write the flat batch cache layout, update ``summary.json``'s
  ``tuning`` table with that batch's running-best kernel.

Cache layout (owned by this module — agents don't manage any paths
themselves):

``<cache_root>/``

* ``nkipy_baseline/``
   * ``nkipy_baseline.py``  — compiler-emitted NKI source
   * ``baseline.json``      — ``{source, mfu, min_ms}``
* ``batch_<bid>/``
   * ``kernel_<kid>/``
      * ``batch_<bid>_kernel_<kid>.py``                  — rendered NKI source
      * ``batch_<bid>_ir_<kid>.md``                      — KernelIR ``repr``
      * ``batch_<bid>_kernel_<kid>_log-neuron-cc.txt``   — compiler log
   * ``batch_<bid>_results.json``                        — backend-shape results for the batch
* ``summary.json``  — ``{"function": str, "baseline": {...}, "tuning": {"batch_<bid>": {...}}}``
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from autotune.runner.api import remote_profile
from autotune.runner.remote import remote_numpy_baseline
from autotune.runner.types import KernelJob, ProfileResult
from nkigym.codegen import render_ir
from nkigym.kernel_ir import KernelIR
from nkigym.search.api import func_source_with_imports, inline_gadgets
from nkigym.search.mac import compute_mac_count

_NEURONX_CC_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")


@dataclass
class BaselineResult:
    """Baseline dump return — what the skill prints + records."""

    mfu: float | None
    min_ms: float | None
    source: str


def dump_baseline(
    f_numpy: Callable[..., np.ndarray],
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_root: str | Path,
    host: str,
    target_mfu: float | None = None,
    source: str = "nkipy",
) -> BaselineResult:
    """Ship the nkipy baseline on ``host`` and seed the tuning cache.

    Writes ``<cache_root>/nkipy_baseline/nkipy_baseline.py`` (compiler
    output) and ``<cache_root>/nkipy_baseline/baseline.json`` (mfu /
    min_ms). Initializes ``<cache_root>/summary.json`` with
    ``function``, ``baseline``, ``target_mfu`` (when provided), and
    an empty ``tuning`` table.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    baseline_dir = cache_root / "nkipy_baseline"

    result = remote_numpy_baseline(
        func=f_numpy,
        input_specs=input_specs,
        mac_count=compute_mac_count(f_nkigym, input_specs),
        host=host,
        kernel_name="nkipy_baseline",
        cache_dir=str(cache_root),
    )

    """``remote_numpy_baseline`` writes <cache>/<stem>/<stem>.py +
    <cache>/results.json; drop the stray results.json (the baseline is
    tracked in summary.json instead) and add baseline.json alongside."""
    (cache_root / "results.json").unlink(missing_ok=True)
    baseline_info = {"source": source, "mfu": result.mfu, "min_ms": result.min_ms}
    (baseline_dir / "baseline.json").write_text(json.dumps(baseline_info, indent=2))

    summary_path = cache_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {"function": f_nkigym.__name__, "baseline": None, "target_mfu": None, "tuning": {}}
    summary["function"] = f_nkigym.__name__
    summary["baseline"] = baseline_info
    if target_mfu is not None:
        summary["target_mfu"] = float(target_mfu)
    summary_path.write_text(json.dumps(summary, indent=2))

    return BaselineResult(mfu=result.mfu, min_ms=result.min_ms, source=source)


def submit_batch(
    irs: list[KernelIR],
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_root: str | Path,
    hosts: list[str],
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> list[ProfileResult]:
    """Render + ship ``irs`` in one SSH batch; persist the batch cache.

    Assigns monotonic ``batch_id`` (from ``summary.json``'s ``tuning``
    keys) and monotonic ``kernel_id`` (from prior batches'
    ``batch_<bid>_results.json``). Rewrites ``summary.json`` with the
    new batch's running-best entry.

    Deduplicates against prior batches: an IR whose ``repr`` matches a
    kernel recorded in any prior ``batch_<bid>_results.json`` is not
    re-rendered or re-profiled — the cached :class:`ProfileResult` is
    returned in that slot (its ``kernel_name`` still points at the
    original batch). If every IR is a cache hit, no new batch is
    created and ``summary.json`` is left untouched.

    Returns :class:`ProfileResult` for each IR in input order.
    """
    if not irs:
        return []

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    summary_path = cache_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else {"function": f_nkigym.__name__, "baseline": None, "tuning": {}}
    )
    summary.setdefault("tuning", {})

    """Partition inputs into cache hits (prior ProfileResult reused) and
    fresh IRs that need rendering + shipping."""
    prior_cache = _load_prior_ir_results(cache_root)
    cached: dict[int, ProfileResult] = {}
    fresh: list[tuple[int, KernelIR]] = []
    for idx, ir in enumerate(irs):
        hit = prior_cache.get(repr(ir))
        if hit is not None:
            cached[idx] = hit
        else:
            fresh.append((idx, ir))

    if not fresh:
        return [cached[i] for i in range(len(irs))]

    bid = _next_batch_id(summary)
    next_kid = _next_kernel_id(cache_root, summary)

    batch_dir = cache_root / f"batch_{bid}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    nkigym_source = func_source_with_imports(f_nkigym)
    mac_count = compute_mac_count(f_nkigym, input_specs)

    """Render every fresh IR, stage .py + ir.md + KernelJob dict."""
    jobs: dict[str, KernelJob] = {}
    assignments: list[tuple[int, int, str, KernelIR]] = []
    for input_idx, ir in fresh:
        kid = next_kid
        next_kid += 1
        stem = f"batch_{bid}_kernel_{kid}"
        kernel_name = f"{stem}.py"
        kdir = batch_dir / f"kernel_{kid}"
        kdir.mkdir(parents=True, exist_ok=True)
        source = inline_gadgets(render_ir(ir))
        (kdir / f"batch_{bid}_ir_{kid}.md").write_text(repr(ir) + "\n")
        jobs[kernel_name] = KernelJob(
            source=source,
            func_name=ir.func_name,
            output_shape=tuple(ir.logical_tensors[ir.return_name].shape),
            input_specs=input_specs,
            nkigym_source=nkigym_source,
            nkigym_func_name=f_nkigym.__name__,
            mac_count=mac_count,
            atol=atol,
            rtol=rtol,
            neuronx_cc_args=_NEURONX_CC_ARGS,
        )
        assignments.append((input_idx, kid, kernel_name, ir))

    """Ship — let the backend do source/log/results writes under batch_dir/<stem>/."""
    output = remote_profile(jobs, hosts=hosts, cache_dir=str(batch_dir))
    by_name = {r.kernel_name: r for r in output.results}

    """Fold the backend's <stem>/ layout into the requested one:
       kernel_<kid>/batch_<bid>_kernel_<kid>.py + _log-neuron-cc.txt."""
    for _input_idx, kid, kernel_name, _ir in assignments:
        stem = Path(kernel_name).stem
        stem_dir = batch_dir / stem
        dst_dir = batch_dir / f"kernel_{kid}"
        if (stem_dir / f"{stem}.py").exists():
            (stem_dir / f"{stem}.py").rename(dst_dir / f"{stem}.py")
        log_src = stem_dir / "log-neuron-cc.txt"
        if log_src.exists():
            log_src.rename(dst_dir / f"{stem}_log-neuron-cc.txt")
        if stem_dir.exists():
            shutil.rmtree(stem_dir)
    backend_results = batch_dir / "results.json"
    if backend_results.exists():
        backend_results.rename(batch_dir / f"batch_{bid}_results.json")

    """Splice fresh results back into input order alongside cache hits."""
    fresh_by_idx: dict[int, ProfileResult] = {
        input_idx: by_name[kernel_name] for input_idx, _kid, kernel_name, _ir in assignments
    }
    merged = {**cached, **fresh_by_idx}
    ordered: list[ProfileResult] = [merged[i] for i in range(len(irs))]

    """Running-best bookkeeping: this batch vs. prior last-batch best.
    Consider all slots (cache + fresh) so a cache hit higher than the
    prior running-best still promotes, guarding against manually edited
    summary.json."""
    prior_best_name, prior_best_mfu = _latest_running_best(summary)
    best_name, best_mfu = prior_best_name, prior_best_mfu
    for r in ordered:
        if r.mfu is not None and (best_mfu is None or r.mfu > best_mfu):
            best_name = r.kernel_name
            best_mfu = r.mfu

    summary["tuning"][f"batch_{bid}"] = {"running_best_kernel": best_name, "mfu": best_mfu}
    summary_path.write_text(json.dumps(summary, indent=2))
    return ordered


def _load_prior_ir_results(cache_root: Path) -> dict[str, ProfileResult]:
    """Scan prior batches and build ``{repr(ir) -> ProfileResult}``.

    Walks every ``batch_<bid>/batch_<bid>_results.json`` under
    ``cache_root``. For each kernel entry, reads the sibling
    ``kernel_<kid>/batch_<bid>_ir_<kid>.md`` (dropping its trailing
    newline) and maps that IR repr to a :class:`ProfileResult`
    reconstructed from the JSON row. Entries whose ``ir.md`` is missing
    are skipped — their IR repr can't be known, so they can't dedup.

    If the same IR repr was run more than once, the latest occurrence
    wins (dicts preserve insertion order; batches iterate in ascending
    id).
    """
    cache: dict[str, ProfileResult] = {}
    batch_dirs = sorted(
        (p for p in cache_root.glob("batch_*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1,
    )
    for batch_dir in batch_dirs:
        try:
            bid = int(batch_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        results_path = batch_dir / f"batch_{bid}_results.json"
        if not results_path.exists():
            continue
        try:
            data = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            continue
        for entry in data.get("kernels", []):
            kernel_name = entry.get("kernel_name", "")
            stem = Path(kernel_name).stem
            tail = stem.rsplit("_", 1)[-1]
            if not tail.isdigit():
                continue
            kid = int(tail)
            ir_md = batch_dir / f"kernel_{kid}" / f"batch_{bid}_ir_{kid}.md"
            if not ir_md.exists():
                continue
            ir_repr = ir_md.read_text().rstrip("\n")
            cache[ir_repr] = ProfileResult(
                kernel_name=kernel_name,
                min_ms=entry.get("min_ms"),
                mean_ms=entry.get("mean_ms"),
                p50_ms=entry.get("p50_ms"),
                p99_ms=entry.get("p99_ms"),
                mac_count=entry.get("mac_count", 0),
                mfu=entry.get("mfu"),
                cpu_sim=entry.get("cpu_sim", {}),
                hardware_output=entry.get("hardware_output", ""),
                mbu_estimated_percent=entry.get("mbu_estimated_percent"),
                mfu_max_achievable_estimated_percent=entry.get("mfu_max_achievable_estimated_percent"),
                roofline_efficiency=entry.get("roofline_efficiency"),
            )
    return cache


def _next_batch_id(summary: dict) -> int:
    """Next ``batch_id`` = count of existing ``tuning.batch_*`` entries."""
    return len(summary.get("tuning", {}))


def _next_kernel_id(cache_root: Path, summary: dict) -> int:
    """Max prior ``kernel_id`` + 1, scanned from ``batch_*_results.json``."""
    max_kid = -1
    for batch_key in summary.get("tuning", {}):
        try:
            bid = int(batch_key.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        rpath = cache_root / batch_key / f"batch_{bid}_results.json"
        if not rpath.exists():
            continue
        try:
            data = json.loads(rpath.read_text())
        except json.JSONDecodeError:
            continue
        for entry in data.get("kernels", []):
            stem = Path(entry.get("kernel_name", "")).stem
            tail = stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                max_kid = max(max_kid, int(tail))
    return max_kid + 1


def _latest_running_best(summary: dict) -> tuple[str | None, float | None]:
    """``(kernel_name, mfu)`` of the latest batch's ``running_best_kernel``."""
    tuning = summary.get("tuning", {})
    if not tuning:
        return None, None
    """dict preserves insertion order; latest batch inserted last."""
    last_key = next(reversed(tuning))
    entry = tuning[last_key]
    return entry.get("running_best_kernel"), entry.get("mfu")
