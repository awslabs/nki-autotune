"""Export benchmark results to nkigym-compatible cache format.

Reorganizes per-job cache directories into the flat structure:
  {cache_root}/nki/nki_v{idx}.py     - kernel source per variant
  {cache_root}/neff/nki_v{idx}/      - compilation artifacts per variant
  {cache_root}/results.json           - unified results file
"""

import json
import os
import shutil

from autotune.job import ProfileJobs


def _build_variant(job_dict: dict, cache_root: str, idx: int) -> dict:
    """Build a single variant entry for results.json.

    Args:
        job_dict: Serialized job attributes from ProfileJob.to_dict().
        cache_root: Root cache directory path.
        idx: Original job index.

    Returns:
        Variant dict matching nkigym results.json format.
    """
    has_error = "error" in job_dict
    has_timing = "min_ms" in job_dict

    if has_error:
        status = "error"
    elif has_timing:
        status = "benchmarked"
    else:
        status = "compiled"

    return {
        "nki_path": f"{cache_root}/nki/nki_v{idx}.py",
        "depth": 0,
        "variant_idx": idx,
        "status": status,
        "min_ms": job_dict.get("min_ms", 0.0),
        "mean_ms": job_dict.get("mean_ms", 0.0),
        "p50_ms": job_dict.get("p50_ms", 0.0),
        "p99_ms": job_dict.get("p99_ms", 0.0),
        "mac_count": job_dict.get("mac_count", 0),
        "mfu": job_dict.get("mfu", 0.0),
        "correct": job_dict.get("correctness_result", False),
        "error": job_dict.get("error"),
    }


def export_results(jobs: ProfileJobs) -> None:
    """Export ProfileJobs to nkigym-compatible cache structure.

    Creates nki/ and neff/ directories, copies artifacts, and writes
    results.json with the same schema as nkigym search results.

    Args:
        jobs: Completed ProfileJobs with benchmark results.
    """
    cache_root = jobs.cache_root_dir
    nki_dir = os.path.join(cache_root, "nki")
    neff_dir = os.path.join(cache_root, "neff")
    os.makedirs(nki_dir, exist_ok=True)
    os.makedirs(neff_dir, exist_ok=True)

    succeeded = 0
    failed = 0
    variants: list[dict] = []

    sorted_indices = sorted(jobs.jobs.keys(), key=lambda ji: jobs.jobs[ji].sort_val)

    for idx in sorted_indices:
        job = jobs.jobs[idx]
        job_dict = job.to_dict()
        variant = _build_variant(job_dict, cache_root, idx)
        variants.append(variant)

        if job.has_error:
            failed += 1
        else:
            succeeded += 1

        _copy_job_artifacts(job, nki_dir, neff_dir, idx)

    results_data = {"compilation": {"succeeded": succeeded, "failed": failed}, "variants": variants}

    results_path = os.path.join(cache_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    _cleanup_internal_dirs(cache_root)


def _cleanup_internal_dirs(cache_root: str) -> None:
    """Remove internal per-workload cache dirs after export.

    Args:
        cache_root: Root cache directory path.
    """
    keep = {"nki", "neff", "results.json", "sources"}
    for item in os.listdir(cache_root):
        if item in keep:
            continue
        path = os.path.join(cache_root, item)
        if os.path.isdir(path):
            shutil.rmtree(path)


def _copy_job_artifacts(job: object, nki_dir: str, neff_dir: str, idx: int) -> None:
    """Copy kernel source and NEFF artifacts to export directories.

    Args:
        job: A ProfileJob instance (typed as object for pyright compat).
        nki_dir: Path to the nki/ output directory.
        neff_dir: Path to the neff/ output directory.
        idx: Variant index for naming.
    """
    cache_dir: str = getattr(job, "cache_dir", "")
    if not cache_dir or not os.path.isdir(cache_dir):
        return

    variant_name = f"nki_v{idx}"
    _copy_nki_sources(cache_dir, nki_dir, variant_name)
    _copy_neff_artifacts(cache_dir, neff_dir, variant_name)


def _copy_nki_sources(cache_dir: str, nki_dir: str, variant_name: str) -> None:
    """Copy the kernel source file to the nki/ directory.

    Args:
        cache_dir: Source job cache directory.
        nki_dir: Destination nki/ directory.
        variant_name: Variant name like 'nki_v0'.
    """
    src = os.path.join(cache_dir, "kernel.py")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(nki_dir, f"{variant_name}.py"))


def _copy_neff_artifacts(cache_dir: str, neff_dir: str, variant_name: str) -> None:
    """Copy compilation artifacts to neff/{variant_name}/.

    Args:
        cache_dir: Source job cache directory.
        neff_dir: Destination neff/ directory.
        variant_name: Variant name like 'nki_v0'.
    """
    neff_variant_dir = os.path.join(neff_dir, variant_name)
    if os.path.isdir(neff_variant_dir):
        shutil.rmtree(neff_variant_dir)
    os.makedirs(neff_variant_dir, exist_ok=True)

    skip = {"kernel.py"}
    for item in os.listdir(cache_dir):
        if item in skip:
            continue
        src = os.path.join(cache_dir, item)
        dst = os.path.join(neff_variant_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
