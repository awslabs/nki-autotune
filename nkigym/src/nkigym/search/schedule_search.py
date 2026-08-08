"""One-call entry point for deterministic heuristic schedule search."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from nkigym.search.engine import HeuristicScheduleSearch, SearchConfig
from nkigym.search.types import InputSpecs, SearchResult


def run_heuristic_search(
    kernel_func: Callable[..., Any],
    input_specs: InputSpecs,
    profile_host: str,
    /,
    *,
    trace_dir: Path | None = None,
    neuronx_cc_args: tuple[str, ...] = (),
    profile_timeout_s: int = 1800,
    lnc: int = 1,
) -> SearchResult:
    """Build and profile deterministic schedule candidates for one kernel."""
    with ExitStack() as stack:
        cache_dir = trace_dir
        if cache_dir is None:
            cache_dir = Path(stack.enter_context(TemporaryDirectory(prefix="nkigym-heuristic-search-")))
        search = HeuristicScheduleSearch(
            kernel_func=kernel_func,
            config=SearchConfig(
                profile_host=profile_host,
                input_specs=input_specs,
                cache_dir=cache_dir,
                neuronx_cc_args=neuronx_cc_args,
                lnc=lnc,
                timeout_s=profile_timeout_s,
            ),
        )
        result = search.run()
    return result


__all__ = ["run_heuristic_search"]
