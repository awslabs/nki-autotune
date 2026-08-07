"""Helpers for Git repository discovery and candidate snapshots."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_BASELINE_REF_PREFIX = "refs/nki-autotune/baselines"


class GitCommandError(RuntimeError):
    """Raised when a required Git operation fails."""


@dataclass(frozen=True)
class FileDiffStat:
    """Added and deleted line counts for one changed path."""

    path: str
    added_lines: int | None
    deleted_lines: int | None


@dataclass(frozen=True)
class CandidateSnapshot:
    """A full base-to-candidate patch, paths, and line counts."""

    patch: str
    changed_files: tuple[str, ...]
    diff_stats: tuple[FileDiffStat, ...]


def _run_git(
    repository: Path, arguments: tuple[str, ...], environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run Git in a repository and raise an error with command context."""
    command = ("git", "-C", str(repository), *arguments)
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GitCommandError(f"{' '.join(command)} failed with exit {completed.returncode}: {detail}")
    return completed


def resolve_repository(path: Path) -> Path:
    """Resolve a path to the root of its containing Git repository."""
    completed = _run_git(path.expanduser().resolve(), ("rev-parse", "--show-toplevel"))
    repository = Path(completed.stdout.strip()).resolve()
    return repository


def resolve_revision(repository: Path, revision: str) -> str:
    """Resolve a revision to a commit SHA."""
    completed = _run_git(repository, ("rev-parse", "--verify", f"{revision}^{{commit}}"))
    sha = completed.stdout.strip()
    return sha


def create_candidate_tree(worktree: Path, baseline: str) -> str:
    """Write and retain the current workspace as a Git tree without changing its index."""
    with tempfile.TemporaryDirectory(prefix="self-evolve-candidate-index-") as temporary:
        environment = dict(os.environ)
        environment["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        _run_git(worktree, ("read-tree", baseline), environment)
        _run_git(worktree, ("add", "--all", "--", "."), environment)
        tree = _run_git(worktree, ("write-tree",), environment).stdout.strip()
    _run_git(worktree, ("update-ref", f"{_BASELINE_REF_PREFIX}/{tree}", tree))
    return tree


def _diff_stats(worktree: Path, base_sha: str, environment: dict[str, str]) -> tuple[FileDiffStat, ...]:
    """Return structured numstat entries relative to one baseline."""
    output = _run_git(
        worktree, ("diff", "--cached", "--numstat", "-z", "--no-renames", base_sha, "--"), environment
    ).stdout
    stats: list[FileDiffStat] = []
    for record in output.split("\0"):
        if not record:
            continue
        added_text, deleted_text, path = record.split("\t", 2)
        added_lines = None if added_text == "-" else int(added_text)
        deleted_lines = None if deleted_text == "-" else int(deleted_text)
        stats.append(FileDiffStat(path=path, added_lines=added_lines, deleted_lines=deleted_lines))
    return tuple(sorted(stats, key=lambda item: item.path))


def snapshot_candidate(worktree: Path, base_sha: str) -> CandidateSnapshot:
    """Capture all committed, staged, unstaged, and non-ignored new changes."""
    with tempfile.TemporaryDirectory(prefix="self-evolve-snapshot-index-") as temporary:
        environment = dict(os.environ)
        environment["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        _run_git(worktree, ("read-tree", base_sha), environment)
        _run_git(worktree, ("add", "--all", "--", "."), environment)
        patch = _run_git(
            worktree, ("diff", "--cached", "--binary", "--no-ext-diff", base_sha, "--"), environment
        ).stdout
        names = _run_git(
            worktree, ("diff", "--cached", "--name-only", "-z", "--no-renames", base_sha, "--"), environment
        ).stdout
        diff_stats = _diff_stats(worktree, base_sha, environment)
    changed_files = tuple(sorted(path for path in names.split("\0") if path))
    return CandidateSnapshot(patch=patch, changed_files=changed_files, diff_stats=diff_stats)
