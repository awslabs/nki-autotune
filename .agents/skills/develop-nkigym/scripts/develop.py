#!/usr/bin/env python3
"""Run the repository's deterministic develop-nkigym support commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repository_root() -> Path:
    """Find the repository containing this repo-local skill."""
    root = None
    for candidate in Path(__file__).resolve().parents:
        skill = candidate / ".agents/skills/develop-nkigym/SKILL.md"
        if skill.is_file() and (candidate / "nkigym/src/nkigym").is_dir():
            root = candidate
            break
    if root is None:
        raise RuntimeError("could not find the nki-autotune repository root")
    return root


def main() -> int:
    """Execute one skill support command with repository imports configured."""
    repository = _repository_root()
    scripts = Path(__file__).resolve().parent
    environment = dict(os.environ)
    entries = [str(scripts), str(repository / "nkigym/src"), str(repository)]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "develop_nkigym", *sys.argv[1:]]
    completed = subprocess.run(command, cwd=repository, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
