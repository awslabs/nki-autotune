"""End-to-end test: scripts/dump_annotated_kernels.py on a cached workload."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_script_writes_four_annotated_kernels(tmp_path: Path) -> None:
    """Running the script writes canonical.py and post-step-{a,b,c}.py."""
    cache_src = Path("/home/ubuntu/cache/rmsnorm_matmul_compile")
    assert (cache_src / "f_nkigym.py").exists(), "fixture cache missing"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shutil.copy(cache_src / "f_nkigym.py", cache_dir / "f_nkigym.py")
    out_dir = tmp_path / "kernels"
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{repo_root}/nkigym/src" + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "scripts/dump_annotated_kernels.py", "--cache-dir", str(cache_dir), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    for fname in ["canonical.py", "post-step-a.py", "post-step-b.py", "post-step-c.py"]:
        p = out_dir / fname
        assert p.exists(), f"missing {p}"
        src = p.read_text()
        assert src.startswith("import nki\n")
        assert "# LoopNode(" in src
        assert "# BodyLeaf(" in src
