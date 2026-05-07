"""End-to-end test: scripts/dump_forest_chain.py on a cached workload."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_script_writes_step_mmds_and_chain_json(tmp_path: Path) -> None:
    """Running the script writes step_0..step_3.mmd and chain.json."""
    cache_src = Path("/home/ubuntu/cache/rmsnorm_matmul_compile")
    assert (cache_src / "f_nkigym.py").exists(), "fixture cache missing — run rmsnorm_matmul.py first"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shutil.copy(cache_src / "f_nkigym.py", cache_dir / "f_nkigym.py")
    out_dir = tmp_path / "forest_chain"
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{repo_root}/nkigym/src" + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(
        [sys.executable, "scripts/dump_forest_chain.py", "--cache-dir", str(cache_dir), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    for k in range(4):
        mmd = out_dir / f"step_{k}.mmd"
        assert mmd.exists(), f"missing {mmd}"
        src = mmd.read_text()
        assert src.startswith("graph TD\n")
    chain = json.loads((out_dir / "chain.json").read_text())
    assert len(chain) == 3
    assert chain[0]["kind"] == "FuseLoops"
    assert chain[1]["kind"] == "FuseLoops"
    assert chain[2]["kind"] == "ReorderLoops"
