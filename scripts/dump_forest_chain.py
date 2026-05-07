"""Deterministic rewrite-chain dumper for the forest-IR walkthrough doc.

Applies the three fixed atoms from
``2026-05-07-forest-ir-visual-walkthrough-design/design.md`` §5:

1. ``FuseLoops(path=(), boundary=(3, 4), dim_id="d0")`` — literal.
2. ``FuseLoops(path=(), boundary=(0, 2), dim_id="d0")`` — topological.
3. ``ReorderLoops(path=(4, 0), outer_dim="d0", inner_dim="d3")``.

After each apply, writes ``<out_dir>/step_K.mmd`` via
:func:`dump_forest_mermaid`. Writes ``<out_dir>/chain.json`` at the
end with a human-readable list of the atoms applied. Aborts if any
atom is illegal on the current forest.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.mermaid import dump_forest_mermaid
from nkigym.tune.fuse_loops import FuseLoops
from nkigym.tune.reorder_loops import ReorderLoops

_ATOMS = [
    FuseLoops(path=(), boundary=(3, 4), dim_id="d0"),
    FuseLoops(path=(), boundary=(0, 2), dim_id="d0"),
    ReorderLoops(path=(4, 0), outer_dim="d0", inner_dim="d3"),
]

_INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs": ((2048, 2048), "bfloat16"),
    "rhs": ((2048, 2048), "bfloat16"),
}


def _load_f_nkigym(path: Path):
    """Load a cached f_nkigym.py module and return its ``f_nkigym`` callable."""
    spec = importlib.util.spec_from_file_location("_chain_f_nkigym", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not spec_from_file_location for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    func = getattr(module, "f_nkigym", None)
    if not callable(func):
        raise ValueError(f"{path} does not define a callable f_nkigym")
    return func


def _atom_to_dict(atom) -> dict:
    """Serialise an atom for chain.json."""
    if isinstance(atom, FuseLoops):
        return {"kind": "FuseLoops", "path": list(atom.path), "boundary": list(atom.boundary), "dim_id": atom.dim_id}
    if isinstance(atom, ReorderLoops):
        return {
            "kind": "ReorderLoops",
            "path": list(atom.path),
            "outer_dim": atom.outer_dim,
            "inner_dim": atom.inner_dim,
        }
    raise TypeError(f"unknown atom type: {type(atom).__name__}")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory containing f_nkigym.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write step_*.mmd and chain.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    f_nkigym = _load_f_nkigym(args.cache_dir / "f_nkigym.py")
    op_graph = parse_and_resolve(f_nkigym, _INPUT_SPECS)
    forest = build_canonical_forest(op_graph)

    (args.out_dir / "step_0.mmd").write_text(dump_forest_mermaid(forest=forest, op_graph=op_graph))

    for k, atom in enumerate(_ATOMS, start=1):
        if not atom.is_legal(op_graph, forest):
            raise ValueError(f"atom #{k - 1} illegal on current forest: {atom!r}")
        op_graph, forest = atom.apply(op_graph, forest)
        (args.out_dir / f"step_{k}.mmd").write_text(dump_forest_mermaid(forest=forest, op_graph=op_graph))

    (args.out_dir / "chain.json").write_text(json.dumps([_atom_to_dict(a) for a in _ATOMS], indent=2) + "\n")
    print(f"wrote step_0..step_{len(_ATOMS)}.mmd + chain.json to {args.out_dir}")


if __name__ == "__main__":
    main()
