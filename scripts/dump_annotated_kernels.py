"""Annotated-kernel dumper for the forest-IR walkthrough doc.

Applies the same three fixed atoms as ``scripts/dump_forest_chain.py``
and dumps an annotated NKI render at each stage via
:func:`render_annotated`. Writes:

- ``<out_dir>/canonical.py``       — canonical forest.
- ``<out_dir>/post-step-a.py``     — after the literal FuseLoops.
- ``<out_dir>/post-step-b.py``     — after the topological FuseLoops.
- ``<out_dir>/post-step-c.py``     — after the ReorderLoops.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.render import render_annotated
from nkigym.tune.fuse_loops import FuseLoops
from nkigym.tune.reorder_loops import ReorderLoops

_ATOMS = [
    ("post-step-a.py", FuseLoops(path=(), boundary=(3, 4), dim_id="d0")),
    ("post-step-b.py", FuseLoops(path=(), boundary=(0, 2), dim_id="d0")),
    ("post-step-c.py", ReorderLoops(path=(4, 0), outer_dim="d0", inner_dim="d3")),
]

_INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs": ((2048, 2048), "bfloat16"),
    "rhs": ((2048, 2048), "bfloat16"),
}


def _load_f_nkigym(path: Path):
    """Load a cached f_nkigym.py module and return its ``f_nkigym`` callable."""
    spec = importlib.util.spec_from_file_location("_annot_f_nkigym", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not spec_from_file_location for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    func = getattr(module, "f_nkigym", None)
    if not callable(func):
        raise ValueError(f"{path} does not define a callable f_nkigym")
    return func


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    f_nkigym = _load_f_nkigym(args.cache_dir / "f_nkigym.py")
    op_graph = parse_and_resolve(f_nkigym, _INPUT_SPECS)
    forest = build_canonical_forest(op_graph)

    (args.out_dir / "canonical.py").write_text(render_annotated(op_graph, forest=forest))

    for fname, atom in _ATOMS:
        if not atom.is_legal(op_graph, forest):
            raise ValueError(f"atom illegal on current forest: {atom!r}")
        op_graph, forest = atom.apply(op_graph, forest)
        (args.out_dir / fname).write_text(render_annotated(op_graph, forest=forest))

    print(f"wrote canonical.py + post-step-*.py ({len(_ATOMS)} rewrites) to {args.out_dir}")


if __name__ == "__main__":
    main()
