# Axis as First-Class Structural Entity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `IterVar.dim_id: str` with `IterVar.axis_id: int`; introduce an `Axis` record owning (opaque integer id, display name, total size, optional source-axes for fused dims). Same-dim Fuse preserves axis identity.

**Architecture:** Atomic axis identity becomes an integer counter on `KernelIR.axis_counter`, analogous to how `var_id` works on IterVar today. `DimInfo` is replaced by `Axis`. All call sites that compared `dim_id: str` now compare `axis_id: int`. The rendered output is byte-identical for canonical kernels because `Axis.name` seeds the same `i_<name>_<ordinal>` loop var spelling.

**Tech Stack:** Python 3.12, dataclasses, existing nkigym IR layers (canonical, atoms, renderer), pytest.

---

## Spec Reference

`docs/superpowers/specs/2026-05-12-axis-first-class-design.md`

## File Map

**Production code (8 files):**
- `nkigym/src/nkigym/codegen/ir.py` — add `Axis` dataclass; `IterVar.dim_id: str → axis_id: int`; `KernelIR.dims: dict[str, DimInfo] → axes: dict[int, Axis]`; add `axis_counter: int`; update `allocate_iter_var` signature; add `allocate_axis` method.
- `nkigym/src/nkigym/codegen/canonical.py` — `_derive_dims` → `_derive_axes`; all dim_id lookups → axis_id lookups; SBlock annotation `axis_map: dict[str, int]`.
- `nkigym/src/nkigym/codegen/lowering/emit_source.py` — innermost-tile identification groups by axis_id; loop-name spelling via `module.axes[iv.axis_id].name`.
- `nkigym/src/nkigym/codegen/lowering/_emit_utils.py` — name lookup updates.
- `nkigym/src/nkigym/codegen/lowering/canonicalize_names.py` — consult Axis.name.
- `nkigym/src/nkigym/codegen/lowering/place_buffers.py` — axis_id-keyed lookups.
- `nkigym/src/nkigym/tune/split.py` — identity checks via axis_id; helpers.
- `nkigym/src/nkigym/tune/fuse.py` — **same-axis fuse preserves axis_id**; cross-axis allocates fresh Axis with source_axes.
- `nkigym/src/nkigym/tune/reorder.py` — axis_id comparisons.
- `nkigym/src/nkigym/tune/compute_at.py` — axis_id comparisons.
- `nkigym/src/nkigym/tune/reverse_compute_at.py` — axis_id comparisons.
- `nkigym/src/nkigym/tune/rfactor.py` — DimInfo → Axis.

**Tests (update ~100 sites):**
- `test/codegen/test_ir.py`, `test_canonical.py`, `test_first_class_buffers.py`, `test_rfactor_rmw.py`, `test_rfactor_slot.py`, `test_batch.py`
- `test/tune/test_split.py`, `test_fuse.py`, `test_reorder.py`, `test_compute_at.py`, `test_reverse_compute_at.py`

**Test helper:** add `KernelIR.axis_id_by_name(name)` for test ergonomics.

## Execution Order

The migration is done in one atomic commit because the IR type changes ripple through every layer — a partial change doesn't compile. Task 1 does all the IR + atom + renderer changes simultaneously behind the same TDD cycle: write a failing repro test (the Fuse+Split reachability test for `kernel_0 → kernel_1`), make the whole refactor land, confirm all existing tests + the new repro test pass.

Tasks 2–N are post-refactor validation + cleanup.

---

## Task 1: Repro test that drives the whole refactor

**Files:**
- Create: `test/tune/test_axis_identity.py`

- [ ] **Step 1: Write the repro test that must pass after the refactor**

Create `/home/ubuntu/nki-autotune/test/tune/test_axis_identity.py`:

```python
"""Repro test: same-axis Fuse preserves axis identity.

Fusing outer + inner ForNode on the same axis must produce a new IterVar
whose axis_id equals the shared axis_id — NOT a fresh synthetic axis.
This unblocks kernel_transforms.py kernel_0 -> kernel_1 (Fuse+Split on
lhs_T load's d1 axis).
"""

from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import ForNode, SBlock
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.fuse import Fuse
from nkigym.tune.split import Split


@nkigym_kernel
def _matmul(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum)
    NKITensorCopy()(src=psum, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SPECS = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _find_lhs_t_load_d1_pair(module):
    """Return (outer_iv, inner_iv) — the two d1 ForNode iter-vars above lhs_T's NKILoad."""
    d1_axis_id = module.axis_id_by_name("d1")

    def walk(node, ancestors_d1):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKILoad":
            for _slot, ba in node.writes.items():
                if "lhs_T_sbuf" in ba.tensor_name:
                    return ancestors_d1
        if isinstance(node, ForNode):
            new_ancestors = ancestors_d1 + [node.iter_var] if node.iter_var.axis_id == d1_axis_id else ancestors_d1
            for c in node.children:
                r = walk(c, new_ancestors)
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None and len(r) == 2:
            return r[0], r[1]
    raise AssertionError("could not find d1 outer/inner pair for lhs_T load")


def test_same_axis_fuse_preserves_axis_id():
    """Fusing d1 outer + d1 inner preserves the axis_id (no synthetic axis)."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    outer_iv, inner_iv = _find_lhs_t_load_d1_pair(module)
    assert outer_iv.axis_id == d1_axis_id
    assert inner_iv.axis_id == d1_axis_id
    atom = Fuse(outer_iter_var_id=outer_iv.var_id, inner_iter_var_id=inner_iv.var_id)
    assert atom.is_legal(module)
    module1 = atom.apply(module)
    # After fuse, there should be exactly one d1 ForNode above the lhs_T load,
    # with extent = 2048, and its axis_id still equal to d1_axis_id.
    found_d1_ivs = []

    def collect(node):
        if isinstance(node, ForNode):
            if node.iter_var.axis_id == d1_axis_id:
                for c in node.children:
                    if isinstance(c, SBlock) and c.body and c.body[0].op_cls.__name__ == "NKILoad":
                        for _slot, ba in c.writes.items():
                            if "lhs_T_sbuf" in ba.tensor_name:
                                found_d1_ivs.append(node.iter_var)
            for c in node.children:
                collect(c)

    for root in module1.body:
        collect(root)
    assert len(found_d1_ivs) == 1
    assert found_d1_ivs[0].extent == 2048
    assert found_d1_ivs[0].axis_id == d1_axis_id


def test_split_after_same_axis_fuse_preserves_axis_id():
    """After Fuse to 2048, a subsequent Split(factors [16, 128]) produces iter-vars on the same axis."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    outer_iv, inner_iv = _find_lhs_t_load_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_iv.var_id, inner_iter_var_id=inner_iv.var_id).apply(module)

    # Find the fused d1 ForNode above lhs_T load and split it.
    def find_fused_path(module):
        def walk(node, path):
            if isinstance(node, ForNode) and node.iter_var.axis_id == d1_axis_id and node.iter_var.extent == 2048:
                # Confirm lhs_T load is a descendant.
                def has_lhs(n):
                    if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
                        for _s, ba in n.writes.items():
                            if "lhs_T_sbuf" in ba.tensor_name:
                                return True
                    if isinstance(n, ForNode):
                        return any(has_lhs(c) for c in n.children)
                    return False
                if has_lhs(node):
                    return path
                return None
            if isinstance(node, ForNode):
                for i, c in enumerate(node.children):
                    r = walk(c, path + (i,))
                    if r is not None:
                        return r
            return None

        for i, root in enumerate(module.body):
            r = walk(root, (i,))
            if r is not None:
                return r
        return None

    fused_path = find_fused_path(module)
    assert fused_path is not None
    module = Split(loop_path=fused_path, factor=128).apply(module)
    # Both resulting ForNodes should have axis_id == d1_axis_id and extents 16 and 128.
    extents = []

    def collect_d1_extents(node):
        if isinstance(node, ForNode):
            if node.iter_var.axis_id == d1_axis_id:
                # Walk to a descendant lhs_T load only.
                def has_lhs(n):
                    if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
                        for _s, ba in n.writes.items():
                            if "lhs_T_sbuf" in ba.tensor_name:
                                return True
                    if isinstance(n, ForNode):
                        return any(has_lhs(c) for c in n.children)
                    return False
                if has_lhs(node):
                    extents.append(node.iter_var.extent)
            for c in node.children:
                collect_d1_extents(c)

    for root in module.body:
        collect_d1_extents(root)
    assert sorted(extents) == [16, 128], f"expected d1 loops (16, 128) above lhs_T load; got {extents}"


def test_axis_rename_does_not_break_logic():
    """Renaming an Axis (display-only) must not affect atom legality or tree walks."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    module.axes[d1_axis_id].name = "row"
    outer_iv, inner_iv = _find_lhs_t_load_d1_pair(module)
    assert outer_iv.axis_id == d1_axis_id
    atom = Fuse(outer_iter_var_id=outer_iv.var_id, inner_iter_var_id=inner_iv.var_id)
    assert atom.is_legal(module)
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_axis_identity.py -v 2>&1 | tail -20
```

Expected: all 3 tests FAIL. `module.axis_id_by_name` doesn't exist; IterVar has no `axis_id`; same-axis fuse creates synthetic `d1_x_d1`.

Do NOT commit yet; proceed to Task 2.

---

## Task 2: IR — Axis dataclass, IterVar.axis_id, KernelIR.axes

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/ir.py`

- [ ] **Step 1: Replace `DimInfo` with `Axis`**

Open `nkigym/src/nkigym/codegen/ir.py`. Find `class DimInfo` (around line 60). Replace with:

```python
@dataclass
class Axis:
    """Logical axis — opaque integer identity plus display name.

    Identity is the ``axis_id`` int; two IterVars are on the same logical
    axis iff their ``axis_id``s are equal. The ``name`` is a cosmetic
    property consumed by the renderer to spell ``i_<name>_<ordinal>``;
    changing it never affects logic.

    For axes created by ``Fuse`` across distinct axes, ``source_axes``
    records the component axis_ids in fuse order (outer-first). Original
    axes have ``source_axes = None``.

    Attributes:
        axis_id: Monotonic unique id per module (allocated via
            :meth:`KernelIR.allocate_axis`).
        name: Display name, e.g. ``"d0"``, ``"row"``. Purely cosmetic.
        total_size: Axis extent in elements.
        source_axes: Tuple of component axis_ids if this axis was created
            by cross-axis Fuse; ``None`` for original axes.
    """

    axis_id: int
    name: str
    total_size: int
    source_axes: tuple[int, ...] | None = None
```

- [ ] **Step 2: Update `IterVar` to use `axis_id`**

Still in `ir.py`, find `class IterVar` (around line 70). Replace `dim_id: str` with `axis_id: int`:

```python
@dataclass(frozen=True)
class IterVar:
    """Stable identity for a loop iteration variable.

    Created by the canonical builder and every Split / Fuse. Never
    mutated — atoms retire iter vars and emit fresh ones.

    Attributes:
        var_id: Monotonic unique id per module.
        axis_id: Logical axis this iter var traverses (integer, stable).
        extent: Trip count (# tiles).
        role: PARALLEL / SEQUENTIAL / ACCUMULATION.
    """

    var_id: int
    axis_id: int
    extent: int
    role: AxisRole
```

- [ ] **Step 3: Update `KernelIR` — replace dims with axes, add axis_counter and allocate_axis**

Find `class KernelIR`. Update the fields and methods:

```python
@dataclass
class KernelIR:
    """Envelope IR — signature + tensor/axis declarations + schedule tree.

    Analog of TVM's PrimFunc + buffer_map.

    Attributes:
        func_name: Emitted kernel name.
        param_names: Signature order.
        return_name: Tensor name of the return.
        tensors: All named tensors, keyed by name.
        axes: All logical axes, keyed by axis_id.
        iter_var_counter: Monotonic counter for allocating IterVar.var_id.
        axis_counter: Monotonic counter for allocating Axis.axis_id.
        body: Schedule tree — list of ForNode / SBlock roots.
        dep: Per-scope dependency cache (lazy).
        fused_iter_var_map: Retired iter-var id -> (fused iter-var id,
            inner extent, is_outer). Populated by the ``Fuse`` atom. The
            renderer consults this map when emitting accesses that still
            reference the retired outer/inner ids: outer decomposes as
            ``(fused // inner_extent)`` and inner as ``(fused % inner_extent)``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    axes: dict[int, Axis]
    iter_var_counter: int = 0
    axis_counter: int = 0
    body: TreeIR = field(default_factory=list)
    dep: DepCache = field(default_factory=lambda: DepCache(scopes={}))
    fused_iter_var_map: dict[int, tuple[int, int, bool]] = field(default_factory=dict)

    def allocate_axis(self, name: str, total_size: int, source_axes: tuple[int, ...] | None = None) -> Axis:
        """Allocate a fresh Axis with a monotonic unique axis_id."""
        axis = Axis(axis_id=self.axis_counter, name=name, total_size=total_size, source_axes=source_axes)
        self.axes[axis.axis_id] = axis
        self.axis_counter += 1
        return axis

    def allocate_iter_var(self, axis_id: int, extent: int, role: AxisRole) -> IterVar:
        """Allocate a fresh IterVar with a monotonic unique var_id.

        Never reuses retired var_ids.

        Args:
            axis_id: The logical axis this iter var traverses.
            extent: The trip count (# tiles).
            role: PARALLEL / SEQUENTIAL / ACCUMULATION.

        Returns:
            A freshly-allocated IterVar.
        """
        iv = IterVar(var_id=self.iter_var_counter, axis_id=axis_id, extent=extent, role=role)
        self.iter_var_counter += 1
        return iv

    def axis_id_by_name(self, name: str) -> int:
        """Return the axis_id for the axis with the given display name.

        Raises ``KeyError`` if no axis has this name, or if multiple axes share it.
        """
        matches = [a.axis_id for a in self.axes.values() if a.name == name]
        if not matches:
            raise KeyError(f"no axis with name {name!r}")
        if len(matches) > 1:
            raise KeyError(f"multiple axes with name {name!r}: {matches}")
        return matches[0]
```

- [ ] **Step 4: Verify ir.py parses and has no syntax errors**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python -c "from nkigym.ir.ir import Axis, IterVar, KernelIR; print('ir.py: ok')"
```

Expected: `ir.py: ok`.

- [ ] **Step 5: Do not commit yet — Task 3 ripples through all consumers**

---

## Task 3: Canonical builder

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/canonical.py`

- [ ] **Step 1: Replace `_derive_dims` with `_derive_axes`**

Find `_derive_dims` in canonical.py (around line 413). Replace:

```python
def _derive_axes(module: KernelIR, dim_sizes: dict[str, int]) -> dict[str, int]:
    """Allocate one Axis per concrete dim, returning a ``name → axis_id`` map.

    Called early in canonical build so subsequent helpers can translate
    abstract-axis strings → concrete axis_ids via the returned map.
    """
    name_to_axis_id: dict[str, int] = {}
    for name, size in dim_sizes.items():
        axis = module.allocate_axis(name=name, total_size=size)
        name_to_axis_id[name] = axis.axis_id
    return name_to_axis_id
```

- [ ] **Step 2: Update `build_initial_ir` to construct KernelIR with axes dict**

Find the place canonical constructs the `KernelIR`. It currently calls `_derive_dims` and passes `dims=...`. Change to: construct an empty `KernelIR` first (with `axes={}`), then call `_derive_axes(module, dim_sizes)` to populate axes and return the name→axis_id map.

Concrete change: locate the `_derive_dims(...)` call (around line 60 of canonical.py based on prior reads) and the `KernelIR(...)` constructor call. Rewrite:

```python
module = KernelIR(
    func_name=...,  # existing args
    param_names=...,
    return_name=...,
    tensors=tensors,
    axes={},
)
name_to_axis_id = _derive_axes(module, dim_sizes)
```

Then thread `name_to_axis_id` through `_build_parsed_ops`, `_make_sblock`, `_derive_op_tiles` so they can translate abstract axis strings / dim names to axis_ids.

- [ ] **Step 3: Update `_derive_op_tiles` signature**

`_derive_op_tiles` currently returns `list[dict[str, int]]` keyed by dim_id. Change it to return `list[dict[int, int]]` keyed by axis_id:

```python
def _derive_op_tiles(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    dim_sizes: dict[str, int],
    name_to_axis_id: dict[str, int],
) -> list[dict[int, int]]:
    """Per-op tile size map: ``op_tiles[i][axis_id] = tile_for_that_op_on_that_axis``.

    For each op, every axis it touches gets a tile size derived from that
    op's ``MAX_TILE_SIZE`` (the largest legal innermost-tile extent). An
    abstract axis with ``MAX_TILE_SIZE[axis] = None`` (or absent entry)
    defaults to the full extent.
    """
    out: list[dict[int, int]] = []
    for raw, axis_map in zip(raws, per_op_axis_maps):
        tiles: dict[int, int] = {}
        for abstract, dim_name in axis_map.items():
            axis_id = name_to_axis_id[dim_name]
            max_tile = raw.op_cls.MAX_TILE_SIZE.get(abstract)
            total = dim_sizes[dim_name]
            if max_tile is None:
                tiles[axis_id] = total
            else:
                if total % max_tile != 0:
                    raise ValueError(
                        f"{raw.op_cls.__name__}.{abstract}: extent {total} not divisible by MAX_TILE_SIZE {max_tile}"
                    )
                tiles[axis_id] = min(max_tile, total)
        out.append(tiles)
    return out
```

- [ ] **Step 4: Update `_ParsedOp.dim_tile` → `axis_tile`, `touched_dims` → `touched_axes`, `dim_role` → `axis_role`**

`_ParsedOp` (the dataclass at ~line 140) currently has fields `touched_dims: tuple[str, ...]`, `axis_map: dict[str, str]`, `dim_tile: dict[str, int]`, `dim_role: dict[str, AxisRole]`. Change to:

```python
@dataclass(frozen=True)
class _ParsedOp:
    """..."""
    idx: int
    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, int]     # abstract → axis_id (was: str dim_id)
    touched_axes: tuple[int, ...]  # was: touched_dims (strs)
    axis_role: dict[int, AxisRole]  # was: dim_role
    axis_tile: dict[int, int]    # was: dim_tile (str→int)
```

Update `_build_parsed_ops`, `_resolve_dim_role → _resolve_axis_role`, `_touched_dims → _touched_axes` accordingly — they just need to translate the string dim_names they compute into axis_ids via `name_to_axis_id`.

- [ ] **Step 5: Update `_make_sblock` to allocate iter-vars with axis_id**

Today `_make_sblock` calls `module.allocate_iter_var(dim_id, extent, role)`. Change the signature expectation: pass `axis_id` instead. The `for dim_id in op.touched_dims` loop becomes `for axis_id in op.touched_axes`.

Also update `SBlock` construction to store `annotations={"axis_map": dict(op.axis_map)}` — the new `axis_map` maps `abstract_name_str → axis_id_int`.

- [ ] **Step 6: Update `_build_buffer_access` / any AccessRange construction**

These use the iter-var maps (dim_to_outer/dim_to_inner). Just change the key type from str to int to match.

- [ ] **Step 7: Verify canonical.py imports `Axis` (not `DimInfo`)**

At the top of canonical.py, change:
```python
from nkigym.ir.ir import DimInfo, ...
```
to:
```python
from nkigym.ir.ir import Axis, ...
```
(or just remove DimInfo if still present; Axis may not be needed to be named directly in this module.)

- [ ] **Step 8: Run canonical tests — expect failures, diagnose from there**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_canonical.py -v 2>&1 | tail -30
```

Expected: failures due to test-side `iv.dim_id` / `module.dims` references. These will be fixed in Task 9 (batch test updates).

If there are production-side import or attribute errors, fix them before moving on.

- [ ] **Step 9: Do not commit yet — rest of production code in Tasks 4–8**

---

## Task 4: Renderer — emit_source, _emit_utils, canonicalize_names

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/emit_source.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/_emit_utils.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/canonicalize_names.py`

- [ ] **Step 1: Update `_innermost_tile_iter_var_ids` to group by axis_id**

In `emit_source.py`, the helper currently does `by_dim: dict[str, list[int]] = {}` and groups by `iv.dim_id`. Change to:

```python
def _innermost_tile_iter_var_ids(module: "KernelIR") -> set[int]:
    """Collect var_ids of iter-vars that are each op's innermost tile loop per axis."""
    from nkigym.ir.ir import ForNode, SBlock

    ids: set[int] = set()

    def visit(node):
        if isinstance(node, SBlock):
            by_axis: dict[int, list[int]] = {}
            for iv in node.iter_vars:
                by_axis.setdefault(iv.axis_id, []).append(iv.var_id)
            for _axis_id, id_list in by_axis.items():
                ids.add(id_list[-1])
        elif isinstance(node, ForNode):
            for c in node.children:
                visit(c)

    for root in module.body:
        visit(root)
    return ids
```

- [ ] **Step 2: Update `_emit_node` name fallback**

Currently: `name = node.name if node.name is not None else f"i_{iv.dim_id}_{iv.var_id}"`. Change to use the axis's display name:

```python
iv = node.iter_var
axis_name = ctx.module.axes[iv.axis_id].name
name = node.name if node.name is not None else f"i_{axis_name}_{iv.var_id}"
```

- [ ] **Step 3: Update `canonicalize_iter_var_names`**

Open `canonicalize_names.py`. Find where it assigns names like `i_<dim>_<ordinal>`. Change `<dim>` to come from `module.axes[iv.axis_id].name`.

- [ ] **Step 4: Update `_emit_utils.py` name helpers**

Any `_resolve_iv_name`-style helper that uses `iv.dim_id` in spelling changes to `module.axes[iv.axis_id].name`. Logic-level checks (innermost elision) already use `iv.var_id` and sets, which don't care about dim_id.

- [ ] **Step 5: Verify the module imports work**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python -c "
from nkigym.codegen.render import render
from nkigym.ir.build import build_initial_ir
print('renderer imports: ok')
"
```

Expected: `renderer imports: ok`.

- [ ] **Step 6: Do not commit yet — continue to Task 5**

---

## Task 5: place_buffers

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/place_buffers.py`

- [ ] **Step 1: Replace all `dim_id`-keyed lookups with `axis_id`**

Open `place_buffers.py`. Grep for `dim_id` and `module.dims`:

```bash
grep -n "dim_id\|module\.dims" /home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/lowering/place_buffers.py
```

Each hit needs mechanical translation:
- `module.dims[dim_id]` → `module.axes[axis_id]`
- `iv.dim_id` → `iv.axis_id`
- `DimInfo` references → `Axis`

Key places to check:
- The LCA walk that groups iter-vars by dim. Switch to grouping by axis_id.
- Buffer-shape computation that uses `module.dims[d].total_size`. Switch to `module.axes[aid].total_size`.

- [ ] **Step 2: Run place_buffers tests (if any) + canonical build**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/codegen/test_place_buffers.py -v 2>&1 | tail -20
```

Expected: some tests may fail on assertions about `module.dims`; production code should import/run without errors.

- [ ] **Step 3: Do not commit yet**

---

## Task 6: Atom migrations — split, fuse, reorder, compute_at, reverse_compute_at

**Files:**
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/split.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/fuse.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/reorder.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/compute_at.py`
- Modify: `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/reverse_compute_at.py`

- [ ] **Step 1: Update split.py**

In `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/split.py`:

- `Split.apply`: `module.allocate_iter_var(old_iv.dim_id, ..., ...)` becomes `module.allocate_iter_var(old_iv.axis_id, ..., ...)`.
- `_check_min_max` uses `target.iter_var.dim_id` (called `target_dim`) to find SBlocks with matching iter-vars. Change to `target.iter_var.axis_id` (renamed `target_axis_id`) and compare `iv.axis_id` instead of `iv.dim_id`.
- Helper `_abstract_axis_for(block, dim_id)` becomes `_abstract_axis_for(block, axis_id)` — takes int, same reverse-lookup logic against `block.annotations["axis_map"]` which now maps `abstract_str → axis_id_int`.

Concrete diff:

```python
def _abstract_axis_for(block: "SBlock", axis_id: int) -> str | None:
    """Look up the op's abstract axis name corresponding to ``axis_id`` for this block."""
    axis_map = block.annotations.get("axis_map", {})
    for abstract, aid in axis_map.items():
        if aid == axis_id:
            return abstract
    return None
```

- [ ] **Step 2: Update fuse.py — same-axis fuse preserves axis_id**

In `/home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/fuse.py`:

Replace the body of `apply` that currently creates a synthetic dim. New logic:

```python
def apply(self, module: KernelIR) -> KernelIR:
    """Execute the fuse."""
    if not self.is_legal(module):
        raise AtomLegalityError(f"Fuse.apply: illegal {self!r}")
    pair = _find_pair(module, self.outer_iter_var_id, self.inner_iter_var_id)
    assert pair is not None
    outer_node, inner_node, outer_path = pair
    v_outer = outer_node.iter_var
    v_inner = inner_node.iter_var

    fused_extent = v_outer.extent * v_inner.extent
    fused_role = max(v_outer.role, v_inner.role, key=_ROLE_RANK.__getitem__)

    if v_outer.axis_id == v_inner.axis_id:
        """Same-axis fuse: reuse the existing axis_id; no new Axis needed."""
        fused_axis_id = v_outer.axis_id
    else:
        """Cross-axis fuse: allocate a fresh Axis with source_axes trace."""
        outer_axis = module.axes[v_outer.axis_id]
        inner_axis = module.axes[v_inner.axis_id]
        fused_axis = module.allocate_axis(
            name=f"{outer_axis.name}_x_{inner_axis.name}",
            total_size=fused_extent,
            source_axes=(v_outer.axis_id, v_inner.axis_id),
        )
        fused_axis_id = fused_axis.axis_id

    v_fused = module.allocate_iter_var(axis_id=fused_axis_id, extent=fused_extent, role=fused_role)

    """Record the decomposition so the renderer can emit
    ``(fused // inner_extent)`` and ``(fused % inner_extent)``."""
    module.fused_iter_var_map[v_outer.var_id] = (v_fused.var_id, v_inner.extent, True)
    module.fused_iter_var_map[v_inner.var_id] = (v_fused.var_id, v_inner.extent, False)

    new_body = _collapse_iter_var_lists(module.body, v_outer.var_id, v_inner.var_id, v_fused)
    new_fornode = ForNode(
        iter_var=v_fused, children=list(inner_node.children), name=None, annotations=dict(outer_node.annotations)
    )
    new_body = replace_at_path(new_body, outer_path, new_fornode)
    return replace(module, body=new_body)
```

Also: `_check_min_max` uses `outer.iter_var.dim_id` — change to `outer.iter_var.axis_id`, and the `fused_dim = d_o` naming becomes `fused_axis_id = v_outer.axis_id`. Import `_abstract_axis_for` from split (already imported; no change needed).

Remove the `from nkigym.ir.ir import DimInfo, ...` import — DimInfo no longer exists.

- [ ] **Step 3: Update reorder.py, compute_at.py, reverse_compute_at.py**

Grep for `dim_id` in each:

```bash
grep -n "dim_id" /home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/reorder.py
grep -n "dim_id" /home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/compute_at.py
grep -n "dim_id" /home/ubuntu/nki-autotune/nkigym/src/nkigym/tune/reverse_compute_at.py
```

For each hit:
- If it's comparing two iter-vars (e.g. `iv1.dim_id == iv2.dim_id`): replace with `iv1.axis_id == iv2.axis_id`.
- If it's looking up something in a map: change the key type to int.
- If it's used for display/error messages: change to `module.axes[iv.axis_id].name`.

- [ ] **Step 4: Update rfactor.py**

`rfactor.py` has heavy `DimInfo` usage (~48 dim_id hits). Walk through each reference and convert. The pattern:
- `DimInfo(dim_id=..., total_size=...)` → `module.allocate_axis(name=..., total_size=...)` (returns `Axis`, but the module stores it; the caller then uses `.axis_id`).
- `module.dims[outer_dim_id] = DimInfo(...)` → `module.allocate_axis(name=outer_dim_id_name, total_size=...)`.
- Any function taking `dims: dict[str, DimInfo]` takes `axes: dict[int, Axis]`.
- Freshness helpers `_fresh_dim_id(dims)` → `_fresh_axis_name(axes)` that returns a display name (still a string, but just cosmetic).

This is the biggest file touch in the refactor.

- [ ] **Step 5: Module-level import cleanup**

In fuse.py and rfactor.py, remove `DimInfo` from imports:

```python
from nkigym.ir.ir import Axis, ForNode, IterVar, KernelIR, SBlock, TreeIR, replace_at_path
```

- [ ] **Step 6: Do not commit yet**

---

## Task 7: Quick smoke test — canonical + render a trivial kernel

- [ ] **Step 1: Try to render the canonical matmul**

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python - <<'PY'
from nkigym.ir.build import build_initial_ir
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy

@nkigym_kernel
def mm(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a)
    NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s)
    NKIStore()(src=s, dst=h)
    return h

module = build_initial_ir(mm, input_specs={
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs":   {"shape": (2048, 2048), "dtype": "bfloat16"},
})
print(f"axes: {[(a.axis_id, a.name, a.total_size) for a in module.axes.values()]}")
print(render(module)[:500])
PY
```

Expected:
- Prints a list like `[(0, "d0", 2048), (1, "d1", 2048), (2, "d3", 2048)]`.
- Prints kernel source starting with `import nki\n...`.

If any production-side error appears, fix it in Tasks 2–6 and re-run.

- [ ] **Step 2: Do not commit yet**

---

## Task 8: CPU-sim regression on the smoke-test kernel

- [ ] **Step 1: Verify the rendered kernel CPU-sims correctly**

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python - <<'PY'
import numpy as np
import nki
from nkigym.ir.build import build_initial_ir
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.verify import _rewrite_to_fp32

@nkigym_kernel
def mm(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a)
    NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s)
    NKIStore()(src=s, dst=h)
    return h

module = build_initial_ir(mm, input_specs={"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}})
source = render(module)
sim_src = _rewrite_to_fp32(source)
ns = {}
exec(sim_src, ns)
rng = np.random.default_rng(0)
lhs_T = rng.standard_normal((2048, 2048)).astype(np.float32)
rhs = rng.standard_normal((2048, 2048)).astype(np.float32)
actual = np.asarray(nki.simulate(ns["mm"])(lhs_T, rhs))
expected = lhs_T.T @ rhs
ok = np.allclose(actual, expected, atol=5e-3, rtol=5e-3)
print(f"CPU-sim pass={ok} max_abs={float(np.abs(actual - expected).max()):.3e}")
PY
```

Expected: `CPU-sim pass=True max_abs=1.297e-04`.

- [ ] **Step 2: Do not commit yet**

---

## Task 9: Migrate tests — mechanical dim_id → axis_id substitution

**Files:**
- `/home/ubuntu/nki-autotune/test/codegen/test_ir.py`
- `/home/ubuntu/nki-autotune/test/codegen/test_canonical.py`
- `/home/ubuntu/nki-autotune/test/codegen/test_first_class_buffers.py`
- `/home/ubuntu/nki-autotune/test/codegen/test_rfactor_rmw.py`
- `/home/ubuntu/nki-autotune/test/codegen/test_rfactor_slot.py`
- `/home/ubuntu/nki-autotune/test/codegen/test_batch.py`
- `/home/ubuntu/nki-autotune/test/codegen/test_place_buffers.py`
- `/home/ubuntu/nki-autotune/test/tune/test_split.py`
- `/home/ubuntu/nki-autotune/test/tune/test_fuse.py`
- `/home/ubuntu/nki-autotune/test/tune/test_reorder.py`
- `/home/ubuntu/nki-autotune/test/tune/test_compute_at.py`
- `/home/ubuntu/nki-autotune/test/tune/test_reverse_compute_at.py`

- [ ] **Step 1: Translation patterns**

For each test file, apply these patterns:

1. `iv.dim_id == "d0"` → `iv.axis_id == module.axis_id_by_name("d0")` (cache the axis_id local at the top of the test).
2. `iv.dim_id` (as display string in error messages or tuple keys) → `module.axes[iv.axis_id].name`.
3. `module.dims` → `module.axes`.
4. `module.dims["d0"]` → `module.axes[module.axis_id_by_name("d0")]`.
5. Test fixtures that instantiate `DimInfo` directly → use `KernelIR.allocate_axis` or remove if unnecessary.

- [ ] **Step 2: Run each test file individually until each passes**

```bash
source ~/venvs/kernel-env/bin/activate
for f in test/codegen/test_ir.py test/codegen/test_canonical.py test/codegen/test_first_class_buffers.py test/codegen/test_rfactor_rmw.py test/codegen/test_rfactor_slot.py test/codegen/test_batch.py test/codegen/test_place_buffers.py test/tune/test_split.py test/tune/test_fuse.py test/tune/test_reorder.py test/tune/test_compute_at.py test/tune/test_reverse_compute_at.py; do
  echo "=== $f ==="
  PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/$f --tb=line 2>&1 | tail -3
done
```

Expected: each file passes (except `test_batch.py` which has a known pre-existing failure we are not fixing).

- [ ] **Step 3: Run the new axis-identity test**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/tune/test_axis_identity.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 4: Full suite**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src pytest /home/ubuntu/nki-autotune/test/ --ignore=/home/ubuntu/nki-autotune/test/codegen/test_batch.py 2>&1 | tail -10
```

Expected: all PASS.

- [ ] **Step 5: Do not commit yet — run kernel regressions first**

---

## Task 10: Regression — kernel_transforms.py + examples

- [ ] **Step 1: All 15 hand kernels**

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python /home/ubuntu/nki-autotune/kernel_transforms.py 2>&1 | tail -20
```

Expected: 15 `pass=True` lines.

- [ ] **Step 2: Examples**

```bash
PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python - <<'PY'
from nkigym.ir.build import build_initial_ir
from nkigym.codegen.render import render
from nkigym.tune.verify import _verify
from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym, K, M, N

INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
INPUT_SPECS_DICT = {"lhs_T": {"shape": (K, M), "dtype": "bfloat16"}, "rhs": {"shape": (K, N), "dtype": "bfloat16"}}
module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs=INPUT_SPECS_DICT)
source = render(module)
_verify(source, matmul_lhsT_rhs_nkigym, INPUT_SPECS)
print("matmul_lhsT_rhs: PASS")

import sys
sys.path.insert(0, "/home/ubuntu/nki-autotune/test/codegen")
from _rmsnorm_matmul_fixture import f_nkigym, INPUT_SPECS as RSN_SPECS
RSN_SPECS_DICT = {name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in RSN_SPECS.items()}
module = build_initial_ir(f_nkigym, input_specs=RSN_SPECS_DICT)
source = render(module)
_verify(source, f_nkigym, RSN_SPECS)
print("rmsnorm_matmul: PASS")
PY
```

Expected: both `PASS`.

- [ ] **Step 3: Grep for stragglers**

```bash
grep -rn "dim_id\|DimInfo" /home/ubuntu/nki-autotune/nkigym/src /home/ubuntu/nki-autotune/test 2>/dev/null | grep -v ".bak" | grep -v "ir_v1.py.bak" | head -20
```

Expected: empty (or only docstrings describing the *old* concept for historical reference — those should be cleaned up).

- [ ] **Step 4: Commit the whole refactor**

```bash
cd /home/ubuntu/nki-autotune
git add -A
git commit -m "refactor: axis as first-class structural entity (dim_id str → axis_id int)

Replaces IterVar.dim_id (str) with IterVar.axis_id (int). Identity
becomes a counter-allocated integer on the module's new axis_counter;
Axis records own display names (cosmetic property for rendering).

Same-axis Fuse now preserves axis identity: fusing outer+inner on the
same axis produces an IterVar with the same axis_id, not a synthetic
'a_x_b' dim. Unblocks kernel_0 -> kernel_1 Fuse+Split reachability.

Cross-axis Fuse allocates a fresh Axis with source_axes=(outer, inner)
trace; display name derived but not semantically load-bearing.

All string identity checks across renderer, atoms, canonical, tests
switched to integer axis_id comparisons. DimInfo replaced by Axis.

Spec: docs/superpowers/specs/2026-05-12-axis-first-class-design.md"
```

---

## Task 11: Documentation cleanup

**Files:**
- `/home/ubuntu/nki-autotune/docs/ir-design.md`

- [ ] **Step 1: Grep for dim_id references**

```bash
grep -n "dim_id\|DimInfo" /home/ubuntu/nki-autotune/docs/ir-design.md
```

- [ ] **Step 2: Update doc**

For each hit:
- "dim_id" → "axis_id"
- "DimInfo" → "Axis"
- Add a sentence or two in the IR section explaining: "Axis identity is an opaque integer `axis_id`; display names live on `Axis.name` and are purely cosmetic."

- [ ] **Step 3: Commit**

```bash
cd /home/ubuntu/nki-autotune
git add docs/ir-design.md
git commit -m "docs: ir-design — Axis replaces DimInfo; axis_id is integer identity"
```

---

## Self-Review Checklist

1. **Spec coverage** — Each requirement in `docs/superpowers/specs/2026-05-12-axis-first-class-design.md`:
   - `Axis` dataclass (spec §"IR Changes") → Task 2.
   - `IterVar.axis_id` replaces `dim_id` (spec §"IR Changes") → Task 2.
   - `KernelIR.axes`, `axis_counter`, `allocate_axis` → Task 2.
   - Same-axis Fuse preserves identity (spec §"Atom Behavior") → Task 6, Step 2.
   - Cross-axis Fuse allocates fresh Axis with source_axes → Task 6, Step 2.
   - Canonical name-to-axis-id routing → Task 3.
   - Renderer uses `Axis.name` for display, `axis_id` for logic → Task 4.
   - `axis_map: dict[str, int]` on SBlock annotations → Task 3, Step 5.
   - Test helper `axis_id_by_name` → Task 2, Step 3.
   - Rename-does-not-break-logic test → Task 1 third test.

2. **Placeholder scan** — No TODO / TBD / "similar to". Every task has concrete code or concrete commands. Translation patterns in Task 9 are explicit.

3. **Type consistency** — `axis_id: int` is consistent across tasks. `Axis(axis_id, name, total_size, source_axes)` shape identical at every reference. `module.axes: dict[int, Axis]` consistent.

4. **Task ordering** — Production code changes (Tasks 2–6) are bundled; tests are migrated in Task 9 after production compiles; regressions in Task 10; then commit. This order keeps individual tasks reviewable while respecting the atomic nature of the refactor.
