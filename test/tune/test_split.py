"""Unit tests for the Split atom on the iter-var IR."""

import pytest

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune import AtomLegalityError
from nkigym.tune.split import Split, enumerate_split_atoms


@nkigym_kernel
def _matmul_large(lhs_T, rhs):
    """2048 matmul fixture - multi-tile on all three dims (num_tiles=16 on K, M, 4 on N)."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SPECS_LARGE = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _find_first_for_path(module, predicate):
    """Return path to the first ForNode matching ``predicate(node)``."""

    def walk(node, path):
        result = None
        if isinstance(node, ForNode) and predicate(node):
            result = path
        elif isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                found = walk(c, path + (i,))
                if found is not None:
                    result = found
                    break
        return result

    for i, root in enumerate(module.body):
        found = walk(root, (i,))
        if found is not None:
            return found
    raise AssertionError("No matching ForNode")


def test_split_divisor_factor_creates_outer_inner_pair() -> None:
    """Split(factor=4) on a trip=16 loop produces trip-4 trip-4 nested pair."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d0 = module.axis_id_by_name("d0")
    path = _find_first_for_path(module, lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16)
    atom = Split(loop_path=path, factor=4)
    assert atom.is_legal(module)
    new_mod = atom.apply(module)

    """Descend to the target path - it should now be the outer ForNode with extent=4."""
    node = new_mod.body[path[0]]
    for idx in path[1:]:
        assert isinstance(node, ForNode)
        node = node.children[idx]
    assert isinstance(node, ForNode)
    assert node.iter_var.extent == 4
    inner = node.children[0]
    assert isinstance(inner, ForNode)
    assert inner.iter_var.extent == 4
    assert inner.iter_var.axis_id == d0


def test_split_non_divisor_factor_rejects() -> None:
    """Split(factor=3) on trip=16 is illegal."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d0 = module.axis_id_by_name("d0")
    path = _find_first_for_path(module, lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16)
    atom = Split(loop_path=path, factor=3)
    assert not atom.is_legal(module)
    with pytest.raises(AtomLegalityError):
        atom.apply(module)


def test_split_factor_1_is_illegal() -> None:
    """factor must be > 1."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d0 = module.axis_id_by_name("d0")
    path = _find_first_for_path(module, lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16)
    assert not Split(loop_path=path, factor=1).is_legal(module)


def test_split_factor_equal_trip_is_illegal() -> None:
    """factor must be < extent."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d0 = module.axis_id_by_name("d0")
    path = _find_first_for_path(module, lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16)
    assert not Split(loop_path=path, factor=16).is_legal(module)


def test_split_rewrites_buffer_access_patterns() -> None:
    """After Split, BufferAccess.pattern references v_outer + v_inner with
    proper coefficients, and the renderer emits the composite expression."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d0 = module.axis_id_by_name("d0")
    path = _find_first_for_path(module, lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16)
    atom = Split(loop_path=path, factor=4)
    new_mod = atom.apply(module)
    source = render(new_mod)
    """Expect the split loop headers."""
    assert "for i_d0_0 in range(4):" in source
    """Expect the inner loop on same dim."""
    assert "for i_d0_1 in range(4):" in source
    """Expect composite indexing: v_outer * inner_extent + v_inner = i_d0_0 * 4 + i_d0_1.
    Rendered as the slot index for lhs_T_sbuf / rhs_sbuf or as tile scaling on HBM."""
    assert "i_d0_0 * 4 + i_d0_1" in source or "i_d0_0 * 4" in source


def test_split_updates_sblock_iter_vars() -> None:
    """After splitting the d0 outer trip ForNode, the SBlock's iter_vars
    list replaces the original d0 outer (extent=16) with the pair
    (v_outer_outer=4, v_outer_inner=4), plus the canonical d0 inner tile
    iter-var (extent=128) stays intact. Total: 3 d0 iter-vars."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    """Find the matmul SBlock's d0 outer-trip iter var (extent=16
    ACCUMULATION). The matmul spine post-Task 4 has d0_outer (extent=16)
    > d0_inner (extent=128); we split the outer."""
    from nkigym.ops.base import AxisRole

    d0 = module.axis_id_by_name("d0")
    mm_path = _find_first_for_path(
        module,
        lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16 and n.iter_var.role == AxisRole.ACCUMULATION,
    )
    atom = Split(loop_path=mm_path, factor=4)
    new_mod = atom.apply(module)

    """Find the matmul SBlock."""

    def collect_blocks(node, blocks):
        if isinstance(node, SBlock):
            blocks.append(node)
        else:
            for c in node.children:
                collect_blocks(c, blocks)

    blocks = []
    for root in new_mod.body:
        collect_blocks(root, blocks)
    mm_block = next(b for b in blocks if any(c.op_cls.__name__ == "NKIMatmul" for c in b.body))
    """d0 appears 3 times: (v_outer_outer=4, v_outer_inner=4) from the
    split + the canonical d0 inner tile (extent=128)."""
    d0_ivs = [iv for iv in mm_block.iter_vars if iv.axis_id == d0]
    assert len(d0_ivs) == 3
    extents = sorted(iv.extent for iv in d0_ivs)
    assert extents == [4, 4, 128]


def test_enumerate_split_atoms_yields_divisor_factors_per_loop() -> None:
    """Enumerator emits Split(loop, factor) for every divisor in (2, extent)."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    atoms = enumerate_split_atoms(module)
    """Every atom must be legal."""
    for atom in atoms:
        assert atom.is_legal(module)
    """At least one atom on d0 (trip=16) with factor=4."""
    d0_factor_4 = [a for a in atoms if a.factor == 4]
    assert len(d0_factor_4) > 0


def test_split_apply_preserves_sblock_annotations() -> None:
    """Annotations on SBlocks and other ForNodes survive Split."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d0 = module.axis_id_by_name("d0")
    path = _find_first_for_path(module, lambda n: n.iter_var.axis_id == d0 and n.iter_var.extent == 16)
    """Attach annotation to target before split."""
    node = module.body[path[0]]
    for idx in path[1:]:
        assert isinstance(node, ForNode)
        node = node.children[idx]
    assert isinstance(node, ForNode)
    node.annotations["test_key"] = "preserved"

    atom = Split(loop_path=path, factor=4)
    new_mod = atom.apply(module)
    """Annotation should land on the outer ForNode (replacing target)."""
    new_outer = new_mod.body[path[0]]
    for idx in path[1:]:
        assert isinstance(new_outer, ForNode)
        new_outer = new_outer.children[idx]
    assert isinstance(new_outer, ForNode)
    """Annotation preserved on outer (or inner - implementation choice).
    Spec is ambiguous; test that it's preserved somewhere in the pair."""
    outer_has = new_outer.annotations.get("test_key") == "preserved"
    inner_has = False
    if new_outer.children:
        inner_child = new_outer.children[0]
        if isinstance(inner_child, ForNode):
            inner_has = inner_child.annotations.get("test_key") == "preserved"
    assert outer_has or inner_has


def test_split_rejects_when_innermost_goes_below_min_tile() -> None:
    """Splitting a fixed-tile innermost loop (matmul M inner, extent=128) further is illegal.

    Matmul M: MIN=MAX=128. Inner tile loop has extent 128 already; any split factor would
    produce factor<128 (since factor must be a proper divisor), violating MIN=128.
    """
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d1 = module.axis_id_by_name("d1")

    def find_inner_matmul_m(module):
        """Return path to the deepest ForNode with axis_id=d1 extent=128 whose
        descendants include a matmul SBlock."""
        candidates: list[tuple[int, ...]] = []

        def has_matmul(n):
            result = False
            if isinstance(n, SBlock):
                result = bool(n.body) and n.body[0].op_cls.__name__ == "NKIMatmul"
            elif isinstance(n, ForNode):
                result = any(has_matmul(c) for c in n.children)
            return result

        def scan(node, path):
            if isinstance(node, ForNode):
                iv = node.iter_var
                if iv.extent == 128 and iv.axis_id == d1 and has_matmul(node):
                    candidates.append(path)
                for i, c in enumerate(node.children):
                    scan(c, path + (i,))

        for i, r in enumerate(module.body):
            scan(r, (i,))
        assert candidates, "no matmul d1 extent=128 ForNode"
        return max(candidates, key=len)

    path = find_inner_matmul_m(module)
    """factor=2 would make the new inner extent 2 < MIN=128; must be rejected."""
    atom = Split(loop_path=path, factor=2)
    assert atom.is_legal(module) is False


def test_split_accepts_outer_loop_split() -> None:
    """Splitting an OUTER trip loop doesn't affect any leaf's innermost; should be legal."""
    module = build_canonical_module(_matmul_large, _SPECS_LARGE)
    d1 = module.axis_id_by_name("d1")

    def find_outer_M(module):
        """Find an outer matmul M loop (extent = 2048/128 = 16, axis=d1, NOT the innermost)."""
        result = None

        def walk(node, path):
            nonlocal result
            if result is not None:
                return
            if isinstance(node, ForNode):
                iv = node.iter_var
                if iv.extent == 16 and iv.axis_id == d1:
                    result = path
                    return
                for i, c in enumerate(node.children):
                    walk(c, path + (i,))

        for i, root in enumerate(module.body):
            walk(root, (i,))
            if result is not None:
                break
        return result

    path = find_outer_M(module)
    assert path is not None
    """16 = 4 * 4, outer split, innermost unchanged."""
    atom = Split(loop_path=path, factor=4)
    assert atom.is_legal(module) is True
