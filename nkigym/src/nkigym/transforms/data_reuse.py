"""Data reuse analysis and transform for tiled compute graphs.

Identifies tensor slices that can be merged across subgraphs, reducing
redundant load operations. Operates on the GymProgram IR.

Example::

    reuse = DataReuseTransform()
    while True:
        pairs = reuse.analyze_ir(program)
        if not pairs:
            break
        program = reuse.transform_ir(program, pairs[0])
"""

from itertools import combinations

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.transforms.base import Transform


def _rename_ref(ref: TensorRef, rename_map: dict[str, str]) -> TensorRef:
    """Replace variable name in a TensorRef according to rename_map.

    Args:
        ref: Tensor reference to rename.
        rename_map: Mapping from old variable names to new names.

    Returns:
        New TensorRef with renamed variable, or original if not in map.
    """
    new_name = rename_map.get(ref.name)
    result = TensorRef(new_name, ref.shape, ref.slices) if new_name is not None else ref
    return result


def _rename_kwargs(
    kwargs: tuple[tuple[str, object], ...], rename_map: dict[str, str]
) -> tuple[tuple[str, object], ...]:
    """Replace variable names in kwargs TensorRef values according to rename_map.

    Args:
        kwargs: Statement keyword argument pairs.
        rename_map: Mapping from old variable names to new names.

    Returns:
        New kwargs tuple with renamed TensorRef values.
    """
    new_kwargs: list[tuple[str, object]] = []
    changed = False
    for key, value in kwargs:
        if isinstance(value, TensorRef) and value.name in rename_map:
            value = TensorRef(rename_map[value.name], value.shape, value.slices)
            changed = True
        new_kwargs.append((key, value))
    return tuple(new_kwargs) if changed else kwargs


def _validate_merge_pair(program: GymProgram, keep: str, drop: str) -> None:
    """Validate that two tensor names can be merged.

    Checks both names exist as np_slice outputs and share identical
    source slices.

    Args:
        program: The GymProgram containing the statements.
        keep: Variable name to keep.
        drop: Variable name to drop.

    Raises:
        ValueError: If names are missing or slices don't match.
    """
    load_sources: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {}
    for stmt in program.stmts:
        if stmt.op == "np_slice":
            src_ref = stmt.kwargs[0][1]
            load_sources[stmt.output.name] = (src_ref.name, src_ref.slices)
    for tensor_name in (keep, drop):
        if tensor_name not in load_sources:
            raise ValueError(f"Tensor '{tensor_name}' not found in program loads")
    if load_sources[keep] != load_sources[drop]:
        raise ValueError(f"Tensors '{keep}' and '{drop}' do not share identical slices")


class DataReuseTransform(Transform):
    """Transform that merges redundant tensor loads across subgraphs.

    Identifies tensor slices that access identical data in different
    subgraphs and merges them into a single load.

    ``analyze_ir()`` returns pairs of tensor variable names that share
    identical slice patterns. ``transform_ir()`` merges a single pair.

    Example::

        reuse = DataReuseTransform()
        while True:
            pairs = reuse.analyze_ir(program)
            if not pairs:
                break
            program = reuse.transform_ir(program, pairs[0])
    """

    name = "data_reuse"

    def analyze_ir(self, program: GymProgram) -> list[tuple[str, str]]:
        """Identify pairs of tensor slices that can be merged across subgraphs.

        Groups np_slice statements by (src_name, src_slices) and returns pairs
        of dst variables that share identical load patterns.

        Args:
            program: GymProgram tuple.

        Returns:
            List of mergeable pairs. Each pair is a tuple of two tensor
            variable names that access identical data
            (e.g., ``('tensor_0', 'tensor_3')``).
        """
        load_groups: dict[tuple[str, tuple[tuple[int, int], ...]], list[str]] = {}
        for stmt in program.stmts:
            if stmt.op != "np_slice":
                continue
            src_ref = None
            for key, value in stmt.kwargs:
                if key == "src":
                    src_ref = value
            if src_ref is None:
                raise ValueError("np_slice statement missing 'src' kwarg")
            key = (src_ref.name, src_ref.slices)
            dst_name = stmt.output.name
            load_groups.setdefault(key, []).append(dst_name)

        pairs: list[tuple[str, str]] = []
        for dst_vars in load_groups.values():
            if len(dst_vars) >= 2:
                pairs.extend(combinations(dst_vars, 2))
        return pairs

    def transform_ir(self, program: GymProgram, pair: tuple[str, str]) -> GymProgram:
        """Merge a single pair of reusable tensor slices.

        Removes the second tensor's load statement and replaces all its
        references with the first tensor.

        Args:
            program: GymProgram to transform.
            pair: A pair of tensor names from ``analyze_ir()``.

        Returns:
            New GymProgram with the pair's redundant load merged.

        Raises:
            ValueError: If tensor names are identical or don't share slices.
        """
        keep, drop = pair
        if keep == drop:
            raise ValueError(f"Cannot merge {keep} with itself")
        _validate_merge_pair(program, keep, drop)

        rename_map = {drop: keep}
        new_stmts: list[GymStatement] = []
        for stmt in program.stmts:
            if stmt.op == "np_slice" and stmt.output.name == drop:
                continue
            new_stmts.append(
                GymStatement(
                    op=stmt.op,
                    kwargs=_rename_kwargs(stmt.kwargs, rename_map),
                    output=_rename_ref(stmt.output, rename_map),
                )
            )

        return GymProgram(
            name=program.name,
            params=program.params,
            input_shapes=program.input_shapes,
            stmts=tuple(new_stmts),
            return_var=program.return_var,
            output_dtype=program.output_dtype,
        )
