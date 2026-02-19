"""Data reuse analysis and transform for tiled compute graphs.

Identifies tensor slices that can be merged across subgraphs, reducing
redundant load operations. Operates on the GymProgram IR.

Example::

    reuse = DataReuseTransform()
    pairs = reuse.analyze_ir(program)
    for pair in pairs:
        program = reuse.transform_ir(program, pair)
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
    new_name = rename_map.get(ref.name, ref.name)
    return TensorRef(new_name, ref.shape, ref.slices)


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
    for key, value in kwargs:
        if isinstance(value, TensorRef):
            value = _rename_ref(value, rename_map)
        new_kwargs.append((key, value))
    return tuple(new_kwargs)


class DataReuseTransform(Transform):
    """Transform that merges redundant tensor loads across subgraphs.

    Identifies tensor slices that access identical data in different
    subgraphs and merges them into a single load.

    ``analyze_ir()`` returns pairs of tensor variable names that share
    identical slice patterns. ``transform_ir()`` merges a single pair.

    Example::

        reuse = DataReuseTransform()
        pairs = reuse.analyze_ir(program)
        for pair in pairs:
            program = reuse.transform_ir(program, pair)
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

        load_sources: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {}
        for stmt in program.stmts:
            if stmt.op == "np_slice":
                dst_name = stmt.output.name
                src_ref = None
                for key, value in stmt.kwargs:
                    if key == "src":
                        src_ref = value
                if src_ref is None:
                    raise ValueError("np_slice statement missing 'src' kwarg")
                load_sources[dst_name] = (src_ref.name, src_ref.slices)

        for tensor_name in (keep, drop):
            if tensor_name not in load_sources:
                raise ValueError(f"Tensor '{tensor_name}' not found in program loads")

        if load_sources[keep] != load_sources[drop]:
            raise ValueError(f"Tensors '{keep}' and '{drop}' do not share identical slices")

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
