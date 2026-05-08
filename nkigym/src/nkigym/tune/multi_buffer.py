"""``MultiBuffer`` rewrite — set the multi-buffer degree of a tensor on a dim.

``MultiBuffer`` mutates only ``KernelModule.tensors[tensor_name].buffer_degree``.
The renderer later consumes this field to size allocations and to build
slot-indexing expressions.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import KernelModule
from nkigym.codegen.lowering.place_buffers import required_tiles
from nkigym.tune import AtomLegalityError


@dataclass(frozen=True)
class MultiBuffer:
    """Set ``module.tensors[tensor_name].buffer_degree[dim_id] = degree``.

    Attributes:
        tensor_name: Target tensor.
        dim_id: Dim on which to set buffer degree.
        degree: New degree (must be >= 1 and <= num_tiles(dim_id)).
    """

    tensor_name: str
    dim_id: str
    degree: int

    def is_legal(self, module: KernelModule) -> bool:
        """Return True iff target tensor exists, dim is bound, and degree is in range.

        Upper bound is ``num_tiles / required_tiles``: more buffer slots than
        there are distinct tiles under the LCA just waste SBUF without
        changing behavior.

        A :class:`ValueError` from :func:`required_tiles` — raised when the
        product of ancestor trip counts does not divide ``num_tiles`` —
        indicates an unsupported forest shape and is treated as "not legal"
        rather than propagated. Well-formed canonical IR never reaches this
        branch; the catch is defensive against malformed input.
        """
        result: bool
        if self.tensor_name not in module.tensors:
            result = False
        else:
            t = module.tensors[self.tensor_name]
            if self.dim_id not in t.dim_ids:
                result = False
            elif self.dim_id not in module.dims:
                result = False
            else:
                try:
                    req = required_tiles(t, self.dim_id, module)
                except ValueError:
                    req = 0
                if req <= 0:
                    result = False
                else:
                    num_t = module.dims[self.dim_id].num_tiles
                    max_degree = num_t // req
                    result = 1 <= self.degree <= max_degree
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Return a new module with ``tensor_name``'s buffer_degree on ``dim_id`` set to ``degree``.

        Re-validates legality against ``module`` at apply time. An atom
        enumerated against a previous state may be stale if a preceding
        rewrite narrowed the tensor's LCA (raising ``required_tiles``) so
        the captured degree no longer fits under
        ``num_tiles // required_tiles``. Applying such an atom would
        produce slot-modulo aliasing in the rendered kernel; instead,
        raise :class:`AtomLegalityError` so the caller can skip.

        Raises:
            AtomLegalityError: If :meth:`is_legal` returns False against
                ``module`` — the atom's captured ``(tensor, dim, degree)``
                no longer satisfies the legality invariants.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(
                f"MultiBuffer.apply: atom stale against current module — "
                f"tensor={self.tensor_name!r}, dim={self.dim_id!r}, degree={self.degree}"
            )
        old_t = module.tensors[self.tensor_name]
        new_degree = dict(old_t.buffer_degree)
        new_degree[self.dim_id] = self.degree
        new_t = replace(old_t, buffer_degree=new_degree)
        new_tensors = {**module.tensors, self.tensor_name: new_t}
        return replace(module, tensors=new_tensors)


def enumerate_multi_buffer_atoms(module: KernelModule) -> list[MultiBuffer]:
    """Return every legal ``(tensor, dim, degree)`` atom on non-HBM tensors.

    Params and returns live in HBM and are not multi-buffered. An
    unsupported forest shape (``required_tiles`` raising
    :class:`ValueError`) yields no atoms for the offending dim.
    """
    atoms: list[MultiBuffer] = []
    for tensor_name, t in module.tensors.items():
        if t.origin in ("param", "return"):
            continue
        for d in t.dim_ids:
            if d not in module.dims:
                continue
            try:
                req = required_tiles(t, d, module)
            except ValueError:
                continue
            if req <= 0:
                continue
            num_t = module.dims[d].num_tiles
            max_degree = num_t // req
            current = t.buffer_degree.get(d, 1)
            for degree in range(1, max_degree + 1):
                if degree == current:
                    continue
                atoms.append(MultiBuffer(tensor_name=tensor_name, dim_id=d, degree=degree))
    return atoms
