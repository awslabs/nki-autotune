"""Sub-pass for the ``buffer_degree`` annotation key.

Reads ``"buffer_degree"`` annotations off alloc ``SBlock`` nodes; widens
``Tensor.buffer_degree[P_dim]`` so that :mod:`place_buffers` derives a
wider P-slot dim on the declared tensor.

Uses ``max(existing, degree)`` when multiple annotations name the same
tensor — later annotations never narrow a previously-widened degree.

Slot-modulo index emission (``(slot) % total_slots``) is deferred: it
only activates when ``total_slots < num_tiles_in_flight`` — an LCA
narrowing from :class:`ComputeAt` or a pipeline offset from
:class:`SoftwarePipeline`. The canonical form has trivial LCA (full
extent) so the widened slot axis fits without modulo rewriting.
See the Bug B followup spec for the modulo-emission recipe.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock


def apply_buffer_degree(module: KernelIR) -> None:
    """Walk every forest root; apply ``buffer_degree`` annotations to tensor metadata.

    Args:
        module: KernelIR to mutate in place.
    """
    for root in module.body:
        _walk(root, module)


def _walk(node: ForNode | SBlock, module: KernelIR) -> None:
    """Depth-first walk; update ``Tensor.buffer_degree`` on each matching SBlock."""
    if isinstance(node, SBlock):
        bd = node.annotations.get("buffer_degree")
        if bd is not None:
            for tname, degree in bd.items():
                tensor = module.tensors.get(tname)
                if tensor is None or not tensor.dim_ids:
                    continue
                p_dim = tensor.dim_ids[0]
                existing = tensor.buffer_degree.get(p_dim, 1)
                tensor.buffer_degree[p_dim] = max(existing, degree)
        return
    for child in node.children:
        _walk(child, module)
