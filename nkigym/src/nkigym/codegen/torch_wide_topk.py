"""Source emission for exact top-k beyond one ``max8`` tile."""

from nkigym.codegen.torch_values import TorchSegments, TorchValue


def emit_wide_topk(
    source: TorchValue, k: int, total_width: int, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Emit a 16-partition hierarchical exact top-k."""
    partitions, width = source.shape
    if partitions != 16 or partitions * width != total_width or k < 1 or k % 8:
        raise ValueError("wide top-k requires 16 equal partitions and a positive multiple-of-eight k")
    imports.update(
        (
            "NKIActivationReduce",
            "NKIFindIndex8",
            "NKIFloat32Cast",
            "NKIFlattenStore",
            "NKIInt32Cast",
            "NKIIota",
            "NKILoad",
            "NKIMatchReplace8",
            "NKIMax8",
            "NKINCGather",
            "NKIStreamShuffleBroadcast",
            "NKITensorScalar",
        )
    )
    base_matrix = TorchValue(f"sbuf_{stem}_partition_base", (partitions, 1))
    base = TorchValue(f"{base_matrix.name}_vector", (partitions,))
    body.extend(
        (
            f"{base_matrix.name} = NKIIota(partitions={partitions}, width=1, pattern=[[0, 1]], "
            f"channel_multiplier={width})()",
            f'{base.name} = NKIActivationReduce(op="copy", reduce_op="add")(data={base_matrix.name})',
        )
    )
    values_out: list[TorchValue] = []
    indices_out: list[TorchValue] = []
    working = source
    for offset in range(0, k, 8):
        local_values = TorchValue(f"sbuf_{stem}_local_values_{offset}", (partitions, 8))
        local_indices = TorchValue(f"sbuf_{stem}_local_indices_{offset}", (partitions, 8))
        local_float = TorchValue(f"{local_indices.name}_float", local_indices.shape)
        global_local = TorchValue(f"sbuf_{stem}_global_local_{offset}", local_indices.shape)
        hbm_values = TorchValue(f"hbm_{stem}_local_values_{offset}", (1, 128))
        hbm_indices = TorchValue(f"hbm_{stem}_local_indices_{offset}", (1, 128))
        flat_values = TorchValue(f"sbuf_{stem}_flat_values_{offset}", (1, 128))
        flat_indices = TorchValue(f"sbuf_{stem}_flat_indices_{offset}", (1, 128))
        values = TorchValue(f"sbuf_{stem}_values_{offset}", (1, 8))
        positions = TorchValue(f"sbuf_{stem}_positions_{offset}", (1, 8))
        indices_float = TorchValue(f"sbuf_{stem}_indices_float_{offset}", (1, 8))
        indices = TorchValue(f"sbuf_{stem}_indices_{offset}", (1, 8))
        body.extend(
            (
                f"{local_values.name} = NKIMax8()(src={working.name})",
                f"{local_indices.name} = NKIFindIndex8()(data={working.name}, vals={local_values.name})",
                f"{local_float.name} = NKIFloat32Cast()(data={local_indices.name})",
                f'{global_local.name} = NKITensorScalar(op0="add")' f"(data={local_float.name}, operand0={base.name})",
                f"{hbm_values.name} = NKIFlattenStore(width=128)(src={local_values.name})",
                f"{hbm_indices.name} = NKIFlattenStore(width=128)(src={global_local.name})",
                f"{flat_values.name} = NKILoad()(src={hbm_values.name})",
                f"{flat_indices.name} = NKILoad()(src={hbm_indices.name})",
                f"{values.name} = NKIMax8()(src={flat_values.name})",
                f"{positions.name} = NKIFindIndex8()(data={flat_values.name}, vals={values.name})",
                f"{indices_float.name} = NKINCGather()(data={flat_indices.name}, indices={positions.name})",
                f"{indices.name} = NKIInt32Cast()(data={indices_float.name})",
            )
        )
        values_out.append(values)
        indices_out.append(indices)
        if offset + 8 < k:
            working = TorchValue(f"sbuf_{stem}_remaining_{offset}", source.shape)
            prior = source.name if offset == 0 else f"sbuf_{stem}_remaining_{offset - 8}"
            broadcast = TorchValue(f"sbuf_{stem}_broadcast_{offset}", (partitions, 8))
            body.extend(
                (
                    f"{broadcast.name} = NKIStreamShuffleBroadcast(partitions={partitions})(src={values.name})",
                    f"{working.name} = NKIMatchReplace8(imm=float('-inf'))" f"(data={prior}, vals={broadcast.name})",
                )
            )
    return TorchSegments(tuple(values_out)), TorchSegments(tuple(indices_out))


__all__ = ["emit_wide_topk"]
