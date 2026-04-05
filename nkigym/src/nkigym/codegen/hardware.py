"""Per-tensor hardware limits of Trainium 2 NeuronCore.

These are per-tensor constraints — a single tensor allocation
must satisfy them. No whole-kernel capacity analysis; whether
all tensors fit simultaneously is the compiler's problem.

NKI source: nki.language.tile_size, nki.isa.validation
Design doc reference: nkigym_ir_guide.md section 2.1.
"""

_PSUM_OPS = frozenset({"nc_matmul", "nc_transpose"})


class Hardware:
    """Per-tensor limits of Trainium 2 NeuronCore.

    Attributes:
        SBUF_PARTITION_MAX: Max partition elements per tile (nl.tile_size.pmax).
        SBUF_PARTITION_BYTES: Usable bytes per SBUF partition (224 KiB - reserved).
        PSUM_FREE_MAX: Max fp32 elements per PSUM bank (nl.tile_size.psum_fmax).
        TRANSPOSE_BLOCK: nc_transpose block size (both P and F <= 128).
    """

    SBUF_PARTITION_MAX: int = 128
    SBUF_PARTITION_BYTES: int = 212984
    PSUM_FREE_MAX: int = 512
    TRANSPOSE_BLOCK: int = 128

    @staticmethod
    def sbuf_free_max(dtype_bytes: int) -> int:
        """Max free dim for a single SBUF tensor.

        Args:
            dtype_bytes: Element size in bytes (e.g. 4 for fp32, 2 for bf16).

        Returns:
            SBUF_PARTITION_BYTES // dtype_bytes.
        """
        return Hardware.SBUF_PARTITION_BYTES // dtype_bytes
