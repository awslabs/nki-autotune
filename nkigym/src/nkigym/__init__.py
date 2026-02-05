"""NKI Gym - Tunable kernel environment for AWS Trainium hardware."""

from nkigym.nki_ops import NKIMatmul, ndarray

nc_matmul = NKIMatmul()

__all__ = ["nc_matmul", "ndarray"]
