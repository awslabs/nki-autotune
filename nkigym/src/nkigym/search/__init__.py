"""Search: build + render + remote-profile a KernelIR on Trainium hosts."""

from nkigym.search.api import dump_ir, func_source_with_imports, inline_gadgets, remote_run

__all__ = ["dump_ir", "func_source_with_imports", "inline_gadgets", "remote_run"]
