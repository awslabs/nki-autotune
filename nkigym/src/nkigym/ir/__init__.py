"""IR layer for the nkigym tune stage.

Defines the IR dataclasses (:mod:`.ir`), the ``f_nkigym`` → :class:`KernelIR`
lowering pass (:mod:`.build`), and the per-scope dependency cache
(:mod:`.dep_cache`). Codegen (``nkigym.codegen``) consumes the IR; the IR
layer itself does not depend on codegen.
"""
