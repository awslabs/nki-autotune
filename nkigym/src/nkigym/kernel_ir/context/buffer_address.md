## Buffer Address Allocation (TODO)

Today, every SBUF tensor in a rendered kernel has its own unique physical address, even if its lifetime doesn't overlap with another tensor's. The reference kernel (`nkilib`'s `attention_cte`) does the opposite: it lets multiple tensors share the same physical address as long as they're never simultaneously alive. Whichever tensor is written later overwrites the (already-dead) bytes of the earlier one. The reclaimed space lets the reference kernel hold larger working tiles with fewer DMAs, contributing to its ~33% MFU on 2048² causal vs our ~15%.

Reference: the allocation primitive is `nkilib/core/utils/modular_allocator.py` (`ModularAllocator.alloc_sbuf_tensor`), and its use for attention buffer placement lives in `nkilib/core/attention/attention_cte.py` starting at line 673 (persistent buffers) and line 1278 (per-section buffers).

### What this feature should do

Let two or more SBUF tensors occupy the **same physical address** when their lifetimes do not overlap.

Today a rendered kernel has one SBUF address per tensor. With this feature, a rendered kernel can have multiple tensors declared at the same address — the tensors have different names, different logical shapes, and different dtypes, but their underlying bytes coincide. Reading and writing any of them touches the same physical bytes.

### What "lifetime" means

A tensor's lifetime runs from the first op that writes it to the last op that reads it, in the kernel's **linearized** op order — the order in which ops will execute when the rendered kernel runs. The linearization is not the op graph; it's the flattened sequence produced by the kernel's loop structure, fusion-group ordering, and any other structural choices baked into the rendered kernel. Two tensors have non-overlapping lifetimes iff, on that linearized sequence, one's last read comes strictly before the other's first write.

Because the linearization depends on structural choices that vary across rendered kernels, a pair of tensors compatible for address sharing in one variant may not be compatible in another.

### The correctness rule

Two tensors may coincide in address only if their lifetimes do not overlap. This is absolute — violating it produces data corruption.

### Expected yield

Rough estimate: **3–6 MFU points** on 2048² causal attention. Shared addresses reduce the kernel's SBUF footprint, which unlocks larger working tiles that were previously unreachable because they'd exceed SBUF capacity.

This feature alone does not close the full gap to the reference. The larger remaining piece (~8–12 pts) concerns running different loop iterations in parallel and is a separate feature.

### Scope boundaries

- **SBUF only.** PSUM has its own allocation rules and smaller, shorter-lived footprints; not in scope for v1.
- **Whole-tensor coincidence only.** A tensor either fully shares another's address or fully doesn't. Partial overlaps (two tensors sharing only some of their bytes) are out of scope.
- **Tensors whose addresses are already fixed by another feature stay where they are.** For example, a tensor kept live across the whole kernel for loop-carried state has a forced placement; this feature cannot move it.

### Open questions

- How should composite ops (e.g. online fusion) expose the lifetimes of tensors they generate internally, so those can participate in address sharing?
- Are there hardware-alignment rules that restrict which tensors may coincide in address beyond just the lifetime check? If so, the feature needs to respect them.
