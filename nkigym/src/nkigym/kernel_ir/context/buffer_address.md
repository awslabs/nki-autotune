## Buffer Address Allocation (TODO)

Today, every SBUF tensor in a rendered kernel has its own unique physical address, even if its lifetime doesn't overlap with another tensor's. The reference kernel (`nkilib`'s `attention_cte`) does the opposite: it lets multiple tensors share the same physical address as long as they're never simultaneously alive. Whichever tensor is written later overwrites the (already-dead) bytes of the earlier one. The reclaimed space lets the reference kernel hold larger working tiles with fewer DMAs, contributing to its ~33% MFU on 2048² causal vs our ~15%.

Reference: the allocation primitive is `nkilib/core/utils/modular_allocator.py` (`ModularAllocator.alloc_sbuf_tensor`), and its use for attention buffer placement lives in `nkilib/core/attention/attention_cte.py` starting at line 673 (persistent buffers) and line 1278 (per-section buffers).

### What this feature should do

Let two or more SBUF tensors occupy the **same physical address** when their lifetimes do not overlap.

Today a rendered kernel has one SBUF address per tensor. With this feature, a rendered kernel can have multiple tensors declared at the same address — the tensors have different names, different logical shapes, and different dtypes, but their underlying bytes coincide. Reading and writing any of them touches the same physical bytes.

### What is sampled

The **partition of SBUF tensors into slots**. Tensors in the same slot share one physical address; tensors in different slots occupy disjoint addresses. The partition is subject to one hard rule: every tensor in a slot must have a lifetime disjoint from every other tensor in that slot.

Nothing else in this feature is sampled:

- **Slot size is derived**: a slot's size is `max(tensor_size for tensor in slot)`. The smaller members sit inside the same bytes as the largest member, using only a prefix during their own lifetimes.
- **Addresses are derived**: slots are laid out end-to-end in SBUF; each slot's starting address is the cumulative sum of preceding slots' sizes. All members of a slot take the slot's starting address.

So the sample space is one dimension: which tensors share slots with which.

### Why there is room to sample

Given a fixed set of tensor lifetimes, many partitions satisfy the lifetime rule. They differ in total SBUF footprint, because a slot holding mixed-size tensors reserves bytes equal to its largest member even when smaller ones are active — coalescing saves a slot but may inflate the one it joins. Different partitions on the same lifetimes yield different footprints, which in turn unlock different downstream tile-size choices. That tradeoff is what makes the partition worth sampling rather than deriving.

### What "lifetime" means

A tensor's lifetime runs from the first op that writes it to the last op that reads it, in the kernel's **linearized** op order — the order in which ops will execute when the rendered kernel runs. Two tensors have non-overlapping lifetimes iff, on that linearized sequence, one's last read comes strictly before the other's first write.

### Relationship to fusion and other structural choices

Fusion-group assignment, loop order, and other structural choices are **upstream** of this feature. They're sampled earlier in the variant-generation pipeline and, once fixed, determine the linearized op order and therefore every tensor's lifetime. This feature runs on the lifetime picture those choices produce; it does not revisit them.

A consequence: the same two tensors may be partition-compatible under one fusion choice and incompatible under another. Each variant samples a partition on its own lifetime picture.

### The correctness rule

Two tensors may coincide in address only if their lifetimes do not overlap on the linearized op order of the variant. This is absolute — violating it produces data corruption.

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
