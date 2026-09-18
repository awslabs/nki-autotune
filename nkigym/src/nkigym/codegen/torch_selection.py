"""Emit exact comparator-based selection from native operations."""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.dynamic_slice_copy import emit_window_updates
from nkigym.ops.dynamic_slice_load import emit_adaptive_window, emit_window_pair
from nkigym.ops.flatten_store import emit_parallel_partition
from nkigym.ops.inplace_max8 import emit_native_prefix, native_prefix_chunk
from nkigym.ops.inplace_tensor_copy import emit_parallel_sort_partitions
from nkigym.ops.max8 import emit_batched_topk
from nkigym.ops.nc_gather import emit_clamped_gather, emit_pair_cursors
from nkigym.ops.nonzero_with_count import emit_compacted_pairs
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.stream_shuffle_broadcast import emit_local_sort_permutation
from nkigym.ops.tensor_scalar import emit_scan_stop
from nkigym.ops.uint16_iota import emit_indices


@dataclass
class SelectionEmitter(ControlEmitter):
    """Build one row's explicit selection state and primitive source."""

    width: int
    full_sort: bool = field(default=False, kw_only=True)
    values: str
    indices: str
    index_cast: Literal["NKIUInt16Cast", "NKIUInt32Cast"] = field(default="NKIUInt32Cast", kw_only=True)

    def gather_index(self, position: str) -> str:
        """Clamp speculative positions and encode the native unsigned indices."""
        return self.cast("NKIUInt32Cast", self.clamp(position, 0.0, float(self.width - 1)))

    def gather(self, data: str, position: str) -> str:
        """Gather one clamped position for speculative scalar evaluation."""
        return emit_clamped_gather(self, data, position, self.width)

    def pair(self, position: str) -> tuple[str, str]:
        """Read a value and its original index before any update."""
        index = self.gather_index(position)
        values = self.emit("NKINCGather", f"data={self.values}, indices={index}")
        indices = self.emit("NKINCGather", f"data={self.indices}, indices={index}")
        return values, indices

    def write_pair(self, position: str, pair: tuple[str, str], enabled: str) -> None:
        """Update one value/index pair with native dynamic window copies."""
        with self.guard(enabled):
            offset = self.cast("NKIUInt32Cast", position)
            self.copy_windows((self.values, self.indices), pair, offset)

    def swap(self, left: str, right: str, enabled: str) -> None:
        """Swap two saved pairs while preserving the original reads."""
        left_pair, right_pair = self.pair(left), self.pair(right)
        self.write_pair(left, right_pair, enabled)
        self.write_pair(right, left_pair, enabled)

    def permute(self, mapping: str) -> None:
        """Gather both state arrays before overwriting either one."""
        pair = self.pair(mapping)
        self.imports.add("NKIInplaceTensorCopy")
        for destination, value in zip((self.values, self.indices), pair, strict=True):
            self.line(
                f"{destination} = NKIInplaceTensorCopy(groups=1, partitions=1, start=0, "
                f"width={self.width}, engine='vector')(src={value}, dst={destination})"
            )


def adjust_heap(
    emit: SelectionEmitter, first: str, top: str, length: str, saved: tuple[str, str], capacity: int
) -> None:
    """Repair a heap inside its caller's active guard using the exact comparator."""
    levels = max(1, capacity.bit_length())
    hole = emit.copy(top)
    with emit.repeat(levels):
        left = emit.scalar("add", emit.scalar("multiply", hole, 2.0), 1.0)
        right = emit.scalar("add", left, 1.0)
        can_descend = emit.binary("greater", length, left)
        right_exists = emit.binary("greater", length, right)
        left_pair, right_pair = emit.pair(emit.binary("add", first, left)), emit.pair(emit.binary("add", first, right))
        right_missing = emit.inverse(right_exists)
        choose_left = emit.binary("maximum", right_missing, emit.before(right_pair[0], left_pair[0]))
        child = emit.select(choose_left, left, right)
        pair = tuple(
            emit.select(choose_left, left_value, right_value)
            for left_value, right_value in zip(left_pair, right_pair, strict=True)
        )
        emit.write_pair(emit.binary("add", first, hole), (pair[0], pair[1]), can_descend)
        emit.write(hole, child, can_descend)
    with emit.repeat(levels):
        parent = emit.half(emit.scalar("subtract", hole, 1.0))
        parent_pair = emit.pair(emit.binary("add", first, parent))
        can_push = emit.binary("multiply", emit.binary("greater", hole, top), emit.before(parent_pair[0], saved[0]))
        emit.write_pair(emit.binary("add", first, hole), parent_pair, can_push)
        emit.write(hole, parent, can_push)
    offset = emit.cast("NKIUInt32Cast", emit.binary("add", first, hole))
    emit.copy_windows((emit.values, emit.indices), saved, offset)


def emit_heap(
    emit: SelectionEmitter, interval: tuple[str, str, str], enabled: str, capacity: int, mode: str, zero: str
) -> None:
    """Emit heap construction followed by selection, draining, or both."""
    first, middle, last = interval
    length = emit.binary("subtract", middle, first)
    parent = emit.copy(emit.half(emit.scalar("subtract", length, 2.0)))
    with emit.repeat(max(1, capacity // 2)):
        active = emit.binary("multiply", enabled, emit.scalar("greater_equal", parent, 0.0))
        active = emit.binary("multiply", active, emit.scalar("greater", length, 1.0))
        with emit.guard(active):
            saved = emit.pair(emit.binary("add", first, parent))
            adjust_heap(emit, first, parent, length, saved, capacity)
        emit.write(parent, emit.scalar("subtract", parent, 1.0), enabled)
    if mode != "sort":
        cursor = emit.copy(middle)
        with emit.repeat(emit.width):
            entry, root = emit.pair(cursor), emit.pair(first)
            active = emit.binary("multiply", enabled, emit.binary("greater", last, cursor))
            active = emit.binary("multiply", active, emit.before(entry[0], root[0]))
            with emit.guard(active):
                emit.write_pair(cursor, root, active)
                adjust_heap(emit, first, zero, length, entry, capacity)
            emit.write(cursor, emit.scalar("add", cursor, 1.0), enabled)
    if mode != "select":
        end = emit.copy(middle)
        with emit.repeat(max(1, capacity - 1)):
            active = emit.binary("multiply", enabled, emit.scalar("greater", emit.binary("subtract", end, first), 1.0))
            next_end = emit.scalar("subtract", end, 1.0)
            with emit.guard(active):
                entry, root = emit.pair(next_end), emit.pair(first)
                emit.write_pair(next_end, root, active)
                adjust_heap(emit, first, zero, emit.binary("subtract", next_end, first), entry, capacity)
            emit.write(end, next_end, active)


def partition(emit: SelectionEmitter, first: str, last: str, enabled: str) -> str:
    """Emit an exact median-of-three partition with bounded window storage."""
    return emit_adaptive_window(emit, (emit.values, emit.indices), (first, last), enabled, emit.width, _partition)


def _partition(
    emitter: ControlEmitter, sources: tuple[str, str], bounds: tuple[str, str], enabled: str, width: int
) -> str:
    """Choose a pivot and partition one materialized interval."""
    emit = SelectionEmitter(emitter.stem, emitter.body, emitter.imports, width, *sources, depth=emitter.depth)
    first, last = bounds
    cut = emit.copy(first)
    with emit.guard(enabled):
        left = emit.scalar("add", first, 1.0)
        middle = emit.binary("add", first, emit.half(emit.binary("subtract", last, first)))
        right = emit.scalar("subtract", last, 1.0)
        a, b, c = (emit.gather(emit.values, index) for index in (left, middle, right))
        ab, ac, bc = emit.before(a, b), emit.before(a, c), emit.before(b, c)
        chosen = emit.select(
            ab,
            emit.select(bc, middle, emit.select(ac, right, left)),
            emit.select(ac, left, emit.select(bc, right, middle)),
        )
        emit.swap(first, chosen, enabled)
        emit.write(cut, _stream_partition(emit, left, last, emit.gather(emit.values, first)), enabled)
    return cut


def _stream_partition(emit: SelectionEmitter, left: str, last: str, pivot: str) -> str:
    """Partition with bounded snapshots and matching overlapping-window updates."""
    if emit.width <= 256:
        return _single_partition(emit, left, last, pivot)
    low, high, boundary = emit.copy(left), emit.copy(last), emit.copy(last)
    window = min(512 if emit.width > 2048 else 256, emit.width)
    local, one = emit.iota(window), emit.scalar("add", emit.iota(1), 1.0)
    with emit.repeat(2 * ((emit.width + window - 1) // window) + 2):
        with emit.guard(emit.binary("greater", high, low)):
            starts = (
                emit.scalar("minimum", low, float(emit.width - window)),
                emit.scalar("maximum", emit.scalar("subtract", high, float(window)), 0.0),
            )
            loaded = [emit_window_pair(emit, (emit.values, emit.indices), start, window) for start in starts]
            pairs = [item[0] for item in loaded]
            grids = [emit.scalar("add", local, start) for start in starts]
            masks = [
                emit_scan_stop(emit, pairs[side][0], grids[side], (low, high), pivot, bool(side)) for side in range(2)
            ]
            tables, counts, partners, swaps, swapped = emit_compacted_pairs(
                emit, (masks[0], masks[1]), grids, starts, window
            )
            updates = emit_window_updates(emit, pairs, grids, starts, swaps, partners, window)
            next_left, next_high, previous = emit_pair_cursors(
                emit, tables, counts, starts, (low, high), swapped, window
            )
            emit.write(boundary, previous, emit.scalar("greater", swapped, 0.0))
            for update, (_, offset) in zip(updates, loaded, strict=True):
                emit.copy_windows((emit.values, emit.indices), update, offset)
            emit.write(low, next_left, one)
            emit.write(high, next_high, one)
    return emit.binary("minimum", low, boundary)


def _single_partition(emit: SelectionEmitter, left: str, last: str, pivot: str) -> str:
    """Partition one bounded tile with one value/index permutation."""
    positions, zero = emit.iota(emit.width), emit.iota(1)
    masks = [emit_scan_stop(emit, emit.values, positions, (left, last), pivot, bool(side)) for side in range(2)]
    tables, counts, partners, swaps, swapped = emit_compacted_pairs(
        emit, (masks[0], masks[1]), [positions, positions], (zero, zero), emit.width
    )
    emit.permute(emit.select(swaps[1], partners[1], emit.select(swaps[0], partners[0], positions)))
    next_left = emit.binary("minimum", emit_clamped_gather(emit, tables[0], swapped, emit.width), last)
    previous = emit_clamped_gather(emit, tables[1], emit.binary("subtract", counts[1], swapped), emit.width)
    return emit.select(emit.scalar("greater", swapped, 0.0), emit.binary("minimum", next_left, previous), next_left)


def insertion(emit: SelectionEmitter, first: str, last: str) -> None:
    """Stable-sort an interval of at most three entries using a bounded window."""
    window = min(3, emit.width)
    active = emit.scalar("greater", emit.binary("subtract", last, first), 1.0)
    with emit.guard(active):
        cursor = emit.copy(emit.scalar("add", first, 1.0))
        local_positions = emit.iota(window)
        with emit.repeat(max(1, window - 1)):
            enabled = emit.binary("greater", last, cursor)
            start = emit.clamp(emit.scalar("subtract", cursor, float(window - 1)), first, float(emit.width - window))
            positions = emit.scalar("add", local_positions, start)
            values, item = emit.gather(emit.values, positions), emit.gather(emit.values, cursor)
            finite, item_finite = emit.binary("equal", values, values), emit.binary("equal", item, item)
            before = emit.binary(
                "maximum", emit.scalar("less", values, item), emit.scalar("multiply", finite, emit.inverse(item_finite))
            )
            inside = emit.binary(
                "multiply", emit.scalar("greater_equal", positions, first), emit.scalar("less", positions, cursor)
            )
            count = emit.slice(emit.prefix(emit.binary("multiply", inside, before)), window - 1)
            insert_at = emit.binary("subtract", cursor, count)
            shift = emit.binary(
                "multiply",
                emit.scalar("greater", positions, insert_at),
                emit.scalar("less", positions, emit.scalar("add", cursor, 1.0)),
            )
            at = emit.scalar("equal", positions, insert_at)
            mapping = emit.select(
                emit.scalar("multiply", at, enabled),
                emit.scalar("add", emit.scalar("multiply", local_positions, 0.0), cursor),
                emit.select(
                    emit.scalar("multiply", shift, enabled), emit.scalar("subtract", positions, 1.0), positions
                ),
            )
            emit.write_pair(start, emit.pair(mapping), active)
            emit.write(cursor, emit.scalar("add", cursor, 1.0), active)


def stable_local_sort(emit: SelectionEmitter, window: int) -> None:
    """Finish short ordered segments using parallel bounded-rank comparisons."""
    span = min(window, emit.width)
    if span > 1:
        emit.permute(emit_local_sort_permutation(emit, emit.values, emit.width, span))


def push_frame(
    emit: SelectionEmitter, stack: tuple[str, str, str], pointer: str, frame: tuple[str, str, str], active: str
) -> None:
    """Push one interval and depth onto the bounded explicit stack."""
    with emit.guard(active):
        offset = emit.cast("NKIUInt32Cast", pointer)
        emit.copy_windows(stack, frame, offset)
        emit.write(pointer, emit.scalar("add", pointer, 1.0), active)


def sort_prefix(emit: SelectionEmitter, stop: int, zero: str, one: str) -> None:
    """Partition bounded prefixes in parallel, then apply exact local finishing."""
    prefix = SelectionEmitter(
        emit.stem,
        emit.body,
        emit.imports,
        stop,
        emit.slice(emit.values, 0, stop),
        emit.slice(emit.indices, 0, stop),
        depth=emit.depth,
        index_cast=emit.index_cast,
    )
    if 16 < stop <= 256 and emit.width <= 1 << 24:
        shape = (max(1, stop // 17), stop)
        bounds, remaining = emit_parallel_sort_partitions(
            prefix,
            (prefix.values, prefix.indices),
            shape,
            lambda sources, interval: emit_parallel_partition(prefix, sources, interval, shape),
            emit.index_cast,
        )
        with prefix.guard(prefix.scalar("greater", remaining, 0.0)):
            cursor = prefix.copy(zero)
            with prefix.repeat(shape[0]):
                first, last = (emit_clamped_gather(prefix, source, cursor, shape[0]) for source in bounds)
                enabled = prefix.scalar("greater", prefix.binary("subtract", last, first), 16.0)
                with prefix.guard(enabled):
                    emit_heap(prefix, (first, last, last), enabled, stop, "sort", zero)
                prefix.write(cursor, prefix.scalar("add", cursor, 1.0), one)
    elif stop > 16:
        depth_limit = 2 * (stop.bit_length() - 1)
        zeros = prefix.scalar("multiply", prefix.iota(depth_limit + 2), 0.0)
        stack = (prefix.copy(zeros), prefix.copy(zeros), prefix.copy(zeros))
        pointer = prefix.copy(zero)
        last = prefix.scalar("add", zero, float(stop))
        depth = prefix.scalar("add", zero, float(depth_limit))
        push_frame(prefix, stack, pointer, (zero, last, depth), one)
        chunk = min(16, 2 * stop)
        with prefix.repeat((2 * stop + chunk - 1) // chunk):
            pending = prefix.scalar("greater", pointer, 0.0)
            with prefix.guard(pending):
                with prefix.repeat(chunk):
                    active = prefix.scalar("greater", pointer, 0.0)
                    with prefix.guard(active):
                        index = prefix.scalar("subtract", pointer, 1.0)
                        slot = prefix.cast("NKIUInt32Cast", index)
                        first, last, depth = tuple(
                            prefix.emit("NKINCGather", f"data={data}, indices={slot}") for data in stack
                        )
                        prefix.write(pointer, index, active)
                        large = prefix.scalar("greater", prefix.binary("subtract", last, first), 16.0)
                        splitting = prefix.binary("multiply", large, prefix.scalar("greater", depth, 0.0))
                        cut = partition(prefix, first, last, splitting)
                        next_depth = prefix.scalar("subtract", depth, 1.0)
                        push_frame(prefix, stack, pointer, (first, cut, next_depth), splitting)
                        push_frame(prefix, stack, pointer, (cut, last, next_depth), splitting)
                        fallback = prefix.binary("multiply", large, prefix.scalar("equal", depth, 0.0))
                        with prefix.guard(fallback):
                            emit_heap(prefix, (first, last, last), fallback, stop, "sort", zero)
    stable_local_sort(prefix, 16)
    emit.write_pair(zero, (prefix.values, prefix.indices), one)


def select_topk(emit: SelectionEmitter, k: int, sorted_output: bool) -> tuple[str, str]:
    """Emit the complete comparator-only top-k algorithm for one row."""
    if not 1 <= k <= emit.width:
        raise ValueError(f"top-k {k} is outside the input width {emit.width}")
    zero = emit.iota(1)
    one = emit.scalar("add", zero, 1.0)
    selected = emit.scalar("add", zero, float(k))
    limit = emit.scalar("add", zero, float(emit.width))
    nth = emit.scalar("add", zero, float(k - 1))
    if emit.full_sort:
        with emit.guard(one):
            sort_prefix(emit, emit.width, zero, one)
    elif k * 64 <= emit.width:
        emit_heap(emit, (zero, selected, limit), one, k, "partial", zero)
    elif emit.width <= 3:
        insertion(emit, zero, limit)
    else:
        first, last = emit.copy(zero), emit.copy(limit)
        with emit.repeat(max(1, 2 * (emit.width.bit_length() - 1))):
            active = emit.scalar("greater", emit.binary("subtract", last, first), 3.0)
            cut = partition(emit, first, last, active)
            right = emit.binary("multiply", active, emit.scalar("less", cut, float(k)))
            left = emit.binary("subtract", active, right)
            emit.write(first, cut, right)
            emit.write(last, cut, left)
        fallback = emit.scalar("greater", emit.binary("subtract", last, first), 3.0)
        with emit.guard(fallback):
            emit_heap(emit, (first, selected, last), fallback, k, "select", zero)
            emit.swap(first, nth, fallback)
            emit.write(first, last, fallback)
        insertion(emit, first, last)
        if sorted_output and k > 1:
            with emit.guard(one) if k > 17 else nullcontext():
                sort_prefix(emit, k - 1, zero, one)
    return emit.slice(emit.values, 0, k), emit.cast("NKIUInt32Cast", emit.slice(emit.indices, 0, k))


def select_rows(
    emit: SelectionEmitter, source: str, rows: int, k: int, sorted_output: bool, initial: tuple[str, str, str] | None
) -> tuple[str, str]:
    """Select HBM rows, optionally preserving previously validated native results."""
    emit.index_cast = "NKIUInt16Cast" if emit.width <= 65536 else "NKIUInt32Cast"
    row = emit.iota(1)
    one = emit.scalar("add", row, 1.0)
    if initial is None:
        zeros = emit.emit("NKIIota", "", f"partitions={rows}, width={k}, pattern=[[0, {k}]], channel_multiplier=0")
        outputs = tuple(emit.emit("NKIStore", f"src={value}") for value in (zeros, emit.cast("NKIUInt32Cast", zeros)))
    else:
        outputs = initial[1:]
    emit.imports.add("NKIHBMScalarRowStore")
    with emit.repeat(rows):
        register = emit.emit("NKIRegisterLoad", f"src={emit.cast('NKIUInt32Cast', row)}", "index=0")
        enabled = one
        if initial is not None:
            valid = emit.emit(
                "NKIHBMScalarRowSlice", f"src={initial[0]}, indices={register}, index=0", "rows=1, width=1"
            )
            enabled = emit.inverse(valid)
        with emit.guard(enabled) if initial is not None else nullcontext():
            loaded = emit.emit(
                "NKIHBMScalarRowSlice", f"src={source}, indices={register}, index=0", f"rows=1, width={emit.width}"
            )
            emit.values = emit.cast("NKIFloat32Cast", loaded)
            if initial is None and sorted_output and native_prefix_chunk(emit.width, k) is not None:
                values, indices, accepted = emit_native_prefix(emit, emit.values, emit.width, k)
                zero = emit.cast("NKIUInt32Cast", emit.iota(1))
                with emit.guard(emit.inverse(accepted)):
                    emit.indices = emit_indices(emit, emit.width)
                    emit.copy_windows((values, indices), select_topk(emit, k, sorted_output), zero)
                selected = values, indices
            else:
                emit.indices = emit_indices(emit, emit.width)
                selected = select_topk(emit, k, sorted_output)
            for destination, value in zip(outputs, selected, strict=True):
                emit.line(f"{destination} = NKIHBMScalarRowStore()(src={value}, indices={register}, dst={destination})")
        emit.write(row, emit.scalar("add", row, 1.0), one)
    return outputs[0], outputs[1]


def emit_torch_topk(
    source: TorchValue,
    k: int,
    sorted_output: bool,
    stem: str,
    body: list[str],
    imports: set[str],
    full_sort: bool = False,
) -> tuple[TorchValue, TorchValue]:
    """Lower a normalized Torch matrix with exact value and index ordering."""
    if source.transposed or len(source.shape) != 2:
        raise ValueError("top-k requires a materialized row-major matrix")
    sorted_output = sorted_output or k * 64 <= source.shape[1]
    emit = SelectionEmitter(f"sbuf_{stem}", body, imports, source.shape[1], "", "")
    emit.full_sort = full_sort
    data = source.name if source.is_hbm else emit.emit("NKIStore", f"src={source.name}")
    selected = emit_batched_topk(
        emit,
        data,
        (source.shape[0], source.shape[1]),
        k,
        sorted_output,
        lambda initial: select_rows(emit, data, source.shape[0], k, sorted_output, initial),
    )
    names = tuple(emit.emit("NKILoad", f"src={value}") for value in selected)
    shape = (source.shape[0], k)
    return (TorchValue(names[0], shape, storage_dtype="float32"), TorchValue(names[1], shape, storage_dtype="uint32"))
