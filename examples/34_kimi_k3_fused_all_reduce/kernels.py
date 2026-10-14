# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# The kernels in this file are adapted from TokenSpeed
# (https://github.com/lightseekorg/tokenspeed, tokenspeed_kernel/ops/communication/iris.py),
# Copyright (c) 2026 LightSeek Foundation, released under the MIT License.

"""
Gluon all-reduce kernels used by Kimi-K3 decode at tp8/ep1 on MI355X (gfx950).

Kernel names are kept identical to TokenSpeed so they match the names in K3 profiles:

* ``iris_reduce_symmetric_two_stage_gluon_kernel``: reduce-scatter then all-gather out of
  symmetric memory. Used for the per-layer attention all-reduce, and for the MoE tail
  when the routed+shared payload is too wide for Lamport.
* ``lamport_all_reduce_bf16``: push-based Lamport all-reduce for the MoE tail at small row
  counts. Peers poll for sentinel-free data instead of handshaking on a flag.
* ``iris_push_one_shot_allreduce_residual_attnres_gluon_kernel``: the attention all-reduce
  fused with the residual add, the AttnRes softmax combine, and the output RMSNorm.

All three take the symmetric heap base of every rank as scalar arguments (padded to eight)
rather than a pointer to a table, so a peer address costs no memory load.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@gluon.jit
def _iris_heap_base(
    rank: gl.constexpr,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
):
    if rank == 0:
        return heap_base_0
    if rank == 1:
        return heap_base_1
    if rank == 2:
        return heap_base_2
    if rank == 3:
        return heap_base_3
    if rank == 4:
        return heap_base_4
    if rank == 5:
        return heap_base_5
    if rank == 6:
        return heap_base_6
    return heap_base_7


@gluon.jit
def _iris_peer_heaps(
    peer_ids,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
):
    """Vector of heap bases indexed by ``peer_ids``."""
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    return peer_heaps


@gluon.jit
def _iris_drain_subgroup_vmem():
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _iris_sync_rank_token(
    flags,
    row,
    token,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Write ``token`` into every peer's flag row, then wait for every peer's token locally."""
    layout: gl.constexpr = gl.BlockedLayout([1], [SUBGROUP_SIZE], [NUM_WARPS], [0])
    peers = gl.arange(0, WORLD_SIZE, layout=layout)
    peer_mask = peers != RANK
    peer_heaps = _iris_peer_heaps(
        peers,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    # Upstream Gluon does not broadcast scalar atomic operands, so splat them explicitly.
    tokens = gl.full([WORLD_SIZE], token, gl.int32, layout=layout)
    flags_heap_offset = tl.cast(flags, gl.uint64) - local_heap
    peer_flags = tl.cast(peer_heaps + flags_heap_offset, gl.pointer_type(gl.int32))
    gl.atomic_xchg(
        peer_flags + row * WORLD_SIZE + RANK,
        tokens,
        mask=peer_mask,
        sem="release",
        scope="sys",
    )
    local_flags = flags + row * WORLD_SIZE + peers
    seen = gl.load(local_flags, mask=peer_mask, other=tokens, cache_modifier=".cv", volatile=True)
    while gl.max(gl.where(peer_mask & (seen != token), 1, 0), axis=0) != 0:
        seen = gl.load(local_flags, mask=peer_mask, other=tokens, cache_modifier=".cv", volatile=True)
    gl.atomic_add(local_flags, gl.zeros_like(tokens), mask=peer_mask, sem="acquire", scope="sys")
    _iris_drain_subgroup_vmem()
    gl.barrier()


@gluon.jit
def _iris_sync_rank_epoch(
    ready_flags,
    block_id,
    epoch,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    PUBLISH: gl.constexpr,
):
    """Wait until every peer has reached ``epoch`` for this block.

    With ``PUBLISH`` the epoch is pushed into each peer's flags and polled locally; otherwise
    each peer's own flag is polled remotely.
    """
    ready_layout: gl.constexpr = gl.BlockedLayout([1], [SUBGROUP_SIZE], [NUM_WARPS], [0])
    peer_ids = gl.arange(0, WORLD_SIZE, layout=ready_layout)
    peer_heaps = _iris_peer_heaps(
        peer_ids,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    flags_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    peer_mask = peer_ids != RANK
    if PUBLISH:
        remote_flags = tl.cast(peer_heaps + flags_heap_offset, gl.pointer_type(gl.int32))
        remote_flags += block_id * WORLD_SIZE + RANK
        gl.store(remote_flags, epoch, mask=peer_mask, cache_modifier=".wt")
        wait_flags = ready_flags + block_id * WORLD_SIZE + peer_ids
    else:
        wait_flags = tl.cast(peer_heaps + flags_heap_offset, gl.pointer_type(gl.int32))
        wait_flags += block_id * WORLD_SIZE + peer_ids
    seen = gl.full([WORLD_SIZE], 0, gl.int32, layout=ready_layout)
    while gl.min(gl.where(peer_mask, seen, epoch), axis=0) < epoch:
        seen = gl.load(wait_flags, mask=peer_mask, other=epoch, cache_modifier=".cv", volatile=True)


@gluon.jit
def _unpack_16bitx4(packed, dtype: gl.constexpr):
    value_0 = (packed & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    value_1 = ((packed >> 16) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    value_2 = ((packed >> 32) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    value_3 = ((packed >> 48) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    return value_0, value_1, value_2, value_3


@gluon.jit
def _pack_16bitx4(value_0, value_1, value_2, value_3, dtype: gl.constexpr):
    bits_0 = value_0.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_1 = value_1.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_2 = value_2.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_3 = value_3.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    return bits_0 | (bits_1 << 16) | (bits_2 << 32) | (bits_3 << 48)


@gluon.jit
def _unpack_word(packed, dtype: gl.constexpr, elements_per_word: gl.constexpr):
    # Explicit branches keep Gluon from type-checking the inactive bitcast.
    if elements_per_word == 4:
        return _unpack_16bitx4(packed, dtype)
    else:
        value_0 = (packed & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        value_1 = ((packed >> 32) & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        return value_0, value_1, value_0, value_1


@gluon.jit
def _pack_word(value_0, value_1, value_2, value_3, dtype: gl.constexpr, elements_per_word: gl.constexpr):
    if elements_per_word == 4:
        return _pack_16bitx4(value_0, value_1, value_2, value_3, dtype)
    else:
        bits_0 = value_0.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        bits_1 = value_1.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        return bits_0 | (bits_1 << 32)


# ---------------------------------------------------------------------------
# Two-stage (reduce-scatter + all-gather) all-reduce
# ---------------------------------------------------------------------------


@gluon.jit
def iris_reduce_symmetric_two_stage_gluon_kernel(
    input_sym_ptr,
    scratch_sym_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    PARTITION_WORDS: gl.constexpr,
    BLOCK_WORDS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_TILES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
    EXIT_BARRIER: gl.constexpr,
):
    """Reduce-scatter symmetric inputs, then all-gather the rank partitions.

    Each rank owns ``PARTITION_WORDS`` 64-bit words of the payload. Stage one pulls that
    partition from every peer, sums it in FP32 and writes it to symmetric scratch. Stage two
    pulls every rank's reduced partition into the local output. Traffic per rank is 2x the
    payload regardless of world size, where one-shot is ``world_size`` x.
    """
    block_id = gl.program_id(0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    scratch_heap_offset = tl.cast(scratch_sym_ptr, gl.uint64) - local_heap
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS_PER_LANE],
        [1, SUBGROUP_SIZE],
        [WORLD_SIZE, NUM_WARPS // WORLD_SIZE],
        [1, 0],
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout([WORLD_SIZE, 1], [1, SUBGROUP_SIZE], [1, NUM_WARPS], [0, 1])
    peer_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    word_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    reduce_word_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    peer_ids = gl.arange(0, WORLD_SIZE, layout=peer_layout)
    words = gl.arange(0, BLOCK_WORDS, layout=word_layout)
    reduce_words = gl.arange(0, BLOCK_WORDS, layout=reduce_word_layout)
    peer_heaps = _iris_peer_heaps(
        peer_ids,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    peer_inputs = tl.cast(gl.expand_dims(peer_heaps, 1) + input_heap_offset, gl.pointer_type(gl.uint64))
    peer_scratch = tl.cast(gl.expand_dims(peer_heaps, 1) + scratch_heap_offset, gl.pointer_type(gl.uint64))
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[32, 4]], [WORLD_SIZE, BLOCK_WORDS], [1, 0])
    peer_values = gl.allocate_shared_memory(gl.uint64, [WORLD_SIZE, BLOCK_WORDS], shared_layout)
    rank_start: gl.constexpr = RANK * PARTITION_WORDS

    # Stage one: reduce only this rank's partition of the full input into symmetric scratch.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        input_offset = rank_start + partition_offset
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_inputs + gl.expand_dims(input_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        # Transpose through LDS so each lane holds one word from every peer.
        peer_values.store(values)
        gl.barrier()

        packed = peer_values.load(reduce_layout)
        value_0, value_1, value_2, value_3 = _unpack_word(packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD)
        reduced = _pack_word(
            gl.sum(value_0, axis=0),
            gl.sum(value_1, axis=0),
            gl.sum(value_2, axis=0),
            gl.sum(value_3, axis=0),
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            reduced,
            tl.cast(scratch_sym_ptr, gl.pointer_type(gl.uint64)),
            (tile_id * BLOCK_WORDS + reduce_words).to(gl.int32),
            mask=tile_id * BLOCK_WORDS + reduce_words < PARTITION_WORDS,
            cache=".wt",
        )
        gl.barrier()
        tile_id += NUM_PROGRAMS

    partitions_ready = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        partitions_ready,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
    )

    # Stage two: gather one reduced partition from every rank into the local output.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_scratch + gl.expand_dims(partition_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=".cg",
        )
        output_offset = gl.expand_dims(peer_ids * PARTITION_WORDS, 1) + gl.expand_dims(partition_offset, 0)
        gl.store(
            tl.cast(output_ptr, gl.pointer_type(gl.uint64)) + output_offset,
            values,
            mask=gl.expand_dims(mask, 0),
        )
        tile_id += NUM_PROGRAMS

    if EXIT_BARRIER:
        # Callers that stage into the symmetric input before launching cannot rotate buffers
        # safely: under graph capture the staging copy records a fixed address and replays it,
        # so a rank one invocation ahead would overwrite an input its slower peers are still
        # reducing. Holding the kernel until every peer has finished reading makes a single
        # staging buffer correct by construction, at the price of one more rendezvous.
        # Producer-direct callers own their input and pass False.
        reads_done = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
        _iris_sync_rank_epoch(
            ready_flags,
            block_id,
            reads_done,
            local_heap,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
            RANK,
            WORLD_SIZE,
            NUM_WARPS,
            SUBGROUP_SIZE,
            PUBLISH=True,
        )


# ---------------------------------------------------------------------------
# Lamport all-reduce (K3 MoE tail, small row counts)
# ---------------------------------------------------------------------------


@gluon.jit
def _iris_sanitize_lamport_bf16(values):
    # -0.0 is the "not yet written" sentinel, so real -0.0 is published as +0.0.
    bits = values.to(gl.uint16, bitcast=True)
    return gl.where(bits == 0x8000, 0, bits).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _iris_wait_lamport_peers(
    region,
    generation,
    offsets,
    valid,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    values = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        values += (gl.full([64, 8], 0, gl.bfloat16, LAYOUT),)
    active = valid
    while gl.max(active.to(gl.int32), 0) != 0:
        loaded = ()
        # A lane reloads every peer until all packs are ready. Issue the independent reads
        # before checking them; retired lanes keep their data. .cv controls hardware caches,
        # not compiler volatility.
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            pointer = region + generation * WORLD_SIZE * MAX_ELEMENTS + peer * MAX_ELEMENTS
            loaded += (
                gl.amd.cdna4.buffer_load(
                    pointer,
                    offsets,
                    mask=active[:, None],
                    other=values[delta - 1],
                    cache=".cv",
                ),
            )
        active = gl.full([64], False, gl.int1, gl.SliceLayout(1, LAYOUT))
        for delta in gl.static_range(0, WORLD_SIZE - 1):
            active |= valid & (gl.max((loaded[delta].to(gl.uint16, bitcast=True) == 0x8000).to(gl.int32), 1) != 0)
        values = loaded
    return values


@gluon.jit
def lamport_all_reduce_bf16(
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """Push BF16 tiles to every peer's mailbox and poll for all peer packs.

    One 64-lane subgroup owns 512 elements; each lane owns one 16-byte pack. The mailbox is
    pre-filled with a pattern whose every pack contains a -0.0 half, and a pack is "arrived"
    once none of its halves is -0.0. That makes the data its own ready flag, so there is no
    separate signal round trip. ``NUM_STAGES`` mailbox generations rotate per tile so a fast
    rank cannot overwrite a generation a slow peer has not consumed.
    """
    gl.static_assert(WORLD_SIZE <= 8)
    gl.static_assert(TOTAL_ELEMENTS % 512 == 0)
    gl.static_assert(TOTAL_ELEMENTS <= MAX_ELEMENTS)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    mask = offsets < TOTAL_ELEMENTS
    generation = gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    stride: gl.constexpr = WORLD_SIZE * MAX_ELEMENTS
    local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    local = _iris_sanitize_lamport_bf16(local)
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = _iris_heap_base(
            peer,
            region_0,
            region_1,
            region_2,
            region_3,
            region_4,
            region_5,
            region_6,
            region_7,
        )
        # Preserve 16-byte publication stores after casting integer addresses.
        destination = gl.multiple_of(destination.to(gl.pointer_type(gl.bfloat16)), 16)
        destination += generation * stride + RANK * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(local, destination, offsets, mask=mask, cache=".wt")

    peers = _iris_wait_lamport_peers(
        region_sym_ptr,
        generation,
        offsets,
        pack * 8 < TOTAL_ELEMENTS,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        layout,
    )
    # All ranks use the same FP32 addition order, then round once to BF16.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            term = peers[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if peer == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(accumulator.to(gl.bfloat16), output_ptr, offsets, mask=mask)

    # Clear only this tile's consumed generation. Skipped tiles keep their own epochs, so
    # mixed row counts need no global counter or tail clearing.
    sentinel = gl.where(offsets % 2 == 0, 0, 0x8000).to(gl.uint16).to(gl.bfloat16, bitcast=True)
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = region_sym_ptr + generation * stride + peer * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(sentinel, destination, offsets, mask=mask, cache=".wt")
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((generation + 1) % NUM_STAGES).to(gl.int32))


# ---------------------------------------------------------------------------
# Fused all-reduce + residual + AttnRes combine + RMSNorm (K3 attention output)
# ---------------------------------------------------------------------------


@gluon.jit
def _iris_attnres_epilogue(
    reduced,
    residual_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    row_offsets,
    weight_offsets,
    mask,
    row,
    HIDDEN: gl.constexpr,
    EPS: gl.constexpr,
):
    """Fold the post-attention prefix into the AttnRes block softmax and normalize.

    ``scratch_{m,s,acc}`` hold an online-softmax partial over the stored block snapshots,
    computed earlier (and overlapped with attention) by a separate kernel: running max logit,
    sum of exp(logit - m), and the matching weighted sum of block values.
    """
    reduced = reduced.to(gl.bfloat16).to(gl.float32)
    residual = gl.amd.cdna4.buffer_load(residual_ptr, row_offsets, mask=mask, other=0.0).to(gl.float32)
    prefix = (reduced + residual).to(gl.bfloat16).to(gl.float32)
    gl.amd.cdna4.buffer_store(prefix.to(residual_out_ptr.dtype.element_ty), residual_out_ptr, row_offsets, mask=mask)
    score_weight = gl.amd.cdna4.buffer_load(score_weight_ptr, weight_offsets, mask=mask, other=0.0).to(gl.float32)
    square_sum = gl.sum(gl.where(mask, prefix * prefix, 0.0), axis=0)
    dot = gl.sum(gl.where(mask, prefix * score_weight, 0.0), axis=0)
    prefix_logit = dot * gl.rsqrt(square_sum / HIDDEN + EPS)
    block_m = gl.load(scratch_m_ptr + row)
    block_s = gl.load(scratch_s_ptr + row)
    maximum = gl.maximum(block_m, prefix_logit)
    block_correction = gl.exp(block_m - maximum)
    prefix_weight = gl.exp(prefix_logit - maximum)
    inverse_sum = 1.0 / (block_s * block_correction + prefix_weight)
    block_acc = gl.amd.cdna4.buffer_load(scratch_acc_ptr, row_offsets, mask=mask, other=0.0).to(gl.float32)
    mixed = ((block_acc * block_correction + prefix_weight * prefix) * inverse_sum).to(gl.bfloat16).to(gl.float32)
    output_square_sum = gl.sum(gl.where(mask, mixed * mixed, 0.0), axis=0)
    inverse_rms = gl.rsqrt(output_square_sum / HIDDEN + EPS)
    output_weight = gl.amd.cdna4.buffer_load(output_weight_ptr, weight_offsets, mask=mask, other=0.0).to(gl.float32)
    gl.amd.cdna4.buffer_store(
        (mixed * inverse_rms * output_weight).to(hidden_ptr.dtype.element_ty),
        hidden_ptr,
        row_offsets,
        mask=mask,
    )


@gluon.jit
def iris_push_one_shot_allreduce_residual_attnres_gluon_kernel(
    partial_ptr,
    residual_ptr,
    inbox_sym_ptr,
    score_weight_ptr,
    output_weight_ptr,
    scratch_m_ptr,
    scratch_s_ptr,
    scratch_acc_ptr,
    hidden_ptr,
    residual_out_ptr,
    generations,
    ready_flags,
    inbox_0: gl.pointer_type(gl.bfloat16),
    inbox_1: gl.pointer_type(gl.bfloat16),
    inbox_2: gl.pointer_type(gl.bfloat16),
    inbox_3: gl.pointer_type(gl.bfloat16),
    inbox_4: gl.pointer_type(gl.bfloat16),
    inbox_5: gl.pointer_type(gl.bfloat16),
    inbox_6: gl.pointer_type(gl.bfloat16),
    inbox_7: gl.pointer_type(gl.bfloat16),
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    HIDDEN: gl.constexpr,
    BLOCK: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    READY_SLOT_STRIDE: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Push attention rows into two-slot rank-ordered inboxes, reduce, then run the epilogue.

    One program per token row. Each rank writes its partial into every peer's inbox, exchanges
    one token per peer, and sums the inbox in rank order so every rank gets bitwise-identical
    results. Peer inboxes are host-computed byte addresses; the explicit BF16 pointer
    annotations preserve their pointer ABI without a Python pointer wrapper.
    """
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([ELEMENTS_PER_THREAD], [SUBGROUP_SIZE], [NUM_WARPS], [0])
    element = gl.arange(0, BLOCK, layout=layout)
    mask = element < HIDDEN
    row_offsets = (row * HIDDEN + element).to(gl.int32)
    weight_offsets = element.to(gl.int32)
    local = gl.amd.cdna4.buffer_load(partial_ptr, row_offsets, mask=mask, other=0.0)
    generation = gl.load(generations + row).to(gl.int32) + 1
    slot = generation & 1
    inbox_slot_offset = slot * WORLD_SIZE * MAX_ELEMENTS
    sync_ready_flags = ready_flags + slot * READY_SLOT_STRIDE
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    for peer_delta in gl.static_range(1, WORLD_SIZE):
        destination = (RANK + peer_delta) % WORLD_SIZE
        destination_inbox = _iris_heap_base(
            destination,
            inbox_0,
            inbox_1,
            inbox_2,
            inbox_3,
            inbox_4,
            inbox_5,
            inbox_6,
            inbox_7,
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination_inbox + inbox_slot_offset + RANK * MAX_ELEMENTS,
            row_offsets,
            mask=mask,
            cache=".wt",
        )
    _iris_drain_subgroup_vmem()
    gl.barrier()

    _iris_sync_rank_token(
        sync_ready_flags,
        row,
        generation,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
    )
    local_inbox = inbox_sym_ptr + inbox_slot_offset

    if RANK == 0:
        reduced = local.to(gl.float32)
    else:
        reduced = gl.amd.cdna4.buffer_load(local_inbox, row_offsets, mask=mask, other=0.0, cache=".cg").to(gl.float32)
    for source in gl.static_range(1, WORLD_SIZE):
        if source == RANK:
            reduced += local.to(gl.float32)
        else:
            reduced += gl.amd.cdna4.buffer_load(
                local_inbox + source * MAX_ELEMENTS,
                row_offsets,
                mask=mask,
                other=0.0,
                cache=".cg",
            ).to(gl.float32)

    gl.store(generations + row, generation)
    _iris_attnres_epilogue(
        reduced,
        residual_ptr,
        score_weight_ptr,
        output_weight_ptr,
        scratch_m_ptr,
        scratch_s_ptr,
        scratch_acc_ptr,
        hidden_ptr,
        residual_out_ptr,
        row_offsets,
        weight_offsets,
        mask,
        row,
        HIDDEN,
        EPS,
    )
