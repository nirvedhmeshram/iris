# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# The kernels in this file are adapted from TokenSpeed
# (https://github.com/lightseekorg/tokenspeed, tokenspeed_kernel/ops/communication/_iris/kimi_k3/allreduce_sharded.py
# and tokenspeed_kernel_amd/ops/gfx950/attention/kda/attn_res.py),
# Copyright (c) 2026 LightSeek Foundation, released under the MIT License.

"""
Gluon row-sharded collectives used by Kimi-K3 at tp8/ep1 on MI355X (gfx950), for token widths M >= 40 (MoE tail)
and M >= 56 (attention tail): prefill, large decode batches, and EAGLE3 verify at C16 and above.

Instead of an all-reduce followed by work that every rank repeats on all M rows, each rank reduces only its own
M / 8 consecutive rows (a pull reduce-scatter), runs the row-wise tail on them, and then pushes the result to every
peer (a push all-gather). Kernel names are kept identical to TokenSpeed so they match the names in K3 profiles:

* ``iris_k3moe_pull_scatter``: reduce the routed ``[M, 3584]`` and shared ``[M, 7168]`` partials into this rank's rows.
* ``iris_k3moe_push_gather``: add the projected routed rows, shared rows and residual, and push them to every peer.
* ``iris_k3attn_pull_scatter``: reduce the attention partial into this rank's rows and add the residual.
* ``iris_k3attn_mix_push_gather``: AttnRes mix and output RMSNorm on this rank's rows, pushed to every peer.
* ``iris_k3attn_push_gather``: the same push for rows that were mixed by a separate kernel (long histories).
* ``gluon_attn_res_fwd_gfx950``: that separate AttnRes mix and output RMSNorm kernel.

Every kernel takes the symmetric heap base of every rank as scalar arguments and hard-codes eight peers.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

cdna4 = gl.amd.cdna4


@gluon.jit
def _iris_drain_subgroup_vmem():
    """Finish this subgroup's outstanding vector memory operations."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


# ---------------------------------------------------------------------------
# AttnRes mixing (device functions)
# ---------------------------------------------------------------------------


@gluon.jit
def _attn_res_accumulate(
    value,
    scorer,
    max_logit,
    denominator,
    mixed,
    H: gl.constexpr,
    SCORE_EPS: gl.constexpr,
):
    """Fold one candidate into a normalized online softmax mixture."""
    square_sum = gl.sum(value * value, axis=0)
    dot = gl.sum(value * scorer, axis=0)
    score = dot * gl.rsqrt(square_sum / H + SCORE_EPS)
    score_delta = score - max_logit
    relative_scale = gl.exp2(-gl.abs(score_delta) * 1.4426950408889634)
    candidate_is_max = score_delta > 0.0
    old_scale = gl.where(candidate_is_max, relative_scale, 1.0)
    candidate_scale = gl.where(candidate_is_max, 1.0, relative_scale)
    denominator = denominator * old_scale + candidate_scale
    candidate_weight = candidate_scale / denominator
    mixed += candidate_weight * (value - mixed)
    return gl.maximum(max_logit, score), denominator, mixed


@gluon.jit
def _attn_res_mix_gfx950(
    prefix,
    block_residual,
    res_weight,
    score_rms_weight,
    output_rms_weight,
    token,
    hidden,
    hidden_mask,
    stride_block_t: gl.constexpr,
    stride_block_n: tl.int64,
    H: gl.constexpr,
    N: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
):
    """Mix candidates and normalize, retaining both existing BF16 boundaries."""
    prefix = prefix.to(gl.float32)
    if N == 1:
        mixed = prefix
    else:
        scorer = cdna4.buffer_load(res_weight, hidden.to(gl.int32), mask=hidden_mask, other=0.0).to(gl.float32)
        scorer *= cdna4.buffer_load(score_rms_weight, hidden.to(gl.int32), mask=hidden_mask, other=0.0).to(gl.float32)
        # Consume the resident prefix first so it becomes the running mixture.
        mixed = prefix
        square_sum = gl.sum(prefix * prefix, axis=0)
        dot = gl.sum(prefix * scorer, axis=0)
        max_logit = dot * gl.rsqrt(square_sum / H + SCORE_EPS)
        denominator = 1.0
        # A runtime loop keeps one block snapshot live at a time; unrolling
        # lets the compiler hoist every snapshot load and halves occupancy.
        for candidate in range(N - 1):
            # The stride fits int32, but candidate * stride can exceed it at large T.
            value = cdna4.buffer_load(
                block_residual + candidate * stride_block_n.to(gl.int64),
                (token * stride_block_t + hidden).to(gl.int32),
                mask=hidden_mask,
                other=0.0,
            ).to(gl.float32)
            max_logit, denominator, mixed = _attn_res_accumulate(
                value, scorer, max_logit, denominator, mixed, H, SCORE_EPS
            )
    mixed = mixed.to(gl.bfloat16).to(gl.float32)
    inverse_rms = gl.rsqrt(gl.sum(mixed * mixed, axis=0) / H + OUTPUT_EPS)
    output_weight = cdna4.buffer_load(output_rms_weight, hidden.to(gl.int32), mask=hidden_mask, other=0.0).to(
        gl.float32
    )
    return (mixed * inverse_rms * output_weight).to(gl.bfloat16)


# Standalone mixer that K3 runs when the history is too long to fuse into the gather.
_BLOCK_H = gl.constexpr(8192)
_LOAD_ELEMS = gl.constexpr(2)


def _attn_res_launch_metadata(grid, kernel, args):
    tensors_per_element = args["N"] + 2 + 2 * int(args["N"] > 1) + 2 * int(args["HAS_DELTA"]) + int(args["WRITE_BLOCK"])
    return {"name": kernel.name, "bytes": grid[0] * args["H"] * tensors_per_element * 2}


@gluon.jit(launch_metadata=_attn_res_launch_metadata)
def gluon_attn_res_fwd_gfx950(
    layer_residual,
    delta,
    block_residual,
    res_weight,
    score_rms_weight,
    output_rms_weight,
    output,
    stride_layer_t: gl.constexpr,
    stride_delta_t: gl.constexpr,
    stride_block_t: gl.constexpr,
    stride_block_n: tl.int64,
    stride_output_t: gl.constexpr,
    H: gl.constexpr,
    N: gl.constexpr,
    BLOCK_WRITE_IDX: gl.constexpr,
    HAS_DELTA: gl.constexpr,
    WRITE_BLOCK: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    token = gl.program_id(0)
    hidden_layout: gl.constexpr = gl.BlockedLayout([_LOAD_ELEMS], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, _BLOCK_H, layout=hidden_layout)
    hidden_mask = hidden < H

    prefix = cdna4.buffer_load(
        layer_residual,
        (token * stride_layer_t + hidden).to(gl.int32),
        mask=hidden_mask,
        other=0.0,
    ).to(gl.float32)
    if HAS_DELTA:
        prefix += cdna4.buffer_load(
            delta,
            (token * stride_delta_t + hidden).to(gl.int32),
            mask=hidden_mask,
            other=0.0,
        ).to(gl.float32)
        prefix = prefix.to(layer_residual.dtype.element_ty).to(gl.float32)
        cdna4.buffer_store(
            prefix.to(layer_residual.dtype.element_ty),
            layer_residual,
            (token * stride_layer_t + hidden).to(gl.int32),
            mask=hidden_mask,
        )
    if WRITE_BLOCK:
        block_write_ptr = block_residual + BLOCK_WRITE_IDX * stride_block_n.to(gl.int64)
        cdna4.buffer_store(
            prefix.to(block_residual.dtype.element_ty),
            block_write_ptr,
            (token * stride_block_t + hidden).to(gl.int32),
            mask=hidden_mask,
        )

    mixed = _attn_res_mix_gfx950(
        prefix,
        block_residual,
        res_weight,
        score_rms_weight,
        output_rms_weight,
        token,
        hidden,
        hidden_mask,
        stride_block_t,
        stride_block_n,
        H,
        N,
        SCORE_EPS,
        OUTPUT_EPS,
    )
    cdna4.buffer_store(
        mixed.to(output.dtype.element_ty),
        output,
        (token * stride_output_t + hidden).to(gl.int32),
        mask=hidden_mask,
    )


# ---------------------------------------------------------------------------
# Row-sharded collectives
# ---------------------------------------------------------------------------


@gluon.jit
def _row_partition_store_completion(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Finish local stores, publish the epoch, and wait for peer completion."""
    _iris_drain_subgroup_vmem()
    gl.barrier()
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    # Upstream Gluon does not broadcast scalar memory operands, so splat the epoch explicitly.
    epochs = gl.full([8], epoch, gl.uint32, layout=layout)
    gl.store(peer_flags + block_id * 8 + RANK, epochs, mask=remote, cache_modifier=".wt")
    local = flags + block_id * 8 + peers
    seen = gl.load(local, mask=remote, other=epochs, cache_modifier=".cv", volatile=True)
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(local, mask=remote, other=epochs, cache_modifier=".cv", volatile=True)
    gl.atomic_add(local, gl.zeros_like(epochs), mask=remote, sem="acquire", scope="sys")


@gluon.jit
def _row_partition_entry_barrier(
    flags,
    peer_flags,
    block_id,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Publish the entry epoch and acquire each peer's producer completion."""
    # Prior producers complete on the calling stream before this entry barrier.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    remote = peers != RANK
    # Upstream Gluon does not broadcast scalar memory operands, so splat the epoch explicitly.
    epochs = gl.full([8], epoch, gl.uint32, layout=layout)
    gl.atomic_xchg(
        peer_flags + block_id * 8 + RANK,
        epochs,
        mask=remote,
        sem="release",
        scope="sys",
    )
    local_flags = flags + block_id * 8 + peers
    seen = gl.load(local_flags, mask=remote, other=epochs, cache_modifier=".cv", volatile=True)
    # Accept newer epochs across uint32 wraparound; peers stay within one call.
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(local_flags, mask=remote, other=epochs, cache_modifier=".cv", volatile=True)
    # Acquire lowering already joins the workgroup after cache invalidation.
    gl.atomic_add(local_flags, gl.zeros_like(epochs), mask=remote, sem="acquire", scope="sys")


@gluon.jit
def _peer_buffers(pointer, heaps, RANK: gl.constexpr):
    """Map a symmetric BF16 buffer to all eight peer heaps."""
    offset = pointer.to(gl.uint64) - heaps[RANK]
    result = ()
    for peer in gl.static_range(8):
        result += (gl.multiple_of((heaps[peer] + offset).to(gl.pointer_type(gl.bfloat16)), 16),)
    return result


@gluon.jit
def _peer_flags(pointer, heaps, RANK: gl.constexpr, NUM_WARPS: gl.constexpr):
    """Map a symmetric flag buffer to all eight peer heaps."""
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, 8, layout=layout)
    bases = gl.full((8,), 0, gl.uint64, layout)
    for peer in gl.static_range(8):
        bases = gl.where(peers == peer, heaps[peer], bases)
    offset = pointer.to(gl.uint64) - heaps[RANK]
    return (bases + offset).to(gl.pointer_type(gl.uint32))


@gluon.jit(do_not_specialize=["ROWS"])
def iris_k3moe_pull_scatter(
    input_ptr,
    scratch_ptr,
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
    ROWS,
    FIRST_WIDTH: gl.constexpr,
    SECOND_WIDTH: gl.constexpr,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Pull K3 routed/shared partials into each rank's consecutive row shard.

    For r = q * (ROWS / 8) + u, scratch_q[u] = BF16(sum_p FP32(partial_p[r])).
    A fixed FP32 tree reduces the routed and shared inputs, packed in that order.
    The matching push-gather completes the input-reuse barrier.
    """
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    inputs = _peer_buffers(input_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    block_id = gl.program_id(0)
    epoch = gl.atomic_add(flags + block_id * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    _row_partition_entry_barrier(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)
    FIRST_ELEMENTS = ROWS // 8 * FIRST_WIDTH
    SECOND_ELEMENTS = ROWS // 8 * SECOND_WIDTH
    PARTITION_ELEMENTS = FIRST_ELEMENTS + SECOND_ELEMENTS
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(block_id, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        source = gl.where(
            offsets < FIRST_ELEMENTS,
            RANK * FIRST_ELEMENTS + offsets,
            ROWS * FIRST_WIDTH + RANK * SECOND_ELEMENTS + offsets - FIRST_ELEMENTS,
        )
        values = ()
        for step in gl.static_range(8):
            values += (gl.amd.cdna4.buffer_load(inputs[(RANK + step) % 8], source, mask, 0, cache=".cg"),)
        sums = ()
        for peer in gl.static_range(2):
            sums += (
                values[(peer - RANK) % 8].to(gl.float32) + values[(peer + 2 - RANK) % 8].to(gl.float32),
                values[(peer + 4 - RANK) % 8].to(gl.float32) + values[(peer + 6 - RANK) % 8].to(gl.float32),
            )
        reduced = ((sums[0] + sums[1]) + (sums[2] + sums[3])).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(reduced, scratch_ptr, offsets, mask, cache=".wt")


@gluon.jit(do_not_specialize=["LOCAL_ROWS"])
def iris_k3moe_push_gather(
    projection_ptr,
    shared_ptr,
    prefix_ptr,
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PREFIX_IS_SHARDED: gl.constexpr,
):
    """Add K3 routed/shared outputs and residual, then push rows to every peer.

    For r = q * LOCAL_ROWS + u, use i = u for sharded prefixes, else i = r:
        output_p[r] = BF16((FP32(prefix_q[i]) + FP32(projected_q[u]))
                          + FP32(shared_q[u]))
    Wait for peer publication before reusing the producer buffers.
    """
    # Preserve row alignment for vector loads/stores without specializing M.
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    # In-place prefixes are safe: ranks read then write disjoint rows.
    # Reduce-scatter entry waits for prior prefix consumers.
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peer_flags = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    block_id = gl.program_id(0)
    epoch = gl.atomic_add(flags + block_id * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(block_id, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        prefix_offsets = offsets
        if not PREFIX_IS_SHARDED:
            prefix_offsets += RANK * PARTITION_ELEMENTS
        a = gl.amd.cdna4.buffer_load(prefix_ptr, prefix_offsets, mask, 0, cache=".cg").to(gl.float32)
        b = gl.amd.cdna4.buffer_load(projection_ptr, offsets, mask, 0, cache=".cg").to(gl.float32)
        c = gl.amd.cdna4.buffer_load(shared_ptr, offsets, mask, 0, cache=".cg").to(gl.float32)
        result = (a + b + c).to(gl.bfloat16)
        # Precomputed peer bases keep stores consecutive without VMEM drains.
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                result,
                outputs[(RANK + step) % 8],
                offsets + RANK * PARTITION_ELEMENTS,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(flags, peer_flags, block_id, epoch, RANK, NUM_WARPS)


def _reduce_metadata(grid, kernel, args):
    elements = args["LOCAL_ROWS"] * 7168
    return {
        "name": kernel.name,
        "bytes": elements * (9 + int(args["HAS_RESIDUAL"])) * 2,
        "flops32": elements * (7 + int(args["HAS_RESIDUAL"])),
    }


def _gather_metadata(grid, kernel, args):
    return {"name": kernel.name, "bytes": args["LOCAL_ROWS"] * 7168 * 9 * 2}


def _mix_gather_metadata(grid, kernel, args):
    return {
        "name": kernel.name,
        "bytes": args["LOCAL_ROWS"]
        * 7168
        * (args["NUM_VALID_BLOCKS"] + 10 + 2 * int(args["NUM_VALID_BLOCKS"] > 0))
        * 2,
    }


@gluon.jit(launch_metadata=_reduce_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3attn_pull_scatter(
    input_ptr,
    residual_ptr,
    prefix_ptr,
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
):
    """Pull K3 attn rows into each rank's residual shard.

    For r = q * LOCAL_ROWS + u:
        reduced_q[u] = BF16(sum_p FP32(partial_p[r]))
        prefix_q[u] = BF16(FP32(reduced_q[u]) + FP32(residual_q[r]))
    Without a residual, store reduced directly. Matching push-gather orders reuse.
    """
    # Whole rows preserve vector alignment without specializing the row count.
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    inputs = _peer_buffers(input_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    _row_partition_entry_barrier(flags, peers, pid, epoch, RANK, NUM_WARPS)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        source = RANK * PARTITION_ELEMENTS + offsets
        values = ()
        for step in gl.static_range(8):
            values += (gl.amd.cdna4.buffer_load(inputs[(RANK + step) % 8], source, mask, 0, cache=".cg"),)
        # Preserve Iris's even/odd FP32 tree and the BF16 boundary before add.
        even = (values[(0 - RANK) % 8].to(gl.float32) + values[(2 - RANK) % 8].to(gl.float32)) + (
            values[(4 - RANK) % 8].to(gl.float32) + values[(6 - RANK) % 8].to(gl.float32)
        )
        odd = (values[(1 - RANK) % 8].to(gl.float32) + values[(3 - RANK) % 8].to(gl.float32)) + (
            values[(5 - RANK) % 8].to(gl.float32) + values[(7 - RANK) % 8].to(gl.float32)
        )
        prefix = (even + odd).to(gl.bfloat16)
        if HAS_RESIDUAL:
            residual = gl.amd.cdna4.buffer_load(residual_ptr, source, mask, 0, cache=".ca")
            prefix = (prefix.to(gl.float32) + residual.to(gl.float32)).to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(prefix, prefix_ptr, offsets, mask, cache=".wb")


@gluon.jit(launch_metadata=_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3attn_push_gather(
    mixed_ptr,
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
    LOCAL_ROWS,
    BLOCK_ELEMENTS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Push mixed K3 attn rows to every peer.

    Each owner q writes output_p[q * LOCAL_ROWS + u] = mixed_q[u] for every p.
    Wait for all peer stores before consuming the replicated output.
    """
    PARTITION_ELEMENTS = LOCAL_ROWS * 7168
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    lanes = gl.arange(0, BLOCK_ELEMENTS, layout=layout)
    for tile in range(pid, gl.cdiv(PARTITION_ELEMENTS, BLOCK_ELEMENTS), NUM_PROGRAMS):
        offsets = tile * BLOCK_ELEMENTS + lanes
        mask = offsets < PARTITION_ELEMENTS
        value = gl.amd.cdna4.buffer_load(mixed_ptr, offsets, mask, 0, cache=".cg")
        # Scalar peer bases let all eight stores issue without intervening drains.
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                value,
                outputs[(RANK + step) % 8],
                RANK * PARTITION_ELEMENTS + offsets,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)


@gluon.jit(launch_metadata=_mix_gather_metadata, do_not_specialize=["LOCAL_ROWS"])
def iris_k3attn_mix_push_gather(
    prefix_ptr,
    output_ptr,
    block_residual,
    res_weight,
    rms_weight,
    out_norm_weight,
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
    LOCAL_ROWS,
    STRIDE_BLOCK_T: gl.constexpr,
    STRIDE_BLOCK_N,
    NUM_VALID_BLOCKS: gl.constexpr,
    SCORE_EPS: gl.constexpr,
    OUTPUT_EPS: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Mix K3 prefixes with AttnRes history, then push normalized rows.

    For r = q * LOCAL_ROWS + u, every peer p receives:
        output_p[r] = BF16(RMSNorm(BF16(AttnRes(prefix_q[u], history[r]))))
    Owners write disjoint rows and wait for all peer stores.
    """
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    outputs = _peer_buffers(output_ptr, heaps, RANK)
    flags = ready_flags.to(gl.pointer_type(gl.uint32))
    peers = _peer_flags(flags, heaps, RANK, NUM_WARPS)
    pid = gl.program_id(0)
    epoch = gl.atomic_add(flags + pid * 8 + RANK, 1, sem="relaxed", scope="gpu") + 1
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_WARPS], [0])
    hidden = gl.arange(0, 8192, layout=layout)
    mask = hidden < 7168
    for row in range(pid, LOCAL_ROWS, NUM_PROGRAMS):
        token = RANK * LOCAL_ROWS + row
        prefix = gl.amd.cdna4.buffer_load(prefix_ptr, row * 7168 + hidden, mask, 0, cache=".ca")
        mixed = _attn_res_mix_gfx950(
            prefix,
            block_residual,
            res_weight,
            rms_weight,
            out_norm_weight,
            token,
            hidden,
            mask,
            STRIDE_BLOCK_T,
            STRIDE_BLOCK_N,
            7168,
            NUM_VALID_BLOCKS + 1,
            SCORE_EPS,
            OUTPUT_EPS,
        )
        for step in gl.static_range(8):
            gl.amd.cdna4.buffer_store(
                mixed,
                outputs[(RANK + step) % 8],
                token * 7168 + hidden,
                mask,
                cache=".wt",
            )
    _row_partition_store_completion(flags, peers, pid, epoch, RANK, NUM_WARPS)
