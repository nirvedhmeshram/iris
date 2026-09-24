# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon TDM reduce-scatter for gfx1250/gfx1260.

Two variants (select via Config.reduce_scatter_tdm_variant):

hoisted — persistent_reduce_scatter_tdm_gfx1250
    Each tile: world_size TDM loads (hoisted input descriptors) + register sum
    + one local TDM store (world_size <= 8).

stepwise — persistent_reduce_scatter_tdm_gfx1250_stepwise
    Same tile assignment as hoisted/Triton: traffic-shaped loads from all ranks
    with dynamically built input descriptors (arbitrary world_size).
"""

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.amd.gfx1250 import tdm as gfx1250_tdm

    GFX1250_TDM_AVAILABLE = True
except ImportError as e:
    raise ValueError("Gluon TDM is not available. Install Triton with Gluon TDM support or set use_tdm=False.") from e

import torch

from iris.ccl.gluon.all_to_all_tdm import TDM_MAX_DIM, TDM_ROW_BYTES, _is_power_of_2, _max_lds_bytes, _validate_tdm_tile
from iris.host.tracing.kernel_artifacts import iris_launch


def _split_block_m(block_m: int):
    """Decompose block_m into the two largest powers of 2 that sum to it.

    384 -> (256, 128).  Returns None when block_m is not expressible as a sum of
    exactly two distinct powers of 2, which is the only shape the split kernel
    stages (two LDS buffers).
    """
    hi = 1 << (block_m.bit_length() - 1)
    lo = block_m - hi
    if hi and lo and (lo & (lo - 1)) == 0:
        return hi, lo
    return None


def _validate_tdm_split_tile(config, elem_size: int, max_lds: int) -> None:
    """Tile check for the split kernel.

    block_m need not be a power of 2 -- that is the point -- but each piece it
    decomposes into must be, since each gets its own PaddedSharedLayout.
    """
    block_m = config.block_size_m
    block_n = config.block_size_n
    parts = _split_block_m(block_m)
    if parts is None:
        raise ValueError(
            f"split variant needs block_size_m to be a sum of two powers of 2 "
            f"(e.g. 384 = 256 + 128), got block_size_m={block_m}."
        )
    hi, lo = parts
    if not _is_power_of_2(block_n):
        raise ValueError(f"TDM block_size_n must be a power of 2, got {block_n}.")
    if max(hi, block_n) > TDM_MAX_DIM:
        raise ValueError(f"TDM tile dims must be <= {TDM_MAX_DIM}.")
    row_bytes = block_n * elem_size
    if row_bytes % TDM_ROW_BYTES != 0:
        raise ValueError(
            f"TDM inner tile width must be a multiple of {TDM_ROW_BYTES} bytes "
            f"(block_size_n={block_n}, elem_size={elem_size} -> {row_bytes} bytes/row)."
        )
    # both staging buffers are resident simultaneously
    smem_bytes = (hi + lo) * (block_n + 8) * elem_size
    if smem_bytes > max_lds:
        raise ValueError(
            f"split TDM tile LDS {smem_bytes} bytes exceeds device max {max_lds} "
            f"(hi={hi}, lo={lo}, block_n={block_n})."
        )


@gluon.jit
def persistent_reduce_scatter_tdm_gfx1250(
    input_ptr,
    output_ptr,
    elem_deltas,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    DISTRIBUTION: gl.constexpr,
):
    """
    Reduce-scatter via TDM: W HBM->LDS loads + register sum + one local HBM store per tile.

    Tile assignment matches the Triton two-shot kernel (block or strided distribution).
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    acc_dtype: gl.constexpr = gl.float32 if dtype != gl.int8 else gl.int32

    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    total_elems: gl.constexpr = block_m * block_n
    elems_per_thread: gl.constexpr = total_elems // (32 * NUM_WARPS)
    tile_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])

    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n
    tiles_per_rank = gl.cdiv(total_tiles, world_size)

    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = gl.maximum(remaining, 0)
        max_tile_offset = gl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = gl.maximum(remaining, 0)
        max_tile_offset = gl.minimum(tiles_per_rank, remaining)

    out_desc = gfx1250_tdm.make_tensor_descriptor(
        base=output_ptr,
        shape=[M, N],
        strides=[stride_out_m, stride_out_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    # Hoist per-rank input descriptors (TDM cannot build descriptors inside nested loops).
    # world_size is constexpr; up to 8 ranks (same traffic shape as all-gather TDM).
    d0 = gl.load(elem_deltas + 0)
    in_desc_0 = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr + d0,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )
    if world_size > 1:
        d1 = gl.load(elem_deltas + 1)
        in_desc_1 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d1,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 2:
        d2 = gl.load(elem_deltas + 2)
        in_desc_2 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d2,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 3:
        d3 = gl.load(elem_deltas + 3)
        in_desc_3 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d3,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 4:
        d4 = gl.load(elem_deltas + 4)
        in_desc_4 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d4,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 5:
        d5 = gl.load(elem_deltas + 5)
        in_desc_5 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d5,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 6:
        d6 = gl.load(elem_deltas + 6)
        in_desc_6 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d6,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 7:
        d7 = gl.load(elem_deltas + 7)
        in_desc_7 = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d7,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )

    # Traffic-shaped load order (same as Triton two-shot): rotate by CTA id
    start_rank_idx = pid % world_size

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n

        acc = gl.zeros([block_m, block_n], dtype=acc_dtype, layout=tile_layout)

        for i in gl.static_range(world_size):
            src_idx = (start_rank_idx + i) % world_size
            in_desc = in_desc_0
            if world_size > 1:
                if src_idx == 1:
                    in_desc = in_desc_1
            if world_size > 2:
                if src_idx == 2:
                    in_desc = in_desc_2
            if world_size > 3:
                if src_idx == 3:
                    in_desc = in_desc_3
            if world_size > 4:
                if src_idx == 4:
                    in_desc = in_desc_4
            if world_size > 5:
                if src_idx == 5:
                    in_desc = in_desc_5
            if world_size > 6:
                if src_idx == 6:
                    in_desc = in_desc_6
            if world_size > 7:
                if src_idx == 7:
                    in_desc = in_desc_7
            gfx1250_tdm.async_load(in_desc, [row_off, col_off], smem)
            gfx1250_tdm.async_wait(0)

            tile = smem.load(tile_layout).to(acc_dtype)
            acc = acc + tile

        smem.store(acc.to(dtype))
        gfx1250_tdm.async_store(out_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_reduce_scatter_tdm_gfx1250_split(
    input_ptr,
    output_ptr,
    elem_deltas,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    block_m: gl.constexpr,
    BLOCK_M_HI: gl.constexpr,
    BLOCK_M_LO: gl.constexpr,
    block_n: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    DISTRIBUTION: gl.constexpr,
):
    """
    Split-tile reduce-scatter via TDM: a block_m that is a SUM OF TWO POWERS
    OF 2 (e.g. 384 = 256 + 128).

    PaddedSharedLayout asserts on non power-of-2 extents, so a 384-row LDS tile
    cannot be allocated directly.  The register accumulator has no such
    restriction (BlockedLayout is fine at 384), so we stage the tile through two
    legal LDS buffers -- HI rows then LO rows -- and keep one accumulator per
    piece.  Logical tile height is BLOCK_M_HI + BLOCK_M_LO.

    Reduce-scatter via TDM: W HBM->LDS loads + register sum + one local HBM store per tile.

    Tile assignment matches the Triton two-shot kernel (block or strided distribution).
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    acc_dtype: gl.constexpr = gl.float32 if dtype != gl.int8 else gl.int32

    smem_layout_hi: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[block_n, 8]], [BLOCK_M_HI, block_n], [1, 0])
    smem_layout_lo: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[block_n, 8]], [BLOCK_M_LO, block_n], [1, 0])
    smem_hi = gl.allocate_shared_memory(dtype, [BLOCK_M_HI, block_n], layout=smem_layout_hi)
    smem_lo = gl.allocate_shared_memory(dtype, [BLOCK_M_LO, block_n], layout=smem_layout_lo)

    tile_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])

    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n
    tiles_per_rank = gl.cdiv(total_tiles, world_size)

    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = gl.maximum(remaining, 0)
        max_tile_offset = gl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = gl.maximum(remaining, 0)
        max_tile_offset = gl.minimum(tiles_per_rank, remaining)

    out_desc_hi = gfx1250_tdm.make_tensor_descriptor(
        base=output_ptr,
        shape=[M, N],
        strides=[stride_out_m, stride_out_n],
        block_shape=[BLOCK_M_HI, block_n],
        layout=smem_layout_hi,


    )

    out_desc_lo = gfx1250_tdm.make_tensor_descriptor(
        base=output_ptr,
        shape=[M, N],
        strides=[stride_out_m, stride_out_n],
        block_shape=[BLOCK_M_LO, block_n],
        layout=smem_layout_lo,


    )

    # Hoist per-rank input descriptors (TDM cannot build descriptors inside nested loops).
    # world_size is constexpr; up to 8 ranks (same traffic shape as all-gather TDM).
    d0 = gl.load(elem_deltas + 0)
    in_desc_0_hi = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr + d0,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[BLOCK_M_HI, block_n],
        layout=smem_layout_hi,

    )
    in_desc_0_lo = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr + d0,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[BLOCK_M_LO, block_n],
        layout=smem_layout_lo,

    )
    if world_size > 1:
        d1 = gl.load(elem_deltas + 1)
        in_desc_1_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d1,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_1_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d1,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )
    if world_size > 2:
        d2 = gl.load(elem_deltas + 2)
        in_desc_2_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d2,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_2_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d2,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )
    if world_size > 3:
        d3 = gl.load(elem_deltas + 3)
        in_desc_3_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d3,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_3_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d3,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )
    if world_size > 4:
        d4 = gl.load(elem_deltas + 4)
        in_desc_4_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d4,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_4_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d4,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )
    if world_size > 5:
        d5 = gl.load(elem_deltas + 5)
        in_desc_5_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d5,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_5_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d5,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )
    if world_size > 6:
        d6 = gl.load(elem_deltas + 6)
        in_desc_6_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d6,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_6_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d6,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )
    if world_size > 7:
        d7 = gl.load(elem_deltas + 7)
        in_desc_7_hi = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d7,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_HI, block_n],
            layout=smem_layout_hi,

        )
        in_desc_7_lo = gfx1250_tdm.make_tensor_descriptor(
            base=input_ptr + d7,
            shape=[M, N],
            strides=[stride_in_m, stride_in_n],
            block_shape=[BLOCK_M_LO, block_n],
            layout=smem_layout_lo,

        )

    # Traffic-shaped load order (same as Triton two-shot): rotate by CTA id
    start_rank_idx = pid % world_size

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n

        acc_hi = gl.zeros([BLOCK_M_HI, block_n], dtype=acc_dtype, layout=tile_layout)
        acc_lo = gl.zeros([BLOCK_M_LO, block_n], dtype=acc_dtype, layout=tile_layout)

        for i in gl.static_range(world_size):
            src_idx = (start_rank_idx + i) % world_size
            in_desc_h = in_desc_0_hi
            in_desc_l = in_desc_0_lo
            if world_size > 1:
                if src_idx == 1:
                    in_desc_h = in_desc_1_hi

                    in_desc_l = in_desc_1_lo
            if world_size > 2:
                if src_idx == 2:
                    in_desc_h = in_desc_2_hi

                    in_desc_l = in_desc_2_lo
            if world_size > 3:
                if src_idx == 3:
                    in_desc_h = in_desc_3_hi

                    in_desc_l = in_desc_3_lo
            if world_size > 4:
                if src_idx == 4:
                    in_desc_h = in_desc_4_hi

                    in_desc_l = in_desc_4_lo
            if world_size > 5:
                if src_idx == 5:
                    in_desc_h = in_desc_5_hi

                    in_desc_l = in_desc_5_lo
            if world_size > 6:
                if src_idx == 6:
                    in_desc_h = in_desc_6_hi

                    in_desc_l = in_desc_6_lo
            if world_size > 7:
                if src_idx == 7:
                    in_desc_h = in_desc_7_hi

                    in_desc_l = in_desc_7_lo
            gfx1250_tdm.async_load(in_desc_h, [row_off, col_off], smem_hi)
            gfx1250_tdm.async_load(in_desc_l, [row_off + BLOCK_M_HI, col_off], smem_lo)
            gfx1250_tdm.async_wait(0)

            acc_hi = acc_hi + smem_hi.load(tile_layout).to(acc_dtype)
            acc_lo = acc_lo + smem_lo.load(tile_layout).to(acc_dtype)

        smem_hi.store(acc_hi.to(dtype))
        gfx1250_tdm.async_store(out_desc_hi, [row_off, col_off], smem_hi)
        smem_lo.store(acc_lo.to(dtype))
        gfx1250_tdm.async_store(out_desc_lo, [row_off + BLOCK_M_HI, col_off], smem_lo)
        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_reduce_scatter_tdm_gfx1250_stepwise(
    input_ptr,
    output_ptr,
    elem_deltas,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    group_rank: gl.constexpr,
    world_size: gl.constexpr,
    block_m: gl.constexpr,
    block_n: gl.constexpr,
    COMM_SMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    DISTRIBUTION: gl.constexpr,
):
    """
    Reduce-scatter via TDM: same structure as hoisted/Triton two-shot.

    Outer loop: assigned tiles per CTA (block or strided distribution).
    Inner loop: traffic-shaped loads from all ranks with a dynamically built
    input descriptor per source (no world_size unroll cap).
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    acc_dtype: gl.constexpr = gl.float32 if dtype != gl.int8 else gl.int32

    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    total_elems: gl.constexpr = block_m * block_n
    elems_per_thread: gl.constexpr = total_elems // (32 * NUM_WARPS)
    tile_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])

    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n
    tiles_per_rank = gl.cdiv(total_tiles, world_size)

    if DISTRIBUTION == 0:
        start_tile = group_rank
        stride = world_size
        remaining = total_tiles - start_tile
        remaining = gl.maximum(remaining, 0)
        max_tile_offset = gl.cdiv(remaining, stride)
    else:
        start_tile = group_rank * tiles_per_rank
        stride = 1
        remaining = total_tiles - start_tile
        remaining = gl.maximum(remaining, 0)
        max_tile_offset = gl.minimum(tiles_per_rank, remaining)

    out_desc = gfx1250_tdm.make_tensor_descriptor(
        base=output_ptr,
        shape=[M, N],
        strides=[stride_out_m, stride_out_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    start_rank_idx = pid % world_size

    for tile_offset in range(pid, max_tile_offset, COMM_SMS):
        tile_id = start_tile + tile_offset * stride
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n

        acc = gl.zeros([block_m, block_n], dtype=acc_dtype, layout=tile_layout)

        for i in gl.static_range(world_size):
            src_idx = (start_rank_idx + i) % world_size
            delta = gl.load(elem_deltas + src_idx)
            in_desc = gfx1250_tdm.make_tensor_descriptor(
                base=input_ptr + delta,
                shape=[M, N],
                strides=[stride_in_m, stride_in_n],
                block_shape=[block_m, block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_load(in_desc, [row_off, col_off], smem)
            gfx1250_tdm.async_wait(0)

            tile = smem.load(tile_layout).to(acc_dtype)
            acc = acc + tile

        smem.store(acc.to(dtype))
        gfx1250_tdm.async_store(out_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)


def _build_elem_deltas(input_tensor, ctx, rank_global, world_size, rank_start, rank_stride):
    heap_bases = ctx.get_heap_bases()
    local_base = heap_bases[rank_global]
    elem_size = input_tensor.element_size()
    elem_deltas = torch.empty(world_size, dtype=torch.int64, device=input_tensor.device)
    for i in range(world_size):
        target_iris_rank = rank_start + i * rank_stride
        elem_deltas[i] = (heap_bases[target_iris_rank] - local_base) // elem_size
    return elem_deltas


def launch(
    output_tensor,
    input_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Gluon TDM reduce-scatter kernel."""
    if not GFX1250_TDM_AVAILABLE:
        raise ValueError("TDM reduce-scatter requires GFX1250 TDM support (gfx1250/gfx1260 + Gluon TDM)")

    tdm_variant = config.reduce_scatter_tdm_variant
    if tdm_variant == "hoisted":
        kernel = persistent_reduce_scatter_tdm_gfx1250
        algorithm = "reduce_scatter_tdm"
    elif tdm_variant == "split":
        kernel = persistent_reduce_scatter_tdm_gfx1250_split
        algorithm = "reduce_scatter_tdm_split"
    elif tdm_variant == "stepwise":
        kernel = persistent_reduce_scatter_tdm_gfx1250_stepwise
        algorithm = "reduce_scatter_tdm_stepwise"
    else:
        raise ValueError(f"Unknown reduce_scatter_tdm_variant: {tdm_variant}")

    M, N = input_tensor.shape[:2]
    if output_tensor.shape[:2] != (M, N):
        raise ValueError(f"Output shape {output_tensor.shape[:2]} does not match input shape {(M, N)}")

    if tdm_variant == "hoisted" and world_size > 8:
        raise ValueError(f"TDM reduce-scatter (hoisted) supports world_size <= 8, got {world_size}")

    elem_size = input_tensor.element_size()
    device_index = input_tensor.device.index
    if device_index is None:
        device_index = 0
    max_lds = _max_lds_bytes(device_index)
    if tdm_variant == "split":
        _validate_tdm_split_tile(config, elem_size, max_lds)
    else:
        _validate_tdm_tile(config, elem_size, max_lds)

    tile_elems = config.block_size_m * config.block_size_n
    threads_per_cta = 32 * config.num_warps
    if tile_elems % threads_per_cta != 0:
        raise ValueError(
            f"TDM reduce-scatter requires block_size_m * block_size_n divisible by "
            f"32 * num_warps ({threads_per_cta}), got {config.block_size_m} * {config.block_size_n} = {tile_elems}."
        )

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    elem_deltas = _build_elem_deltas(input_tensor, ctx, rank_global, world_size, rank_start, rank_stride)

    distribution = config.all_reduce_distribution

    # The split kernel takes the two power-of-2 pieces block_m decomposes into,
    # since each needs its own PaddedSharedLayout.
    split_args = []
    if tdm_variant == "split":
        hi, lo = _split_block_m(config.block_size_m)
        split_args = [hi, lo]

    iris_launch(
        kernel,
        (config.comm_sms,),
        input_tensor,
        output_tensor,
        elem_deltas,
        M,
        N,
        stride_in_m,
        stride_in_n,
        stride_out_m,
        stride_out_n,
        rank_in_group,
        world_size,
        config.block_size_m,
        *split_args,
        config.block_size_n,
        config.comm_sms,
        config.num_warps,
        distribution,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm=algorithm,
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
