# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon TDM all-gather for gfx1250/gfx1260.

Three variants (select via Config.all_gather_tdm_variant):

hoisted — persistent_all_gather_tdm_gfx1250
    Each tile loop iteration: 1 HBM->LDS load + world_size stores (descriptors
    hoisted at kernel entry; world_size <= 8).

stepwise — persistent_all_gather_tdm_gfx1250_stepwise
    Same tile-parallel CTA assignment as hoisted/Triton: each CTA loads a tile
    once then stores to all ranks with dynamically built output descriptors
    (arbitrary world_size).

warp_team — persistent_all_gather_tdm_gfx1250_warp_team
    Single-wave CTAs (num_warps=1 at launch) each doing full TDM load+store on a
    sub-tile of block_size_m/warps_per_tile rows. config.num_warps sets how many
    independent single-wave CTAs stripe one logical tile (barrier-free async_wait).
    Requires block_size_m and num_warps powers of 2 with block_size_m divisible
    by num_warps.

warp_specialized — persistent_all_gather_tdm_gfx1250_warp_specialized
    warp_specialize gives each row slice its own 1-warp worker partition with a
    private smem.index(slice_id) buffer (TransferBench-style). config.num_warps in
    {1,2,4,8} splits block_size_m / num_warps rows per warp. The compiler requires
    a 4-warp default partition, so num_warps>1 uses an empty default plus one worker
    per slice (launch num_warps=4).

warp_specialized_local_smem — persistent_all_gather_tdm_gfx1250_warp_specialized_local_smem
    Same warp_specialize dispatch as warp_specialized, but each worker partition
    calls allocate_shared_memory locally (no parent smem[num_slices, ...]) to help
    the compiler prove per-partition LDS non-aliasing.

warp_specialized_improved — persistent_all_gather_tdm_gfx1250_warp_specialized_improved
    Same warp_specialize dispatch as warp_specialized_local_smem (empty 4-warp
    default + 1-warp workers) but with warp_team-style sub-tile striping so worker
    warps are not lockstep on the same tile_id each loop iteration.
"""

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.amd.gfx1250 import tdm as gfx1250_tdm
    from triton.language.core import _aggregate as aggregate

    GFX1250_TDM_AVAILABLE = True
except ImportError as e:
    raise ValueError("Gluon TDM is not available. Install Triton with Gluon TDM support or set use_tdm=False.") from e

import torch

from iris.ccl.gluon.all_to_all_tdm import TDM_MAX_DIM, TDM_ROW_BYTES, _max_lds_bytes, _validate_tdm_tile
from iris.host.tracing.kernel_artifacts import iris_launch


@gluon.jit
def persistent_all_gather_tdm_gfx1250(
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
):
    """
    All-gather via TDM: 1 HBM->LDS load + world_size HBM/XGMI stores per tile.

    Rank g writes its input tiles to output[g*M : (g+1)*M, :] on every rank.
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    out_m = M * world_size
    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    # Hoist per-destination output descriptors (traffic-shaped store order).
    if world_size > 0:
        d0 = gl.load(elem_deltas + ((group_rank + 0) % world_size))
        out_desc_0 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d0,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 1:
        d1 = gl.load(elem_deltas + ((group_rank + 1) % world_size))
        out_desc_1 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d1,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 2:
        d2 = gl.load(elem_deltas + ((group_rank + 2) % world_size))
        out_desc_2 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d2,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 3:
        d3 = gl.load(elem_deltas + ((group_rank + 3) % world_size))
        out_desc_3 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d3,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 4:
        d4 = gl.load(elem_deltas + ((group_rank + 4) % world_size))
        out_desc_4 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d4,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 5:
        d5 = gl.load(elem_deltas + ((group_rank + 5) % world_size))
        out_desc_5 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d5,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 6:
        d6 = gl.load(elem_deltas + ((group_rank + 6) % world_size))
        out_desc_6 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d6,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )
    if world_size > 7:
        d7 = gl.load(elem_deltas + ((group_rank + 7) % world_size))
        out_desc_7 = gfx1250_tdm.make_tensor_descriptor(
            base=output_ptr + d7,
            shape=[out_m, N],
            strides=[stride_out_m, stride_out_n],
            block_shape=[block_m, block_n],
            layout=smem_layout,
        )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n
        out_row_off = group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        if world_size > 0:
            gfx1250_tdm.async_store(out_desc_0, [out_row_off, col_off], smem)
        if world_size > 1:
            gfx1250_tdm.async_store(out_desc_1, [out_row_off, col_off], smem)
        if world_size > 2:
            gfx1250_tdm.async_store(out_desc_2, [out_row_off, col_off], smem)
        if world_size > 3:
            gfx1250_tdm.async_store(out_desc_3, [out_row_off, col_off], smem)
        if world_size > 4:
            gfx1250_tdm.async_store(out_desc_4, [out_row_off, col_off], smem)
        if world_size > 5:
            gfx1250_tdm.async_store(out_desc_5, [out_row_off, col_off], smem)
        if world_size > 6:
            gfx1250_tdm.async_store(out_desc_6, [out_row_off, col_off], smem)
        if world_size > 7:
            gfx1250_tdm.async_store(out_desc_7, [out_row_off, col_off], smem)

        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_all_gather_tdm_gfx1250_stepwise(
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
):
    """
    All-gather via TDM: same structure as hoisted/Triton all-gather.

    Outer loop: tile_id = pid, pid + COMM_SMS, ... (one load per tile per CTA).
    Inner loop: traffic-shaped stores to all ranks with a dynamically built
    output descriptor per destination (no world_size unroll cap).
    """
    pid = gl.program_id(0)

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[block_n, 8]], [block_m, block_n], [1, 0])
    smem = gl.allocate_shared_memory(dtype, [block_m, block_n], layout=smem_layout)

    out_m = M * world_size
    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = num_tiles_m * num_tiles_n

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[block_m, block_n],
        layout=smem_layout,
    )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row_off = tile_m * block_m
        col_off = tile_n * block_n
        out_row_off = group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        for dest_idx in range(world_size):
            dest_group_rank = (group_rank + dest_idx) % world_size
            delta = gl.load(elem_deltas + dest_group_rank)
            out_desc = gfx1250_tdm.make_tensor_descriptor(
                base=output_ptr + delta,
                shape=[out_m, N],
                strides=[stride_out_m, stride_out_n],
                block_shape=[block_m, block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_store(out_desc, [out_row_off, col_off], smem)

        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_all_gather_tdm_gfx1250_warp_team(
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
    warps_per_tile: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """
    All-gather via single-wave TDM on row sub-tiles.

    Each CTA runs with one warp (forced at launch) and owns a private LDS tile of
    sub_block_m = block_m / warps_per_tile rows. Full TDM async_load + async_store
    on PaddedSharedLayout without cross-wave WG barriers at async_wait.
    """
    pid = gl.program_id(0)

    sub_block_m: gl.constexpr = block_m // warps_per_tile

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[block_n, 8]], [sub_block_m, block_n], [1, 0]
    )
    smem = gl.allocate_shared_memory(dtype, [sub_block_m, block_n], layout=smem_layout)

    out_m = M * world_size
    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)
    total_sub_tiles = num_tiles_m * num_tiles_n * warps_per_tile

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[sub_block_m, block_n],
        layout=smem_layout,
    )

    for sub_tile_id in range(pid, total_sub_tiles, COMM_SMS):
        tile_flat = sub_tile_id // warps_per_tile
        warp_slice = sub_tile_id - tile_flat * warps_per_tile
        tile_m = tile_flat // num_tiles_n
        tile_n = tile_flat - tile_m * num_tiles_n
        row_off = tile_m * block_m + warp_slice * sub_block_m
        col_off = tile_n * block_n
        out_row_off = group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        for dest_idx in range(world_size):
            dest_group_rank = (group_rank + dest_idx) % world_size
            delta = gl.load(elem_deltas + dest_group_rank)
            out_desc = gfx1250_tdm.make_tensor_descriptor(
                base=output_ptr + delta,
                shape=[out_m, N],
                strides=[stride_out_m, stride_out_n],
                block_shape=[sub_block_m, block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_store(out_desc, [out_row_off, col_off], smem)

        gfx1250_tdm.async_wait(0)


@aggregate
class _WarpSliceLocalArgs:
    input_ptr: gl.tensor
    output_ptr: gl.tensor
    elem_deltas: gl.tensor
    group_rank: gl.constexpr
    world_size: gl.constexpr
    block_m: gl.constexpr
    block_n: gl.constexpr
    sub_block_m: gl.constexpr
    COMM_SMS: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        input_ptr,
        output_ptr,
        elem_deltas,
        group_rank,
        world_size,
        block_m,
        block_n,
        sub_block_m,
        COMM_SMS,
    ):
        self.input_ptr = input_ptr
        self.output_ptr = output_ptr
        self.elem_deltas = elem_deltas
        self.group_rank = gl.constexpr(group_rank)
        self.world_size = gl.constexpr(world_size)
        self.block_m = gl.constexpr(block_m)
        self.block_n = gl.constexpr(block_n)
        self.sub_block_m = gl.constexpr(sub_block_m)
        self.COMM_SMS = gl.constexpr(COMM_SMS)


@gluon.jit
def _ag_warp_slice_loop_local_smem(
    args: _WarpSliceLocalArgs,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    pid,
    num_tiles_n,
    total_tiles,
    slice_id: gl.constexpr,
):
    """Worker partition with smem allocated locally (not shared via parent)."""
    dtype: gl.constexpr = args.input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[args.block_n, 8]], [args.sub_block_m, args.block_n], [1, 0]
    )
    smem = gl.allocate_shared_memory(dtype, [args.sub_block_m, args.block_n], layout=smem_layout)

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=args.input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[args.sub_block_m, args.block_n],
        layout=smem_layout,
    )

    out_m = M * args.world_size

    for tile_id in range(pid, total_tiles, args.COMM_SMS):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id - tile_m * num_tiles_n
        row_off = tile_m * args.block_m + slice_id * args.sub_block_m
        col_off = tile_n * args.block_n
        out_row_off = args.group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        for dest_idx in range(args.world_size):
            dest_group_rank = (args.group_rank + dest_idx) % args.world_size
            delta = gl.load(args.elem_deltas + dest_group_rank)
            out_desc = gfx1250_tdm.make_tensor_descriptor(
                base=args.output_ptr + delta,
                shape=[out_m, N],
                strides=[stride_out_m, stride_out_n],
                block_shape=[args.sub_block_m, args.block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_store(out_desc, [out_row_off, col_off], smem)

        gfx1250_tdm.async_wait(0)


@gluon.jit
def _ag_warp_slice_loop_improved(
    args: _WarpSliceLocalArgs,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    pid,
    num_tiles_n,
    num_tiles_m,
    num_slices: gl.constexpr,
    slice_id: gl.constexpr,
):
    """
    Improved warp slice worker: local smem + sub-tile striping.

    Striping assigns sub_tile_id = pid * num_slices + slice_id, pid * num_slices +
    slice_id + COMM_SMS * num_slices, ... so warps in a CTA work on different tiles
    instead of sharing the same tile_id each iteration.
    """
    dtype: gl.constexpr = args.input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[args.block_n, 8]], [args.sub_block_m, args.block_n], [1, 0]
    )
    smem = gl.allocate_shared_memory(dtype, [args.sub_block_m, args.block_n], layout=smem_layout)

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=args.input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[args.sub_block_m, args.block_n],
        layout=smem_layout,
    )

    out_m = M * args.world_size
    total_sub_tiles = num_tiles_m * num_tiles_n * num_slices

    for sub_tile_id in range(pid * num_slices + slice_id, total_sub_tiles, args.COMM_SMS * num_slices):
        tile_flat = sub_tile_id // num_slices
        tile_m = tile_flat // num_tiles_n
        tile_n = tile_flat - tile_m * num_tiles_n
        row_off = tile_m * args.block_m + slice_id * args.sub_block_m
        col_off = tile_n * args.block_n
        out_row_off = args.group_rank * M + row_off

        gfx1250_tdm.async_load(input_desc, [row_off, col_off], smem)
        gfx1250_tdm.async_wait(0)

        for dest_idx in range(args.world_size):
            dest_group_rank = (args.group_rank + dest_idx) % args.world_size
            delta = gl.load(args.elem_deltas + dest_group_rank)
            out_desc = gfx1250_tdm.make_tensor_descriptor(
                base=args.output_ptr + delta,
                shape=[out_m, N],
                strides=[stride_out_m, stride_out_n],
                block_shape=[args.sub_block_m, args.block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_store(out_desc, [out_row_off, col_off], smem)

        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_all_gather_tdm_gfx1250_warp_specialized_improved(
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
    num_slices: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """warp_specialize + sub-tile striping (same dispatch as warp_specialized_local_smem)."""
    pid = gl.program_id(0)
    sub_block_m: gl.constexpr = block_m // num_slices

    num_tiles_m = gl.cdiv(M, block_m)
    num_tiles_n = gl.cdiv(N, block_n)

    args = _WarpSliceLocalArgs(
        input_ptr,
        output_ptr,
        elem_deltas,
        group_rank,
        world_size,
        block_m,
        block_n,
        sub_block_m,
        COMM_SMS,
    )

    if num_slices == 1:
        _ag_warp_slice_loop_improved(
            args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
            pid, num_tiles_n, num_tiles_m, num_slices, 0,
        )
    elif num_slices == 2:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 0),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 1),
                ),
            ],
            [1, 1],
        )
    elif num_slices == 4:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 0),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 1),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 2),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 3),
                ),
            ],
            [1, 1, 1, 1],
        )
    elif num_slices == 8:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 0),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 1),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 2),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 3),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 4),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 5),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 6),
                ),
                (
                    _ag_warp_slice_loop_improved,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n,
                     pid, num_tiles_n, num_tiles_m, num_slices, 7),
                ),
            ],
            [1, 1, 1, 1, 1, 1, 1, 1],
        )


@gluon.jit
def persistent_all_gather_tdm_gfx1250_warp_specialized_local_smem(
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
    num_slices: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """
    warp_specialized variant: each worker allocates its own smem inside the
    partition (no parent smem[num_slices, ...]).
    """
    pid = gl.program_id(0)
    sub_block_m: gl.constexpr = block_m // num_slices

    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = gl.cdiv(M, block_m) * num_tiles_n

    args = _WarpSliceLocalArgs(
        input_ptr,
        output_ptr,
        elem_deltas,
        group_rank,
        world_size,
        block_m,
        block_n,
        sub_block_m,
        COMM_SMS,
    )

    if num_slices == 1:
        _ag_warp_slice_loop_local_smem(
            args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0
        )
    elif num_slices == 2:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 1),
                ),
            ],
            [1, 1],
        )
    elif num_slices == 4:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 1),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 2),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 3),
                ),
            ],
            [1, 1, 1, 1],
        )
    elif num_slices == 8:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 1),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 2),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 3),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 4),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 5),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 6),
                ),
                (
                    _ag_warp_slice_loop_local_smem,
                    (args, M, N, stride_in_m, stride_in_n, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 7),
                ),
            ],
            [1, 1, 1, 1, 1, 1, 1, 1],
        )


@aggregate
class _WarpSliceArgs:
    smem: gl.shared_memory_descriptor
    input_desc: gfx1250_tdm.tensor_descriptor
    output_ptr: gl.tensor
    elem_deltas: gl.tensor
    group_rank: gl.constexpr
    world_size: gl.constexpr
    block_m: gl.constexpr
    block_n: gl.constexpr
    sub_block_m: gl.constexpr
    COMM_SMS: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        smem,
        input_desc,
        output_ptr,
        elem_deltas,
        group_rank,
        world_size,
        block_m,
        block_n,
        sub_block_m,
        COMM_SMS,
    ):
        self.smem = smem
        self.input_desc = input_desc
        self.output_ptr = output_ptr
        self.elem_deltas = elem_deltas
        self.group_rank = gl.constexpr(group_rank)
        self.world_size = gl.constexpr(world_size)
        self.block_m = gl.constexpr(block_m)
        self.block_n = gl.constexpr(block_n)
        self.sub_block_m = gl.constexpr(sub_block_m)
        self.COMM_SMS = gl.constexpr(COMM_SMS)


@gluon.jit
def _ag_warp_epilogue():
    """Empty default warp_specialize partition (compiler requires num_warps % 4 == 0)."""


@gluon.jit
def _ag_warp_slice_loop(
    args: _WarpSliceArgs,
    M,
    N,
    stride_out_m,
    stride_out_n,
    pid,
    num_tiles_n,
    total_tiles,
    slice_id: gl.constexpr,
):
    """One warp partition: private smem slice, independent TDM load+store sub-tile loop."""
    smem_slice = args.smem.index(slice_id)
    smem_layout: gl.constexpr = smem_slice.layout
    out_m = M * args.world_size

    for tile_id in range(pid, total_tiles, args.COMM_SMS):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id - tile_m * num_tiles_n
        row_off = tile_m * args.block_m + slice_id * args.sub_block_m
        col_off = tile_n * args.block_n
        out_row_off = args.group_rank * M + row_off

        gfx1250_tdm.async_load(args.input_desc, [row_off, col_off], smem_slice)
        gfx1250_tdm.async_wait(0)

        for dest_idx in range(args.world_size):
            dest_group_rank = (args.group_rank + dest_idx) % args.world_size
            delta = gl.load(args.elem_deltas + dest_group_rank)
            out_desc = gfx1250_tdm.make_tensor_descriptor(
                base=args.output_ptr + delta,
                shape=[out_m, N],
                strides=[stride_out_m, stride_out_n],
                block_shape=[args.sub_block_m, args.block_n],
                layout=smem_layout,
            )
            gfx1250_tdm.async_store(out_desc, [out_row_off, col_off], smem_slice)

        gfx1250_tdm.async_wait(0)


@gluon.jit
def persistent_all_gather_tdm_gfx1250_warp_specialized(
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
    num_slices: gl.constexpr,
    COMM_SMS: gl.constexpr,
):
    """
    All-gather via warp_specialize: each row slice runs in a dedicated 1-warp worker
    partition with private smem.index(slice_id) (TransferBench-style per-warp LDS).
    """
    pid = gl.program_id(0)
    sub_block_m: gl.constexpr = block_m // num_slices

    dtype: gl.constexpr = input_ptr.dtype.element_ty
    smem_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[block_n, 8]], [sub_block_m, block_n], [1, 0]
    )
    smem = gl.allocate_shared_memory(dtype, [num_slices, sub_block_m, block_n], layout=smem_layout)

    num_tiles_n = gl.cdiv(N, block_n)
    total_tiles = gl.cdiv(M, block_m) * num_tiles_n

    input_desc = gfx1250_tdm.make_tensor_descriptor(
        base=input_ptr,
        shape=[M, N],
        strides=[stride_in_m, stride_in_n],
        block_shape=[sub_block_m, block_n],
        layout=smem_layout,
    )

    args = _WarpSliceArgs(
        smem,
        input_desc,
        output_ptr,
        elem_deltas,
        group_rank,
        world_size,
        block_m,
        block_n,
        sub_block_m,
        COMM_SMS,
    )

    if num_slices == 1:
        _ag_warp_slice_loop(args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0)
    elif num_slices == 2:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 1)),
            ],
            [1, 1],
        )
    elif num_slices == 4:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 1)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 2)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 3)),
            ],
            [1, 1, 1, 1],
        )
    elif num_slices == 8:
        gl.warp_specialize(
            [
                (_ag_warp_epilogue, ()),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 0)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 1)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 2)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 3)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 4)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 5)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 6)),
                (_ag_warp_slice_loop, (args, M, N, stride_out_m, stride_out_n, pid, num_tiles_n, total_tiles, 7)),
            ],
            [1, 1, 1, 1, 1, 1, 1, 1],
        )


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _validate_tdm_warp_subtile(config, elem_size: int, max_lds: int, *, lds_factor: int) -> None:
    """Validate row sub-tiles for warp_team TDM variant."""
    warps_per_tile = config.num_warps
    block_m = config.block_size_m
    block_n = config.block_size_n

    if warps_per_tile not in (1, 2, 4, 8):
        raise ValueError(
            f"TDM all-gather (warp_team/warp_specialized) requires num_warps in {{1, 2, 4, 8}}, "
            f"got {warps_per_tile}."
        )

    if not _is_power_of_2(warps_per_tile):
        raise ValueError(
            f"TDM all-gather (warp_team) requires num_warps to be a power of 2, got {warps_per_tile}."
        )

    if block_m % warps_per_tile != 0:
        raise ValueError(
            f"TDM all-gather (warp_team) requires block_size_m divisible by num_warps, "
            f"got block_size_m={block_m}, num_warps={warps_per_tile}."
        )

    sub_block_m = block_m // warps_per_tile
    if not _is_power_of_2(sub_block_m):
        raise ValueError(
            f"TDM all-gather (warp_team) requires block_size_m/num_warps to be a power of 2 "
            f"(PaddedSharedLayout sub-tile requirement), got {sub_block_m}."
        )

    if not _is_power_of_2(block_n):
        raise ValueError(
            f"TDM block_size_n must be a power of 2 (PaddedSharedLayout requirement), got {block_n}."
        )

    if sub_block_m > TDM_MAX_DIM or block_n > TDM_MAX_DIM:
        raise ValueError(
            f"TDM sub-tile dimensions must be <= {TDM_MAX_DIM}, "
            f"got sub_block_m={sub_block_m}, block_n={block_n}."
        )

    row_bytes = block_n * elem_size
    if row_bytes % TDM_ROW_BYTES != 0:
        raise ValueError(
            f"TDM inner tile width must be a multiple of {TDM_ROW_BYTES} bytes "
            f"(block_size_n={block_n}, elem_size={elem_size} -> {row_bytes} bytes/row)."
        )

    smem_bytes = lds_factor * sub_block_m * (block_n + 8) * elem_size
    if smem_bytes > max_lds:
        raise ValueError(
            f"TDM warp sub-tile LDS {smem_bytes} bytes exceeds device max {max_lds} bytes "
            f"(lds_factor={lds_factor}, sub_block_m={sub_block_m}, block_n={block_n}, elem_size={elem_size})."
        )


def _validate_tdm_warp_team_tile(config, elem_size: int, max_lds: int) -> None:
    _validate_tdm_warp_subtile(config, elem_size, max_lds, lds_factor=1)


def _validate_tdm_warp_specialized_tile(config, elem_size: int, max_lds: int) -> None:
    _validate_tdm_warp_subtile(config, elem_size, max_lds, lds_factor=config.num_warps)


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
    input_tensor,
    output_tensor,
    ctx,
    rank_in_group,
    rank_global,
    world_size,
    rank_start,
    rank_stride,
    config,
):
    """Launch the Gluon TDM all-gather kernel."""
    if not GFX1250_TDM_AVAILABLE:
        raise ValueError("TDM all-gather requires GFX1250 TDM support (gfx1250/gfx1260 + Gluon TDM)")

    if config.all_gather_variant != "persistent":
        raise ValueError(
            f"TDM all_gather only supports all_gather_variant='persistent', got '{config.all_gather_variant}'."
        )

    tdm_variant = config.all_gather_tdm_variant
    if tdm_variant == "hoisted":
        kernel = persistent_all_gather_tdm_gfx1250
        algorithm = "all_gather_tdm"
    elif tdm_variant == "stepwise":
        kernel = persistent_all_gather_tdm_gfx1250_stepwise
        algorithm = "all_gather_tdm_stepwise"
    elif tdm_variant == "warp_team":
        kernel = persistent_all_gather_tdm_gfx1250_warp_team
        algorithm = "all_gather_tdm_warp_team"
    elif tdm_variant == "warp_specialized":
        kernel = persistent_all_gather_tdm_gfx1250_warp_specialized
        algorithm = "all_gather_tdm_warp_specialized"
    elif tdm_variant == "warp_specialized_local_smem":
        kernel = persistent_all_gather_tdm_gfx1250_warp_specialized_local_smem
        algorithm = "all_gather_tdm_warp_specialized_local_smem"
    elif tdm_variant == "warp_specialized_improved":
        kernel = persistent_all_gather_tdm_gfx1250_warp_specialized_improved
        algorithm = "all_gather_tdm_warp_specialized_improved"
    else:
        raise ValueError(f"Unknown all_gather_tdm_variant: {tdm_variant}")

    M, N = input_tensor.shape[:2]
    expected_output_shape = (world_size * M, N)
    if output_tensor.shape[:2] != expected_output_shape:
        raise ValueError(f"Output shape {output_tensor.shape[:2]} does not match expected {expected_output_shape}")

    if tdm_variant == "hoisted" and world_size > 8:
        raise ValueError(f"TDM all-gather (hoisted) supports world_size <= 8, got {world_size}")

    elem_size = input_tensor.element_size()
    device_index = input_tensor.device.index
    if device_index is None:
        device_index = 0
    max_lds = _max_lds_bytes(device_index)
    if tdm_variant == "warp_team":
        _validate_tdm_warp_team_tile(config, elem_size, max_lds)
    elif tdm_variant in ("warp_specialized", "warp_specialized_local_smem"):
        _validate_tdm_warp_specialized_tile(config, elem_size, max_lds)
    elif tdm_variant == "warp_specialized_improved":
        _validate_tdm_warp_specialized_tile(config, elem_size, max_lds)
    else:
        _validate_tdm_tile(config, elem_size, max_lds)

    stride_in_m, stride_in_n = input_tensor.stride(0), input_tensor.stride(1)
    stride_out_m, stride_out_n = output_tensor.stride(0), output_tensor.stride(1)

    elem_deltas = _build_elem_deltas(input_tensor, ctx, rank_global, world_size, rank_start, rank_stride)

    if tdm_variant == "warp_team":
        launch_num_warps = 1
    elif tdm_variant == "warp_specialized_improved":
        launch_num_warps = 4 if config.num_warps > 1 else 1
    elif tdm_variant in ("warp_specialized", "warp_specialized_local_smem"):
        launch_num_warps = 4 if config.num_warps > 1 else 1
    else:
        launch_num_warps = config.num_warps

    launch_args = [
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
        config.block_size_n,
    ]
    if tdm_variant == "warp_team":
        launch_args.extend([config.num_warps, config.comm_sms])
    elif tdm_variant in ("warp_specialized", "warp_specialized_local_smem", "warp_specialized_improved"):
        launch_args.extend([config.num_warps, config.comm_sms])
    else:
        launch_args.append(config.comm_sms)

    iris_launch(
        kernel,
        (config.comm_sms,),
        *launch_args,
        num_stages=config.num_stages,
        num_warps=launch_num_warps,
        waves_per_eu=config.waves_per_eu,
        algorithm=algorithm,
        rank=rank_global,
        dtype=input_tensor.dtype,
    )
