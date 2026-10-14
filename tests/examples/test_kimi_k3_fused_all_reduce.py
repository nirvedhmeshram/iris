#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import gc
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import iris


# Load the real example file so this test follows the same code CI will run.
root = Path(__file__).resolve()
while not (root / "tests").is_dir() or not (root / "examples").is_dir():
    if root == root.parent:
        raise FileNotFoundError("Could not find project root")
    root = root.parent

path = root / "examples" / "34_kimi_k3_fused_all_reduce" / "example.py"
spec = importlib.util.spec_from_file_location("kimi_k3_fused_all_reduce_example", path)
k3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(k3)


def _is_gfx950():
    return torch.cuda.is_available() and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")


# Widths 1 and 4 take Lamport for the MoE reduce, 8 and 16 take two-stage, and every width
# also runs the fused AttnRes kernel. Each case makes three distinct calls, eagerly and then
# under CUDA graph replay, so a buffer reused before a slower peer finished reading it shows
# up as a wrong answer on some rank.
@pytest.mark.parametrize("tokens", [[1, 4], [8, 16]])
def test_kimi_k3_fused_all_reduce_matches_reference(tokens):
    if not dist.is_initialized():
        pytest.skip("torch.distributed is not initialized")
    if dist.get_world_size() not in (2, 4, 8):
        pytest.skip("the K3 kernels support 2, 4 or 8 ranks")
    if not _is_gfx950():
        pytest.skip("the K3 kernels use CDNA4 (gfx950) buffer intrinsics")

    ctx = None
    try:
        ctx = iris.iris(heap_size=1 << 28)
        op = k3.K3FusedAllReduce(ctx, max_tokens=max(tokens))
        seeds = [0, 1, 2]
        for case in k3.make_cases(op, ["attention", "moe", "attnres"], tokens, seeds):
            for _ in range(2):
                for i in range(len(seeds)):
                    case.run(i)
            assert k3.validate(case, ctx, "eager"), f"{case.name} M={case.m} eager"

            ctx.barrier()
            graph = k3.capture(case.run, len(seeds))
            ctx.barrier()
            case.poison()
            for _ in range(2):
                graph.replay()
            assert k3.validate(case, ctx, "graph"), f"{case.name} M={case.m} graph"
            del graph
    finally:
        if ctx is not None:
            ctx.barrier()
        del ctx
        gc.collect()
        torch.cuda.empty_cache()
