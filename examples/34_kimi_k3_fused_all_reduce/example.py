#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: the Iris all-reduces in Kimi-K3 decode at tp8/ep1.

Each K3 decoder layer at tp8/ep1 issues two collectives per rank:

* attention all-reduce over a ``[M, 7168]`` partial. It runs either as the plain two-stage
  kernel, or (for M <= 16, when the model enables it) fused with the residual add, the
  AttnRes softmax combine and the next RMSNorm.
* MoE all-reduce over the routed-expert latent output ``[M, 3584]`` and the shared-expert
  output ``[M, 7168]``. Both producers write directly into consecutive symmetric-heap views
  and one launch reduces both ("producer-direct"): Lamport for M <= 6, two-stage above.

M is the token width of the forward: concurrency for plain decode, 4 x concurrency for the
EAGLE3 verify forward (concurrency 8 -> M = 32).

Run with:
    torchrun --nproc_per_node=8 --standalone example.py --validate --graph --benchmark
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.distributed as dist
import triton
from triton.experimental.gluon import language as gl

import iris

# Load the sibling kernels.py by path so this works both as a script and when the test imports it.
_spec = importlib.util.spec_from_file_location(
    "k3_fused_all_reduce_kernels", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
)
kernels = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kernels
_spec.loader.exec_module(kernels)


# Kimi-K3 shapes.
K3_HIDDEN = 7168
K3_ROUTED_HIDDEN = 3584  # Latent MoE: routed experts run at half width, projected back after the reduce.
K3_RMS_EPS = 1e-6

# Launch parameters, as tuned for gfx950 in TokenSpeed.
SUBGROUP_SIZE = 64
PACKED_WORD_BYTES = 8
TWO_STAGE_MAX_PROGRAMS = 84
TWO_STAGE_NUM_WARPS = 8
TWO_STAGE_WORDS_PER_LANE = 2
LAMPORT_MAX_ROWS = 6
LAMPORT_STAGES = 3
LAMPORT_BLOCK_ELEMENTS = 512
ATTNRES_MAX_ROWS = 16
ATTNRES_NUM_WARPS = 16
ATTNRES_ELEMENTS_PER_THREAD = 8

_GL_DTYPES = {torch.bfloat16: gl.bfloat16, torch.float16: gl.float16, torch.float32: gl.float32}


class K3FusedAllReduce:
    """Symmetric-heap workspaces and launchers for the three K3 decode collectives.

    Args:
        ctx: Iris context; every rank in it participates.
        max_tokens: Largest token width M any call will use.
        hidden: Attention / shared-expert output width.
        routed_hidden: Routed-expert (latent) output width.
    """

    def __init__(self, ctx, max_tokens, hidden=K3_HIDDEN, routed_hidden=K3_ROUTED_HIDDEN):
        self.ctx = ctx
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        if self.world_size not in (2, 4, 8):
            raise ValueError(f"these kernels support 2, 4 or 8 ranks, got {self.world_size}")
        if TWO_STAGE_NUM_WARPS % self.world_size:
            raise ValueError("two-stage subgroups must divide evenly across ranks")
        self.max_tokens = max_tokens
        self.hidden = hidden
        self.routed_hidden = routed_hidden
        self.row_numel = routed_hidden + hidden
        dtype = torch.bfloat16
        self.dtype = dtype
        self.elements_per_word = PACKED_WORD_BYTES // dtype.itemsize
        ws = self.world_size

        # Attention all-reduce (two-stage). The input is staged into one symmetric buffer
        # before launch, which is safe only because the kernel holds an exit barrier.
        attn_numel = max_tokens * hidden
        self.attn_input = ctx.zeros((attn_numel,), dtype=dtype)
        self.attn_scratch = ctx.zeros((triton.cdiv(attn_numel, ws),), dtype=dtype)
        self.attn_flags = ctx.zeros((TWO_STAGE_MAX_PROGRAMS, ws), dtype=torch.int32)

        # MoE producer-direct all-reduce. Producers write routed+shared rows into moe_input.
        moe_numel = max_tokens * self.row_numel
        if self.row_numel % LAMPORT_BLOCK_ELEMENTS:
            raise ValueError("MoE row width must be a whole number of Lamport tiles")
        self.moe_input = ctx.zeros((moe_numel,), dtype=dtype)
        self.moe_scratch = ctx.zeros((triton.cdiv(moe_numel, ws),), dtype=dtype)
        # Separate epochs from the attention reduce: both run inside one layer, and a shared
        # counter would let one path's epoch satisfy the other's barrier.
        self.moe_flags = ctx.zeros((TWO_STAGE_MAX_PROGRAMS, ws), dtype=torch.int32)
        self.moe_output = torch.empty((moe_numel,), dtype=dtype, device="cuda")
        self.lamport_max_numel = min(max_tokens, LAMPORT_MAX_ROWS) * self.row_numel
        self.lamport_region = ctx.zeros((LAMPORT_STAGES, ws, self.lamport_max_numel), dtype=dtype)
        # Alternating +0/-0 halves: every lane's 16-byte pack holds a -0 sentinel.
        self.lamport_region.view(torch.int32).fill_(-(2**31))
        self.lamport_epochs = torch.zeros(
            (self.lamport_max_numel // LAMPORT_BLOCK_ELEMENTS,), dtype=torch.int32, device="cuda"
        )

        # Fused attention all-reduce + residual + AttnRes. Two inbox slots, one per row parity.
        self.attnres_max_rows = min(max_tokens, ATTNRES_MAX_ROWS)
        self.attnres_max_numel = self.attnres_max_rows * hidden
        self.attnres_inbox = ctx.zeros((2, ws, self.attnres_max_numel), dtype=dtype)
        self.attnres_generations = torch.zeros((self.attnres_max_rows,), dtype=torch.int32, device="cuda")
        self.attnres_flags = ctx.zeros((2, self.attnres_max_rows, ws), dtype=torch.int32)

        torch.cuda.synchronize()
        ctx.barrier()

        heap_bases = [int(address) for address in ctx.get_heap_bases().tolist()]
        self.heap_bases = tuple(heap_bases + [heap_bases[-1]] * (8 - ws))
        self.lamport_peers = self._peer_addresses(self.lamport_region)
        self.attnres_peers = self._peer_addresses(self.attnres_inbox)

    def _peer_addresses(self, tensor):
        offset = tensor.data_ptr() - self.heap_bases[self.rank]
        return tuple(base + offset for base in self.heap_bases)

    # -- two-stage ---------------------------------------------------------------------------

    def _two_stage(self, input_sym, scratch, output, flags, numel, exit_barrier):
        ws = self.world_size
        assert numel % (ws * self.elements_per_word) == 0, "payload must split into whole words per rank"
        assert output.data_ptr() % PACKED_WORD_BYTES == 0
        partition_words = numel // ws // self.elements_per_word
        block_words = TWO_STAGE_NUM_WARPS * SUBGROUP_SIZE * TWO_STAGE_WORDS_PER_LANE // ws
        num_tiles = triton.cdiv(partition_words, block_words)
        num_programs = min(num_tiles, TWO_STAGE_MAX_PROGRAMS)
        kernels.iris_reduce_symmetric_two_stage_gluon_kernel[(num_programs,)](
            input_sym,
            scratch,
            output,
            flags,
            *self.heap_bases,
            RANK=self.rank,
            WORLD_SIZE=ws,
            PARTITION_WORDS=partition_words,
            BLOCK_WORDS=block_words,
            NUM_PROGRAMS=num_programs,
            NUM_TILES=num_tiles,
            NUM_WARPS=TWO_STAGE_NUM_WARPS,
            SUBGROUP_SIZE=SUBGROUP_SIZE,
            WORDS_PER_LANE=TWO_STAGE_WORDS_PER_LANE,
            ELEMENT_DTYPE=_GL_DTYPES[self.dtype],
            ELEMENTS_PER_WORD=self.elements_per_word,
            EXIT_BARRIER=exit_barrier,
            num_warps=TWO_STAGE_NUM_WARPS,
        )

    def attention_all_reduce(self, partial, out):
        """Plain attention all-reduce: ``out = sum over ranks of partial``."""
        numel = partial.numel()
        assert partial.shape[-1] == self.hidden and numel <= self.attn_input.numel()
        self.attn_input[:numel].copy_(partial.view(-1))
        self._two_stage(self.attn_input, self.attn_scratch, out.view(-1), self.attn_flags, numel, exit_barrier=True)
        return out

    # -- MoE producer-direct -----------------------------------------------------------------

    def moe_outputs(self, num_tokens):
        """Symmetric views the routed and shared expert GEMMs write into directly."""
        assert 0 < num_tokens <= self.max_tokens
        routed_numel = num_tokens * self.routed_hidden
        routed = self.moe_input[:routed_numel].view(num_tokens, self.routed_hidden)
        shared = self.moe_input[routed_numel : num_tokens * self.row_numel].view(num_tokens, self.hidden)
        return routed, shared

    def uses_lamport(self, num_tokens):
        return num_tokens * self.row_numel <= self.lamport_max_numel

    def moe_all_reduce(self, num_tokens):
        """Reduce both producer outputs in one launch; returns views of a local output buffer."""
        numel = num_tokens * self.row_numel
        if self.uses_lamport(num_tokens):
            kernels.lamport_all_reduce_bf16[(numel // LAMPORT_BLOCK_ELEMENTS,)](
                self.moe_input,
                self.lamport_region,
                self.moe_output,
                self.lamport_epochs,
                *self.lamport_peers,
                RANK=self.rank,
                WORLD_SIZE=self.world_size,
                TOTAL_ELEMENTS=numel,
                MAX_ELEMENTS=self.lamport_max_numel,
                NUM_STAGES=LAMPORT_STAGES,
                num_warps=1,
            )
        else:
            # The producer owns moe_input, so no staging copy and no exit barrier. Peers have
            # finished reading this rank's input by the mid-kernel barrier, before any rank
            # can return and let the next producer overwrite it.
            self._two_stage(self.moe_input, self.moe_scratch, self.moe_output, self.moe_flags, numel, False)
        routed_numel = num_tokens * self.routed_hidden
        routed = self.moe_output[:routed_numel].view(num_tokens, self.routed_hidden)
        shared = self.moe_output[routed_numel:numel].view(num_tokens, self.hidden)
        return routed, shared

    # -- fused attention all-reduce + residual + AttnRes -------------------------------------

    def attention_residual_attnres(
        self, partial, residual, score_weight, output_weight, scratch, hidden_out, residual_out, eps=K3_RMS_EPS
    ):
        """Reduce the attention partial and finish the AttnRes mix in the same kernel.

        Returns ``(hidden_out, residual_out)``: the RMS-normalized AttnRes mixture that feeds the
        MoE, and ``residual + all_reduce(partial)``.
        """
        num_tokens = partial.shape[0]
        assert 0 < num_tokens <= self.attnres_max_rows
        assert partial.shape == residual.shape == (num_tokens, self.hidden)
        m, s, acc = scratch
        assert m.dtype == s.dtype == acc.dtype == torch.float32
        kernels.iris_push_one_shot_allreduce_residual_attnres_gluon_kernel[(num_tokens,)](
            partial,
            residual,
            self.attnres_inbox,
            score_weight,
            output_weight,
            m,
            s,
            acc,
            hidden_out,
            residual_out,
            self.attnres_generations,
            self.attnres_flags,
            *self.attnres_peers,
            *self.heap_bases,
            RANK=self.rank,
            WORLD_SIZE=self.world_size,
            HIDDEN=self.hidden,
            BLOCK=triton.next_power_of_2(self.hidden),
            MAX_ELEMENTS=self.attnres_max_numel,
            READY_SLOT_STRIDE=self.attnres_max_rows * self.world_size,
            EPS=eps,
            ELEMENTS_PER_THREAD=ATTNRES_ELEMENTS_PER_THREAD,
            NUM_WARPS=ATTNRES_NUM_WARPS,
            SUBGROUP_SIZE=SUBGROUP_SIZE,
            num_warps=ATTNRES_NUM_WARPS,
        )
        return hidden_out, residual_out


# ---------------------------------------------------------------------------
# Inputs and torch references
# ---------------------------------------------------------------------------


def rank_input(seed, rank, shape, scale=1.0):
    """Deterministic per-rank input, so every rank can rebuild every peer's contribution."""
    generator = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    return (torch.randn(shape, generator=generator, device="cuda") * scale).to(torch.bfloat16)


def reference_sum(seed, world_size, shape, scale=1.0):
    """FP32 sum over ranks in rank order, rounded once to BF16 (the kernels' contract)."""
    total = torch.zeros(shape, dtype=torch.float32, device="cuda")
    for peer in range(world_size):
        total += rank_input(seed, peer, shape, scale).float()
    return total.to(torch.bfloat16)


def attnres_scratch(seed, num_tokens, hidden, score_weight, num_blocks=4, eps=K3_RMS_EPS):
    """Online-softmax partial over stored block snapshots, as K3's attnres_partial produces it."""
    generator = torch.Generator(device="cuda").manual_seed(seed * 7919 + 1)
    blocks = torch.randn((num_blocks, num_tokens, hidden), generator=generator, device="cuda")
    blocks = blocks.to(torch.bfloat16).float()
    logits = (blocks * score_weight.float()).sum(-1) * torch.rsqrt(blocks.pow(2).mean(-1) + eps)
    m = logits.max(0).values
    weights = torch.exp(logits - m)
    s = weights.sum(0)
    acc = (weights.unsqueeze(-1) * blocks).sum(0)
    return m.contiguous(), s.contiguous(), acc.contiguous()


def reference_attnres(reduced, residual, score_weight, output_weight, scratch, eps=K3_RMS_EPS):
    """Torch mirror of the fused epilogue, including its BF16 rounding points."""
    m, s, acc = scratch
    prefix = (reduced.float() + residual.float()).to(torch.bfloat16).float()
    logit = (prefix * score_weight.float()).sum(-1) * torch.rsqrt(prefix.pow(2).mean(-1) + eps)
    maximum = torch.maximum(m, logit)
    correction = torch.exp(m - maximum)
    prefix_weight = torch.exp(logit - maximum)
    inverse_sum = 1.0 / (s * correction + prefix_weight)
    mixed = (acc * correction[:, None] + prefix_weight[:, None] * prefix) * inverse_sum[:, None]
    mixed = mixed.to(torch.bfloat16).float()
    hidden = mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + eps) * output_weight.float()
    return hidden.to(torch.bfloat16), prefix.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Cases: each sets up K distinct calls so validation catches cross-call buffer reuse.
# ---------------------------------------------------------------------------


class AttentionCase:
    name = "attention two-stage"

    def __init__(self, op, num_tokens, seeds):
        self.op, self.m, self.seeds = op, num_tokens, seeds
        shape = (num_tokens, op.hidden)
        self.inputs = [rank_input(seed, op.rank, shape) for seed in seeds]
        self.outputs = [torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in seeds]
        self.payload_bytes = num_tokens * op.hidden * 2

    def run(self, i):
        self.op.attention_all_reduce(self.inputs[i], self.outputs[i])

    bench = run  # The staging copy is part of the real path, so it is timed.

    def check(self, i):
        expected = reference_sum(self.seeds[i], self.op.world_size, self.inputs[i].shape)
        return [("out", self.outputs[i], expected)]

    def poison(self):
        for out in self.outputs:
            out.fill_(float("nan"))

    def rccl(self, group):
        dist.all_reduce(self.inputs[0], group=group)


class MoeCase:
    def __init__(self, op, num_tokens, seeds):
        self.op, self.m, self.seeds = op, num_tokens, seeds
        self.name = f"moe {'lamport' if op.uses_lamport(num_tokens) else 'two-stage'}"
        self.routed_shape = (num_tokens, op.routed_hidden)
        self.shared_shape = (num_tokens, op.hidden)
        self.routed = [rank_input(seed, op.rank, self.routed_shape) for seed in seeds]
        self.shared = [rank_input(2 * seed + 1, op.rank, self.shared_shape) for seed in seeds]
        self.outputs = [
            (
                torch.empty(self.routed_shape, dtype=torch.bfloat16, device="cuda"),
                torch.empty(self.shared_shape, dtype=torch.bfloat16, device="cuda"),
            )
            for _ in seeds
        ]
        self.payload_bytes = num_tokens * op.row_numel * 2
        self._rccl_buf = torch.cat([self.routed[0].view(-1), self.shared[0].view(-1)])

    def run(self, i):
        # Stand-in for the expert GEMM epilogues writing straight into symmetric memory.
        routed_sym, shared_sym = self.op.moe_outputs(self.m)
        routed_sym.copy_(self.routed[i])
        shared_sym.copy_(self.shared[i])
        routed, shared = self.op.moe_all_reduce(self.m)
        self.outputs[i][0].copy_(routed)
        self.outputs[i][1].copy_(shared)

    def bench(self, i):
        # In K3 the expert GEMMs write the symmetric views directly; time only the reduce.
        self.op.moe_all_reduce(self.m)

    def check(self, i):
        ws = self.op.world_size
        return [
            ("routed", self.outputs[i][0], reference_sum(self.seeds[i], ws, self.routed_shape)),
            ("shared", self.outputs[i][1], reference_sum(2 * self.seeds[i] + 1, ws, self.shared_shape)),
        ]

    def poison(self):
        for routed, shared in self.outputs:
            routed.fill_(float("nan"))
            shared.fill_(float("nan"))

    def rccl(self, group):
        dist.all_reduce(self._rccl_buf, group=group)


class AttnResCase:
    name = "attention+residual+attnres"

    def __init__(self, op, num_tokens, seeds):
        self.op, self.m, self.seeds = op, num_tokens, seeds
        shape = (num_tokens, op.hidden)
        generator = torch.Generator(device="cuda").manual_seed(4242)
        # Weights and residual stream are replicated across TP ranks.
        self.score_weight = (torch.randn(op.hidden, generator=generator, device="cuda") * 0.05).to(torch.bfloat16)
        self.output_weight = (1 + 0.1 * torch.randn(op.hidden, generator=generator, device="cuda")).to(torch.bfloat16)
        self.partials = [rank_input(seed, op.rank, shape, scale=0.25) for seed in seeds]
        self.residuals = [rank_input(seed, 1000, shape) for seed in seeds]
        self.scratch = [attnres_scratch(seed, num_tokens, op.hidden, self.score_weight) for seed in seeds]
        self.hidden = [torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in seeds]
        self.residual_out = [torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in seeds]
        self.payload_bytes = num_tokens * op.hidden * 2

    def run(self, i):
        self.op.attention_residual_attnres(
            self.partials[i],
            self.residuals[i],
            self.score_weight,
            self.output_weight,
            self.scratch[i],
            self.hidden[i],
            self.residual_out[i],
        )

    bench = run

    def check(self, i):
        reduced = reference_sum(self.seeds[i], self.op.world_size, self.partials[i].shape, scale=0.25)
        hidden, residual_out = reference_attnres(
            reduced, self.residuals[i], self.score_weight, self.output_weight, self.scratch[i]
        )
        return [("hidden", self.hidden[i], hidden), ("residual_out", self.residual_out[i], residual_out)]

    def poison(self):
        for out in self.hidden + self.residual_out:
            out.fill_(float("nan"))

    def rccl(self, group):
        dist.all_reduce(self.partials[0], group=group)


def make_cases(op, kernels_to_run, tokens, seeds):
    cases = []
    for num_tokens in tokens:
        if "attention" in kernels_to_run:
            cases.append(AttentionCase(op, num_tokens, seeds))
        if "moe" in kernels_to_run:
            cases.append(MoeCase(op, num_tokens, seeds))
        if "attnres" in kernels_to_run and num_tokens <= op.attnres_max_rows:
            cases.append(AttnResCase(op, num_tokens, seeds))
    return cases


# ---------------------------------------------------------------------------
# Validation and timing
# ---------------------------------------------------------------------------


def _gather_device():
    # Host-side checks run under gloo when launched directly and under NCCL from the test runner.
    return "cuda" if dist.get_backend() == "nccl" else "cpu"


def validate(case, ctx, label):
    """Check every call against the torch reference and that all ranks agree bitwise."""
    torch.cuda.synchronize()
    failures = []
    for i in range(len(case.seeds)):
        for name, got, expected in case.check(i):
            diff = (got.float() - expected.float()).abs()
            tolerance = 1e-2 + 1e-2 * expected.float().abs()
            if not torch.isfinite(got).all() or (diff > tolerance).any():
                failures.append(f"call {i} {name}: max |diff| {diff.max().item():.3e}")
            # Every kernel here sums in one fixed order, so ranks must match exactly.
            local = got.view(torch.int16).to(torch.int32).to(_gather_device())  # gloo has no int16
            gathered = [torch.empty_like(local) for _ in range(ctx.get_num_ranks())]
            dist.all_gather(gathered, local)
            if any(not torch.equal(local, peer) for peer in gathered):
                failures.append(f"call {i} {name}: ranks disagree")
    status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    if ctx.get_rank() == 0:
        ctx.info(f"[{label}] {case.name:28s} M={case.m:3d}  {status}")
    return not failures


def capture(fn, calls):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i)
    return graph


def time_graph(graph, calls, reps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * calls)


def slowest_rank(value):
    # Collective time is set by the slowest rank; rank 0 alone is not representative.
    tensor = torch.tensor([value], dtype=torch.float64, device=_gather_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kimi-K3 tp8/ep1 fused all-reduce examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["attention", "moe", "attnres"],
        default=["attention", "moe", "attnres"],
        help="Which collectives to run",
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32], help="Token widths M (attnres skips M > 16)"
    )
    parser.add_argument("--calls", type=int, default=3, help="Distinct calls per round; odd counts catch slot reuse")
    parser.add_argument("--rounds", type=int, default=4, help="Eager rounds / graph replays to validate")
    parser.add_argument("-v", "--validate", action="store_true", help="Check against a torch reference")
    parser.add_argument("-g", "--graph", action="store_true", help="Also validate under CUDA graph replay")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Time graph replay per call")
    parser.add_argument("--rccl", action="store_true", help="Also time RCCL all_reduce on the same payload")
    parser.add_argument("--reps", type=int, default=200, help="Graph replays when benchmarking")
    parser.add_argument("--heap_size", type=int, default=1 << 30, help="Iris heap size")
    parser.add_argument("--seed", type=int, default=0)
    return vars(parser.parse_args())


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")

    ctx = iris.iris(heap_size=args["heap_size"])
    rank = ctx.get_rank()
    world_size = ctx.get_num_ranks()
    op = K3FusedAllReduce(ctx, max_tokens=max(args["tokens"]))
    seeds = [args["seed"] * 100 + i for i in range(args["calls"])]
    cases = make_cases(op, args["kernels"], args["tokens"], seeds)
    rccl_group = dist.new_group(backend="nccl") if args["rccl"] else None
    if rank == 0:
        ctx.info(f"world_size={world_size} hidden={op.hidden} routed_hidden={op.routed_hidden}")

    ok = True
    for case in cases:
        # Eager rounds also compile every kernel before any graph capture.
        for _ in range(args["rounds"]):
            for i in range(len(seeds)):
                case.run(i)
        if args["validate"]:
            ok &= validate(case, ctx, "eager")

        if args["graph"]:
            ctx.barrier()
            graph = capture(case.run, len(seeds))
            ctx.barrier()
            # Poison outputs so a replay that silently skips a write cannot pass.
            case.poison()
            for _ in range(args["rounds"]):
                graph.replay()
            if args["validate"]:
                ok &= validate(case, ctx, "graph")
            del graph

        if args["benchmark"]:
            ctx.barrier()
            graph = capture(case.bench, len(seeds))
            ctx.barrier()
            us = slowest_rank(time_graph(graph, len(seeds), args["reps"]))
            del graph
            line = f"[bench] {case.name:28s} M={case.m:3d}  {case.payload_bytes / 1024:7.1f} KiB  {us:7.2f} us/call"
            if rccl_group is not None:
                # RCCL builds its communicator lazily, which is not allowed during capture.
                case.rccl(rccl_group)
                ctx.barrier()
                rccl_graph = capture(lambda _: case.rccl(rccl_group), len(seeds))
                rccl_us = slowest_rank(time_graph(rccl_graph, len(seeds), args["reps"]))
                line += f"  | RCCL all_reduce {rccl_us:7.2f} us/call"
                del rccl_graph
            if rank == 0:
                ctx.info(line)

    ctx.barrier()
    if args["validate"] and rank == 0:
        ctx.info("All validations passed" if ok else "VALIDATION FAILED")
    if rccl_group is not None:
        # Tear the communicator down while every rank is still here; leaving it to the
        # global teardown hangs on ROCm.
        torch.cuda.synchronize()
        dist.destroy_process_group(rccl_group)
    dist.destroy_process_group()
    if args["validate"] and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
