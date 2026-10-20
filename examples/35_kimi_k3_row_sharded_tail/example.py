#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Example: the row-sharded reduce-scatter + all-gather tails in Kimi-K3 at tp8/ep1.

For wide batches (prefill, and decode or EAGLE3 verify forwards with M >= 40 for the MoE tail and M >= 56 for the
attention tail) K3 does not all-reduce and then repeat the row-wise tail on all M rows on every rank. Each rank instead

1. pulls and reduces only its own M / 8 consecutive rows (reduce-scatter),
2. runs the tail on those rows, and
3. pushes the result rows to every peer (all-gather).

* MoE tail: reduce the routed ``[M, 3584]`` and shared ``[M, 7168]`` partials, RMSNorm and project the routed rows back
  to the hidden width, then add the shared rows and the residual while gathering. Only the ``[M, 7168]`` sum is gathered,
  one third less than gathering the ``[M, 10752]`` reduced pair.
* Attention tail: reduce the attention partial and add the residual (the residual shard stays local), then run the
  AttnRes mix and output RMSNorm on this rank's rows while gathering the normalized activation.

M is the token width of the forward and must be a multiple of 8. The kernels hard-code eight peers.

Run with:
    torchrun --nproc_per_node=8 --standalone example.py --validate --graph --benchmark
"""

import argparse
import gc
import importlib.util
import os
import sys

import torch
import torch.distributed as dist

import iris

# Load the sibling kernels.py by path so this works both as a script and when the test imports it.
_spec = importlib.util.spec_from_file_location(
    "k3_row_sharded_tail_kernels", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py")
)
kernels = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = kernels
_spec.loader.exec_module(kernels)


# Kimi-K3 shapes.
K3_HIDDEN = 7168
K3_ROUTED_HIDDEN = 3584  # Latent MoE: routed experts run at half width, projected back after the reduce.
K3_RMS_EPS = 1e-6
WORLD_SIZE = 8

# Launch parameters, as tuned for gfx950 in TokenSpeed.
REDUCE_PROGRAMS = 24
GATHER_PROGRAMS = 128
SEPARATE_MIX_GATHER_PROGRAMS = 32
BLOCK_ELEMENTS = 2048
NUM_WARPS = 4
MAX_FUSED_BLOCKS = 11  # The mix kernel supports up to 11 stored block snapshots.


class K3RowShardedTail:
    """Symmetric-heap workspaces and launchers for the two K3 row-sharded tails.

    The attention and MoE tails run one after the other inside a layer, so they share one producer input buffer, one
    scratch buffer and one gathered output buffer. Each gather finishes every peer's reads of the input before it
    returns, which is what makes the reuse safe.

    Args:
        ctx: Iris context; all eight ranks participate.
        max_rows: Largest token width M any call will use; a multiple of 8.
    """

    def __init__(self, ctx, max_rows, hidden=K3_HIDDEN, routed_hidden=K3_ROUTED_HIDDEN):
        self.ctx = ctx
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        if self.world_size != WORLD_SIZE:
            raise ValueError(f"the row-sharded kernels hard-code {WORLD_SIZE} peers, got {self.world_size} ranks")
        if max_rows <= 0 or max_rows % WORLD_SIZE:
            raise ValueError(f"max_rows must be a positive multiple of {WORLD_SIZE}, got {max_rows}")
        if (hidden, routed_hidden) != (K3_HIDDEN, K3_ROUTED_HIDDEN):
            raise ValueError("the kernels hard-code K3's hidden and routed widths")
        self.max_rows = max_rows
        self.hidden = hidden
        self.routed_hidden = routed_hidden
        self.row_numel = routed_hidden + hidden
        dtype = torch.bfloat16
        self.dtype = dtype

        # Producers write the routed rows and then the shared rows (or just the attention partial) here.
        self.input = ctx.zeros((max_rows * self.row_numel,), dtype=dtype)
        # This rank's reduced rows, routed then shared.
        self.scratch = ctx.zeros((max_rows // WORLD_SIZE * self.row_numel,), dtype=dtype)
        # Reduce-scatter and gather keep separate epochs: a shared counter would let one satisfy the other's barrier.
        self.reduce_flags = ctx.zeros((REDUCE_PROGRAMS, WORLD_SIZE), dtype=torch.int32)
        self.gather_flags = ctx.zeros((GATHER_PROGRAMS, WORLD_SIZE), dtype=torch.int32)
        # Every peer pushes its rows here, so each rank ends up with all M rows.
        self.output = ctx.zeros((max_rows, hidden), dtype=dtype)

        torch.cuda.synchronize()
        ctx.barrier()
        self.heap_bases = tuple(int(address) for address in ctx.get_heap_bases().tolist())

    # -- MoE tail ----------------------------------------------------------------------------

    def moe_inputs(self, rows):
        """Symmetric views the routed and shared expert GEMMs write into directly."""
        assert 0 < rows <= self.max_rows and rows % WORLD_SIZE == 0
        routed_numel = rows * self.routed_hidden
        routed = self.input[:routed_numel].view(rows, self.routed_hidden)
        shared = self.input[routed_numel : rows * self.row_numel].view(rows, self.hidden)
        return routed, shared

    def moe_tail(self, rows, prefix, projection_weight, norm_weight, eps, prefix_is_sharded):
        """Reduce, normalize and project this rank's rows, then gather ``prefix + projected + shared`` everywhere.

        Args:
            rows: Token width M.
            prefix: Residual stream, ``[M / 8, 7168]`` if ``prefix_is_sharded`` else ``[M, 7168]``. A sharded prefix
                is the shard the attention tail left on this rank.
            projection_weight: ``[7168, 3584]`` latent up-projection.
            norm_weight: ``[3584]`` routed RMSNorm weight.
            eps: RMSNorm epsilon.
            prefix_is_sharded: Whether ``prefix`` holds only this rank's rows.

        Returns:
            A view of the shared output buffer, ``[M, 7168]``. The next gather overwrites it.
        """
        local_rows = rows // WORLD_SIZE
        routed_numel = local_rows * self.routed_hidden
        routed = self.scratch[:routed_numel].view(local_rows, self.routed_hidden)
        shared = self.scratch[routed_numel : local_rows * self.row_numel].view(local_rows, self.hidden)
        output = self.output[:rows]

        kernels.iris_k3moe_pull_scatter[(REDUCE_PROGRAMS,)](
            self.input,
            self.scratch,
            self.reduce_flags,
            *self.heap_bases,
            RANK=self.rank,
            ROWS=rows,
            FIRST_WIDTH=self.routed_hidden,
            SECOND_WIDTH=self.hidden,
            BLOCK_ELEMENTS=BLOCK_ELEMENTS,
            NUM_PROGRAMS=REDUCE_PROGRAMS,
            NUM_WARPS=NUM_WARPS,
            num_warps=NUM_WARPS,
        )
        # The row-local tail runs on M / 8 rows instead of M.
        normalized = rmsnorm(routed, norm_weight, eps)
        projected = normalized @ projection_weight.t()
        kernels.iris_k3moe_push_gather[(GATHER_PROGRAMS,)](
            projected,
            shared,
            prefix,
            output,
            self.gather_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            BLOCK_ELEMENTS=BLOCK_ELEMENTS,
            NUM_PROGRAMS=GATHER_PROGRAMS,
            NUM_WARPS=NUM_WARPS,
            PREFIX_IS_SHARDED=prefix_is_sharded,
            num_warps=NUM_WARPS,
        )
        return output

    # -- attention tail ----------------------------------------------------------------------

    def attention_input(self, rows):
        """Symmetric view the attention output projection writes into directly."""
        assert 0 < rows <= self.max_rows and rows % WORLD_SIZE == 0
        return self.input[: rows * self.hidden].view(rows, self.hidden)

    @staticmethod
    def fuses_mix(rows, num_valid_blocks):
        """Whether the AttnRes mix runs inside the gather kernel.

        The separate mixer is faster in the middle token range, and longer histories need enough rows to amortize the
        live peer pointers the fused kernel holds.
        """
        return (rows < 1024 or rows >= 4096) and (num_valid_blocks <= 6 or (rows >= 7680 and num_valid_blocks <= 8))

    def mix(self, prefix, history, num_valid_blocks, res_weight, rms_weight, eps, out_norm_weight, out_norm_eps):
        """AttnRes mix and output RMSNorm of this rank's rows in their own launch, for histories too long to fuse."""
        rows = prefix.shape[0]
        output = torch.empty_like(prefix)
        warps = 4 if rows >= 256 or num_valid_blocks <= 1 else 8
        kernels.gluon_attn_res_fwd_gfx950[(rows,)](
            prefix,
            prefix,  # No delta is added here; the kernel ignores this operand.
            history,
            res_weight,
            rms_weight,
            out_norm_weight,
            output,
            stride_layer_t=prefix.stride(0),
            stride_delta_t=prefix.stride(0),
            stride_block_t=history.stride(1),
            stride_block_n=history.stride(0),
            stride_output_t=output.stride(0),
            H=self.hidden,
            N=num_valid_blocks + 1,
            BLOCK_WRITE_IDX=0,
            HAS_DELTA=False,
            WRITE_BLOCK=False,
            SCORE_EPS=eps,
            OUTPUT_EPS=out_norm_eps,
            NUM_WARPS=warps,
            num_warps=warps,
        )
        return output

    def attention_tail(
        self,
        rows,
        residual,
        block_residual,
        res_weight,
        rms_weight,
        out_norm_weight,
        eps,
        out_norm_eps,
        num_valid_blocks,
    ):
        """Reduce this rank's rows, add the residual, run the AttnRes mix, and gather the normalized rows.

        Args:
            rows: Token width M.
            residual: Replicated ``[M, 7168]`` residual stream, or None at block-write layers.
            block_residual: ``[blocks, M, 7168]`` stored block snapshots; the mix reads the first ``num_valid_blocks``.
            res_weight, rms_weight: ``[7168]`` AttnRes score weights. Their product is the scorer.
            out_norm_weight: ``[7168]`` output RMSNorm weight.
            eps, out_norm_eps: Score and output RMSNorm epsilons.
            num_valid_blocks: Stored snapshots the mix folds in, 0 to 8.

        Returns:
            ``(prefix, output)``: this rank's ``[M / 8, 7168]`` residual rows, which stay local, and a view of the
            shared output buffer ``[M, 7168]`` with the normalized mix of every row.
        """
        assert 0 <= num_valid_blocks <= min(MAX_FUSED_BLOCKS, block_residual.shape[0])
        local_rows = rows // WORLD_SIZE
        partial = self.attention_input(rows)
        prefix = torch.empty((local_rows, self.hidden), dtype=self.dtype, device=partial.device)
        output = self.output[:rows]

        kernels.iris_k3attn_pull_scatter[(REDUCE_PROGRAMS,)](
            partial,
            residual,
            prefix,
            self.reduce_flags,
            *self.heap_bases,
            RANK=self.rank,
            LOCAL_ROWS=local_rows,
            BLOCK_ELEMENTS=BLOCK_ELEMENTS,
            NUM_PROGRAMS=REDUCE_PROGRAMS,
            NUM_WARPS=NUM_WARPS,
            HAS_RESIDUAL=residual is not None,
            num_warps=NUM_WARPS,
        )
        if self.fuses_mix(rows, num_valid_blocks):
            warps = 8 if num_valid_blocks <= 7 else 4
            kernels.iris_k3attn_mix_push_gather[(GATHER_PROGRAMS,)](
                prefix,
                output,
                block_residual,
                res_weight,
                rms_weight,
                out_norm_weight,
                self.gather_flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=local_rows,
                STRIDE_BLOCK_T=block_residual.stride(1),
                STRIDE_BLOCK_N=block_residual.stride(0),
                NUM_VALID_BLOCKS=num_valid_blocks,
                SCORE_EPS=eps,
                OUTPUT_EPS=out_norm_eps,
                NUM_PROGRAMS=GATHER_PROGRAMS,
                NUM_WARPS=warps,
                num_warps=warps,
            )
        else:
            first_row = self.rank * local_rows
            history = block_residual[:, first_row : first_row + local_rows]
            mixed = self.mix(
                prefix, history, num_valid_blocks, res_weight, rms_weight, eps, out_norm_weight, out_norm_eps
            )
            kernels.iris_k3attn_push_gather[(SEPARATE_MIX_GATHER_PROGRAMS,)](
                mixed,
                output,
                self.gather_flags,
                *self.heap_bases,
                RANK=self.rank,
                LOCAL_ROWS=local_rows,
                BLOCK_ELEMENTS=BLOCK_ELEMENTS,
                NUM_PROGRAMS=SEPARATE_MIX_GATHER_PROGRAMS,
                NUM_WARPS=NUM_WARPS,
                num_warps=NUM_WARPS,
            )
        return prefix, output


# ---------------------------------------------------------------------------
# Torch tails and references. The kernels round to BF16 at fixed points; these mirror them.
# ---------------------------------------------------------------------------


def rmsnorm(x, weight, eps):
    """RMSNorm in FP32 with one BF16 rounding at the end."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16)


def attnres_mix(prefix, history, num_valid_blocks, res_weight, rms_weight, eps, out_norm_weight, out_norm_eps):
    """AttnRes: softmax-weighted mix of the prefix and stored block snapshots, then output RMSNorm.

    Each candidate is scored by ``dot(candidate, res_weight * rms_weight) * rsqrt(mean(candidate^2) + eps)``. The
    mixture is rounded to BF16 before the output norm.
    """
    candidates = torch.stack([prefix.float()] + [history[j].float() for j in range(num_valid_blocks)])
    scorer = res_weight.float() * rms_weight.float()
    scores = (candidates * scorer).sum(-1) * torch.rsqrt(candidates.pow(2).mean(-1) + eps)
    weights = torch.softmax(scores, dim=0)
    mixed = (weights.unsqueeze(-1) * candidates).sum(0).to(torch.bfloat16)
    return rmsnorm(mixed, out_norm_weight, out_norm_eps)


def rank_input(seed, rank, shape, scale=1.0):
    """Deterministic per-rank input, so every rank can rebuild every peer's contribution."""
    generator = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    return (torch.randn(shape, generator=generator, device="cuda") * scale).to(torch.bfloat16)


def reference_sum(seed, world_size, shape, scale=1.0):
    """FP32 sum over ranks in the kernels' fixed tree order, rounded once to BF16.

    The kernels add ``((r0 + r2) + (r4 + r6)) + ((r1 + r3) + (r5 + r7))``. A different order can round a few
    elements one BF16 step apart, and the norm and projection that follow can amplify that.
    """
    assert world_size == WORLD_SIZE
    r = [rank_input(seed, peer, shape, scale).float() for peer in range(world_size)]
    return (((r[0] + r[2]) + (r[4] + r[6])) + ((r[1] + r[3]) + (r[5] + r[7]))).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Cases: each sets up K distinct calls so validation catches cross-call buffer reuse.
# Each check item is (name, got, expected, replicated); a sharded result differs by rank on purpose.
# ---------------------------------------------------------------------------


class MoeTailCase:
    def __init__(self, op, rows, seeds, prefix_is_sharded):
        self.op, self.m, self.seeds = op, rows, seeds
        self.prefix_is_sharded = prefix_is_sharded
        self.name = f"moe tail ({'sharded' if prefix_is_sharded else 'replicated'} prefix)"
        local_rows = rows // WORLD_SIZE
        generator = torch.Generator(device="cuda").manual_seed(31337)
        # Weights are replicated across TP ranks.
        self.projection_weight = (
            torch.randn((op.hidden, op.routed_hidden), generator=generator, device="cuda") * 0.02
        ).to(torch.bfloat16)
        self.norm_weight = (1 + 0.1 * torch.randn(op.routed_hidden, generator=generator, device="cuda")).to(
            torch.bfloat16
        )
        self.routed = [rank_input(seed, op.rank, (rows, op.routed_hidden)) for seed in seeds]
        self.shared = [rank_input(2 * seed + 1, op.rank, (rows, op.hidden)) for seed in seeds]
        if prefix_is_sharded:
            self.prefix = [rank_input(3 * seed + 2, op.rank, (local_rows, op.hidden)) for seed in seeds]
        else:
            self.prefix = [rank_input(3 * seed + 2, 1000, (rows, op.hidden)) for seed in seeds]
        self.outputs = [torch.empty((rows, op.hidden), dtype=torch.bfloat16, device="cuda") for _ in seeds]
        self.payload_bytes = rows * op.row_numel * 2

    def run(self, i):
        # Stand-in for the expert GEMM epilogues writing straight into symmetric memory.
        routed_sym, shared_sym = self.op.moe_inputs(self.m)
        routed_sym.copy_(self.routed[i])
        shared_sym.copy_(self.shared[i])
        out = self.bench(i)
        self.outputs[i].copy_(out)

    def bench(self, i):
        # In K3 the expert GEMMs write the symmetric views directly; time only the tail.
        return self.op.moe_tail(
            self.m, self.prefix[i], self.projection_weight, self.norm_weight, K3_RMS_EPS, self.prefix_is_sharded
        )

    def check(self, i):
        ws = self.op.world_size
        routed = reference_sum(self.seeds[i], ws, (self.m, self.op.routed_hidden))
        shared = reference_sum(2 * self.seeds[i] + 1, ws, (self.m, self.op.hidden))
        # Normalize and project in M / 8-row shards, as the ranks do: torch can split a reduction or a GEMM differently
        # for a different row count, which shifts FP32 rounding and can flip a BF16 result.
        local_rows = self.m // ws
        projected = torch.cat(
            [
                rmsnorm(routed[q * local_rows : (q + 1) * local_rows], self.norm_weight, K3_RMS_EPS)
                @ self.projection_weight.t()
                for q in range(ws)
            ]
        )
        if self.prefix_is_sharded:
            # Rank q's shard holds rows q * M/8 ... (q + 1) * M/8 - 1.
            prefix = torch.cat(
                [rank_input(3 * self.seeds[i] + 2, q, (self.m // ws, self.op.hidden)) for q in range(ws)]
            )
        else:
            prefix = self.prefix[i]
        expected = (prefix.float() + projected.float() + shared.float()).to(torch.bfloat16)
        return [("out", self.outputs[i], expected, True)]

    def poison(self):
        for out in self.outputs:
            out.fill_(float("nan"))


class AttentionTailCase:
    def __init__(self, op, rows, seeds, num_valid_blocks, has_residual):
        self.op, self.m, self.seeds = op, rows, seeds
        self.num_valid_blocks, self.has_residual = num_valid_blocks, has_residual
        mix = "fused mix" if op.fuses_mix(rows, num_valid_blocks) else "separate mix"
        self.name = f"attention tail ({num_valid_blocks} blocks, {mix}{'' if has_residual else ', no residual'})"
        shape = (rows, op.hidden)
        generator = torch.Generator(device="cuda").manual_seed(4242)
        # Weights and block snapshots are replicated across TP ranks.
        self.res_weight = (torch.randn(op.hidden, generator=generator, device="cuda") * 0.05).to(torch.bfloat16)
        self.rms_weight = (1 + 0.1 * torch.randn(op.hidden, generator=generator, device="cuda")).to(torch.bfloat16)
        self.out_norm_weight = (1 + 0.1 * torch.randn(op.hidden, generator=generator, device="cuda")).to(torch.bfloat16)
        self.block_residual = torch.randn(
            (max(num_valid_blocks, 1), rows, op.hidden), generator=generator, device="cuda"
        ).to(torch.bfloat16)
        self.partials = [rank_input(seed, op.rank, shape, scale=0.25) for seed in seeds]
        self.residuals = [rank_input(seed, 1000, shape) if has_residual else None for seed in seeds]
        self.hidden = [torch.empty(shape, dtype=torch.bfloat16, device="cuda") for _ in seeds]
        self.prefix = [torch.empty((rows // WORLD_SIZE, op.hidden), dtype=torch.bfloat16, device="cuda") for _ in seeds]
        self.payload_bytes = rows * op.hidden * 2

    def run(self, i):
        self.op.attention_input(self.m).copy_(self.partials[i])
        prefix, hidden = self.bench(i)
        self.prefix[i].copy_(prefix)
        self.hidden[i].copy_(hidden)

    def bench(self, i):
        return self.op.attention_tail(
            self.m,
            self.residuals[i],
            self.block_residual,
            self.res_weight,
            self.rms_weight,
            self.out_norm_weight,
            K3_RMS_EPS,
            K3_RMS_EPS,
            self.num_valid_blocks,
        )

    def _expected_prefix(self, i):
        reduced = reference_sum(self.seeds[i], self.op.world_size, (self.m, self.op.hidden), scale=0.25)
        if self.residuals[i] is None:
            return reduced
        return (reduced.float() + self.residuals[i].float()).to(torch.bfloat16)

    def check(self, i):
        prefix = self._expected_prefix(i)
        mixed = attnres_mix(
            prefix,
            self.block_residual,
            self.num_valid_blocks,
            self.res_weight,
            self.rms_weight,
            K3_RMS_EPS,
            self.out_norm_weight,
            K3_RMS_EPS,
        )
        local_rows = self.m // WORLD_SIZE
        shard = prefix[self.op.rank * local_rows : (self.op.rank + 1) * local_rows]
        return [("hidden", self.hidden[i], mixed, True), ("prefix shard", self.prefix[i], shard, False)]

    def poison(self):
        for out in self.hidden + self.prefix:
            out.fill_(float("nan"))


# (num_valid_blocks, has_residual): a fused mix, a block-write layer with no residual, and a history long enough to
# use the separate mixer at any width below 4096.
ATTENTION_VARIANTS = [(4, True), (0, False), (8, True)]


def make_cases(op, kernels_to_run, tokens, seeds):
    cases = []
    for rows in tokens:
        if "moe" in kernels_to_run:
            cases.append(MoeTailCase(op, rows, seeds, prefix_is_sharded=True))
            cases.append(MoeTailCase(op, rows, seeds, prefix_is_sharded=False))
        if "attention" in kernels_to_run:
            for num_valid_blocks, has_residual in ATTENTION_VARIANTS:
                cases.append(AttentionTailCase(op, rows, seeds, num_valid_blocks, has_residual))
    return cases


# ---------------------------------------------------------------------------
# Validation and timing
# ---------------------------------------------------------------------------


def _gather_device():
    # Host-side checks run under gloo when launched directly and under NCCL from the test runner.
    return "cuda" if dist.get_backend() == "nccl" else "cpu"


def validate(case, ctx, label):
    """Check every call against the torch reference and that ranks agree bitwise on replicated results."""
    torch.cuda.synchronize()
    failures = []
    for i in range(len(case.seeds)):
        for name, got, expected, replicated in case.check(i):
            diff = (got.float() - expected.float()).abs()
            tolerance = 1e-2 + 1e-2 * expected.float().abs()
            if not torch.isfinite(got).all() or (diff > tolerance).any():
                failures.append(f"call {i} {name}: max |diff| {diff.max().item():.3e}")
            if not replicated:
                continue
            # Every kernel here sums in one fixed order, so every rank must hold the same bits.
            local = got.view(torch.int16).to(torch.int32).to(_gather_device())  # gloo has no int16
            gathered = [torch.empty_like(local) for _ in range(ctx.get_num_ranks())]
            dist.all_gather(gathered, local)
            if any(not torch.equal(local, peer) for peer in gathered):
                failures.append(f"call {i} {name}: ranks disagree")
    status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    if ctx.get_rank() == 0:
        ctx.info(f"[{label}] {case.name:52s} M={case.m:5d}  {status}")
    return not failures


def release_graph_memory():
    """Return a deleted graph's private memory pool before the next case captures its own."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


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
        description="Kimi-K3 tp8/ep1 row-sharded reduce-scatter + all-gather tails",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kernels", nargs="+", choices=["attention", "moe"], default=["attention", "moe"], help="Which tails to run"
    )
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[40, 56, 64, 128, 512, 2048], help="Token widths M (multiples of 8)"
    )
    parser.add_argument("--calls", type=int, default=3, help="Distinct calls per round; odd counts catch slot reuse")
    parser.add_argument("--rounds", type=int, default=4, help="Eager rounds / graph replays to validate")
    parser.add_argument("-v", "--validate", action="store_true", help="Check against a torch reference")
    parser.add_argument("-g", "--graph", action="store_true", help="Also validate under CUDA graph replay")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Time graph replay per call")
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
    op = K3RowShardedTail(ctx, max_rows=max(args["tokens"]))
    seeds = [args["seed"] * 100 + i for i in range(args["calls"])]
    cases = make_cases(op, args["kernels"], args["tokens"], seeds)
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
            release_graph_memory()

        if args["benchmark"]:
            ctx.barrier()
            graph = capture(case.bench, len(seeds))
            ctx.barrier()
            us = slowest_rank(time_graph(graph, len(seeds), args["reps"]))
            del graph
            release_graph_memory()
            line = f"[bench] {case.name:52s} M={case.m:5d}  {case.payload_bytes / 1024:8.1f} KiB  {us:8.2f} us/call"
            if rank == 0:
                ctx.info(line)

    ctx.barrier()
    if args["validate"] and rank == 0:
        ctx.info("All validations passed" if ok else "VALIDATION FAILED")
    dist.destroy_process_group()
    if args["validate"] and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
