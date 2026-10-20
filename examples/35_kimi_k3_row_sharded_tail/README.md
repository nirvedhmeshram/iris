<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Kimi-K3 row-sharded reduce-scatter + all-gather tails

These are the Iris collectives that Kimi-K3 runs with 8-way tensor parallelism and no expert parallelism (tp8/ep1) on MI355X when the batch is wide: prefill, and decode or EAGLE3 verify forwards with enough tokens. They are written in Gluon and adapted from [TokenSpeed](https://github.com/lightseekorg/tokenspeed). The kernel names are unchanged, so each one here matches its entry in a K3 profile.

They complement [example 34](../34_kimi_k3_fused_all_reduce/README.md), which covers the small-batch all-reduces. Example 34 reduces everything and then lets every rank repeat the row-wise work on all `M` rows. Here each rank owns `M / 8` consecutive rows instead:

1. **Pull reduce-scatter.** Each rank reads its own rows from all eight peers' symmetric input and reduces them in FP32 in a fixed order.
2. **Rank-local tail** on those `M / 8` rows.
3. **Push all-gather.** Each rank writes its finished rows into every peer's output buffer and waits until all peers have done the same.

`M` is the token width of the forward pass and must be a multiple of 8. K3 takes this path for `M >= 40` on the MoE tail and `M >= 56` on the attention tail. For EAGLE3 verify forwards `M` is 4 × concurrency, so concurrency 16 gives `M = 64`.

| K3 call site | Rank-local work on `M / 8` rows | Kernels |
|---|---|---|
| MoE tail: routed `[M, 3584]` and shared `[M, 7168]` partials | routed RMSNorm and up-projection to 7168 | `iris_k3moe_pull_scatter`, `iris_k3moe_push_gather` |
| Attention tail: `[M, 7168]` partial | residual add, AttnRes mix, output RMSNorm | `iris_k3attn_pull_scatter`, then `iris_k3attn_mix_push_gather` or `iris_k3attn_push_gather` |

How each tail works:

- **MoE.** The routed and shared partials are consecutive in one symmetric buffer, so one pull kernel reduces both. The routed rows are normalized and projected back to the hidden width on this rank only, which removes 87.5% of that work per GPU. The gather kernel then adds the projected routed rows, the shared rows and the residual and pushes the `[M, 7168]` sum. Gathering the sum instead of the reduced `[M, 10752]` pair cuts the all-gather payload by one third.
  - The residual is either a `[M / 8, 7168]` shard, which is what the attention tail leaves on each rank, or a replicated `[M, 7168]` tensor.
- **Attention.** The pull kernel reduces this rank's rows and adds the residual. The residual shard stays local until the MoE tail consumes it. The normalized AttnRes mix is what gets gathered.
  - `iris_k3attn_mix_push_gather` runs the mix inside the gather kernel. It is used when `M < 1024` or `M >= 4096`, and the history has at most 6 stored blocks (at most 8 once `M >= 7680`).
  - Otherwise `gluon_attn_res_fwd_gfx950` runs the mix in its own launch and `iris_k3attn_push_gather` pushes its output.

The attention and MoE tails run one after the other in a layer, so they share one input buffer, one scratch buffer and one output buffer. Each gather finishes every peer's reads of the input before it returns, which is what makes the reuse safe, including under CUDA-graph replay.

Every kernel sums in FP32 in a fixed order, so all ranks hold bitwise-identical gathered results. The example checks this.

## Run

```terminal
torchrun --standalone --nproc_per_node=8 examples/35_kimi_k3_row_sharded_tail/example.py --validate --graph --benchmark
```

| Option | Default | Effect |
|---|---|---|
| `--kernels` | `attention moe` | Which tails to run. |
| `--tokens` | `40 56 64 128 512 2048` | Token widths to run; multiples of 8. |
| `--validate` | off | Checks every call against a PyTorch reference, and checks that all ranks agree bitwise on gathered results. |
| `--graph` | off | Repeats the validation under CUDA graph replay, with outputs poisoned to NaN before each replay. |
| `--benchmark` | off | Reports µs per call for the slowest rank. The tail compute (norm, projection, mix) is included. |
| `--calls` | `3` | Distinct inputs per round. An odd count catches buffer reuse across calls. |

The cases cover both residual layouts for the MoE tail, and three attention variants: a fused mix with four stored blocks, a block-write layer with no residual and no history, and an eight-block history that forces the separate mixer below `M = 4096`.

The kernels need gfx950 because they use CDNA4 buffer intrinsics, and they hard-code eight peers, so unlike example 34 they run only at 8 ranks.

## What the example leaves out

- The projection and RMSNorm run as plain torch ops. K3 uses its own fused kernels for them.
- In the model the expert GEMMs write directly into the symmetric input buffer. The example copies into it, and the benchmark times only the tail.
- There is no RCCL comparison. Example 34 times RCCL against its all-reduces, but the row-sharded tails replace an all-reduce plus rank-local work, so a like-for-like baseline needs K3's own unsharded tail kernels.

## Results

Measured on 8 × MI355X (K3's tp8 group) with graph replay, slowest rank, Triton 3.7.1. Each time covers the whole tail: the pull reduce-scatter, the rank-local compute and the push all-gather. The payload is the producer input, `[M, 10752]` for the MoE tail and `[M, 7168]` for the attention tail.

| Tail | M | Payload | µs per call |
|---|---:|---:|---:|
| MoE | 40 | 840 KiB | 49.3 |
| | 64 | 1.3 MiB | 50.5 |
| | 128 | 2.6 MiB | 56.7 |
| | 512 | 10.5 MiB | 95.2 |
| | 2048 | 42 MiB | 231.0 |
| Attention, 4 blocks, fused mix | 56 | 784 KiB | 22.1 |
| | 64 | 896 KiB | 22.7 |
| | 128 | 1.8 MiB | 25.9 |
| | 512 | 7 MiB | 52.1 |
| Attention, no residual, fused mix | 56 | 784 KiB | 17.2 |
| | 512 | 7 MiB | 47.0 |
| Attention, 8 blocks, separate mix | 56 | 784 KiB | 26.1 |
| | 512 | 7 MiB | 56.0 |
| Attention, 4 blocks, separate mix | 2048 | 28 MiB | 152.0 |

The MoE tail includes the torch RMSNorm and up-projection on `M / 8` rows, which K3 replaces with fused kernels, so treat the MoE numbers as an upper bound on K3's tail. The two MoE residual layouts time the same. These come from back-to-back calls with nothing in between; in a model the time is often dominated by waiting for the slowest rank to arrive.

## Test

```terminal
python tests/run_tests_distributed.py tests/examples/test_kimi_k3_row_sharded_tail.py --num_ranks 8 -v
```

The test covers every case at `M` in {40, 64, 128, 512}, eagerly and under graph replay. It is skipped unless there are exactly 8 ranks on gfx950.
