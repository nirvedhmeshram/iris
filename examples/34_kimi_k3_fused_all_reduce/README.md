<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Kimi-K3 fused all-reduce kernels

These are the Iris all-reduces that Kimi-K3 runs during decode with 8-way tensor parallelism and no expert parallelism (tp8/ep1) on MI355X. They are written in Gluon and adapted from [TokenSpeed](https://github.com/lightseekorg/tokenspeed). The kernel names are unchanged, so each one here matches its entry in a K3 profile.

Each K3 decoder layer issues two collectives per rank. `M` is the token width of the forward pass. For plain decode that is the concurrency; for an EAGLE3 verify forward it is 4 × concurrency, so concurrency 8 gives `M = 32`.

| K3 call site | Payload | Kernel | Used when |
|---|---|---|---|
| Attention output | `[M, 7168]` | `iris_reduce_symmetric_two_stage_gluon_kernel` | default |
| Attention output, fused with residual add, AttnRes combine and RMSNorm | `[M, 7168]` | `iris_push_one_shot_allreduce_residual_attnres_gluon_kernel` | `M <= 16` and the model enables the fusion |
| MoE output: routed latent `[M, 3584]` and shared `[M, 7168]`, reduced in one launch | `[M, 10752]` | `lamport_all_reduce_bf16` | `M <= 6` |
| | | `iris_reduce_symmetric_two_stage_gluon_kernel` | `M > 6` |

How each kernel works:

- **Two-stage.** Each rank reduces its own 1/world-size partition of every peer's symmetric input into symmetric scratch. It then gathers every rank's reduced partition. Traffic per rank is 2 × the payload, compared with world-size × for a one-shot reduce.
  - For attention, the input is copied into a single staging buffer before launch. Reusing that one buffer is safe under CUDA-graph replay only because the kernel holds an exit barrier (`EXIT_BARRIER=True`).
  - For MoE, the expert kernels write their outputs directly into symmetric memory (the "producer-direct" pattern). There is no staging copy and no exit barrier.
- **Lamport.** Each rank pushes 16-byte packs into every peer's mailbox. The mailbox starts out holding a `-0.0` sentinel, so a pack counts as arrived once none of its halves is `-0.0`. Real `-0.0` values are sent as `+0.0`. The data acts as its own ready flag, which removes a signalling round trip. Three mailbox generations rotate so that a fast rank cannot overwrite data a slow peer has not yet read.
- **Fused attention + AttnRes.** The kernel pushes each row to every peer and swaps one token with each peer to signal arrival. It then sums the inbox in rank order and runs the epilogue in the same kernel:
  - residual add;
  - an AttnRes softmax over the stored block snapshots. The `(m, s, acc)` partial is computed earlier and overlapped with attention;
  - the output RMSNorm.

Every kernel sums in FP32 in a fixed order, so all ranks get bitwise-identical results. The example checks this.

## Run

```terminal
torchrun --standalone --nproc_per_node=8 examples/34_kimi_k3_fused_all_reduce/example.py --validate --graph --benchmark --rccl
```

| Option | Default | Effect |
|---|---|---|
| `--kernels` | `attention moe attnres` | Which collectives to run. |
| `--tokens` | `1 4 8 16 32` | Token widths to run. The fused AttnRes kernel skips `M > 16`. |
| `--validate` | off | Checks every call against a PyTorch reference, and checks that all ranks agree bitwise. |
| `--graph` | off | Repeats the validation under CUDA graph replay, with outputs poisoned to NaN before each replay. |
| `--benchmark` | off | Reports µs per call for the slowest rank. |
| `--rccl` | off | Also times `torch.distributed.all_reduce` on the same payload. |
| `--calls` | `3` | Distinct inputs per round. An odd count catches buffer-slot reuse across calls. |

The kernels need gfx950 because they use CDNA4 buffer intrinsics. They support 2, 4 or 8 ranks and have been validated at all three. K3 itself runs at 8; the smaller world sizes exist so the examples can run in CI.

## Results

Measured on 8 × MI355X (K3's tp8 group) with graph replay, slowest rank, Triton 3.7.1:

| Kernel | M | Payload | Iris µs | RCCL `all_reduce` µs |
|---|---:|---:|---:|---:|
| attention two-stage (incl. staging copy) | 1 | 14 KiB | 14.5 | 20.3 |
| | 8 | 112 KiB | 15.5 | 18.1 |
| | 32 | 448 KiB | 17.0 | 25.0 |
| attention + residual + AttnRes | 1 | 14 KiB | 13.3 | 20.6 |
| | 8 | 112 KiB | 14.3 | 18.2 |
| | 16 | 224 KiB | 16.6 | 22.6 |
| MoE Lamport | 1 | 21 KiB | 7.4 | 21.0 |
| | 6 | 126 KiB | 10.1 | 18.9 |
| MoE two-stage | 8 | 168 KiB | 11.4 | 21.8 |
| | 32 | 672 KiB | 12.6 | 27.7 |

The RCCL column covers only the all-reduce. The fused AttnRes kernel also does the epilogue, which would otherwise be a separate kernel.

These numbers come from back-to-back calls with nothing in between. In a model, a collective's time is often dominated by waiting for the slowest rank to arrive, so treat them as the cost of the transfer, not as end-to-end decode savings.

## Test

```terminal
python tests/run_tests_distributed.py tests/examples/test_kimi_k3_fused_all_reduce.py --num_ranks 2 -v
```

The test covers every kernel at `M` in {1, 4, 8, 16}, eagerly and under graph replay. It is skipped on hardware other than gfx950.
