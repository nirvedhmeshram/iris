# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Manual test: put to an inter-node peer over rocSHMEM's GDA QueuePair.

Covers the half of rocshmem_provider that the peer-base table cannot reach. The
provider translates addresses for peers reachable by direct load/store; for the
peers it reports as indirect, this drives the transfer over the NIC through the
shim in csrc/rocshmem_gda/.

Requires 2 nodes -- with every rank on one node there is no indirect peer and
the shim is never exercised, which the test asserts rather than passing
vacuously.

Prerequisites:
  * rocSHMEM built with USE_GDA=ON and a concrete GDA provider (not GDA_MUX),
    and USE_IPC=ON so the provider can classify same-node peers at all.
  * The shim built from that SAME rocSHMEM source tree:
      ROCSHMEM_SRC_TREE=... BUILD_INC=... bash csrc/rocshmem_gda/build.sh

Run:
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<r> \
      --master_addr=<n0-ip> --master_port=<p> tests/manual_rocshmem_gda.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import rocshmem4py as rshmem

from iris.experimental.rocshmem_gda import LIB_NAME, SHIM_BITCODE_PATH, gda_put_nbi, gda_quiet, resolve_queue_pairs
from iris.experimental.rocshmem_provider import RocshmemProvider

MAGIC = 0x0BADCAFE
NELEM = 256

# Triton keys its compile cache on the extern_libs path, not the bitcode's
# contents, so a rebuilt shim at the same path would reuse a stale hsaco.
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-cache-{os.environ.get('SLURM_JOB_ID', 'local')}")


@triton.jit
def put_and_quiet_kernel(qp, dest_va, src_va, nbytes, peer):
    """One symmetric-address put through rocSHMEM's own QueuePair code."""
    gda_put_nbi(qp, dest_va, src_va, nbytes.to(tl.int64), peer)
    gda_quiet(qp, peer)


def main():
    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rshmem.init_rocshmem_by_uniqueid(dist.group.WORLD)

    provider = RocshmemProvider()
    me, n = provider.get_rank(), provider.get_num_ranks()

    # Both ranks allocate in the same order, so these are symmetric.
    src, amap = provider.allocate_symmetric_map(NELEM, dtype=torch.int32)
    dst, _ = provider.allocate_symmetric_map(NELEM, dtype=torch.int32)
    src.fill_(MAGIC if me == 0 else 0)
    dst.fill_(-1)
    torch.cuda.synchronize()
    provider.barrier()

    indirect = amap.indirect_peers()
    print(f"[pe{me}] direct={amap.direct} indirect_peers={indirect}", flush=True)

    # The point of running on 2 nodes. On a single node every peer is direct,
    # the shim is never called, and a pass would mean nothing.
    assert indirect, (
        f"[pe{me}] no indirect peer -- every rank is on one node, so this test "
        "cannot exercise the GDA shim. Run it across 2 nodes."
    )

    qps = resolve_queue_pairs(n)
    print(f"[pe{me}] qps_base={qps.qps_base:#x} sizeof(QueuePair)={qps.stride}", flush=True)

    # Rank 0 sends to its first indirect peer. Broadcast which one, so the
    # receiver knows to check and the result can name it -- rather than assuming
    # a 2-rank layout where "the peer" is implicit.
    chosen = [indirect[0] if me == 0 else None]
    dist.broadcast_object_list(chosen, src=0)
    peer = chosen[0]

    if me == 0:
        nbytes = NELEM * 4
        print(f"[pe0] put {nbytes}B -> pe{peer} over the GDA QueuePair", flush=True)
        # extern_libs must be at the launch, and the key must prefix the called
        # symbols; omitting either surfaces as HIP 209, not a link error.
        #
        # dst.data_ptr() is OUR address, not the peer's: put_nbi takes symmetric
        # addresses and translates internally. This is where it differs from
        # iris.store, which wants an already-translated peer pointer.
        put_and_quiet_kernel[(1,)](
            qps.for_peer(peer),
            dst.data_ptr(),
            src.data_ptr(),
            nbytes,
            peer,
            num_warps=1,
            extern_libs={LIB_NAME: SHIM_BITCODE_PATH},
        )
        torch.cuda.synchronize()
        print("[pe0] kernel returned without fault", flush=True)

    ok = None
    if me == peer:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            torch.cuda.synchronize()
            if bool((dst == MAGIC).all().item()):
                ok = True
                break
            time.sleep(0.25)
        ok = bool(ok)
        got = [hex(v & 0xFFFFFFFF) for v in dst[:4].tolist()]
        print(f"[pe{me}] dst first4={got} want={hex(MAGIC)} all_match={ok}", flush=True)

    res = [None] * n
    dist.all_gather_object(res, ok)
    if me == 0:
        # Only the receiving rank's verdict counts; the others never looked.
        print(f"ROCSHMEM_GDA_RESULT(pe{peer}):", "PASS" if res[peer] else "FAIL", flush=True)

    provider.barrier()
    provider.free(src)
    provider.free(dst)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
