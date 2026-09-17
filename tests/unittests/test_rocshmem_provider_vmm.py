# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""The rocSHMEM provider over a VMM-allocated symmetric heap.

rocSHMEM picks its symmetric-heap allocator at run time from
ROCSHMEM_HEAP_ALLOCATOR_TYPE. Most values obtain the heap with an ordinary
device allocation; ``vmm_posix`` and ``vmm_fabric`` obtain it through HIP's
virtual memory management -- hipMemCreate, hipMemAddressReserve, hipMemMap --
and share it between peers with POSIX file descriptors or fabric handles.

That matters here because the provider queries ``rocshmem_ptr`` once and caches
the per-peer byte offset, which is only sound while the whole heap is one
mapping. These tests pin that down on the VMM path.

``vmm_fabric`` is deliberately not covered: it needs a GPU with fabric handle
support (MI455/gfx1250). On others rocSHMEM reports
"Fabric handle type is not supported on device N" and aborts, so there is nothing
to skip on -- the process is already gone.

Run under the usual launcher:

    python tests/run_tests_distributed.py \
        tests/unittests/test_rocshmem_provider_vmm.py --num_ranks 2 -v

Note this file sets an environment variable that rocSHMEM reads at init, and
rocSHMEM initialises once per process. run_tests.sh gives every test file its own
torchrun, so this is isolated from the default-allocator tests next door; running
both in a single pytest process would silently give the second one whichever
allocator the first selected.
"""

import os

import pytest
import torch
import torch.distributed as dist

# Spread the allocations out rather than taking three small ones from the same
# corner of the heap. HIP's VMM granularity is typically 2 MiB, so 32 MiB apart
# puts them in different chunks -- if a future rocSHMEM ever mapped the heap
# incrementally instead of once, adjacent 4 KiB allocations would not notice and
# these would.
ALLOC_BYTES = 32 * 1024 * 1024
ALLOC_ELEMS = ALLOC_BYTES // 4  # float32
NALLOC = 3


def _rocm_version():
    """(major, minor) of the ROCm runtime, or None if it cannot be determined."""
    raw = getattr(torch.version, "hip", None)
    if not raw:
        try:
            raw = open("/opt/rocm/.info/version").read()
        except OSError:
            return None
    parts = raw.strip().split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


@pytest.fixture(scope="module")
def vmm_provider():
    if not dist.is_initialized():
        pytest.skip("needs torch.distributed; run via tests/run_tests_distributed.py")
    if dist.get_world_size() < 2:
        pytest.skip("needs at least 2 ranks (--num_ranks 2)")

    # Guard the version *before* setting the variable. rocSHMEM does not raise
    # when vmm_posix is unavailable, it calls LOG_ERROR_ABORT -- which takes the
    # whole pytest process with it, so there would be nothing left to skip.
    ver = _rocm_version()
    if ver is None:
        pytest.skip("cannot determine the ROCm version; vmm_posix needs 7.2+")
    if ver < (7, 2):
        pytest.skip(f"vmm_posix needs ROCm 7.2+, found {ver[0]}.{ver[1]}")

    rshmem = pytest.importorskip("rocshmem4py", reason="rocSHMEM provider tests need rocshmem4py installed")
    from iris.experimental.rocshmem_provider import RocshmemProvider

    # Read at init, so it has to be set before the call below. vmm_posix also
    # requires a TCP bootstrap, which init_rocshmem_by_uniqueid uses.
    os.environ["ROCSHMEM_HEAP_ALLOCATOR_TYPE"] = "vmm_posix"
    rshmem.init_rocshmem_by_uniqueid(dist.group.WORLD)
    return RocshmemProvider()


@pytest.fixture
def spread_allocations(vmm_provider):
    """Allocations far enough apart to sit in different VMM chunks."""
    tensors, tables = [], []
    for _ in range(NALLOC):
        t, bases = vmm_provider.allocate_symmetric(ALLOC_ELEMS, dtype=torch.float32)
        tensors.append(t)
        tables.append(bases)
    yield vmm_provider, tensors, tables
    vmm_provider.barrier()
    for t in tensors:
        vmm_provider.free(t)


def test_invariant_holds_per_allocation(spread_allocations):
    """peer_bases[local_rank] is each allocation's own base, as on any allocator."""
    provider, tensors, tables = spread_allocations
    me = provider.get_rank()

    for i, (t, bases) in enumerate(zip(tensors, tables)):
        assert bases.numel() == provider.get_num_ranks()
        assert int(bases[me].item()) == t.data_ptr(), f"allocation {i}"


def test_offsets_uniform_across_a_spread_heap(spread_allocations):
    """The property the provider's cached offsets depend on, over a wide span.

    rocSHMEM allocates the symmetric heap once and sub-allocates it with
    dlmalloc, so one mapping covers every allocation and the per-peer offset is
    constant. If that ever changes -- a heap mapped in pieces, or peer memory
    mapped per allocation the way torch symmetric memory does it -- the
    provider's cached deltas would silently produce wrong addresses, and this is
    the test that should fail.
    """
    provider, tensors, tables = spread_allocations
    ws = provider.get_num_ranks()

    spans = [t.data_ptr() for t in tensors]
    assert max(spans) - min(spans) >= ALLOC_BYTES, (
        "allocations landed closer together than expected; this test is only "
        "meaningful if they are in different VMM chunks"
    )

    direct = provider.symmetric_address_map(tensors[0]).direct
    reference = None
    for i, (t, bases) in enumerate(zip(tensors, tables)):
        offsets = tuple(int(bases[p].item()) - t.data_ptr() if direct[p] else None for p in range(ws))
        if reference is None:
            reference = offsets
        else:
            assert offsets == reference, (
                f"allocation {i} has per-peer offsets {offsets}, expected {reference}; "
                "the heap is no longer a single mapping and the provider's cached "
                "offsets are unsound"
            )


def test_peers_are_addressable_under_vmm(spread_allocations):
    """VMM sharing actually resolved, rather than falling back to no peers."""
    provider, tensors, _tables = spread_allocations
    amap = provider.symmetric_address_map(tensors[0])
    me, ws = provider.get_rank(), provider.get_num_ranks()

    assert amap.direct[me], "a rank must be able to reach itself"
    if ws > 1:
        assert any(amap.direct[p] for p in range(ws) if p != me), (
            "no peer was addressable under vmm_posix; the POSIX fd exchange did not resolve"
        )
