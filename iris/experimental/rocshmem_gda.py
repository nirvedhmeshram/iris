# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Inter-node transport for rocSHMEM-allocated memory, from Triton kernels.

Companion to ``rocshmem_provider``, which stops exactly where this starts. That
module builds a peer-base table from ``rocshmem_ptr()``, so Iris kernels can
load and store rocSHMEM memory directly -- but only for peers reachable that
way, which means same node. It reports the rest through
``SymmetricAddressMap.direct`` and leaves them to a transport it does not
provide. This is that transport.

The two are meant to be used together, split by peer::

    tensor, amap = provider.allocate_symmetric_map(n, dtype=torch.float32)
    qps = resolve_queue_pairs(provider.get_num_ranks())

    # amap.direct[peer]  -> iris.store / iris.load against amap.peer_bases
    # otherwise          -> gda_put_nbi over qps.for_peer(peer)

Sending a same-node peer over the QueuePair would work, but takes the fabric
where a direct store would do. The split is not an optimisation detail: this
shim enters rocSHMEM below the layer that would have made that choice, so
nothing downstream will correct a bad routing decision. Note also that a
mis-routed peer still produces *correct data*, so a value check cannot detect
it -- assert on the routing.

Addresses differ between the two paths. iris.store takes an already-translated
peer address; ``gda_put_nbi`` takes symmetric addresses in the CALLER's own
address space and translates internally, so pass local pointers to it.

How it works: rocSHMEM's QueuePair device code is header-only as of
ROCm/rocm-systems#7217, so we compile a thin shim against those headers
(``csrc/rocshmem_gda/``) and link the resulting bitcode into the Triton module.
Nothing from rocSHMEM's own device bitcode is linked, and there is no
``rocshmem_hipmodule_init`` step, because the shim references no device globals
-- see the header comment in ``iris_gda_shim.hip`` for the conditions that rests
on.

Three things about the launch are load-bearing. Each one fails as HIP 209 "no
kernel image is available for execution on the device" -- which reads like an
architecture mismatch, not a linking problem:

  1. ``extern_libs`` must be passed at the LAUNCH, not only inside
     ``extern_elementwise``. The AMD backend links from ``options.extern_libs``.
  2. The dict KEY must be a prefix of the symbols called: ``iris_gda`` matches
     ``iris_gda_*``. A mismatched key silently drops the library.
  3. Triton's compile cache keys on the extern_libs PATH, not the bitcode's
     contents, so rebuilding in place leaves a stale hsaco. Set
     ``TRITON_CACHE_DIR`` per run while iterating on the shim.

Like ``rocshmem_provider``, this module is deliberately not imported by
``iris/experimental/__init__.py``: importing it must not make rocshmem4py or a
built shim mandatory for every Iris user.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

import rocshmem4py as rshmem

# Built by csrc/rocshmem_gda/build.sh. Not linked here -- callers pass these in
# extern_libs at launch (see module docstring).
_DEFAULT_CSRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "csrc", "rocshmem_gda"
)

SHIM_BITCODE_PATH = os.environ.get("IRIS_GDA_BC", os.path.join(_DEFAULT_CSRC, "iris_gda_shim_gfx950.bc"))

# Separate object: the only one that includes GDAContext. Keeping it out of the
# hot-path shim is what holds that shim's include set down to the QueuePair
# headers.
PROBE_BITCODE_PATH = os.environ.get("IRIS_GDA_PROBE_BC", os.path.join(_DEFAULT_CSRC, "iris_gda_ctx_probe_gfx950.bc"))

# Also the extern_libs key callers must use: it has to prefix the symbol names.
LIB_NAME = "iris_gda"

# Triton cannot type a void result -- extern_elementwise rejects it -- so the
# shim returns int32 and callers discard it. The callee never writes a return
# register, which is fine precisely because nothing reads it.
_VOID = tl.int32


def _extern(fn_name: str, arg_types: list, bitcode: str = None):
    """Declare one extern "C" shim function to Triton."""
    path = bitcode or SHIM_BITCODE_PATH

    @tl.core.extern
    def _wrapper(*args, _semantic=None):
        return tl.core.extern_elementwise(
            LIB_NAME,
            path,
            list(args),
            {tuple(arg_types): (fn_name, _VOID)},
            is_pure=False,
            _semantic=_semantic,
        )

    return _wrapper


gda_put_nbi = _extern("iris_gda_put_nbi", [tl.int64, tl.int64, tl.int64, tl.int64, tl.int32])
"""Non-blocking put of nbytes from source to dest on PE `pe`, over QueuePair `qp`.

   Args are (qp, dest, source, nbytes, pe). Addresses are symmetric addresses in
   THIS PE's space -- unlike iris.store, do not pre-translate them. `pe` is
   needed alongside `qp` because it feeds the QueuePair's own accounting. Follow
   with gda_quiet() to guarantee completion.
"""

gda_get_nbi = _extern("iris_gda_get_nbi", [tl.int64, tl.int64, tl.int64, tl.int64, tl.int32])
"""Non-blocking get of nbytes from source on PE `pe` into local dest."""

gda_quiet = _extern("iris_gda_quiet", [tl.int64, tl.int32])
"""Complete outstanding operations on QueuePair `qp`.

   Per-QP, not global: a kernel that put to several peers must call this for
   each of them.
"""

gda_probe_ctx = _extern("iris_gda_probe_ctx", [tl.int64, tl.int64], bitcode=PROBE_BITCODE_PATH)
"""Setup-time: write (qps base, sizeof(QueuePair)) to out[0..1]. See resolve_queue_pairs."""


@triton.jit
def _probe_kernel(ctx, out):
    gda_probe_ctx(ctx, out)


@dataclass(frozen=True)
class QueuePairMap:
    """Where each peer's rocSHMEM QueuePair lives.

    Held as a base and a stride rather than a materialised per-peer table: it
    would be the same arithmetic on the same two numbers for every peer.
    """

    qps_base: int
    stride: int  # sizeof(rocshmem::QueuePair)
    num_ranks: int

    def for_peer(self, pe: int) -> int:
        """Address of `pe`'s QueuePair, to pass to gda_put_nbi / gda_quiet.

        Correct while there is one QP per PE, which is the default
        (NUM_QPS_PER_PE_DEFAULT_CTX=1). The general mapping lives in
        GDAContext::get_qp_index, which is private and so unreachable from here.
        """
        if not 0 <= pe < self.num_ranks:
            raise IndexError(f"peer {pe} out of range for {self.num_ranks} ranks")
        return self.qps_base + pe * self.stride


def resolve_queue_pairs(num_ranks: int, device: str = "cuda") -> QueuePairMap:
    """Resolve the current rocSHMEM context to its QueuePair array. Call once.

    Runs a one-thread kernel because the answer lives in device-resident
    rocSHMEM internals. The stride comes back from the same device code rather
    than being assumed here, so it cannot disagree with the shim's view of
    sizeof(QueuePair).

    rocSHMEM must already be initialised with the GDA backend active.
    """
    for label, path in (("shim", SHIM_BITCODE_PATH), ("ctx-probe", PROBE_BITCODE_PATH)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} bitcode not found at {path}; build it with csrc/rocshmem_gda/build.sh")

    ctx = rshmem.rocshmem_get_device_ctx()
    if not ctx:
        raise RuntimeError("rocshmem_get_device_ctx() returned 0 -- is rocSHMEM initialised with the GDA backend?")

    out = torch.zeros(2, dtype=torch.int64, device=device)
    _probe_kernel[(1,)](ctx, out.data_ptr(), num_warps=1, extern_libs={LIB_NAME: PROBE_BITCODE_PATH})
    torch.cuda.synchronize()

    qps_base = out[0].item() & 0xFFFFFFFFFFFFFFFF
    stride = out[1].item()
    if not qps_base:
        raise RuntimeError(
            "GDAContext::qps was null. Either rocSHMEM is not using the GDA "
            "backend, or the shim and librocshmem came from different rocSHMEM "
            "source trees and qps was read at the wrong offset -- these are "
            "internal headers and nothing checks that at load time."
        )
    if stride <= 0:
        raise RuntimeError(f"sizeof(QueuePair) came back as {stride}")

    return QueuePairMap(qps_base=qps_base, stride=stride, num_ranks=num_ranks)
