/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * extern "C" __device__ entry points onto rocSHMEM's GDA QueuePair, callable
 * from a Triton kernel.
 *
 * This is the inter-node transport that rocshmem_provider.py does not provide.
 * That module gives Iris kernels a peer-base table built from rocshmem_ptr(),
 * which works only for peers reachable by direct load/store -- i.e. same node.
 * For the peers it reports as indirect, this shim issues the transfer over the
 * NIC instead.
 *
 * Why a shim rather than rocSHMEM's own device API: the public entry points
 * (rocshmem_putmem_nbi and friends) are compiled into rocSHMEM's device
 * bitcode, so calling them means linking that bitcode into the JIT'd module and
 * initialising its device globals with rocshmem_hipmodule_init. Since the
 * QueuePair refactor (ROCm/rocm-systems#7217) the QueuePair device code is
 * header-only, so entering one layer lower -- at QueuePairSHMEM::put_nbi --
 * needs no prebuilt bitcode and no initialisation step at all. We compile these
 * headers ourselves into the small object this file produces.
 *
 * These entry points take a QueuePair*, not a rocSHMEM context. That keeps this
 * translation unit's include set down to the QueuePair headers, with none of the
 * context.hpp -> host.hpp -> rocshmem.hpp -> mpi.h chain that including
 * GDAContext would drag in. Resolving a context to its per-peer QueuePair is
 * done once at setup, in iris_gda_ctx_probe.hip.
 *
 * This module needs no device globals of its own, which is what removes the
 * rocshmem_hipmodule_init step. That is a claim about THIS module, not about
 * rocSHMEM, and it holds for four specific reasons, each of which can stop
 * holding:
 *
 *   1. The QueuePair arrives as a kernel argument, so nothing reads
 *      ROCSHMEM_CTX_DEFAULT.
 *   2. The provider is fixed at compile time (see the GDA_MUX guard below), so
 *      nothing reads constmem.gda_provider to dispatch.
 *   3. We never enter the Context layer, so we miss both its dispatch on
 *      constmem.backend_type and GDAContext::putmem's IPC check, which reads
 *      constmem.my_pe.
 *   4. log.hpp is replaced (replacements/log.hpp), so the LOGD_* macros never
 *      pull in the extern __constant__ logd_constants.
 *
 * __constant__ globals are per-hipModule: a JIT'd kernel is loaded as its own
 * module, so a global that rocshmem_init() initialised inside rocSHMEM's own
 * fatbinary is neither visible nor initialised here, however we link. Any global
 * this module referenced would have to be defined and populated locally.
 *
 * Signatures are int64-flavoured because Triton's extern_elementwise types its
 * arguments by value and cannot express void*; callers pass device addresses as
 * plain integers. Returns are int32 rather than void for the same reason --
 * extern_elementwise cannot type a void result -- and the value is meaningless.
 *
 * HARD REQUIREMENT: this must be compiled from the SAME rocSHMEM source tree
 * that built the librocshmem it runs against. These are internal headers, so
 * they fix sizeof(QueuePair) and every field offset, and nothing checks it at
 * load time: a mismatched pair links, loads, and then faults at address 0x0 on
 * first use. iris_gda_probe_ctx exists partly to tell that case apart from a
 * genuinely null QueuePair array.
 */
#include <cstdint>

#include "rocshmem/rocshmem_config.h"
#include "log.hpp"
#include "gda/queue_pair_provider.hpp"
#include "gda/queue_pair/queue_pair_common.hpp"

// The provider is resolved at COMPILE time: queue_pair_provider.hpp aliases
// rocshmem::QueuePair to a concrete type based on the GDA_* define in
// rocshmem_config.h, so there is no runtime dispatch and nothing reads constmem.
//
// GDA_MUX is the exception. QueuePairMux::get_provider() returns
// constmem.gda_provider, a __constant__ device global defined in rocSHMEM's own
// translation unit and therefore absent from a JIT'd consumer module: it would
// either fail to link (surfacing as HIP 209 "no kernel image is available",
// which reads like an architecture mismatch) or resolve to a zeroed copy and
// dispatch on a bogus tag -- the switch's default arm is __builtin_unreachable,
// so that is undefined behaviour rather than a clean failure.
//
// It is not only the missing global: QueuePairUnion is a union of the provider
// QueuePairs, so a mux build also changes sizeof(QueuePair) and the caller's
// stride arithmetic would be wrong regardless. The tag cannot be sidestepped by
// calling the concrete member either -- QueuePairUnion is private.
//
// Supporting mux is possible but not free: define constmem here and populate
// gda_provider into the JIT'd module from the host via hipModuleGetGlobal. That
// reintroduces a per-module host init step, which is the thing this path exists
// to avoid, so we reject it instead. Revisit if a mux build ever matters.
#if defined(GDA_MUX)
#error "The Iris GDA shim does not support the multiplexing GDA provider: \
QueuePairMux dispatches on the constmem __constant__ device global, which is \
not initialised in a JIT'd consumer module. Build rocSHMEM with a concrete \
provider (e.g. GDA_IONIC)."
#endif

using rocshmem::ActiveWFInfo;
using rocshmem::QueuePair;

namespace {

__device__ inline QueuePair *as_qp(int64_t qp) { return reinterpret_cast<QueuePair *>(qp); }

}  // namespace

extern "C" {

/*
 * Non-blocking put of nbytes from `source` to `dest` on PE `pe`, over the
 * QueuePair at `qp` -- which the caller resolved for that peer (see
 * QueuePairMap.for_peer).
 *
 * Both addresses are symmetric-heap addresses in THIS PE's address space;
 * put_nbi translates the destination for the peer, so callers on the rocSHMEM
 * symmetric heap need no registration or address arithmetic of their own. Note
 * that differs from iris.store, which needs an already-translated peer address.
 *
 * `pe` is still required even though the QueuePair is peer-specific: it goes
 * into ActiveWFInfo, which the QueuePair uses for its own accounting.
 *
 * This always goes over the NIC. It is meant for the peers
 * SymmetricAddressMap.direct reports as unreachable; sending a same-node peer
 * here would work but would take the fabric instead of a direct store.
 */
__device__ int32_t iris_gda_put_nbi(int64_t qp, int64_t dest, int64_t source, int64_t nbytes, int32_t pe) {
  ActiveWFInfo wf_info{static_cast<int>(pe)};
  as_qp(qp)->put_nbi(reinterpret_cast<void *>(dest), reinterpret_cast<const void *>(source),
                     static_cast<size_t>(nbytes), wf_info);
  return 0;
}

/* Non-blocking get: `dest` local, `source` on PE `pe`; both symmetric. */
__device__ int32_t iris_gda_get_nbi(int64_t qp, int64_t dest, int64_t source, int64_t nbytes, int32_t pe) {
  ActiveWFInfo wf_info{static_cast<int>(pe)};
  as_qp(qp)->get_nbi(reinterpret_cast<void *>(dest), reinterpret_cast<const void *>(source),
                     static_cast<size_t>(nbytes), wf_info);
  return 0;
}

/*
 * Complete outstanding operations on this QueuePair.
 *
 * Per-QP, not global: quieting one peer's queue says nothing about any other's,
 * so a kernel that put to several peers must call this for each of them.
 */
__device__ int32_t iris_gda_quiet(int64_t qp, int32_t pe) {
  ActiveWFInfo wf_info{static_cast<int>(pe)};
  as_qp(qp)->quiet(wf_info);
  return 0;
}

}  // extern "C"
