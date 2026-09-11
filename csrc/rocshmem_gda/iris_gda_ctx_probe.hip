/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Setup-time probe: resolve a rocSHMEM context to its QueuePair array.
 *
 * Deliberately a separate translation unit from iris_gda_shim.hip. The shim is
 * the hot path and includes only the QueuePair headers; this file is the one
 * place that includes context_gda_device.hpp, which drags in
 * context.hpp -> host/host.hpp -> rocshmem.hpp -> mpi.h.
 *
 * Splitting them confines the part upstream considers a hack to one function
 * called once per process. Reaching into GDAContext::qps is how a consumer gets
 * a QueuePair today; the author of the refactor calls that "kind of a hack" and
 * intends a supported accessor as a follow-up. When it lands, this file is all
 * that changes -- the shim and everything above it are untouched.
 *
 * Writes into out[]:
 *   out[0] = GDAContext::qps             -- base of the QueuePair array
 *   out[1] = sizeof(rocshmem::QueuePair) -- stride for indexing it
 *
 * The caller derives each peer's QueuePair as qps + pe * stride. That holds
 * while there is one QP per PE, which is the default
 * (NUM_QPS_PER_PE_DEFAULT_CTX=1). The general mapping is
 * GDAContext::get_qp_index(pe, wf_info), which is private and so unavailable to
 * a consumer -- the same gap an accessor would close.
 *
 * Returning the stride from device code rather than hardcoding it host-side is
 * the point: sizeof(QueuePair) is fixed by the headers this was compiled
 * against, so it cannot drift from the shim's view of the same type.
 *
 * out[0] also discriminates the failure this path is prone to. If the shim and
 * librocshmem came from different rocSHMEM trees, the module still loads and the
 * context pointer still looks plausible, but qps is read at the wrong offset and
 * the first transfer faults at 0x0. A healthy value sits a fixed offset above
 * ctx_opaque and is identical across ranks; 0 means either a genuinely null
 * array or a tree mismatch.
 */
#include <cstdint>

#include "rocshmem/rocshmem_config.h"
#include "log.hpp"
#include "gda/context_gda_device.hpp"
#include "gda/queue_pair_provider.hpp"
#include "gda/queue_pair/queue_pair_common.hpp"

using rocshmem::GDAContext;
using rocshmem::QueuePair;

extern "C" {

__device__ int32_t iris_gda_probe_ctx(int64_t ctx, int64_t out) {
  GDAContext *gda = reinterpret_cast<GDAContext *>(ctx);
  int64_t *o = reinterpret_cast<int64_t *>(out);
  o[0] = reinterpret_cast<int64_t>(gda->qps);
  o[1] = static_cast<int64_t>(sizeof(QueuePair));
  return 0;
}

}  // extern "C"
