#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Inside the CI container on every node (started by node.sh): one torchrun per
# node, ranks node-major (0-7 on the first node), for each of
#   - tests/unittests/test_rccl_gin_provider.py, with IRIS_REQUIRE_GIN=1 so the
#     GIN tests fail instead of skipping when GIN is missing: running them is
#     the point of this job;
#   - examples/35_wide_ep_rccl_gin/gin_smoke.py, GIN to every rank;
#   - examples/35_wide_ep_rccl_gin/example_run.py, a checked MoE layer.
# Every step runs under a timeout, so a hang fails the job instead of holding
# the nodes until the job's time limit. All steps run; the exit code is
# nonzero if any failed.

set -uo pipefail

NODE=${IRIS_MN_NODE:?} NNODES=${IRIS_MN_NNODES:?}
GPUS=${IRIS_MN_GPUS_PER_NODE:-8}
STEP_TIMEOUT=${IRIS_MN_STEP_TIMEOUT:-600}
log() { echo "[node $NODE] $*"; }

mkdir -p "$HOME"

# Iris is pure Python, so the checkout goes on PYTHONPATH rather than being
# pip-installed: the checkout is mounted read-only (node.sh), and both nodes
# share it. Only the dependencies the image lacks are installed, into this
# container.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
pip install --quiet --no-warn-script-location --root-user-action=ignore \
    numpy requests "tritonblas @ git+https://github.com/ROCm/tritonBLAS.git@muhaawad/iris" pytest \
    || { log "dependency install failed"; exit 1; }

# The GIN IB-proxy backend; RCCL offers its device API only with cuMem.
export NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-2} NCCL_CUMEM_ENABLE=1 NCCL_DMABUF_ENABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} PYTHONFAULTHANDLER=1
# The checkout is read-only here, so every cache lives in the container.
export TRITON_CACHE_DIR=/tmp/triton-cache IRIS_RCCL_GIN_CACHE=/tmp/rccl-gin PYTHONDONTWRITEBYTECODE=1
export IRIS_REQUIRE_GIN=1

log "$(hostname -s): $NNODES nodes x $GPUS GPUs, master $MASTER_ADDR, $(python3 -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__)')"
ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ' | sed "s/^/[node $NODE] RDMA devices: /"; echo

failed=()
step=0
run() {
    local name=$1; shift
    local port=$((IRIS_MN_PORT + step))
    step=$((step + 1))
    log "===== $name"
    timeout --kill-after=30 "$STEP_TIMEOUT" \
        torchrun --nnodes="$NNODES" --node_rank="$NODE" --nproc_per_node="$GPUS" \
            --master_addr="$MASTER_ADDR" --master_port="$port" "$@"
    local rc=$?
    log "===== $name rc=$rc$([ $rc = 124 ] && echo ' (timed out)')"
    [ $rc = 0 ] || failed+=("$name")
}

run "unit tests" tests/run_tests_distributed.py tests/unittests/test_rccl_gin_provider.py -v --tb=short -rs -p no:cacheprovider
run "GIN smoke" examples/35_wide_ep_rccl_gin/gin_smoke.py
run "wide-EP MoE" examples/35_wide_ep_rccl_gin/example_run.py --bench_iters 0

if [ ${#failed[@]} -gt 0 ]; then
    log "FAILED: ${failed[*]}"
    exit 1
fi
log "all passed"
