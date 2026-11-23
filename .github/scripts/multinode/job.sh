#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# The batch script launch.sh submits: one node.sh per node, as a job step.
# Arguments are passed to node.sh (e.g. --image-only).

cd "${IRIS_MN_CHECKOUT:?set by launch.sh}" || exit 1
# First, every node checks that its GPUs are free; if one is busy, stop here
# rather than leave the other nodes waiting for it. launch.sh reads which.
srun --ntasks-per-node=1 --label bash .github/scripts/multinode/node.sh --preflight
# srun's exit status has not reflected a failed task (it returned 0 after one
# exited 75), so go by the result files the nodes write instead.
if grep -qs '^busy' "${IRIS_MN_RESULTS:?set by launch.sh}"/node*.rc; then
    exit 75
fi
# Then every node gets the image, as its own step: a node loading the saved
# image can take minutes longer than one that already has it, and a node that
# started the tests alone would wait out each test's timeout for its peers.
if [ "${1:-}" != "--image-only" ]; then
    srun --ntasks-per-node=1 --label bash .github/scripts/multinode/node.sh --image-only
    if [ "$(cat "$IRIS_MN_RESULTS"/node*.rc 2>/dev/null | grep -c '^0$')" != "${SLURM_NNODES:?}" ]; then
        echo "[job] the image did not reach every node"
        exit 1
    fi
    rm -f "$IRIS_MN_RESULTS"/node*.rc
fi
srun --ntasks-per-node=1 --label bash .github/scripts/multinode/node.sh "$@"
