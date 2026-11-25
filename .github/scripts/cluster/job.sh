#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# The batch script launch.sh submits. Arguments are passed to node.sh
# (--image-only, or --run <command...>), which runs once per node as a job step.

cd "${IRIS_CLUSTER_CHECKOUT:?set by launch.sh}" || exit 1
RESULTS=${IRIS_CLUSTER_RESULTS:?set by launch.sh}

# First, every node checks that its GPUs are free; if one is busy, stop here
# rather than leave the other nodes waiting for it. launch.sh reads which.
srun --ntasks-per-node=1 --label bash .github/scripts/cluster/node.sh --preflight
# srun's exit status has not reflected a failed task (it returned 0 after one
# exited 75), so go by the result files the nodes write instead.
if grep -qs '^busy' "$RESULTS"/node*.rc; then
    exit 75
fi
# Then every node gets the image, as its own step: a node loading the saved
# image can take minutes longer than one that already has it, and on multiple
# nodes the first one ready would otherwise start alone and wait for the rest.
if [ "${1:-}" != "--image-only" ]; then
    srun --ntasks-per-node=1 --label bash .github/scripts/cluster/node.sh --image-only
    if [ "$(cat "$RESULTS"/node*.rc 2>/dev/null | grep -c '^0$')" != "${SLURM_NNODES:?}" ]; then
        echo "[job] the image did not reach every node"
        exit 1
    fi
    rm -f "$RESULTS"/node*.rc
fi
srun --ntasks-per-node=1 --label bash .github/scripts/cluster/node.sh "$@"
