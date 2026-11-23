#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Multi-node CI entry point, run on the GitHub runner's host. The runner does
# no GPU work: it submits a Slurm batch job of IRIS_MN_NODES nodes (job.sh),
# which runs node.sh on each of them, in the checkout; the checkout must be on
# a filesystem the compute nodes share with this host. It then follows the
# job's log until the job ends, and exits 0 only if the job completed with exit
# code 0.
#
# Submitted with sbatch and polled, rather than run under a foreground srun or
# `sbatch --wait`: a scheduler client held open for the whole run has been
# seen to be cut off after a few minutes, which ends the job as a timeout.
#
# Usage: launch.sh
#
# Scheduler settings come from the environment, so the workflow can take them
# from repository variables rather than naming a site's accounts:
#   IRIS_MN_ACCOUNT    Slurm account (-A)                    required
#   IRIS_MN_QOS        Slurm QoS (--qos)                     required
#   IRIS_MN_PARTITION  Slurm partition (-p)                  default: the cluster's default
#   IRIS_MN_GRES       GPUs per node (--gres)                default: gpu:8
#   IRIS_MN_NODES      nodes                                 default: 2
#   IRIS_MN_TIME       job time limit                        default: 0:45:00
#   IRIS_MN_CACHE      shared dir for the saved image (see node.sh)
#   IRIS_MN_ENV        the site's RCCL settings, as space-separated KEY=VALUE
#                      pairs (NCCL_*/RCCL_* only), e.g. which NICs and RoCE GID
#                      index to use and the topology file. They reach every rank.
#
# If the image is not saved yet, it is first built in a 1-node job (time limit
# IRIS_MN_BUILD_TIME, default 1:30:00): the build compiles Triton, and should
# not hold every node while it does.
#
# A cancelled workflow signals this script, which cancels the job; node.sh's
# container goes with it.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export IRIS_MN_CHECKOUT=$PWD

: "${IRIS_MN_ACCOUNT:?set IRIS_MN_ACCOUNT (Slurm account)}"
: "${IRIS_MN_QOS:?set IRIS_MN_QOS (Slurm QoS)}"
NODES=${IRIS_MN_NODES:-2}
for kv in ${IRIS_MN_ENV:-}; do
    if [[ ! "$kv" =~ ^(NCCL|RCCL)_[A-Z0-9_]+=[^[:space:]]*$ ]]; then
        echo "[launch] IRIS_MN_ENV: '$kv' is not an NCCL_*/RCCL_* KEY=VALUE pair" >&2
        exit 1
    fi
    export "${kv?}"
done
LOGS="$PWD/.multinode-logs"
mkdir -p "$LOGS"

args=(
    --ntasks-per-node=1
    -A "$IRIS_MN_ACCOUNT" --qos="$IRIS_MN_QOS"
    --gres="${IRIS_MN_GRES:-gpu:8}"
    -c "${IRIS_MN_CPUS:-64}"
    -J "iris-ci-${GITHUB_RUN_ID:-manual}"
)
if [ -n "${IRIS_MN_PARTITION:-}" ]; then
    args+=(-p "$IRIS_MN_PARTITION")
fi

JOB=""
cancel() {
    if [ -n "$JOB" ]; then
        echo "[launch] cancelling job $JOB"
        scancel "$JOB" || true
    fi
    exit 143
}
trap cancel TERM INT

# job_state <id>: the job's state from the accounting record, or empty.
job_state() {
    sacct -j "$1" -n -o State 2>/dev/null | awk 'NF { print $1; exit }'
}

# run_job <nodes> <time> [node.sh args]: submit job.sh, follow its log, and
# return 0 only if every node reported exit code 0, 2 if some node's GPUs
# were already in use (those nodes are added to EXCLUDE), else 1. Each node
# writes its own result file, since the scheduler has reported a job whose
# steps failed as COMPLETED with exit code 0.
run_job() {
    local nodes=$1 time=$2; shift 2
    local log tailer state
    export IRIS_MN_RESULTS
    IRIS_MN_RESULTS=$(mktemp -d "$LOGS/results.XXXXXX")
    local exclude=()
    if [ -n "$EXCLUDE" ]; then
        exclude=(-x "$EXCLUDE")
    fi
    JOB=$(sbatch --parsable -N "$nodes" -t "$time" "${args[@]}" "${exclude[@]}" \
        -o "$LOGS/job-%j.log" .github/scripts/multinode/job.sh "$@")
    JOB=${JOB%%;*}
    log="$LOGS/job-$JOB.log"
    echo "[launch] job $JOB: $nodes node(s), limit $time, log $log"
    touch "$log"
    tail -n +1 -F "$log" 2>/dev/null &
    tailer=$!
    while :; do
        state=$(job_state "$JOB")
        case "$state" in
            ""|PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED) sleep 20 ;;
            *) break ;;
        esac
    done
    sleep 5  # let the last lines reach the log
    kill "$tailer" 2>/dev/null || true
    local n rc busy=0 failed=0 result=0
    for ((n = 0; n < nodes; n++)); do
        rc=$(cat "$IRIS_MN_RESULTS/node$n.rc" 2>/dev/null || echo "no result")
        echo "[launch] job $JOB node $n: $rc"
        case "$rc" in
            0) ;;
            busy\ *) EXCLUDE="${EXCLUDE:+$EXCLUDE,}${rc#busy }"; busy=1 ;;
            *) failed=1 ;;
        esac
    done
    # A busy node stops the whole job early, so the others report nothing.
    # A cancelled job is not retried: someone wanted it stopped.
    if [ "$state" = CANCELLED ]; then
        result=1
    elif [ "$busy" = 1 ]; then
        result=2
    elif [ "$failed" = 1 ]; then
        result=1
    fi
    echo "[launch] job $JOB ended: $state"
    rm -rf "$IRIS_MN_RESULTS"
    JOB=""
    return "$result"
}

# run_with_retries <nodes> <time> [node.sh args]: run_job, resubmitting
# without busy nodes up to IRIS_MN_RETRIES (default 2) times.
EXCLUDE=""
run_with_retries() {
    local attempt rc
    for ((attempt = 0; ; attempt++)); do
        rc=0
        run_job "$@" || rc=$?
        if [ "$rc" != 2 ] || [ "$attempt" -ge "${IRIS_MN_RETRIES:-2}" ]; then
            return "$rc"
        fi
        echo "[launch] resubmitting without busy nodes: $EXCLUDE"
    done
}

# shellcheck source=image.sh
source .github/scripts/multinode/image.sh
if [ ! -f "$IMAGE_SAVED" ]; then
    echo "[launch] no saved $IMAGE; building it on one node first"
    run_with_retries 1 "${IRIS_MN_BUILD_TIME:-1:30:00}" --image-only
fi

# One port per job; in_container.sh offsets it per torchrun.
export IRIS_MN_PORT=$((29500 + RANDOM % 2000))

echo "[launch] checkout $PWD at $(git rev-parse --short HEAD)"
run_with_retries "$NODES" "${IRIS_MN_TIME:-0:45:00}"
