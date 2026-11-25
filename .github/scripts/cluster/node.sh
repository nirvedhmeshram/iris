#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs on every node of a cluster CI job (job.sh starts it with srun, in the
# shared checkout).
#
#   node.sh --preflight      check that the node's GPUs are free, and stop
#   node.sh --image-only     make sure the CI image is on this node, and stop
#   node.sh --run <command...>  both, then run <command> in one container with all
#                               of the node's GPUs and RDMA NICs
#
# The container runs as the invoking user, with the checkout mounted read-write
# at its own path (the working directory) and at /iris_workspace, where
# container_exec.sh-based scripts expect it. IRIS_IN_CONTAINER=1 tells those
# scripts that they are already inside the CI image, so they run their command
# directly. For multi-node commands, MASTER_ADDR, MASTER_PORT,
# IRIS_CLUSTER_NODE and IRIS_CLUSTER_NNODES describe the job.
#
# The image (docker/Dockerfile.cluster, i.e. apptainer/iris.def plus RDMA
# userspace) is tagged by the checksum of its inputs. Images are node-local and
# building one compiles Triton, so the first node to need a new image builds it
# and saves it to IRIS_CLUSTER_CACHE, a directory every node shares; the
# others, then and later, load that copy. A lock on the cache keeps two nodes
# from building at once.

set -euo pipefail

NODE=${SLURM_NODEID:?run under srun (see job.sh)}
NNODES=${SLURM_NNODES:?}
JOB=${SPUR_JOB_ID:-${SLURM_JOB_ID:?}}
log() { echo "[node $NODE $(hostname -s)] $*"; }

# report <rc>: this node's result for launch.sh (see the end of this script).
report() {
    if [ -n "${IRIS_CLUSTER_RESULTS:-}" ]; then
        echo "$1" > "$IRIS_CLUSTER_RESULTS/node$NODE.rc"
    fi
}

MODE=${1:-}
case "$MODE" in
    --preflight|--image-only) ;;
    # Not "--": sbatch consumes that before passing arguments on to job.sh.
    --run) shift; [ $# -gt 0 ] || { log "no command given after --run"; report 1; exit 1; } ;;
    *) log "usage: node.sh --preflight | --image-only | --run <command...>"; report 1; exit 1 ;;
esac

# ── preflight ────────────────────────────────────────────────────────────────

# The scheduler can hand out a node whose GPUs are held by containers started
# outside any job, and the job then dies in its first large allocation. Check
# before doing anything else; launch.sh resubmits without busy nodes.
MAX_VRAM_PCT=${IRIS_CLUSTER_MAX_VRAM_PCT:-5}
busy=$(rocm-smi --showmemuse --json 2>/dev/null | python3 -I -c '
import json, sys
key = "GPU Memory Allocated (VRAM%)"
cards = json.load(sys.stdin)
print(" ".join(c + "=" + v[key] + "%" for c, v in sorted(cards.items()) if float(v.get(key) or 0) > float(sys.argv[1])))
' "$MAX_VRAM_PCT" || true)
if [ -n "$busy" ]; then
    log "GPU memory already in use: $busy; containers: $(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')"
    report "busy $(hostname -s)"
    exit 75
fi
if [ "$MODE" = "--preflight" ]; then
    exit 0
fi

# ── image ────────────────────────────────────────────────────────────────────

# shellcheck source=image.sh
source .github/scripts/cluster/image.sh
BASE=$IMAGE_BASE CACHE=$IMAGE_CACHE SAVED=$IMAGE_SAVED

ensure_image() {
    docker image inspect "$IMAGE" >/dev/null 2>&1 && { log "image $IMAGE present"; return; }
    # Loading the saved image touches only this node's Docker, so nodes load in
    # parallel (it takes about ten minutes); only a build needs the lock.
    if [ -f "$SAVED" ]; then
        log "loading $SAVED"
        zstd -dc "$SAVED" | docker load
        return
    fi
    mkdir -p "$CACHE"
    exec 9>"$CACHE/lock"
    log "waiting for the image cache lock"
    flock 9
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        if [ -f "$SAVED" ]; then
            log "loading $SAVED (saved by another node while this one waited)"
            zstd -dc "$SAVED" | docker load
        else
            log "building $IMAGE from $BASE (first use of this image: compiles Triton)"
            ctx=$(mktemp -d)
            cp apptainer/iris.def docker/Dockerfile.cluster .github/scripts/install_rocshmem.sh "$ctx/"
            docker build --build-arg BASE="$BASE" -f "$ctx/Dockerfile.cluster" -t "$IMAGE" "$ctx"
            rm -rf "$ctx"
            log "saving to $SAVED"
            docker save "$IMAGE" | zstd -T0 -3 -q -o "$SAVED.tmp.$JOB"
            mv "$SAVED.tmp.$JOB" "$SAVED"
            # Keep this image and the one before it; older ones are dead weight.
            ls -1t "$CACHE"/iris-cluster-*.tar.zst | tail -n +3 | xargs -r rm -f
        fi
    fi
    flock -u 9
}

ensure_image
if [ "$MODE" = "--image-only" ]; then
    report 0
    exit 0
fi

# ── run ──────────────────────────────────────────────────────────────────────

MASTER=$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -1)
IFNAME=${IRIS_CLUSTER_IFNAME:-$(ip -o -4 route show default | awk '{print $5; exit}')}
IONIC=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)
NAME="iris-ci-$JOB-$NODE"
gid() { getent group "$1" | cut -d: -f3; }

ionic_mounts=()
if [ -n "$IONIC" ]; then
    # Both names RCCL's libibverbs looks for: the provider plugin, and the
    # library it links.
    ionic_mounts=(-v "$IONIC:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro"
                  -v "$IONIC:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
fi

# RCCL settings given to the job (IRIS_CLUSTER_ENV in launch.sh, e.g.
# NCCL_IB_HCA, NCCL_IB_GID_INDEX, NCCL_TOPO_FILE) reach the ranks. A topology
# file named by NCCL_TOPO_FILE is a host file, so it is mounted in too.
nccl_env=()
while IFS= read -r var; do
    nccl_env+=(-e "$var")
done < <(env | grep -E '^(NCCL|RCCL)_[A-Z0-9_]+=' | cut -d= -f1)
if [ -n "${NCCL_TOPO_FILE:-}" ]; then
    if [ ! -f "$NCCL_TOPO_FILE" ]; then
        log "NCCL_TOPO_FILE=$NCCL_TOPO_FILE does not exist on this node"
        report 1
        exit 1
    fi
    nccl_env+=(-v "$NCCL_TOPO_FILE:$NCCL_TOPO_FILE:ro")
fi

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

log "image $IMAGE; command: $*"
# As the invoking user, so whatever the command writes to the checkout belongs
# to the runner. spur_job_id lets the cluster's epilog reap the container if
# this script is killed before its trap runs.
docker run -d --init --name "$NAME" \
    --label "spur_job_id=$JOB" \
    --user "$(id -u):$(id -g)" \
    --group-add "$(gid video)" --group-add "$(gid render)" \
    --network host --ipc host --shm-size 64G \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
    --security-opt seccomp=unconfined \
    "${ionic_mounts[@]}" \
    -v "$PWD:$PWD" -v "$PWD:/iris_workspace" -w "$PWD" \
    -e HOME=/tmp/home -e IRIS_IN_CONTAINER=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e IRIS_CLUSTER_NODE="$NODE" -e IRIS_CLUSTER_NNODES="$NNODES" \
    -e MASTER_ADDR="$MASTER" -e MASTER_PORT="${IRIS_CLUSTER_PORT:-29500}" \
    -e NCCL_SOCKET_IFNAME="$IFNAME" -e GLOO_SOCKET_IFNAME="$IFNAME" \
    "${nccl_env[@]}" \
    "$IMAGE" bash -c 'mkdir -p "$HOME" && exec "$@"' bash "$@" >/dev/null

docker logs -f "$NAME" 2>&1 &
# In the background and waited on, so a TERM from a cancelled job runs the
# trap now rather than after the container exits.
rcfile=$(mktemp)
docker wait "$NAME" >"$rcfile" &
wait $!
rc=$(cat "$rcfile"); rm -f "$rcfile"
wait || true
rc=${rc:-1}
log "container exited $rc"
# launch.sh reads these: the scheduler's own record of the job's exit code is
# not reliable (a job whose steps failed has been reported COMPLETED 0:0).
report "$rc"
exit "$rc"
