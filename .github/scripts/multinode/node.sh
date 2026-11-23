#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs on every node of the multi-node CI job (launched by launch.sh through
# srun, in the shared checkout): makes sure the CI image is on this node, then
# runs in_container.sh in one container with all of the node's GPUs and RDMA
# NICs. With --preflight it only checks that the node's GPUs are free; with
# --image-only it stops once the image is there.
#
# The image (docker/Dockerfile.multinode, i.e. apptainer/iris.def plus RDMA
# userspace) is tagged by the checksum of its inputs. Images are node-local and
# building one compiles Triton, so the first node to need a new image builds
# it and saves it to IRIS_MN_CACHE, a directory every node shares; the others,
# then and later, load that copy. A lock on the cache keeps two nodes from
# building at once.

set -euo pipefail

NODE=${SLURM_NODEID:?run under srun (see launch.sh)}
NNODES=${SLURM_NNODES:?}
JOB=${SPUR_JOB_ID:-${SLURM_JOB_ID:?}}
log() { echo "[node $NODE $(hostname -s)] $*"; }

# report <rc>: this node's result for launch.sh (see the end of this script).
report() {
    if [ -n "${IRIS_MN_RESULTS:-}" ]; then
        echo "$1" > "$IRIS_MN_RESULTS/node$NODE.rc"
    fi
}

# ── preflight ────────────────────────────────────────────────────────────────

# The scheduler can hand out a node whose GPUs are held by containers started
# outside any job, and the tests then die in RCCL's first allocation. Check
# before doing anything else; launch.sh resubmits without busy nodes.
MAX_VRAM_PCT=${IRIS_MN_MAX_VRAM_PCT:-5}
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
if [ "${1:-}" = "--preflight" ]; then
    exit 0
fi

# ── image ────────────────────────────────────────────────────────────────────

# shellcheck source=image.sh
source .github/scripts/multinode/image.sh
BASE=$IMAGE_BASE CACHE=$IMAGE_CACHE SAVED=$IMAGE_SAVED

ensure_image() {
    docker image inspect "$IMAGE" >/dev/null 2>&1 && { log "image $IMAGE present"; return; }
    mkdir -p "$CACHE"
    exec 9>"$CACHE/lock"
    log "waiting for the image cache lock"
    flock 9
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        if [ -f "$SAVED" ]; then
            log "loading $SAVED"
            zstd -dc "$SAVED" | docker load
        else
            log "building $IMAGE from $BASE (first use of this image: compiles Triton)"
            ctx=$(mktemp -d)
            cp apptainer/iris.def docker/Dockerfile.multinode .github/scripts/install_rocshmem.sh "$ctx/"
            docker build --build-arg BASE="$BASE" -f "$ctx/Dockerfile.multinode" -t "$IMAGE" "$ctx"
            rm -rf "$ctx"
            log "saving to $SAVED"
            docker save "$IMAGE" | zstd -T0 -3 -q -o "$SAVED.tmp.$JOB"
            mv "$SAVED.tmp.$JOB" "$SAVED"
            # Keep this image and the one before it; older ones are dead weight.
            ls -1t "$CACHE"/iris-multinode-*.tar.zst | tail -n +3 | xargs -r rm -f
        fi
    fi
    flock -u 9
}

ensure_image
if [ "${1:-}" = "--image-only" ]; then
    report 0
    exit 0
fi

# ── run ──────────────────────────────────────────────────────────────────────

MASTER=$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -1)
IFNAME=${IRIS_MN_IFNAME:-$(ip -o -4 route show default | awk '{print $5; exit}')}
IONIC=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)
NAME="iris-ci-$JOB-$NODE"

ionic_mounts=()
if [ -n "$IONIC" ]; then
    # Both names RCCL's libibverbs looks for: the provider plugin, and the
    # library it links.
    ionic_mounts=(-v "$IONIC:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro"
                  -v "$IONIC:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
else
    log "no libionic on the host; GIN will use whatever provider the image has"
fi

# RCCL settings given to the job (IRIS_MN_ENV in launch.sh, e.g. NCCL_IB_HCA,
# NCCL_IB_GID_INDEX, NCCL_TOPO_FILE) reach the ranks. A topology file named by
# NCCL_TOPO_FILE is a host file, so it is mounted in too.
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

log "master $MASTER, socket interface $IFNAME, image $IMAGE"
# Root with SYS_PTRACE: with cuMem on, RCCL hands GPU memory between a node's
# ranks as file descriptors taken with pidfd_getfd, which sibling processes may
# only do with ptrace rights (Yama ptrace_scope=1 here); as an ordinary user it
# fails with "ncclProxyClientGetFd ... failed : 3". So that root never writes
# to the shared filesystem, the checkout is mounted read-only and every cache
# stays in the container. spur_job_id lets the cluster's epilog reap the
# container if this script is killed before its trap runs.
docker run -d --init --name "$NAME" \
    --label "spur_job_id=$JOB" \
    --network host --ipc host --shm-size 64G \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
    --security-opt seccomp=unconfined \
    "${ionic_mounts[@]}" \
    -v "$(pwd):$(pwd):ro" -w "$(pwd)" \
    -e HOME=/tmp/home \
    -e IRIS_MN_NODE="$NODE" -e IRIS_MN_NNODES="$NNODES" \
    -e MASTER_ADDR="$MASTER" -e IRIS_MN_PORT="${IRIS_MN_PORT:-29500}" \
    -e NCCL_SOCKET_IFNAME="$IFNAME" -e GLOO_SOCKET_IFNAME="$IFNAME" \
    "${nccl_env[@]}" \
    "$IMAGE" bash .github/scripts/multinode/in_container.sh >/dev/null

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
