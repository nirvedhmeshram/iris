#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Compile the rocSHMEM GDA shim to LLVM bitcode for linking into a Triton module.
#
# Two objects are produced:
#   iris_gda_shim_<arch>.bc       hot path; QueuePair headers only
#   iris_gda_ctx_probe_<arch>.bc  setup only; the one object that sees GDAContext
#
# Must run where a ROCm clang exists (inside the ROCm container, not a login
# node). Requires a rocSHMEM SOURCE tree: rocSHMEM does not install the QueuePair
# headers today, since they are internal. Upstream is discussing making the whole
# device layer header-only and installing it, which would remove that need.
#
# Usage:
#   ROCSHMEM_SRC_TREE=/path/to/rocshmem \
#   BUILD_INC=/path/to/rocshmem-install/include \
#     bash csrc/rocshmem_gda/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SRC_TREE must be the SAME tree that built the librocshmem this bitcode will run
# against -- these are internal headers, so a mismatch changes struct layouts and
# faults at 0x0 with no diagnostic. Nothing checks it at load time.
SRC_TREE="${ROCSHMEM_SRC_TREE:?set ROCSHMEM_SRC_TREE to the rocSHMEM source tree (…/projects/rocshmem)}"
ARCH="${ARCH:-gfx950}"
SHIM="${SHIM:-${HERE}/iris_gda_shim.hip}"
PROBE="${PROBE:-${HERE}/iris_gda_ctx_probe.hip}"
OUT="${OUT:-${HERE}/iris_gda_shim_${ARCH}.bc}"
PROBE_OUT="${PROBE_OUT:-${HERE}/iris_gda_ctx_probe_${ARCH}.bc}"
BUILD_INC="${BUILD_INC:-}"   # dir holding the generated rocshmem_config.h

# Without `|| true` this assignment fails under `set -e` and the script exits 2
# with no output at all -- the error branch below never runs.
CLANG=$(ls -d /opt/rocm*/lib/llvm/bin/clang++ /opt/rocm*/llvm/bin/clang++ 2>/dev/null | head -1 || true)
if [ -z "$CLANG" ]; then
  echo "FATAL: no clang++ under /opt/rocm* -- run this inside the ROCm container" >&2
  exit 1
fi
echo "==> clang:    $CLANG"
echo "==> src tree: $SRC_TREE"
echo "==> arch:     $ARCH"

# Replacements FIRST. replacements/log.hpp must win over rocSHMEM's own log.hpp,
# which declares an extern __constant__ logd_constants that is defined only
# inside rocSHMEM's device bitcode -- which we deliberately do not link. Include
# order is the whole mechanism: put SRC_TREE first and rocSHMEM's header wins,
# the extern comes back, and the module fails to load with HIP 209.
INCS=(
  -I "${HERE}/replacements"
  -I "${SRC_TREE}/include"
  -I "${SRC_TREE}/include/rocshmem"
  -I "${SRC_TREE}/src"
)
[ -n "$BUILD_INC" ] && INCS+=( -I "$BUILD_INC" -I "${BUILD_INC}/rocshmem" )

# rocshmem.hpp includes <mpi.h> when the install was configured with
# HAVE_EXTERNAL_MPI, and context_gda_device.hpp pulls that in transitively, so
# the probe needs it even though the shim does not.
MPI_INC=$(mpicc --showme:incdirs 2>/dev/null | tr ' ' '\n' | head -1 || true)
[ -z "$MPI_INC" ] && MPI_INC=$(ls -d /usr/lib/x86_64-linux-gnu/openmpi/include \
                                    /usr/include/openmpi-x86_64 \
                                    /usr/include/x86_64-linux-gnu/openmpi 2>/dev/null | head -1 || true)
if [ -n "$MPI_INC" ]; then
  echo "==> mpi include: $MPI_INC"
  INCS+=( -I "$MPI_INC" )
else
  echo "==> WARNING: no mpi.h found; the probe will fail to compile if the config defines HAVE_EXTERNAL_MPI"
fi

# Mirror the flags rocSHMEM uses for its own device bitcode (DeviceBitcode.cmake):
# device-only, emit LLVM, default visibility so the symbols survive, and
# -fgpu-rdc so functions with no caller in this TU are not internalised away.
compile_bc() {
  local src="$1" dst="$2" label="$3"
  "$CLANG" \
    -x hip --cuda-device-only -emit-llvm -std=c++17 \
    --offload-arch="${ARCH}" \
    -fvisibility=default -fgpu-rdc -O3 \
    -Xclang -mcode-object-version=none \
    "${INCS[@]}" \
    -c "$src" -o "$dst" 2>&1 | head -40

  if [ ! -f "$dst" ]; then
    echo "==> BUILD FAILED (${label})" >&2; exit 1
  fi
  echo "==> wrote $dst ($(stat -c %s "$dst") bytes)  [${label}]"

  local dis
  dis=$(ls -d /opt/rocm*/lib/llvm/bin/llvm-dis 2>/dev/null | head -1 || true)
  if [ -n "$dis" ]; then
    "$dis" -o /tmp/iris-gda-bc.ll "$dst"
    echo "    exported iris_gda symbols:"
    grep -o '^define[^@]*@iris_gda[a-z_]*' /tmp/iris-gda-bc.ll | sed 's/.*@/      /'
    # The split only pays off if it holds: the hot path must not pull in
    # GDAContext. Without this check it decays silently the first time someone
    # adds an include, and no test would fail.
    if [ "$label" = "shim" ] && grep -q "GDAContext" /tmp/iris-gda-bc.ll; then
      echo "==> ERROR: GDAContext appears in the hot-path shim bitcode;" >&2
      echo "    it belongs only in the context probe." >&2
      rm -f /tmp/iris-gda-bc.ll
      exit 1
    fi
    rm -f /tmp/iris-gda-bc.ll
  fi
}

compile_bc "$SHIM"  "$OUT"       "shim"
compile_bc "$PROBE" "$PROBE_OUT" "ctx-probe"
