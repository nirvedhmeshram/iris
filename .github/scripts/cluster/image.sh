#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Sourced by launch.sh and node.sh, from the checkout's root: the cluster CI
# image's name and where its saved copy lives. The tag is a checksum of every
# input to the image, so editing any of them builds a new one.
# shellcheck disable=SC2034  # used by the scripts that source this

IMAGE_BASE=$(sed -n 's/^From: *//p' apptainer/iris.def)
IMAGE_SUM=$(cat apptainer/iris.def docker/Dockerfile.cluster .github/scripts/install_rocshmem.sh | sha256sum | cut -c1-16)
IMAGE="iris-cluster:$IMAGE_SUM"
IMAGE_CACHE=${IRIS_CLUSTER_CACHE:-$HOME/.cache/iris-cluster}
IMAGE_SAVED="$IMAGE_CACHE/iris-cluster-$IMAGE_SUM.tar.zst"
