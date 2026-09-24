#!/usr/bin/env python3
"""Compile-only register/spill report for the reduce-scatter TDM kernel.

Tests the claim that block_size_m=512 spills registers while 256 does not.
Compiles each block size and reports, from the generated AMDGCN:

  vgpr / sgpr counts, scratch (private segment) bytes, and the number of
  scratch_/buffer_ spill instructions actually emitted.

Scratch bytes > 0 is the authoritative spill signal on AMDGPU: the compiler
only allocates a private segment when it has to spill.

Usage (inside the FFM container):
  IRIS_SIMULATION=1 python3 spill_check.py --block-m 256 384 512
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "../../scripts/roccap_wrapper.py"
EXAMPLE = HERE / "example.py"
KERNEL_BASE = "persistent_reduce_scatter_tdm_gfx1250"

SPILL_INSNS = ("scratch_store", "scratch_load", "buffer_store_dword", "buffer_load_dword")


def compile_one(block_m: int, block_n: int, warps: int, m: int, n: int, variant: str):
    artifacts = Path(tempfile.mkdtemp(prefix="rs_spill_"))
    env = os.environ.copy()
    env["IRIS_KERNEL_ARTIFACTS_DIR"] = str(artifacts)
    env.setdefault("IRIS_SIMULATION", "1")
    cmd = [
        "torchrun", "--nproc_per_node=1", "--standalone", str(WRAPPER), "--skip-roccap",
        "-k", KERNEL_BASE + ("_split" if variant == "split" else ("_stepwise" if variant == "stepwise" else "")), str(EXAMPLE),
        "-m", str(m), "-n", str(n),
        "--block_size_m", str(block_m), "--block_size_n", str(block_n),
        "--num_warps", str(warps), "--comm_sms", "64",
        "--use_gluon", "--use_tdm", "--reduce_scatter_tdm_variant", variant,
    ]
    proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    isa = list(artifacts.rglob("kernel.amdgcn"))
    if not isa:
        err = (proc.stderr or proc.stdout).strip().splitlines()
        reason = next((l for l in reversed(err) if l.strip()), "no output")
        return {"ok": False, "reason": reason[:150]}

    text = isa[0].read_text(errors="replace")
    meta_path = isa[0].parent / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def grab(pat):
        m_ = re.search(pat, text)
        return int(m_.group(1)) if m_ else None

    return {
        "ok": True,
        "vgpr": grab(r"\.vgpr_count:\s*(\d+)") or meta.get("num_registers"),
        "sgpr": grab(r"\.sgpr_count:\s*(\d+)"),
        "vgpr_spill": grab(r"\.vgpr_spill_count:\s*(\d+)"),
        "sgpr_spill": grab(r"\.sgpr_spill_count:\s*(\d+)"),
        "scratch": grab(r"\.private_segment_fixed_size:\s*(\d+)"),
        "lds": grab(r"\.group_segment_fixed_size:\s*(\d+)"),
        "spill_insns": sum(text.count(i) for i in SPILL_INSNS),
        "occupancy_warps": meta.get("num_warps"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-m", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("-m", type=int, default=1024)
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--variant", default="hoisted")
    a = ap.parse_args()

    print(f"reduce_scatter TDM '{a.variant}'  block_n={a.block_n}  num_warps={a.num_warps}  M={a.m} N={a.n}")
    print(f"{'block_m':>8} {'vgpr':>6} {'sgpr':>6} {'vspill':>7} {'sspill':>7} {'scratch':>8} {'lds':>8} {'spill_insns':>12}")
    for bm in a.block_m:
        r = compile_one(bm, a.block_n, a.num_warps, a.m, a.n, a.variant)
        if not r["ok"]:
            print(f"{bm:>8}  FAILED: {r['reason']}")
            continue
        fmt = lambda v: "-" if v is None else v
        print(f"{bm:>8} {fmt(r['vgpr']):>6} {fmt(r['sgpr']):>6} {fmt(r['vgpr_spill']):>7} "
              f"{fmt(r['sgpr_spill']):>7} {fmt(r['scratch']):>8} {fmt(r['lds']):>8} {r['spill_insns']:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
