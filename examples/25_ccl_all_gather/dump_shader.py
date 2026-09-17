#!/usr/bin/env python3
"""
Fast local shader dump for all-gather TDM kernels (no roccap capture).

Compile-only path (~20s in FFM):
  IRIS_SIMULATION=1 python dump_shader.py --variant warp_specialized_local_smem

Writes under shader/:
  <name>.amdgcn, <name>.llir, <name>.ttgir, <name>.metadata.json
and prints barrier/TDM opcode counts from the AMDGPU text.

Optional SP3 disassembly from an existing roccap .cap (no re-capture):
  python dump_shader.py --sp3-from-cap path/to/kernel_rank0.cap --name mykernel

Requires FFM env (source /ffm/ffmlite_env.sh) for TDM compile and /ffm/sp3disasm.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
ROCCAP_WRAPPER = EXAMPLE_DIR / "../../scripts/roccap_wrapper.py"
EXAMPLE_SCRIPT = EXAMPLE_DIR / "example.py"
SHADER_DIR = EXAMPLE_DIR / "shader"
SP3DISASM = Path("/ffm/sp3disasm")

VARIANT_KERNEL = {
    "hoisted": "persistent_all_gather_tdm_gfx1250",
    "stepwise": "persistent_all_gather_tdm_gfx1250_stepwise",
    "warp_team": "persistent_all_gather_tdm_gfx1250_warp_team",
    "warp_specialized": "persistent_all_gather_tdm_gfx1250_warp_specialized",
    "warp_specialized_local_smem": "persistent_all_gather_tdm_gfx1250_warp_specialized_local_smem",
    "warp_specialized_improved": "persistent_all_gather_tdm_gfx1250_warp_specialized_improved",
}

BARRIER_PATTERNS = [
    "s_barrier_join",
    "s_barrier_signal",
    "s_barrier_wait",
    "s_waitcnt",
    "s_wait_tensorcnt",
    "tensor_load_to_lds",
    "tensor_store_from_lds",
]


def _run_compile(args: argparse.Namespace) -> Path:
    kernel = VARIANT_KERNEL[args.variant]
    artifacts_root = Path(tempfile.mkdtemp(prefix="iris_shader_"))
    env = os.environ.copy()
    env["IRIS_KERNEL_ARTIFACTS_DIR"] = str(artifacts_root)
    env.setdefault("IRIS_SIMULATION", "1")

    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--standalone",
        str(ROCCAP_WRAPPER),
        "--skip-roccap",
        "-k",
        kernel,
        str(EXAMPLE_SCRIPT),
        "-m",
        str(args.m),
        "-n",
        str(args.n),
        "--datatype",
        args.datatype,
        "--block_size_m",
        str(args.block_size_m),
        "--block_size_n",
        str(args.block_size_n),
        "--comm_sms",
        str(args.comm_sms),
        "--num_warps",
        str(args.num_warps),
        "--use_gluon",
        "--use_tdm",
        "--all_gather_tdm_variant",
        args.variant,
    ]
    print("Compiling:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=EXAMPLE_DIR, env=env)

    matches = list(artifacts_root.rglob("kernel.amdgcn"))
    if not matches:
        raise RuntimeError(f"No kernel.amdgcn under {artifacts_root}")
    return matches[0].parent


def _copy_artifacts(artifact_dir: Path, out_stem: Path) -> None:
    SHADER_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "kernel.amdgcn": f"{out_stem.name}.amdgcn",
        "kernel.llir": f"{out_stem.name}.llir",
        "kernel.ttgir": f"{out_stem.name}.ttgir",
        "metadata.json": f"{out_stem.name}.metadata.json",
    }
    for src_name, dst_name in mapping.items():
        src = artifact_dir / src_name
        if src.exists():
            shutil.copy2(src, SHADER_DIR / dst_name)


def _summarize_amdgcn(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = {pat: text.count(pat) for pat in BARRIER_PATTERNS}
    counts["lines"] = text.count("\n") + 1
    meta = path.with_suffix(".metadata.json")
    if meta.exists():
        md = json.loads(meta.read_text(encoding="utf-8"))
        counts["num_warps_launch"] = md.get("num_warps", 0)
        counts["vgpr_hint"] = md.get("num_registers", 0)
    return counts


def _parse_pgm_rsrc1(cap_path: Path) -> int | None:
    """Parse compute_pgm_rsrc1 from roccap extract stdout."""
    proc = subprocess.run(
        ["roccap", "extract", "-s", "--sp3", "0-", str(cap_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"compute_pgm_rsrc1\s*=\s*(0x[0-9a-fA-F]+)", proc.stdout)
    if not match:
        return None
    return int(match.group(1), 16)


def _sp3_from_cap(cap_path: Path, out_sp3: Path) -> None:
    if not cap_path.exists():
        raise FileNotFoundError(cap_path)
    pgm_rsrc1 = _parse_pgm_rsrc1(cap_path)
    if pgm_rsrc1 is None:
        raise RuntimeError(f"Could not parse compute_pgm_rsrc1 from {cap_path}")

    with tempfile.TemporaryDirectory(prefix="roccap_sp3_") as tmp:
        prefix = Path(tmp) / "out"
        subprocess.run(
            ["roccap", "extract", "--sp3", "0-", "-o", str(prefix), str(cap_path)],
            check=True,
        )
        bins = sorted(glob.glob(str(prefix) + "*.bin"))
        if not bins:
            raise RuntimeError(f"No isa-data.bin from roccap extract on {cap_path}")
        sp3disasm = SP3DISASM if SP3DISASM.exists() else Path(shutil.which("sp3disasm") or "")
        if not sp3disasm:
            raise RuntimeError("sp3disasm not found (expected /ffm/sp3disasm in FFM)")
        os.chmod(sp3disasm, os.stat(sp3disasm).st_mode | 0o111)
        subprocess.run([str(sp3disasm), bins[0], str(out_sp3), hex(pgm_rsrc1)], check=True)


def _default_name(args: argparse.Namespace) -> str:
    return (
        f"{VARIANT_KERNEL[args.variant]}_{args.m}x{args.n}_"
        f"{args.block_size_m}x{args.block_size_n}_{args.comm_sms}sms_"
        f"{args.datatype}_{args.num_warps}warps"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast compile-only shader dump for all-gather TDM")
    parser.add_argument("--name", type=str, default="", help="Output basename under shader/ (default: auto)")
    parser.add_argument(
        "--variant",
        type=str,
        default="warp_specialized_local_smem",
        choices=sorted(VARIANT_KERNEL),
    )
    parser.add_argument("-m", type=int, default=512)
    parser.add_argument("-n", type=int, default=256)
    parser.add_argument("--datatype", type=str, default="fp32", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--block_size_m", type=int, default=512)
    parser.add_argument("--block_size_n", type=int, default=256)
    parser.add_argument("--comm_sms", type=int, default=64)
    parser.add_argument("--num_warps", type=int, default=8)
    parser.add_argument(
        "--sp3-from-cap",
        type=Path,
        default=None,
        help="Optional: disassemble SP3 from an existing .cap (uses roccap extract + sp3disasm)",
    )
    parser.add_argument("--compile-only", action="store_true", help="Skip compile; only run --sp3-from-cap")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_name = args.name or _default_name(args)

    if not args.compile_only:
        artifact_dir = _run_compile(args)
        out_stem = SHADER_DIR / out_name
        _copy_artifacts(artifact_dir, out_stem)
        counts = _summarize_amdgcn(SHADER_DIR / f"{out_name}.amdgcn")
        print(f"Wrote {SHADER_DIR / (out_name + '.amdgcn')}")
        print("Summary:")
        for key, val in counts.items():
            print(f"  {key}: {val}")

    if args.sp3_from_cap:
        out_sp3 = SHADER_DIR / f"{out_name}.shader_0.sp3"
        _sp3_from_cap(args.sp3_from_cap, out_sp3)
        counts = _summarize_amdgcn(out_sp3)
        print(f"Wrote {out_sp3}")
        print("SP3 summary:")
        for key in BARRIER_PATTERNS + ["lines"]:
            if key in counts:
                print(f"  {key}: {counts[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
