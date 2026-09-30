#!/usr/bin/env python3
"""
Sweep roccap captures for the CCL reduce-scatter example (Gluon TDM).

Edit the parameter arrays below, then run from this directory:
    python sweep_roccap.py
    python sweep_roccap.py --dry-run

Each run executes torchrun + roccap_wrapper, then renames generated .cap and
.json files to unique names encoding the sweep parameters, e.g.:
    persistent_reduce_scatter_tdm_gfx1250_64x64_8x64_4sms_1stage_fp32_4warps_2nproc_rank0.cap
    persistent_reduce_scatter_tdm_gfx1250_stepwise_...  (reduce_scatter_tdm_variant=stepwise)
    persistent_reduce_scatter_two_shot_64x64_2x64_4sms_1stage_fp32_1warps_4nproc_rank0.cap

Notes for TDM sweeps:
  - Use fp32; block_size_m/block_size_n must be powers of 2 (PaddedSharedLayout).
  - Inner block_size_n > 256 may fail LLVM PassManager in FFM.
  - Skip --validate in FFM (cross-rank RMA is not functional in simulation).
  - Per tile: world_size TDM loads + fp32 sum + one local TDM store (not W stores).
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Sweep parameter arrays — edit these to define your sweep
# ---------------------------------------------------------------------------

NPROC_PER_NODE = [8]

M_SIZES = [16384]
N_SIZES = [8192]

DATATYPES = ["fp32"]

# (block_size_m, block_size_n) pairs — each entry is one sweep point
BLOCK_SIZES: list[tuple[int, int]] = [
    # (256, 128),
    # (256, 256),
    # Production-ish (528 KiB LDS tile):
    (512, 256),
    (256, 256),
    # (256, 128),
    # (128, 128),
]

COMM_SMS = [64, 80, 96, 128]
NUM_STAGES = [1]
NUM_WARPS = [8]

WAVES_PER_EU = [0]
HEAP_SIZE = [1 << 31]  # [1 << 31]
VALIDATE = [False]
USE_GLUON = [True]
USE_TDM = [True]
REDUCE_SCATTER_TDM_VARIANT = ["stepwise"]  # "hoisted" | "stepwise"

# Minimum .cap file size (MiB) for a capture to count as successful.
MIN_CAP_MB = 4.0

EXTRA_EXAMPLE_ARGS: list[str] = []


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path(__file__).resolve().parent
ROCCAP_WRAPPER = EXAMPLE_DIR / "../../scripts/roccap_wrapper.py"
EXAMPLE_SCRIPT = EXAMPLE_DIR / "example.py"


# Roccap -k filter must match the Triton kernel function name.


def roccap_kernel(use_gluon: bool, use_tdm: bool, reduce_scatter_tdm_variant: str = "hoisted") -> str:
    if use_tdm:
        if not use_gluon:
            raise ValueError("use_tdm=True requires use_gluon=True")
        if reduce_scatter_tdm_variant == "stepwise":
            return "persistent_reduce_scatter_tdm_gfx1250_stepwise"
        if reduce_scatter_tdm_variant == "split":
            return "persistent_reduce_scatter_tdm_gfx1250_split"
        if reduce_scatter_tdm_variant == "padded":
            return "persistent_reduce_scatter_tdm_gfx1250_padded"
        if reduce_scatter_tdm_variant != "hoisted":
            raise ValueError(f"Unknown reduce_scatter_tdm_variant: {reduce_scatter_tdm_variant}")
        return "persistent_reduce_scatter_tdm_gfx1250"
    if use_gluon:
        raise ValueError("reduce-scatter non-TDM Gluon is not implemented")
    return "persistent_reduce_scatter_two_shot"


@dataclass(frozen=True)
class SweepConfig:
    nproc_per_node: int
    m: int
    n: int
    datatype: str
    block_size_m: int
    block_size_n: int
    comm_sms: int
    num_stages: int
    num_warps: int
    waves_per_eu: int
    heap_size: int
    validate: bool
    use_gluon: bool
    use_tdm: bool
    reduce_scatter_tdm_variant: str

    @property
    def kernel(self) -> str:
        return roccap_kernel(self.use_gluon, self.use_tdm, self.reduce_scatter_tdm_variant)

    @property
    def matrix_label(self) -> str:
        return f"{self.m}x{self.n}"

    def basename(self, rank: int) -> str:
        return (
            f"{self.kernel}_"
            f"{self.matrix_label}_"
            f"{self.block_size_m}x{self.block_size_n}_"
            f"{self.comm_sms}sms_"
            f"{self.num_stages}stage_"
            f"{self.datatype}_"
            f"{self.num_warps}warps_"
            f"{self.nproc_per_node}nproc_rank{rank}"
        )

    def example_args(self) -> list[str]:
        args = [
            "-m",
            str(self.m),
            "-n",
            str(self.n),
            "--heap_size",
            str(self.heap_size),
            "--datatype",
            self.datatype,
            "--block_size_m",
            str(self.block_size_m),
            "--block_size_n",
            str(self.block_size_n),
            "--comm_sms",
            str(self.comm_sms),
            "--num_stages",
            str(self.num_stages),
            "--num_warps",
            str(self.num_warps),
            "--waves_per_eu",
            str(self.waves_per_eu),
        ]
        if self.validate:
            args.append("--validate")
        if self.use_gluon:
            args.append("--use_gluon")
        if self.use_tdm:
            args.append("--use_tdm")
            args.extend(["--reduce_scatter_tdm_variant", self.reduce_scatter_tdm_variant])
        args.extend(EXTRA_EXAMPLE_ARGS)
        return args

    def torchrun_cmd(self) -> list[str]:
        wrapper = ROCCAP_WRAPPER.resolve()
        example = EXAMPLE_SCRIPT.resolve()
        return [
            "torchrun",
            f"--nproc_per_node={self.nproc_per_node}",
            "--standalone",
            str(wrapper),
            "-k",
            self.kernel,
            str(example),
            *self.example_args(),
        ]


def iter_sweep_configs() -> Iterable[SweepConfig]:
    for (
        nproc_per_node,
        m,
        n,
        datatype,
        block_size,
        comm_sms,
        num_stages,
        num_warps,
        waves_per_eu,
        heap_size,
        validate,
        use_gluon,
        use_tdm,
        reduce_scatter_tdm_variant,
    ) in itertools.product(
        NPROC_PER_NODE,
        M_SIZES,
        N_SIZES,
        DATATYPES,
        BLOCK_SIZES,
        COMM_SMS,
        NUM_STAGES,
        NUM_WARPS,
        WAVES_PER_EU,
        HEAP_SIZE,
        VALIDATE,
        USE_GLUON,
        USE_TDM,
        REDUCE_SCATTER_TDM_VARIANT,
    ):
        if use_tdm and not use_gluon:
            continue
        if use_gluon and not use_tdm:
            continue
        if reduce_scatter_tdm_variant != "hoisted" and not use_tdm:
            continue
        block_size_m, block_size_n = block_size
        yield SweepConfig(
            nproc_per_node,
            m,
            n,
            datatype,
            block_size_m,
            block_size_n,
            comm_sms,
            num_stages,
            num_warps,
            waves_per_eu,
            heap_size,
            validate,
            use_gluon,
            use_tdm,
            reduce_scatter_tdm_variant,
        )


def resolve_cap_path(cfg: SweepConfig, workdir: Path, rank: int) -> Path | None:
    pattern = f"{cfg.kernel}_rank_{rank}*.cap"
    candidates = list(workdir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def capture_passed(cfg: SweepConfig, workdir: Path, min_cap_bytes: int) -> tuple[bool, list[str]]:
    messages: list[str] = []
    all_ok = True

    for rank in range(cfg.nproc_per_node):
        cap_path = resolve_cap_path(cfg, workdir, rank)
        if cap_path is None:
            all_ok = False
            messages.append(f"rank{rank}: missing {cfg.kernel}_rank_{rank}*.cap")
            continue

        size = cap_path.stat().st_size
        size_mb = size / (1024 * 1024)
        if size <= min_cap_bytes:
            all_ok = False
            min_cap_mb = min_cap_bytes / (1024 * 1024)
            messages.append(f"rank{rank}: {cap_path.name} is {size_mb:.2f} MiB (need > {min_cap_mb:g} MiB)")
        else:
            messages.append(f"rank{rank}: {cap_path.name} is {size_mb:.2f} MiB (pass)")

    return all_ok, messages


def rename_outputs(cfg: SweepConfig, workdir: Path, output_dir: Path | None) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    dest_root = output_dir if output_dir is not None else workdir

    for rank in range(cfg.nproc_per_node):
        basename = cfg.basename(rank)
        cap_src = resolve_cap_path(cfg, workdir, rank)

        renames: list[tuple[Path, Path]] = []
        if cap_src is not None:
            renames.append((cap_src, dest_root / f"{basename}.cap"))
        renames.extend(
            [
                (
                    workdir / f"{cfg.kernel}_rank_{rank}_heap_bases.json",
                    dest_root / f"{basename}_heap_bases.json",
                ),
                (
                    workdir / f"iris_rank_{rank}_allocator_views.json",
                    dest_root / f"{basename}_allocator_views.json",
                ),
            ]
        )

        for src, dst in renames:
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moves.append((src, dst))

    return moves


def run_one(
    cfg: SweepConfig,
    workdir: Path,
    output_dir: Path | None,
    dry_run: bool,
    min_cap_bytes: int,
) -> int:
    cmd = cfg.torchrun_cmd()
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        for rank in range(cfg.nproc_per_node):
            print(f"  -> {cfg.basename(rank)}.cap")
            print(f"  -> {cfg.basename(rank)}_heap_bases.json")
            print(f"  -> {cfg.basename(rank)}_allocator_views.json")
        return 0

    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        print(f"Command exited with code {result.returncode} (checking .cap sizes anyway)")

    passed, cap_messages = capture_passed(cfg, workdir, min_cap_bytes)
    for line in cap_messages:
        print(line)

    if not passed:
        print("Capture failed: one or more .cap files missing or too small", file=sys.stderr)
        return 1

    moves = rename_outputs(cfg, workdir, output_dir)
    if not moves:
        print("Warning: capture passed but no outputs found to rename", file=sys.stderr)
        return 1

    for _, dst in moves:
        print(f"Renamed -> {dst}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep roccap captures for CCL reduce-scatter (Gluon TDM)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=EXAMPLE_DIR,
        help="Directory to run torchrun in (default: this example directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Move renamed artifacts here (default: keep in workdir)",
    )
    parser.add_argument(
        "--min-cap-mb",
        type=float,
        default=None,
        help=f"Minimum .cap file size in MiB (default: MIN_CAP_MB={MIN_CAP_MB:g})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workdir = args.workdir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None

    configs = list(iter_sweep_configs())
    min_cap_mb = MIN_CAP_MB if args.min_cap_mb is None else args.min_cap_mb
    min_cap_bytes = int(min_cap_mb * 1024 * 1024)
    print(f"Planned sweep runs: {len(configs)}")
    print(f"Pass criteria: each rank .cap file > {min_cap_mb:g} MiB")

    failures = 0
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n[{idx}/{len(configs)}]")
        rc = run_one(cfg, workdir, output_dir, args.dry_run, min_cap_bytes)
        if rc != 0:
            failures += 1
            if not args.dry_run:
                print("Stopping sweep after failure.", file=sys.stderr)
                break

    if failures:
        print(f"\nSweep finished with {failures} failure(s).", file=sys.stderr)
        return 1

    print(f"\nSweep finished successfully ({len(configs)} run(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
