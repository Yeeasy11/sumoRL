#!/usr/bin/env python3
"""Move non-4way-turns output artifacts into waste folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


KEEP_KEYWORDS = [
    "4way",
    "fourdir",
    "turn",
    "left",
    "right",
    "straight",
    "三流向",
    "图11",
    "图12",
    "图13",
    "图14",
    "图15",
    "图16",
    "图17",
    "图18",
    "图19",
    "图20",
    "图21",
    "图22",
    "ttc",
    "trajectory",
    "all_runs_matrix",
    "all_evaluation_results",
    "all_ablation_results",
    "master_data_table",
    "composite_summary",
    "significance_results",
]

SKIP_DIRS = {"waste", ".git", ".idea", ".cursor", "__pycache__"}


def _should_keep(path: Path, outputs_root: Path) -> bool:
    rel = str(path.relative_to(outputs_root)).replace("\\", "/").lower()
    if rel.endswith(".csv") and "all_runs_matrix" in rel:
        return True
    return any(k.lower() in rel for k in KEEP_KEYWORDS)


def _collect_move_candidates(outputs_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for p in outputs_root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if _should_keep(p, outputs_root):
            continue
        candidates.append(p)
    return candidates


def _move_files(files: list[Path], outputs_root: Path, waste_root: Path) -> int:
    moved = 0
    for src in files:
        rel = src.relative_to(outputs_root)
        dst = waste_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            i = 1
            while True:
                alt = dst.with_name(f"{stem}__dup{i}{suffix}")
                if not alt.exists():
                    dst = alt
                    break
                i += 1
        shutil.move(str(src), str(dst))
        moved += 1
    return moved


def main() -> None:
    p = argparse.ArgumentParser(description="Move non-4way-turn outputs into waste.")
    p.add_argument("--outputs-root", type=str, default="outputs")
    p.add_argument("--waste-root", type=str, default="waste/outputs_cleanup")
    p.add_argument("--execute", action="store_true", help="Actually move files; default is dry-run only.")
    args = p.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    waste_root = Path(args.waste_root).resolve()

    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs root not found: {outputs_root}")

    candidates = _collect_move_candidates(outputs_root)
    print(f"Found {len(candidates)} files to move.")
    for pth in candidates[:80]:
        print(f"  MOVE -> {pth.relative_to(outputs_root)}")
    if len(candidates) > 80:
        print(f"  ... and {len(candidates) - 80} more")

    if not args.execute:
        print("Dry-run only. Add --execute to move files.")
        return

    moved = _move_files(candidates, outputs_root, waste_root)
    print(f"Moved {moved} files to {waste_root}")


if __name__ == "__main__":
    main()

