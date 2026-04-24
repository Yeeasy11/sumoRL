#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


SETTINGS = {
    "full_full",
    "full_phase_only",
    "default_full",
    "default_phase_only",
}


def _stamp_has_variation(eval_run_files: list[Path]) -> bool:
    metrics = ["avg_travel_time", "mean_waiting_time", "throughput", "collisions"]
    vals: dict[str, list[float]] = {m: [] for m in metrics}
    for fp in eval_run_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            return False
        if df.empty:
            return False
        for m in metrics:
            if m in df.columns:
                vals[m].append(float(pd.to_numeric(df[m], errors="coerce").mean()))
    for m in metrics:
        arr = vals[m]
        if len(arr) >= 2 and any(abs(v - arr[0]) > 1e-9 for v in arr[1:]):
            return True
    return False


def _normalize_method_name(v: object) -> str:
    s = str(v).strip().lower()
    mapping = {
        "rule-based": "idm",
        "rule_based": "idm",
        "idm": "idm",
        "fixed_speed": "fixed_speed",
        "yield": "yield",
        "ppo": "ppo",
        "dqn": "dqn",
    }
    return mapping.get(s, s)


def _find_latest_complete_ablation_root(ablation_root: Path) -> tuple[str | None, list[Path]]:
    candidates: dict[str, set[str]] = {}
    file_map: dict[str, list[Path]] = {}
    pat = re.compile(r"^(\d{8}_\d{6})_(full|default)_(full|phase_only)$")

    for p in ablation_root.glob("20*_*/eval/eval_runs.csv"):
        name = p.parent.parent.name
        m = pat.match(name)
        if m is None:
            continue
        stamp = m.group(1)
        reward = m.group(2)
        obs = m.group(3)
        setting = f"{reward}_{obs}"
        if setting not in SETTINGS:
            continue
        candidates.setdefault(stamp, set()).add(setting)
        file_map.setdefault(stamp, []).append(p)

    complete_stamps = [s for s, settings in candidates.items() if settings == SETTINGS]
    if not complete_stamps:
        return None, []

    complete_sorted = sorted(complete_stamps)
    latest = complete_sorted[-1]
    latest_files = sorted(file_map[latest])

    if _stamp_has_variation(latest_files):
        return latest, latest_files

    varied = [s for s in complete_sorted if _stamp_has_variation(sorted(file_map[s]))]
    if varied:
        chosen = varied[-1]
        return chosen, sorted(file_map[chosen])
    return latest, latest_files


def _load_ablation_runs(eval_run_files: Iterable[Path], stamp: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pat = re.compile(r"^(\d{8}_\d{6})_(full|default)_(full|phase_only)$")

    for fp in eval_run_files:
        df = pd.read_csv(fp)
        setting_dir = fp.parent.parent.name
        m = pat.match(setting_dir)
        if m is None:
            reward_mode, obs_mode = ("unknown", "unknown")
        else:
            reward_mode, obs_mode = (m.group(2), m.group(3))

        out = df.copy()
        if "method" not in out.columns:
            out["method"] = "ppo_ablation"
        out["method"] = out["method"].map(_normalize_method_name)
        out["experiment_group"] = "ablation_4way"
        out["ablation_stamp"] = stamp
        out["reward_mode"] = out.get("reward_mode", reward_mode)
        out["obs_mode"] = out.get("obs_mode", obs_mode)
        out["scenario"] = "4way_turns_balanced"
        out["source_eval_runs"] = str(fp)

        if "flow" not in out.columns:
            out["flow"] = 600
        if "dist" not in out.columns:
            out["dist"] = "balanced"
        if "sim_seconds" not in out.columns:
            out["sim_seconds"] = 600
        if "delta_time" not in out.columns:
            out["delta_time"] = 3

        if "throughput" not in out.columns and "total_arrived" in out.columns:
            out["throughput"] = pd.to_numeric(out["total_arrived"], errors="coerce") / 600.0 * 3600.0

        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_main_eval_runs(main_eval_root: Path) -> pd.DataFrame:
    files = sorted(main_eval_root.glob("seed*/eval_runs.csv"))
    frames: list[pd.DataFrame] = []
    for fp in files:
        seed_tag = fp.parent.name
        df = pd.read_csv(fp)
        out = df.copy()
        if "method" not in out.columns:
            out["method"] = "unknown"
        out["method"] = out["method"].map(_normalize_method_name)
        out["experiment_group"] = "main_eval"
        out["scenario"] = "4way_turns_balanced"
        out["source_eval_runs"] = str(fp)
        out["seed_group"] = seed_tag

        if "flow" not in out.columns:
            out["flow"] = 600
        if "dist" not in out.columns:
            out["dist"] = "balanced"
        if "sim_seconds" not in out.columns:
            out["sim_seconds"] = 600
        if "delta_time" not in out.columns:
            out["delta_time"] = 3
        if "throughput" not in out.columns and "total_arrived" in out.columns:
            out["throughput"] = pd.to_numeric(out["total_arrived"], errors="coerce") / 600.0 * 3600.0

        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_eval_matrix_runs(eval_matrix_root: Path) -> pd.DataFrame:
    files = sorted(eval_matrix_root.glob("flow*_*/*/eval_runs.csv"))
    frames: list[pd.DataFrame] = []
    pat = re.compile(r"flow(\d+)_(uniform|poisson|burst|balanced)")

    for fp in files:
        df = pd.read_csv(fp)
        out = df.copy()

        if "method" not in out.columns:
            out["method"] = fp.parent.name
        out["method"] = out["method"].map(_normalize_method_name)

        scenario_name = fp.parent.parent.name
        m = pat.match(scenario_name)
        if m is not None:
            flow = int(m.group(1))
            dist = m.group(2)
        else:
            flow = 600
            dist = "balanced"

        if "flow" not in out.columns:
            out["flow"] = flow
        if "dist" not in out.columns:
            out["dist"] = dist

        if "flow_n" not in out.columns:
            out["flow_n"] = flow
        if "flow_e" not in out.columns:
            out["flow_e"] = flow
        if "flow_s" not in out.columns:
            out["flow_s"] = flow
        if "flow_w" not in out.columns:
            out["flow_w"] = flow

        out["experiment_group"] = "main_eval"
        out["scenario"] = "4way_turns_matrix"
        out["source_eval_runs"] = str(fp)

        if "sim_seconds" not in out.columns:
            out["sim_seconds"] = 600
        if "delta_time" not in out.columns:
            out["delta_time"] = 3
        if "throughput" not in out.columns and "total_arrived" in out.columns:
            out["throughput"] = pd.to_numeric(out["total_arrived"], errors="coerce") / 600.0 * 3600.0

        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    keys = [k for k in ["experiment_group", "method", "flow", "dist", "reward_mode", "obs_mode"] if k in df.columns]
    metric_cols = [
        c
        for c in [
            "total_arrived",
            "mean_waiting_time",
            "mean_speed",
            "avg_travel_time",
            "collisions",
            "min_ttc",
            "harsh_brake_rate",
            "mean_abs_jerk",
            "gini_waiting_time",
            "throughput",
        ]
        if c in df.columns
    ]
    if not keys or not metric_cols:
        return pd.DataFrame()

    agg = {m: ["mean", "std"] for m in metric_cols}
    out = df.groupby(keys, as_index=False).agg(agg)
    out.columns = [c if isinstance(c, str) else f"{c[0]}_{c[1]}" for c in out.columns]
    return out


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip("_") for col in out.columns]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build unified all_runs_matrix (CSV + XLSX with multi-sheets).")
    p.add_argument("--eval-matrix-root", type=str, default="outputs/eval_repro")
    p.add_argument("--main-eval-root", type=str, default="outputs/4way-single-intersection/multiseed_eval")
    p.add_argument("--ablation-root", type=str, default="outputs/4way-single-intersection/ablation")
    p.add_argument("--seed3-csv", type=str, default="outputs/figures/all_runs_matrix_3seed.csv")
    p.add_argument("--summary-csv", type=str, default="outputs/composite_summary.csv")
    p.add_argument("--sig-csv", type=str, default="outputs/significance_results.csv")
    p.add_argument("--out-csv", type=str, default="outputs/all_runs_matrix.csv")
    p.add_argument("--out-xlsx", type=str, default="outputs/all_runs_matrix.xlsx")
    args = p.parse_args()

    root = Path.cwd()
    eval_matrix_root = root / args.eval_matrix_root
    main_eval_root = root / args.main_eval_root
    ablation_root = root / args.ablation_root
    seed3_csv = root / args.seed3_csv
    summary_csv = root / args.summary_csv
    sig_csv = root / args.sig_csv
    out_csv = root / args.out_csv
    out_xlsx = root / args.out_xlsx

    main_df = _load_eval_matrix_runs(eval_matrix_root)
    if main_df.empty:
        main_df = _load_main_eval_runs(main_eval_root)
    stamp, ablation_files = _find_latest_complete_ablation_root(ablation_root)
    ablation_df = _load_ablation_runs(ablation_files, stamp or "unknown")

    merged_parts = [df for df in [main_df, ablation_df] if not df.empty]
    merged_df = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame()

    if not merged_df.empty:
        merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    seed3_df = pd.DataFrame()
    if seed3_csv.exists():
        try:
            seed3_df = pd.read_csv(seed3_csv, header=[0, 1])
        except Exception:
            seed3_df = pd.read_csv(seed3_csv)
    seed3_df = _flatten_multiindex_columns(seed3_df)

    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    sig_df = pd.read_csv(sig_csv) if sig_csv.exists() else pd.DataFrame()
    merged_summary_df = _summarize(merged_df)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        merged_df.to_excel(writer, sheet_name="all_runs_matrix", index=False)
        merged_summary_df.to_excel(writer, sheet_name="all_runs_summary", index=False)
        if not main_df.empty:
            main_df.to_excel(writer, sheet_name="main_eval_raw", index=False)
        if not ablation_df.empty:
            ablation_df.to_excel(writer, sheet_name="ablation_latest_runs", index=False)
        if not seed3_df.empty:
            seed3_df.to_excel(writer, sheet_name="matrix_3seed_source", index=False)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="composite_summary", index=False)
        if not sig_df.empty:
            sig_df.to_excel(writer, sheet_name="significance", index=False)

    print(f"Wrote unified CSV: {out_csv}")
    print(f"Wrote workbook: {out_xlsx}")
    print(f"Latest complete ablation stamp: {stamp}")
    print(f"Ablation files merged: {len(ablation_files)}")
    print(f"Rows in unified matrix: {len(merged_df)}")


if __name__ == "__main__":
    main()
