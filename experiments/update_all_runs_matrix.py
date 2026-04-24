#!/usr/bin/env python3
"""Rebuild outputs/all_runs_matrix.csv from all available result sources."""

from pathlib import Path
import re

import pandas as pd


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return pd.DataFrame()


def _enrich_common(df: pd.DataFrame, source: Path, source_type: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["source"] = str(source).replace("/", "\\")
    out["source_type"] = source_type
    if "method" not in out.columns:
        out["method"] = ""
    out["method"] = out["method"].fillna("").astype(str).str.strip()
    out.loc[out["method"] == "", "method"] = "PPO"

    if "experiment" not in out.columns:
        out["experiment"] = source_type

    src_lower = str(source).lower()
    out["parsed_seed"] = "unknown"
    m_seed = re.search(r"seed(\d+)", src_lower)
    if m_seed:
        out["parsed_seed"] = int(m_seed.group(1))

    out["parsed_distribution"] = "unknown"
    for dist in ("uniform", "poisson", "burst"):
        if dist in src_lower:
            out["parsed_distribution"] = dist
            break

    out["parsed_intersection"] = "4way" if "4way" in src_lower else "unknown"
    return out


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_cols = [
        "flow",
        "flow_n",
        "flow_e",
        "flow_s",
        "flow_w",
        "total_arrived",
        "mean_waiting_time",
        "mean_speed",
        "avg_travel_time",
        "collisions",
        "throughput",
        "min_ttc",
        "harsh_brake_rate",
        "mean_abs_jerk",
        "gini_waiting_time",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "flow" not in df.columns:
        if {"flow_n", "flow_e", "flow_s", "flow_w"}.issubset(df.columns):
            df["flow"] = (
                pd.to_numeric(df["flow_n"], errors="coerce").fillna(0)
                + pd.to_numeric(df["flow_e"], errors="coerce").fillna(0)
                + pd.to_numeric(df["flow_s"], errors="coerce").fillna(0)
                + pd.to_numeric(df["flow_w"], errors="coerce").fillna(0)
            ) / 4.0
        else:
            df["flow"] = pd.NA

    if "dist" not in df.columns:
        if "parsed_distribution" in df.columns:
            df["dist"] = df["parsed_distribution"]
        else:
            df["dist"] = "unknown"

    if "setting" not in df.columns:
        df["setting"] = "default"

    if {"reward_mode", "obs_mode"}.issubset(df.columns):
        setting_mask = df["setting"].isna() | (df["setting"].astype(str).str.strip() == "") | (df["setting"] == "default")
        df.loc[setting_mask, "setting"] = (
            df["reward_mode"].fillna("unknown").astype(str)
            + "|"
            + df["obs_mode"].fillna("unknown").astype(str)
        )

    preferred = [
        "experiment",
        "source_type",
        "source",
        "method",
        "setting",
        "reward_mode",
        "obs_mode",
        "flow",
        "flow_n",
        "flow_e",
        "flow_s",
        "flow_w",
        "dist",
        "total_arrived",
        "mean_waiting_time",
        "mean_speed",
        "avg_travel_time",
        "collisions",
        "throughput",
        "min_ttc",
        "harsh_brake_rate",
        "mean_abs_jerk",
        "gini_waiting_time",
        "parsed_seed",
        "parsed_distribution",
        "parsed_intersection",
    ]
    existing = [c for c in preferred if c in df.columns]
    extra = [c for c in df.columns if c not in existing]
    return df[existing + extra]


def main() -> None:
    outputs_dir = Path("outputs")
    pieces: list[pd.DataFrame] = []

    # Source 1: previously aggregated full tables.
    for name, tag in [
        ("all_evaluation_results.csv", "evaluation_full"),
        ("all_ablation_results.csv", "ablation_full"),
        ("master_data_table.csv", "master_full"),
    ]:
        fp = outputs_dir / name
        df = _read_csv_if_exists(fp)
        if not df.empty:
            pieces.append(_enrich_common(df, fp, tag))

    # Source 1b: legacy full matrix from waste (often contains safety/comfort metrics).
    legacy_matrix = Path("waste") / "outputs" / "all_runs_matrix.csv"
    df_legacy = _read_csv_if_exists(legacy_matrix)
    if not df_legacy.empty:
        pieces.append(_enrich_common(df_legacy, legacy_matrix, "legacy_waste_matrix"))

    # Source 2: direct eval summary.
    eval_summary = outputs_dir / "4way-single-intersection" / "eval" / "eval" / "eval_summary.csv"
    df_eval = _read_csv_if_exists(eval_summary)
    if not df_eval.empty:
        if "experiment" not in df_eval.columns:
            df_eval["experiment"] = "evaluation"
        pieces.append(_enrich_common(df_eval, eval_summary, "evaluation_summary"))

    # Source 3: each ablation setting eval summary.
    ablation_root = outputs_dir / "4way-single-intersection" / "ablation" / "ablation"
    for fp in sorted(ablation_root.glob("*_*_*/eval/eval_summary.csv")):
        df = _read_csv_if_exists(fp)
        if df.empty:
            continue
        parts = fp.parts
        setting_name = ""
        if len(parts) >= 4:
            setting_name = parts[-3]
        m = re.search(r"_(full|default)_(full|phase_only)$", setting_name)
        if m:
            df["reward_mode"] = m.group(1)
            df["obs_mode"] = m.group(2)
            df["setting"] = f"{m.group(1)}|{m.group(2)}"
        if "experiment" not in df.columns:
            df["experiment"] = "ablation"
        pieces.append(_enrich_common(df, fp, "ablation_setting_summary"))

    if not pieces:
        raise FileNotFoundError("No result sources found under outputs/")

    result_df = pd.concat(pieces, ignore_index=True)
    result_df = _normalize_columns(result_df)

    dedup_cols = [
        c
        for c in [
            "source",
            "method",
            "setting",
            "flow",
            "dist",
            "total_arrived",
            "mean_waiting_time",
            "mean_speed",
            "avg_travel_time",
            "collisions",
            "throughput",
            "min_ttc",
            "harsh_brake_rate",
            "mean_abs_jerk",
            "gini_waiting_time",
        ]
        if c in result_df.columns
    ]
    if dedup_cols:
        result_df = result_df.drop_duplicates(subset=dedup_cols).copy()

    sort_cols = [c for c in ["experiment", "source_type", "method", "setting", "flow", "dist"] if c in result_df.columns]
    if sort_cols:
        result_df = result_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    output_path = outputs_dir / "all_runs_matrix.csv"
    figures_path = outputs_dir / "figures" / "all_runs_matrix.csv"
    figures_path.parent.mkdir(parents=True, exist_ok=True)

    # utf-8-sig avoids Chinese garbling in Excel/WPS on Windows.
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    result_df.to_csv(figures_path, index=False, encoding="utf-8-sig")

    print(f"Generated: {output_path}")
    print(f"Backup:    {figures_path}")
    print(f"Rows:      {len(result_df)}")
    print(f"Columns:   {len(result_df.columns)}")
    print(f"Sources:   {result_df['source_type'].nunique() if 'source_type' in result_df.columns else 0}")


if __name__ == "__main__":
    main()
