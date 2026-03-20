import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _read_runs_csv_from_dir(d: Path) -> Path:
    p = d / "eval_runs.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing eval_runs.csv in: {d}")
    return p


def _compute_throughput(total_arrived: pd.Series, seconds: int) -> pd.Series:
    # veh/h for paper tables
    return (total_arrived / float(seconds)) * 3600.0


def load_and_merge_runs(eval_dirs: List[str], runs_csvs: List[str]) -> pd.DataFrame:
    csv_paths: List[Path] = []
    for d in eval_dirs:
        csv_paths.append(_read_runs_csv_from_dir(Path(d)))
    for c in runs_csvs:
        csv_paths.append(Path(c))

    if not csv_paths:
        raise ValueError("Provide --eval-dirs and/or --runs-csvs")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    required = {"method", "flow_ns", "flow_we", "total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions"}
    missing = required.difference(set(merged.columns))
    if missing:
        raise ValueError(f"Merged runs missing columns: {sorted(missing)}")
    return merged


def summarize_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    # mean over runs per (flow, method)
    return (
        df_runs.groupby(["flow_ns", "flow_we", "method"], as_index=False)[
            ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions"]
        ]
        .mean()
        .sort_values(["flow_ns", "flow_we", "method"])
        .reset_index(drop=True)
    )


def to_paper_markdown_table(df_summary: pd.DataFrame, seconds: int) -> str:
    df = df_summary.copy()
    # use one flow column for paper display; assume ns==we in our generated routes
    df["flow"] = df["flow_ns"].astype(int)
    df["throughput_veh_h"] = _compute_throughput(df["total_arrived"], seconds=seconds)
    df["collisions"] = df["collisions"].round().astype(int)

    # match your paper table order/labels
    order_method = ["Rule-based", "PPO"]
    df["method"] = pd.Categorical(df["method"], categories=order_method, ordered=True)
    df = df.sort_values(["flow", "method"])

    lines = []
    lines.append("| Flow (veh/h) | Method | Travel Time (s) | Waiting Time (s) | Throughput (veh/h) | Avg Speed (m/s) | Collisions |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            "| {flow} | {method} | {tt:.3f} | {wt:.3f} | {thr:.1f} | {spd:.3f} | {col} |".format(
                flow=int(r["flow"]),
                method=str(r["method"]),
                tt=float(r["avg_travel_time"]),
                wt=float(r["mean_waiting_time"]),
                thr=float(r["throughput_veh_h"]),
                spd=float(r["mean_speed"]),
                col=int(r["collisions"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Merge multiple E4 eval_runs.csv and generate a thesis-ready table + merged CSVs.",
    )
    prs.add_argument(
        "--eval-dirs",
        nargs="*",
        default=[],
        help="Evaluation directories that contain eval_runs.csv (e.g., outputs/.../eval_flow300).",
    )
    prs.add_argument(
        "--runs-csvs",
        nargs="*",
        default=[],
        help="Direct paths to eval_runs.csv files (alternative to --eval-dirs).",
    )
    prs.add_argument("--seconds", type=int, default=600, help="Simulation seconds (for throughput veh/h).")
    prs.add_argument("--outdir", type=str, default="outputs/single-intersection/eval_merged", help="Output directory.")
    prs.add_argument(
        "--table-out",
        type=str,
        default="",
        help="Optional markdown table output path. If empty, writes to {outdir}/table_e4.md",
    )
    args = prs.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_runs = load_and_merge_runs(args.eval_dirs, args.runs_csvs)
    df_summary = summarize_runs(df_runs)

    merged_runs_path = outdir / "merged_eval_runs.csv"
    merged_summary_path = outdir / "merged_eval_summary.csv"
    df_runs.to_csv(merged_runs_path, index=False)
    df_summary.to_csv(merged_summary_path, index=False)

    md = to_paper_markdown_table(df_summary, seconds=args.seconds)
    table_path = Path(args.table_out) if args.table_out else (outdir / "table_e4.md")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(md, encoding="utf-8")

    print(f"Wrote: {merged_runs_path}")
    print(f"Wrote: {merged_summary_path}")
    print(f"Wrote: {table_path}")
    print("\n=== Table X: Performance comparison under different traffic flows ===")
    print(md)


if __name__ == "__main__":
    main()

