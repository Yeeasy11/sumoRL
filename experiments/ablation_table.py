import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _read_summary(path: Path) -> pd.DataFrame:
    if path.is_dir():
        path = path / "eval_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_summary.csv at: {path}")
    return pd.read_csv(path)


def _throughput_veh_h(total_arrived: pd.Series, seconds: int) -> pd.Series:
    return (total_arrived / float(seconds)) * 3600.0


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create ablation markdown table from multiple eval_summary.csv files.",
    )
    prs.add_argument(
        "--summaries",
        nargs="+",
        required=True,
        help="Paths to eval_summary.csv or directories containing it (one per ablation setting).",
    )
    prs.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels in the same order as --summaries (e.g., 'PPO (full)' 'PPO (no collision penalty)' ...)",
    )
    prs.add_argument("--seconds", type=int, default=600)
    prs.add_argument("--out", type=str, default="outputs/single-intersection/ablation_table.md")
    args = prs.parse_args()

    if len(args.summaries) != len(args.labels):
        raise ValueError("--summaries and --labels must have same length")

    rows = []
    for s, label in zip(args.summaries, args.labels):
        df = _read_summary(Path(s))
        # Expect one PPO row; if both methods exist, keep PPO
        if "method" in df.columns:
            df = df[df["method"].astype(str).str.contains("PPO", case=False)]
        if df.empty:
            raise ValueError(f"No PPO row found in summary: {s}")
        r = df.iloc[0]
        rows.append(
            {
                "Method": label,
                "Travel Time": float(r["avg_travel_time"]),
                "Waiting Time": float(r["mean_waiting_time"]),
                "Throughput": float(_throughput_veh_h(pd.Series([r["total_arrived"]]), args.seconds).iloc[0]),
                "Collision": int(round(float(r["collisions"]))),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("| Method | Travel Time | Waiting Time | Throughput | Collision |")
    md.append("|---|---:|---:|---:|---:|")
    for row in rows:
        md.append(
            "| {Method} | {tt:.3f} | {wt:.3f} | {thr:.1f} | {col} |".format(
                Method=row["Method"],
                tt=row["Travel Time"],
                wt=row["Waiting Time"],
                thr=row["Throughput"],
                col=row["Collision"],
            )
        )
    md_text = "\n".join(md)
    out_path.write_text(md_text, encoding="utf-8")

    print(f"Wrote: {out_path}")
    print("\n=== Table X: Ablation study on reward and state design ===")
    print(md_text)


if __name__ == "__main__":
    main()

