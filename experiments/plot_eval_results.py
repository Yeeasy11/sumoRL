import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting.\n"
            "Install with: pip install matplotlib"
        ) from e


def load_eval_data(summary_csv: Optional[str], runs_csv: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if not summary_csv and not runs_csv:
        raise ValueError("Provide at least one of --summary-csv or --runs-csv")

    df_runs = None
    if runs_csv:
        df_runs = pd.read_csv(runs_csv)

    if summary_csv:
        df_summary = pd.read_csv(summary_csv)
    else:
        # derive summary from runs
        assert df_runs is not None
        df_summary = (
            df_runs.groupby(["method", "flow_ns", "flow_we"], as_index=False)[
                ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions"]
            ]
            .mean()
        )

    return df_summary, df_runs


def compute_throughput(df: pd.DataFrame, seconds: int, unit: str) -> pd.Series:
    thr = df["total_arrived"] / float(seconds)
    if unit.lower() in {"veh/h", "vehs/h", "vph"}:
        return thr * 3600.0
    return thr  # veh/s


def print_paper_table(df_summary: pd.DataFrame, seconds: int, unit: str) -> None:
    df = df_summary.copy()
    df["throughput"] = compute_throughput(df, seconds=seconds, unit=unit)

    cols = ["method", "flow_ns", "flow_we", "avg_travel_time", "mean_waiting_time", "throughput", "mean_speed", "collisions"]
    df = df[cols].sort_values(["flow_ns", "flow_we", "method"])

    print("\n=== Paper Table Values (copy into thesis) ===")
    print(f"Throughput unit: {unit}")
    for _, r in df.iterrows():
        print(
            f"- {r['method']} (flow_ns={int(r['flow_ns'])}, flow_we={int(r['flow_we'])}): "
            f"TravelTime={r['avg_travel_time']:.3f}, "
            f"WaitingTime={r['mean_waiting_time']:.3f}, "
            f"Throughput={r['throughput']:.3f}, "
            f"Speed={r['mean_speed']:.3f}, "
            f"Collision={int(r['collisions'])}"
        )


def plot_method_comparison(
    df_summary: pd.DataFrame,
    df_runs: Optional[pd.DataFrame],
    seconds: int,
    unit: str,
    out_dir: Path,
    title_prefix: str,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df_s = df_summary.copy()
    df_s["throughput"] = compute_throughput(df_s, seconds=seconds, unit=unit)

    # If multiple flows exist, create one figure per flow
    flows = df_s[["flow_ns", "flow_we"]].drop_duplicates().sort_values(["flow_ns", "flow_we"])
    for _, fr in flows.iterrows():
        fns, fwe = int(fr["flow_ns"]), int(fr["flow_we"])
        sub = df_s[(df_s["flow_ns"] == fns) & (df_s["flow_we"] == fwe)].copy()

        # Prefer ordering: Rule-based then PPO
        order = ["Rule-based", "PPO"]
        sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
        sub = sub.sort_values("method")

        metrics = [
            ("avg_travel_time", "Travel Time (s) ↓"),
            ("mean_waiting_time", "Waiting Time (s) ↓"),
            ("throughput", f"Throughput ({unit}) ↑"),
            ("mean_speed", "Speed (m/s) ↑"),
            ("collisions", "Collision ↓"),
        ]

        # error bars from runs, if available
        err = None
        if df_runs is not None:
            dr = df_runs.copy()
            dr["throughput"] = compute_throughput(dr, seconds=seconds, unit=unit)
            dr = dr[(dr["flow_ns"] == fns) & (dr["flow_we"] == fwe)]
            if not dr.empty:
                err = dr.groupby("method")[["avg_travel_time", "mean_waiting_time", "throughput", "mean_speed", "collisions"]].std()

        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.2))
        fig.suptitle(f"{title_prefix} (flow_ns={fns}, flow_we={fwe})", fontsize=12)

        for ax, (col, ylabel) in zip(axes, metrics):
            vals = sub.set_index("method")[col]
            x = list(vals.index.astype(str))
            y = vals.values
            yerr = None
            if err is not None and col in err.columns:
                # Use 1D std array aligned with methods (same order as vals)
                idx = [str(m) for m in vals.index.tolist()]
                yerr = err.reindex(idx)[col].values
            ax.bar(x, y, yerr=yerr, capsize=4, color=["#4C72B0", "#55A868"])
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)

        plt.tight_layout()
        out_path = out_dir / f"method_compare_flow_{fns}_{fwe}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote: {out_path}")


def plot_flow_sweep(
    df_summary: pd.DataFrame,
    seconds: int,
    unit: str,
    out_dir: Path,
    title: str,
) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df_summary.copy()
    df["flow"] = df["flow_ns"]  # for this net, ns and we are often set equal; we plot by ns
    df["throughput"] = compute_throughput(df, seconds=seconds, unit=unit)

    if df["flow"].nunique() < 2:
        return  # nothing to sweep

    # Two typical thesis plots: Travel Time vs Flow, Throughput vs Flow
    for ycol, ylabel, fname in [
        ("avg_travel_time", "Average Travel Time (s) ↓", "flow_sweep_travel_time.png"),
        ("throughput", f"Throughput ({unit}) ↑", "flow_sweep_throughput.png"),
    ]:
        fig = plt.figure(figsize=(6.8, 4.2))
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method].sort_values("flow")
            plt.plot(sub["flow"], sub[ycol], marker="o", linewidth=2, label=method)
        plt.title(title)
        plt.xlabel("Traffic Flow (veh/h)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote: {out_path}")


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot thesis-ready figures from eval_runs.csv / eval_summary.csv.",
    )
    prs.add_argument("--summary-csv", type=str, default="", help="Path to eval_summary.csv")
    prs.add_argument("--runs-csv", type=str, default="", help="Path to eval_runs.csv (for error bars)")
    prs.add_argument("--seconds", type=int, default=600, help="Simulation seconds (for throughput conversion)")
    prs.add_argument("--throughput-unit", type=str, default="veh/h", help="veh/s or veh/h")
    prs.add_argument("--outdir", type=str, default="outputs/figures", help="Output directory for figures")
    prs.add_argument("--title-prefix", type=str, default="Performance Comparison Between Rule-based and PPO", help="Title prefix")
    args = prs.parse_args()

    df_summary, df_runs = load_eval_data(args.summary_csv or None, args.runs_csv or None)
    out_dir = Path(args.outdir)

    print_paper_table(df_summary, seconds=args.seconds, unit=args.throughput_unit)
    plot_method_comparison(
        df_summary=df_summary,
        df_runs=df_runs,
        seconds=args.seconds,
        unit=args.throughput_unit,
        out_dir=out_dir,
        title_prefix=args.title_prefix,
    )
    plot_flow_sweep(
        df_summary=df_summary,
        seconds=args.seconds,
        unit=args.throughput_unit,
        out_dir=out_dir,
        title="Performance Under Different Traffic Densities",
    )


if __name__ == "__main__":
    main()

