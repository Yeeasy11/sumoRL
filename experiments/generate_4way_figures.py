import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("matplotlib is required for plotting. Install with: pip install matplotlib") from e


def _save(fig, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _find_monitor_files(monitor_dir: Path) -> List[Path]:
    if not monitor_dir.exists():
        return []
    return sorted(monitor_dir.glob("*.monitor.csv"), key=lambda p: p.stat().st_mtime)


def _load_monitor(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"Monitor file missing r column: {file_path}")
    out = df.copy()
    out["episode"] = np.arange(1, len(out) + 1)
    return out


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=max(1, window // 3)).mean()


def _align_runs(runs: List[pd.DataFrame], smooth_window: int) -> pd.DataFrame:
    min_len = min(len(x) for x in runs)
    rows = []
    for idx, run in enumerate(runs, start=1):
        sub = run.iloc[:min_len].copy()
        sub["smooth"] = _rolling_mean(sub["r"], smooth_window)
        sub["seed_id"] = idx
        rows.append(sub[["episode", "r", "smooth", "seed_id"]])
    return pd.concat(rows, ignore_index=True)


def plot_training_curves(monitor_root: Path, out_dir: Path, smooth_window: int) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    algo_dirs = {"PPO": monitor_root / "ppo" / "monitor", "DQN": monitor_root / "dqn" / "monitor"}
    colors = {"PPO": "#0072B2", "DQN": "#D55E00"}
    for algo, monitor_dir in algo_dirs.items():
        files = _find_monitor_files(monitor_dir)
        if not files:
            continue
        runs = [_load_monitor(f) for f in files]
        aligned = _align_runs(runs, smooth_window)

        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for sid in sorted(aligned["seed_id"].unique()):
            sub = aligned[aligned["seed_id"] == sid]
            ax.plot(sub["episode"], sub["smooth"], color=colors[algo], alpha=0.25, linewidth=1.1)
        agg = aligned.groupby("episode", as_index=False).agg(mean_smooth=("smooth", "mean"), std_smooth=("smooth", "std")).fillna(0.0)
        ax.plot(agg["episode"], agg["mean_smooth"], color=colors[algo], linewidth=2.4, label=f"{algo} mean")
        ax.fill_between(agg["episode"], agg["mean_smooth"] - agg["std_smooth"], agg["mean_smooth"] + agg["std_smooth"], color=colors[algo], alpha=0.15)
        ax.set_title(f"{algo} training convergence")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Smoothed return")
        ax.legend()
        _save(fig, out_dir / f"4way_{algo.lower()}_convergence.png")


def plot_method_comparison(eval_summary: Path, out_dir: Path, seconds: int) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not eval_summary.exists():
        return
    df = pd.read_csv(eval_summary)
    if "throughput" not in df.columns:
        df["throughput"] = (df["total_arrived"] / float(seconds)) * 3600.0

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    fig.suptitle("4way single intersection: Rule-based vs PPO vs DQN", fontsize=14, fontweight="bold")
    metrics = [
        ("avg_travel_time", "Average travel time (s)"),
        ("mean_waiting_time", "Average waiting time (s)"),
        ("throughput", "Throughput (veh/h)"),
        ("collisions", "Collisions"),
    ]
    order = [m for m in ["Rule-based", "PPO", "DQN"] if m in set(df["method"].astype(str).tolist())]
    colors = {"Rule-based": "#6E6E6E", "PPO": "#0072B2", "DQN": "#D55E00"}
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        sub = df[df["method"].isin(order)].copy()
        sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
        sub = sub.sort_values("method")
        x = np.arange(len(sub))
        vals = sub[metric].to_numpy()
        methods = sub["method"].astype(str).tolist()
        ax.bar(x, vals, color=[colors[m] for m in methods], edgecolor="#333333", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    _save(fig, out_dir / "4way_method_comparison.png")


def plot_ablation(summary_csv: Path, out_dir: Path) -> None:
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    df["setting"] = df.apply(lambda r: f"{r['reward_mode']} / {r['obs_mode']}", axis=1)
    order = ["full / full", "full / phase_only", "default / full", "default / phase_only"]
    df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)
    df = df.sort_values("setting")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    fig.suptitle("4way PPO ablation study", fontsize=14, fontweight="bold")
    metrics = [
        ("avg_travel_time", "Average travel time (s)"),
        ("mean_waiting_time", "Average waiting time (s)"),
        ("throughput", "Throughput (veh/h)"),
        ("collisions", "Collisions"),
    ]
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        x = np.arange(len(df))
        y = df[metric].to_numpy()
        ax.bar(x, y, color=colors[: len(df)], edgecolor="#333333", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df["setting"].astype(str).tolist(), rotation=15)
        ax.set_ylabel(ylabel)
        for xi, yi in zip(x, y):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    _save(fig, out_dir / "4way_ablation_summary.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 4-way single-intersection figures from outputs.")
    p.add_argument("--monitor-root", type=str, default="logs/4way_single_intersection")
    p.add_argument("--eval-summary", type=str, default="outputs/4way-single-intersection/eval/eval_summary.csv")
    p.add_argument("--ablation-summary", type=str, default="outputs/4way-single-intersection/ablation/ablation_summary.csv")
    p.add_argument("--outdir", type=str, default="outputs/figures")
    p.add_argument("--seconds", type=int, default=1200)
    p.add_argument("--smooth", type=int, default=25)
    args = p.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(Path(args.monitor_root), out_dir, args.smooth)
    plot_method_comparison(Path(args.eval_summary), out_dir, args.seconds)
    plot_ablation(Path(args.ablation_summary), out_dir)

    print(f"Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()