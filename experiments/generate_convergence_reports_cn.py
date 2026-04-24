#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.25

METHOD_ORDER = ["idm", "fixed_speed", "yield", "ppo", "dqn"]
METHOD_CN = {
    "idm": "IDM",
    "fixed_speed": "固定速度",
    "yield": "礼让规则",
    "ppo": "PPO",
    "dqn": "DQN",
}
METHOD_COLOR = {
    "idm": "#6E6E6E",
    "fixed_speed": "#E69F00",
    "yield": "#009E73",
    "ppo": "#0072B2",
    "dqn": "#D55E00",
}


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=max(1, window // 3)).mean()


def _find_monitor_files(monitor_dir: Path) -> List[Path]:
    if not monitor_dir.exists():
        return []
    return sorted(monitor_dir.glob("*.monitor.csv"), key=lambda p: p.stat().st_mtime)


def _load_monitor(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"监控文件缺少 r 列: {file_path}")
    out = df.copy()
    out["episode"] = np.arange(1, len(out) + 1)
    return out


def _align_runs(runs: List[pd.DataFrame], smooth_window: int) -> pd.DataFrame:
    min_len = min(len(x) for x in runs)
    rows = []
    for idx, run in enumerate(runs, start=1):
        sub = run.iloc[:min_len].copy()
        sub["smooth"] = _rolling_mean(sub["r"], smooth_window)
        sub["seed_id"] = idx
        rows.append(sub[["episode", "r", "smooth", "seed_id"]])
    return pd.concat(rows, ignore_index=True)


def plot_single_algo_convergence(algo: str, runs_aligned: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for sid in sorted(runs_aligned["seed_id"].unique()):
        sub = runs_aligned[runs_aligned["seed_id"] == sid]
        ax.plot(sub["episode"], sub["smooth"], color=METHOD_COLOR[algo], alpha=0.25, linewidth=1.2)

    agg = (
        runs_aligned.groupby("episode", as_index=False)
        .agg(mean_smooth=("smooth", "mean"), std_smooth=("smooth", "std"))
        .fillna(0.0)
    )
    ax.plot(agg["episode"], agg["mean_smooth"], color=METHOD_COLOR[algo], linewidth=2.4, label=f"{METHOD_CN[algo]} 平均")
    ax.fill_between(
        agg["episode"],
        agg["mean_smooth"] - agg["std_smooth"],
        agg["mean_smooth"] + agg["std_smooth"],
        color=METHOD_COLOR[algo],
        alpha=0.15,
        label="均值±标准差",
    )

    ax.set_title(f"{METHOD_CN[algo]} 训练收敛曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("平滑回报")
    ax.legend()
    _save(fig, out_path)


def plot_ppo_dqn_compare(ppo_aligned: pd.DataFrame, dqn_aligned: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ppo = ppo_aligned.groupby("episode", as_index=False).agg(mean=("smooth", "mean")).rename(columns={"mean": "ppo"})
    dqn = dqn_aligned.groupby("episode", as_index=False).agg(mean=("smooth", "mean")).rename(columns={"mean": "dqn"})
    merged = ppo.merge(dqn, on="episode", how="inner")

    ax.plot(merged["episode"], merged["ppo"], color=METHOD_COLOR["ppo"], linewidth=2.4, label="PPO")
    ax.plot(merged["episode"], merged["dqn"], color=METHOD_COLOR["dqn"], linewidth=2.4, label="DQN")

    ppo_final = float(merged["ppo"].iloc[-1])
    dqn_final = float(merged["dqn"].iloc[-1])
    if abs(dqn_final) > 1e-12:
        delta = (ppo_final - dqn_final) / abs(dqn_final) * 100.0
        ax.text(
            0.03,
            0.95,
            f"末段收敛回报: PPO 相对 DQN {delta:+.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", alpha=0.95),
        )

    ax.set_title("PPO 与 DQN 训练收敛对比")
    ax.set_xlabel("Episode")
    ax.set_ylabel("平滑回报")
    ax.legend()
    _save(fig, out_path)


def summarize_convergence(algo: str, aligned: pd.DataFrame) -> Dict[str, float]:
    grouped = aligned.groupby("seed_id")["smooth"]
    finals = grouped.apply(lambda s: float(s.tail(30).mean())).to_numpy()
    bests = grouped.max().to_numpy()
    stabilities = grouped.apply(lambda s: float(s.tail(100).std(ddof=0))).to_numpy()

    return {
        "算法": METHOD_CN[algo],
        "曲线数量": float(len(finals)),
        "末段平均回报": float(np.mean(finals)),
        "末段平均回报标准差": float(np.std(finals, ddof=0)),
        "峰值平滑回报": float(np.mean(bests)),
        "末段波动": float(np.mean(stabilities)),
    }


def write_markdown_table(df: pd.DataFrame, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_method_cards(df: pd.DataFrame, out_path: Path) -> None:
    g = (
        df.groupby("method", as_index=False)
        .agg(
            throughput=("throughput", "mean"),
            avg_travel_time=("avg_travel_time", "mean"),
            min_ttc=("min_ttc", "mean"),
            mean_abs_jerk=("mean_abs_jerk", "mean"),
        )
    )
    g = g[g["method"].isin(METHOD_ORDER)].copy()
    g["method"] = pd.Categorical(g["method"], categories=METHOD_ORDER, ordered=True)
    g = g.sort_values("method")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    fig.suptitle("五类模型论文主指标对比", fontsize=13, fontweight="bold")
    metrics = [
        ("throughput", "吞吐量 (veh/h)", True, axes[0, 0]),
        ("avg_travel_time", "平均旅行时间 (s)", False, axes[0, 1]),
        ("min_ttc", "最小TTC (s)", True, axes[1, 0]),
        ("mean_abs_jerk", "平均绝对Jerk", False, axes[1, 1]),
    ]

    base = g[g["method"] == "idm"]
    base_vals = {m: float(base[m].iloc[0]) for m, _, _, _ in metrics} if not base.empty else {}

    for metric, ylabel, higher_better, ax in metrics:
        x = np.arange(len(g))
        vals = g[metric].to_numpy()
        methods = g["method"].astype(str).tolist()
        ax.bar(x, vals, color=[METHOD_COLOR[m] for m in methods], alpha=0.85, edgecolor="#333333", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_CN[m] for m in methods])
        ax.set_ylabel(ylabel)

        for xi, yi, m in zip(x, vals, methods):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
            if m == "idm" or metric not in base_vals or base_vals[metric] == 0:
                continue
            if higher_better:
                delta = (yi - base_vals[metric]) / base_vals[metric] * 100.0
            else:
                delta = (base_vals[metric] - yi) / base_vals[metric] * 100.0
            ax.text(xi, yi * 0.92, f"{delta:+.1f}%", ha="center", va="top", fontsize=8)

    _save(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="生成训练收敛图与模型论文表格")
    p.add_argument("--ppo-monitor-dir", type=str, default="logs/repro/ppo/monitor")
    p.add_argument("--dqn-monitor-dir", type=str, default="logs/repro/dqn/monitor")
    p.add_argument("--matrix-csv", type=str, default="outputs/figures_repro/all_runs_matrix.csv")
    p.add_argument("--out-fig-dir", type=str, default="outputs/figures_repro")
    p.add_argument("--out-report-dir", type=str, default="outputs/reports_repro")
    p.add_argument("--smooth", type=int, default=25)
    args = p.parse_args()

    out_fig = Path(args.out_fig_dir)
    out_rep = Path(args.out_report_dir)
    out_fig.mkdir(parents=True, exist_ok=True)
    out_rep.mkdir(parents=True, exist_ok=True)

    ppo_files = _find_monitor_files(Path(args.ppo_monitor_dir))
    dqn_files = _find_monitor_files(Path(args.dqn_monitor_dir))
    if not ppo_files or not dqn_files:
        raise FileNotFoundError("未找到 PPO 或 DQN 监控日志，请检查 monitor 目录")

    ppo_runs = [_load_monitor(f) for f in ppo_files]
    dqn_runs = [_load_monitor(f) for f in dqn_files]
    ppo_aligned = _align_runs(ppo_runs, args.smooth)
    dqn_aligned = _align_runs(dqn_runs, args.smooth)

    plot_single_algo_convergence("ppo", ppo_aligned, out_fig / "图19_PPO训练收敛曲线.png")
    plot_single_algo_convergence("dqn", dqn_aligned, out_fig / "图20_DQN训练收敛曲线.png")
    plot_ppo_dqn_compare(ppo_aligned, dqn_aligned, out_fig / "图21_PPO_DQN收敛对比.png")

    conv_rows = [summarize_convergence("ppo", ppo_aligned), summarize_convergence("dqn", dqn_aligned)]
    conv_df = pd.DataFrame(conv_rows)
    conv_csv = out_rep / "训练收敛汇总表.csv"
    conv_md = out_rep / "训练收敛汇总表.md"
    conv_df.to_csv(conv_csv, index=False, encoding="utf-8-sig")
    write_markdown_table(conv_df, conv_md)

    matrix_df = pd.read_csv(Path(args.matrix_csv))
    matrix_df["method"] = matrix_df["method"].str.lower()
    matrix_df = matrix_df[matrix_df["method"].isin(METHOD_ORDER)].copy()
    plot_method_cards(matrix_df, out_fig / "图22_五模型主指标对比.png")

    model_tbl = (
        matrix_df.groupby("method", as_index=False)
        .agg(
            平均旅行时间=("avg_travel_time", "mean"),
            平均等待时间=("mean_waiting_time", "mean"),
            吞吐量=("throughput", "mean"),
            最小TTC=("min_ttc", "mean"),
            急减速率=("harsh_brake_rate", "mean"),
            平均绝对Jerk=("mean_abs_jerk", "mean"),
            等待公平性Gini=("gini_waiting_time", "mean"),
        )
        .sort_values(by="method")
    )
    model_tbl["method"] = model_tbl["method"].map(METHOD_CN)
    model_tbl = model_tbl.rename(columns={"method": "模型"})
    model_csv = out_rep / "五模型指标汇总表.csv"
    model_md = out_rep / "五模型指标汇总表.md"
    model_tbl.to_csv(model_csv, index=False, encoding="utf-8-sig")
    write_markdown_table(model_tbl, model_md)

    print(f"Wrote: {out_fig / '图19_PPO训练收敛曲线.png'}")
    print(f"Wrote: {out_fig / '图20_DQN训练收敛曲线.png'}")
    print(f"Wrote: {out_fig / '图21_PPO_DQN收敛对比.png'}")
    print(f"Wrote: {out_fig / '图22_五模型主指标对比.png'}")
    print(f"Wrote: {conv_csv}")
    print(f"Wrote: {model_csv}")


if __name__ == "__main__":
    main()
