#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=max(1, window // 3)).mean()


def _collect_latest_monitors_by_seed(monitor_dir: Path) -> list[Path]:
    pattern = re.compile(r"seed(\d+)")
    latest: dict[int, Path] = {}
    for p in sorted(monitor_dir.glob("*.monitor.csv"), key=lambda x: x.stat().st_mtime):
        m = pattern.search(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        latest[seed] = p
    return [latest[k] for k in sorted(latest.keys())]


def _align_runs(runs: list[pd.DataFrame], smooth_window: int) -> pd.DataFrame:
    min_len = min(len(x) for x in runs)
    rows = []
    for idx, run in enumerate(runs, start=1):
        sub = run.iloc[:min_len].copy()
        sub["smooth"] = _rolling_mean(sub["r"], smooth_window)
        sub["seed_id"] = idx
        rows.append(sub[["episode", "smooth", "seed_id"]])
    return pd.concat(rows, ignore_index=True)


def _load_monitor(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"监控文件缺少 r 列: {path}")
    out = df.copy()
    out["episode"] = np.arange(1, len(out) + 1)
    return out


def plot_figure19(ppo_monitor_dir: Path, dqn_monitor_dir: Path, out_path: Path, smooth: int) -> None:
    ppo_files = _collect_latest_monitors_by_seed(ppo_monitor_dir)
    dqn_files = _collect_latest_monitors_by_seed(dqn_monitor_dir)
    if len(ppo_files) < 3 or len(dqn_files) < 3:
        raise FileNotFoundError("PPO/DQN 监控文件不足3个随机种子，无法绘制图19。")

    ppo_runs = [_load_monitor(f) for f in ppo_files]
    dqn_runs = [_load_monitor(f) for f in dqn_files]
    ppo_aligned = _align_runs(ppo_runs, smooth)
    dqn_aligned = _align_runs(dqn_runs, smooth)

    ppo = ppo_aligned.groupby("episode", as_index=False).agg(mean=("smooth", "mean"), std=("smooth", "std")).fillna(0.0)
    dqn = dqn_aligned.groupby("episode", as_index=False).agg(mean=("smooth", "mean"), std=("smooth", "std")).fillna(0.0)
    merged = ppo.merge(dqn, on="episode", how="inner", suffixes=("_ppo", "_dqn"))

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(merged["episode"], merged["mean_ppo"], color="#0072B2", linewidth=2.5, label="PPO（三随机种子均值）")
    ax.fill_between(
        merged["episode"],
        merged["mean_ppo"] - merged["std_ppo"],
        merged["mean_ppo"] + merged["std_ppo"],
        color="#0072B2",
        alpha=0.16,
        label="PPO 均值±标准差",
    )

    ax.plot(merged["episode"], merged["mean_dqn"], color="#D55E00", linewidth=2.5, label="DQN（三随机种子均值）")
    ax.fill_between(
        merged["episode"],
        merged["mean_dqn"] - merged["std_dqn"],
        merged["mean_dqn"] + merged["std_dqn"],
        color="#D55E00",
        alpha=0.16,
        label="DQN 均值±标准差",
    )

    ppo_final = float(merged["mean_ppo"].iloc[-1])
    dqn_final = float(merged["mean_dqn"].iloc[-1])
    if abs(dqn_final) > 1e-12:
        delta = (ppo_final - dqn_final) / abs(dqn_final) * 100.0
        ax.text(
            0.03,
            0.95,
            f"末段收敛回报：PPO 相对 DQN {delta:+.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", alpha=0.95),
        )

    ax.set_title("PPO 与 DQN 训练收敛曲线（3随机种子）")
    ax.set_xlabel("Episode")
    ax.set_ylabel("平滑回报")
    ax.legend(loc="best", fontsize=8.5)
    _save(fig, out_path)


def plot_figure20(ablation_summary_csv: Path, out_path: Path) -> None:
    if not ablation_summary_csv.exists():
        raise FileNotFoundError(f"未找到消融汇总文件: {ablation_summary_csv}")

    df = pd.read_csv(ablation_summary_csv)
    if df.empty:
        raise ValueError("消融汇总文件为空，无法绘图。")

    setting_cn = {
        ("full", "full"): "完整奖励+完整状态",
        ("full", "phase_only"): "完整奖励+相位状态",
        ("default", "full"): "默认奖励+完整状态",
        ("default", "phase_only"): "默认奖励+相位状态",
    }
    df["setting"] = df.apply(lambda r: setting_cn.get((str(r["reward_mode"]), str(r["obs_mode"])), f"{r['reward_mode']}+{r['obs_mode']}"), axis=1)

    order = [
        "完整奖励+完整状态",
        "完整奖励+相位状态",
        "默认奖励+完整状态",
        "默认奖励+相位状态",
    ]
    df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)
    df = df.sort_values("setting")

    colors = {
        "完整奖励+完整状态": "#0072B2",
        "完整奖励+相位状态": "#56B4E9",
        "默认奖励+完整状态": "#D55E00",
        "默认奖励+相位状态": "#E69F00",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fig.suptitle("消融实验总览（四设置对比）", fontsize=14, fontweight="bold")

    metrics = [
        ("avg_travel_time", "平均旅行时间 (s)"),
        ("mean_waiting_time", "平均等待时间 (s)"),
        ("throughput", "吞吐量 (veh/h)"),
        ("collisions", "碰撞次数"),
    ]

    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        x = np.arange(len(df))
        y = df[metric].to_numpy()
        bar_colors = [colors[str(s)] for s in df["setting"].astype(str).tolist()]
        ax.bar(x, y, color=bar_colors, edgecolor="#333333", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df["setting"].astype(str).tolist(), rotation=18)
        ax.set_ylabel(ylabel)
        for xi, yi in zip(x, y):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[k], edgecolor="#333333", label=k)
        for k in order
        if k in set(df["setting"].astype(str).tolist())
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.97), title="消融设置")
    fig.subplots_adjust(top=0.88)
    _save(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser(description="按指定命名导出图11/12/13/16/17/18/19/20到新文件夹。")
    p.add_argument("--matrix-csv", type=str, default="outputs/all_runs_matrix.csv")
    p.add_argument("--ppo-monitor-dir", type=str, default="logs/4way_single_intersection/ppo/monitor")
    p.add_argument("--dqn-monitor-dir", type=str, default="logs/4way_single_intersection/dqn/monitor")
    p.add_argument("--ablation-summary", type=str, default="outputs/4way-single-intersection/ablation/ablation_summary.csv")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--smooth", type=int, default=25)
    p.add_argument("--skip-ttc", action="store_true", default=True)
    args = p.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"figures_random3_cn_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "experiments/generate_thesis_figures.py",
        "--matrix-csv",
        str(args.matrix_csv),
        "--out-dir",
        str(out_dir),
        "--skip-ttc",
    ]
    subprocess.run(cmd, check=True)

    plot_figure19(
        ppo_monitor_dir=Path(args.ppo_monitor_dir),
        dqn_monitor_dir=Path(args.dqn_monitor_dir),
        out_path=out_dir / "图19_PPO与dqn训练收敛曲线.png",
        smooth=args.smooth,
    )

    plot_figure20(
        ablation_summary_csv=Path(args.ablation_summary),
        out_path=out_dir / "图20消融实验总览图.png",
    )

    print(f"图像已输出到: {out_dir}")


if __name__ == "__main__":
    main()
