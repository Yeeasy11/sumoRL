import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from plot_font_cn import configure_chinese_font


configure_chinese_font()


def load_all_runs(root: Path) -> pd.DataFrame:
    files = list(root.glob("flow*_*/*/eval_runs.csv"))
    files += list(root.glob("2way-single-intersection/ablation/*/eval_runs.csv"))
    files += list(root.glob("2way-single-intersection/ablation_legacy/*/eval_runs.csv"))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No eval_runs.csv under {root}")
    frames: List[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)

        tag = f.parent.parent.name  # e.g., flow600_poisson or ablation
        if tag.startswith("flow") and "_" in tag:
            flow_str, dist = tag.split("_", 1)
            flow = int(flow_str.replace("flow", ""))
            df["flow"] = flow
            df["dist"] = dist
        else:
            # Ablation runs do not follow flow{n}_{dist} naming. Normalize into matrix schema.
            if "flow" not in df.columns:
                df["flow"] = 600
            if "dist" not in df.columns:
                df["dist"] = "ablation_2way"

        if "method" not in df.columns:
            if {"reward_mode", "obs_mode"}.issubset(df.columns):
                df["method"] = df.apply(
                    lambda r: f"ablation_{str(r['reward_mode'])}_{str(r['obs_mode'])}",
                    axis=1,
                )
            else:
                df["method"] = "unknown_method"

        if "flow_ns" not in df.columns:
            df["flow_ns"] = (pd.to_numeric(df["flow"], errors="coerce").fillna(600).astype(int) // 2)
        if "flow_we" not in df.columns:
            df["flow_we"] = (pd.to_numeric(df["flow"], errors="coerce").fillna(600).astype(int) // 2)

        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "reward_mode" in out.columns:
        out = out[out["reward_mode"].astype(str) != "no_collision"].copy()
    out = out[~out["method"].astype(str).str.contains("no_collision", na=False)].copy()
    out["throughput"] = out["total_arrived"] / 600.0 * 3600.0
    return out


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_method_by_flow(df: pd.DataFrame, outdir: Path) -> None:
    agg = (
        df.groupby(["flow", "method"], as_index=False)
        .agg(
            平均旅行时间=("avg_travel_time", "mean"),
            平均等待时间=("mean_waiting_time", "mean"),
            吞吐量=("throughput", "mean"),
            碰撞次数=("collisions", "mean"),
        )
    )
    methods = [m for m in ["Rule-based", "PPO", "DQN"] if m in agg["method"].unique()]
    method_cn = {
        "Rule-based": "规则基线",
        "PPO": "PPO",
        "DQN": "DQN",
        "idm": "IDM默认模型",
        "fixed_speed": "固定目标速度",
        "yield": "礼让规则",
    }
    for extra in ["idm", "fixed_speed", "yield"]:
        if extra in agg["method"].unique() and extra not in methods:
            methods.append(extra)
    colors = {"Rule-based": "#F39C12", "PPO": "#2ECC71", "DQN": "#3498DB"}
    colors.update({"idm": "#9B59B6", "fixed_speed": "#E74C3C", "yield": "#1ABC9C"})

    for metric, ylabel, fname in [
        ("平均旅行时间", "平均旅行时间 (s)", "图1_方法对比_平均旅行时间.png"),
        ("平均等待时间", "平均等待时间 (s)", "图2_方法对比_平均等待时间.png"),
        ("吞吐量", "吞吐量 (veh/h)", "图3_方法对比_吞吐量.png"),
        ("碰撞次数", "碰撞次数", "图4_方法对比_碰撞次数.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for m in methods:
            sub = agg[agg["method"] == m].sort_values(by="flow")
            ax.plot(sub["flow"], sub[metric], marker="o", linewidth=2, label=method_cn.get(m, m), color=colors[m])
        ax.set_title(f"不同流量下方法对比：{metric}")
        ax.set_xlabel("交通流量 (veh/h)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(title="方法")
        _save(fig, outdir / fname)


def plot_distribution_impact(df: pd.DataFrame, outdir: Path) -> None:
    agg = (
        df.groupby(["dist", "method"], as_index=False)
        .agg(
            平均旅行时间=("avg_travel_time", "mean"),
            平均等待时间=("mean_waiting_time", "mean"),
            吞吐量=("throughput", "mean"),
            碰撞次数=("collisions", "mean"),
        )
    )
    methods = [m for m in ["Rule-based", "PPO", "DQN", "idm", "fixed_speed", "yield"] if m in agg["method"].unique()]
    method_cn = {
        "Rule-based": "规则基线",
        "PPO": "PPO",
        "DQN": "DQN",
        "idm": "IDM默认模型",
        "fixed_speed": "固定目标速度",
        "yield": "礼让规则",
    }
    dist_order = [d for d in ["uniform", "poisson", "burst"] if d in agg["dist"].unique()]
    dist_cn = {"uniform": "均匀", "poisson": "泊松", "burst": "突发"}

    for metric, ylabel, fname in [
        ("平均旅行时间", "平均旅行时间 (s)", "图5_到达分布影响_平均旅行时间.png"),
        ("吞吐量", "吞吐量 (veh/h)", "图6_到达分布影响_吞吐量.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = range(len(dist_order))
        width = 0.12
        for i, m in enumerate(methods):
            sub = agg[agg["method"] == m].set_index("dist").reindex(dist_order)
            y = sub[metric].values
            ax.bar([xi + (i - (len(methods) - 1) / 2.0) * width for xi in x], y, width=width, label=method_cn.get(m, m))
        ax.set_xticks(list(x))
        ax.set_xticklabels([dist_cn.get(d, d) for d in dist_order])
        ax.set_xlabel("到达分布")
        ax.set_ylabel(ylabel)
        ax.set_title(f"不同到达分布下方法对比：{metric}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="方法")
        _save(fig, outdir / fname)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="从实验矩阵评估结果生成中文图例与图注图表。",
    )
    p.add_argument("--eval-root", type=str, default="outputs/eval_repro")
    p.add_argument("--outdir", type=str, default="outputs/figures_repro")
    args = p.parse_args()

    df = load_all_runs(Path(args.eval_root))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_method_by_flow(df, outdir)
    plot_distribution_impact(df, outdir)

    # New safety/comfort/collaboration plots from vehicle-level metrics
    if all(c in df.columns for c in ["min_ttc", "harsh_brake_rate", "mean_abs_jerk", "gini_waiting_time"]):
        agg2 = (
            df.groupby(["flow", "method"], as_index=False)
            .agg(
                最小TTC=("min_ttc", "mean"),
                急减速率=("harsh_brake_rate", "mean"),
                平均绝对Jerk=("mean_abs_jerk", "mean"),
                等待公平性Gini=("gini_waiting_time", "mean"),
            )
        )
        method_cn = {
            "Rule-based": "规则基线",
            "PPO": "PPO",
            "DQN": "DQN",
            "idm": "IDM默认模型",
            "fixed_speed": "固定目标速度",
            "yield": "礼让规则",
        }
        colors = {
            "Rule-based": "#F39C12",
            "PPO": "#2ECC71",
            "DQN": "#3498DB",
            "idm": "#9B59B6",
            "fixed_speed": "#E74C3C",
            "yield": "#1ABC9C",
        }
        methods = [m for m in ["idm", "fixed_speed", "yield", "PPO", "DQN", "Rule-based"] if m in agg2["method"].unique()]

        for metric, ylabel, fname, title in [
            ("最小TTC", "最小TTC (s)", "图7_安全性_最小TTC.png", "安全性对比：最小TTC（越大越安全）"),
            ("急减速率", "急减速率", "图8_安全性_急减速率.png", "安全性对比：急减速率（越低越安全）"),
            ("平均绝对Jerk", "平均绝对Jerk", "图9_舒适性_jerk.png", "舒适性对比：平均绝对Jerk（越低越舒适）"),
            ("等待公平性Gini", "等待公平性Gini", "图10_协同性_Gini.png", "协同性对比：等待公平性Gini（越低越公平）"),
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            for m in methods:
                sub = agg2[agg2["method"] == m].sort_values(by="flow")
                ax.plot(sub["flow"], sub[metric], marker="o", linewidth=2, label=method_cn.get(m, m), color=colors[m])
            ax.set_title(title)
            ax.set_xlabel("交通流量 (veh/h)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(title="方法")
            _save(fig, outdir / fname)

    df.to_csv(outdir / "all_runs_matrix.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote figures to: {outdir}")


if __name__ == "__main__":
    main()
