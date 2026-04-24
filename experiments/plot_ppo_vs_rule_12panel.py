#!/usr/bin/env python3
"""Generate 12-panel (4x3) trajectory and TTC figures comparing PPO vs Rule-based only."""

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _init_cn_font() -> None:
    windows_font_paths = [
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ]
    for font_path in windows_font_paths:
        if os.path.exists(font_path):
            try:
                prop = fm.FontProperties(fname=font_path)
                matplotlib.rcParams["font.sans-serif"] = [prop.get_name()]
                matplotlib.rcParams["font.family"] = prop.get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


_init_cn_font()

METHOD_COLORS = {
    "Rule-based": "#E69F00",
    "PPO": "#0072B2",
}
METHOD_LABELS = {
    "Rule-based": "IDM基线",
    "PPO": "PPO",
}
DIR_CN = {"n": "北向", "e": "东向", "s": "南向", "w": "西向"}
TURN_CN = {"left": "左转", "straight": "直行", "right": "右转"}
DIR_ORDER = ["n", "e", "s", "w"]
TURN_ORDER = ["left", "straight", "right"]


def _scatter_subplot(ax, sub, turn):
    methods = ["Rule-based", "PPO"]
    for method in methods:
        msub = sub[sub["method"] == method].copy()
        if msub.empty:
            continue
        if turn == "left":
            marker_size = 2.0
            marker_alpha = 0.20
            near_stop = msub[msub["s_from_stopline"].abs() < 12.0]
            far_stop = msub[msub["s_from_stopline"].abs() >= 12.0]
            if len(near_stop) > 2000:
                near_stop = near_stop.sample(2000, random_state=7)
            msub = pd.concat([near_stop, far_stop], ignore_index=True)
        else:
            marker_size = 2.5
            marker_alpha = 0.16
            if len(msub) > 14000:
                msub = msub.sample(14000, random_state=7)
        ax.scatter(
            msub["time"],
            msub["s_from_stopline"],
            s=marker_size,
            alpha=marker_alpha,
            c=METHOD_COLORS[method],
            edgecolors="none",
            label=METHOD_LABELS[method],
            rasterized=True,
        )


def plot_trajectory_12panel(traj: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 16.5), sharex=True, sharey=True)

    for r, in_dir in enumerate(DIR_ORDER):
        for c, turn in enumerate(TURN_ORDER):
            ax = axes[r, c]
            sub = traj[(traj["in_dir"] == in_dir) & (traj["turn"] == turn)]
            _scatter_subplot(ax, sub, turn)

            if sub.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=9)

            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]}", fontsize=10)
            if r == 3:
                ax.set_xlabel("时间 (s)", fontsize=9)
            if c == 0:
                ax.set_ylabel("相对停止线位置 (m)", fontsize=9)
            ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=0.9)
            ax.grid(alpha=0.20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=11, markerscale=6)
    fig.suptitle("轨迹时空图 4×3 全流向汇总（PPO 与 IDM基线）", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.94])
    fig.savefig(outdir / "图21_PPO与IDM基线_轨迹时空图_4x3汇总.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir / '图21_PPO与IDM基线_轨迹时空图_4x3汇总.png'}")


def plot_trajectory_by_direction(traj: pd.DataFrame, outdir: Path) -> None:
    for in_dir in DIR_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharex=True, sharey=True)
        for ax, turn in zip(axes, TURN_ORDER):
            sub = traj[(traj["in_dir"] == in_dir) & (traj["turn"] == turn)]
            _scatter_subplot(ax, sub, turn)
            if sub.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]}")
            ax.set_xlabel("时间 (s)")
            ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=0.9)
            ax.grid(alpha=0.20)
        axes[0].set_ylabel("相对停止线位置 (m)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.85), markerscale=5)
        fig.suptitle(f"轨迹时空图：{DIR_CN[in_dir]}进口三流向（PPO 与 IDM基线）", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.78])
        fig.savefig(outdir / f"图21_PPO与IDM基线_轨迹时空图_{DIR_CN[in_dir]}三流向.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved triple panel: {outdir / f'图21_PPO与IDM基线_轨迹时空图_{DIR_CN[in_dir]}三流向.png'}")


def plot_ttc_by_direction(ttc: pd.DataFrame, outdir: Path) -> None:
    methods = ["Rule-based", "PPO"]
    d = ttc[np.isfinite(ttc["ttc"]) & (ttc["ttc"] > 0) & (ttc["ttc"] <= 20.0)].copy()
    for in_dir in DIR_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharex=True, sharey=True)
        for ax, turn in zip(axes, TURN_ORDER):
            sub = d[(d["in_dir"] == in_dir) & (d["turn"] == turn)]
            for method in methods:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                s = msub.groupby("time", as_index=False)["ttc"].min().sort_values("time")
                s["ttc_smooth"] = s["ttc"].rolling(window=9, min_periods=1).mean()
                ax.plot(s["time"], s["ttc_smooth"], color=METHOD_COLORS[method], linewidth=1.6, alpha=0.9, label=METHOD_LABELS[method])
            ax.axhline(3.0, color="#DC2626", linestyle="--", linewidth=0.9, alpha=0.8)
            if sub.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=10)
            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]}")
            ax.set_xlabel("时间 (s)")
            ax.set_ylim(0, 20)
            ax.grid(alpha=0.20)
        axes[0].set_ylabel("最小TTC (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.85), markerscale=4)
        fig.suptitle(f"TTC时序图：{DIR_CN[in_dir]}进口三流向（PPO 与 IDM基线）", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.78])
        fig.savefig(outdir / f"图22_PPO与IDM基线_TTC时序图_{DIR_CN[in_dir]}三流向.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved triple panel: {outdir / f'图22_PPO与IDM基线_TTC时序图_{DIR_CN[in_dir]}三流向.png'}")


def plot_ttc_12panel(ttc: pd.DataFrame, outdir: Path) -> None:
    methods = ["Rule-based", "PPO"]
    d = ttc[np.isfinite(ttc["ttc"]) & (ttc["ttc"] > 0) & (ttc["ttc"] <= 20.0)].copy()
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 16.5), sharex=True, sharey=True)

    for r, in_dir in enumerate(DIR_ORDER):
        for c, turn in enumerate(TURN_ORDER):
            ax = axes[r, c]
            sub = d[(d["in_dir"] == in_dir) & (d["turn"] == turn)]

            for method in methods:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                s = msub.groupby("time", as_index=False)["ttc"].min().sort_values("time")
                s["ttc_smooth"] = s["ttc"].rolling(window=9, min_periods=1).mean()
                ax.plot(s["time"], s["ttc_smooth"], color=METHOD_COLORS[method], linewidth=1.4, alpha=0.9, label=METHOD_LABELS[method])

            if sub.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=9)

            ax.axhline(3.0, color="#DC2626", linestyle="--", linewidth=0.9, alpha=0.8)
            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]}", fontsize=10)
            if r == 3:
                ax.set_xlabel("时间 (s)", fontsize=9)
            if c == 0:
                ax.set_ylabel("最小TTC (s)", fontsize=9)
            ax.set_ylim(0, 20)
            ax.grid(alpha=0.20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=11)
    fig.suptitle("TTC时序图 4×3 全流向汇总（PPO 与 IDM基线）", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.94])
    fig.savefig(outdir / "图22_PPO与IDM基线_TTC时序图_4x3汇总.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir / '图22_PPO与IDM基线_TTC时序图_4x3汇总.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 12-panel PPO vs Rule-based trajectory/TTC figures.")
    p.add_argument("--traj-csv", type=str, default="outputs/figures/trajectory_samples_by_flow12.csv")
    p.add_argument("--ttc-csv", type=str, default="outputs/figures/ttc_samples_by_flow12.csv")
    p.add_argument("--outdir", type=str, default="outputs/figures")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    traj_df = pd.read_csv(args.traj_csv)
    ttc_df = pd.read_csv(args.ttc_csv)

    # Filter to PPO and Rule-based only
    traj_sub = traj_df[traj_df["method"].isin(["Rule-based", "PPO"])].copy()
    ttc_sub = ttc_df[ttc_df["method"].isin(["Rule-based", "PPO"])].copy()

    plot_trajectory_12panel(traj_sub, outdir)
    plot_ttc_12panel(ttc_sub, outdir)
    plot_trajectory_by_direction(traj_sub, outdir)
    plot_ttc_by_direction(ttc_sub, outdir)

    print("Done.")


if __name__ == "__main__":
    main()
