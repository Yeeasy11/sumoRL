#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional
import matplotlib.font_manager as fm
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib import rcParams


# 强制使用支持中文的字体
def _init_fonts():
    """Initialize Chinese font support."""
    # Try to find a Chinese font that's actually available
    font_names = ["SimHei", "Microsoft YaHei", "SimSun", "FangSong", "KaiTi"]
    
    for fname in font_names:
        try:
            # Try to load the font
            prop = fm.FontProperties(family=fname)
            # Verify it actually works
            test_path = fm.findfont(prop)
            if test_path and "DejaVu" not in test_path:
                rcParams["font.sans-serif"] = [fname]
                rcParams["font.family"] = "sans-serif"
                print(f"Loaded font: {fname} from {test_path}", flush=True)
                return
        except Exception as e:
            pass
    
    # If no Chinese font found, try to use a CJK font
    try:
        rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
    except:
        pass


_init_fonts()
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 10
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.25
rcParams["axes.facecolor"] = "#FFFFFF"
rcParams["figure.facecolor"] = "#FFFFFF"
rcParams["axes.edgecolor"] = "#333333"
rcParams["axes.linewidth"] = 0.8
rcParams["grid.color"] = "#D1D5DB"
rcParams["figure.autolayout"] = True
rcParams["axes.labelsize"] = 11
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10

METHOD_ORDER = ["idm", "fixed_speed", "yield", "ppo", "dqn"]
METHOD_CN = {
    "idm": "IDM基线",
    "fixed_speed": "固定速度规则",
    "yield": "礼让规则",
    "ppo": "PPO方法",
    "dqn": "DQN方法",
}
METHOD_COLOR = {
    "idm": "#6E6E6E",
    "fixed_speed": "#E69F00",
    "yield": "#009E73",
    "ppo": "#0072B2",
    "dqn": "#D55E00",
}
DIST_ORDER = ["uniform", "poisson", "burst", "balanced"]
DIST_CN = {"uniform": "均匀", "poisson": "泊松", "burst": "突发", "balanced": "平衡"}


def _save(fig: Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use tight_layout with padding to prevent label overlap
    fig.tight_layout(pad=1.5)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def load_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到评估矩阵: {path}")
    df = pd.read_csv(path)
    if "method" not in df.columns:
        raise ValueError("all_runs_matrix.csv 缺少 method 列")
    df = df.copy()
    df["method"] = df["method"].str.lower()
    df["method"] = df["method"].replace({"rule-based": "idm", "rule_based": "idm"})

    required_metrics = [
        "avg_travel_time",
        "mean_waiting_time",
        "mean_speed",
        "throughput",
        "collisions",
        "min_ttc",
        "harsh_brake_rate",
        "mean_abs_jerk",
        "gini_waiting_time",
    ]
    for c in required_metrics:
        if c not in df.columns:
            df[c] = np.nan

    if "flow" not in df.columns:
        df["flow"] = np.nan
    if {"flow_n", "flow_e", "flow_s", "flow_w"}.issubset(df.columns):
        flow_from_cardinal = (
            pd.to_numeric(df["flow_n"], errors="coerce")
            + pd.to_numeric(df["flow_e"], errors="coerce")
            + pd.to_numeric(df["flow_s"], errors="coerce")
            + pd.to_numeric(df["flow_w"], errors="coerce")
        ) / 4.0
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(flow_from_cardinal)
    elif {"flow_ns", "flow_we"}.issubset(df.columns):
        flow_from_nswe = (
            pd.to_numeric(df["flow_ns"], errors="coerce")
            + pd.to_numeric(df["flow_we"], errors="coerce")
        ) / 2.0
        df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(flow_from_nswe)
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce").fillna(600)

    if "dist" not in df.columns:
        df["dist"] = np.nan
    if "arrival_dist" in df.columns:
        dist_from_arrival = df["arrival_dist"].astype(str).str.lower().replace({"nan": np.nan})
        df["dist"] = df["dist"].astype(str).str.lower().replace({"nan": np.nan}).fillna(dist_from_arrival)
    else:
        df["dist"] = df["dist"].astype(str).str.lower().replace({"nan": np.nan})
    df["dist"] = df["dist"].fillna("balanced")

    df = df[df["method"].isin(METHOD_ORDER)]
    return df


def _main_eval_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "experiment_group" in out.columns:
        out = out[out["experiment_group"] == "main_eval"]
    out = out[out["dist"].isin(["uniform", "poisson", "burst"])].copy()
    out["flow"] = pd.to_numeric(out["flow"], errors="coerce")
    out = out[out["flow"].notna()]
    return out


def _bar_panel(
    ax: Axes,
    g: pd.DataFrame,
    metric: str,
    ylabel: str,
    better: str,
    higher_is_better: bool,
    baseline_method: str = "idm",
) -> None:
    sub = g[g["method"].isin(METHOD_ORDER)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
    sub = sub.iloc[np.argsort(sub["method"].cat.codes)]
    x = np.arange(len(sub))
    means = sub[(metric, "mean")].to_numpy()
    stds = sub[(metric, "std")].fillna(0.0).to_numpy()
    means_clean = np.nan_to_num(means, nan=0.0)
    labels = [METHOD_CN[m] for m in sub["method"].astype(str)]
    colors = [METHOD_COLOR[m] for m in sub["method"].astype(str)]

    ax.bar(x, means_clean, yerr=stds, capsize=4, color=colors, alpha=0.88, edgecolor="#333333", linewidth=1)
    ax.set_xticks(x)
    # Rotate labels and use ha='right' to prevent overlap
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(better, fontsize=12, fontweight='bold')
    
    # Adjust y-axis to have more space for values
    if np.isfinite(means).any():
        y_max = float(np.nanmax(means) + np.nanmax(stds))
    else:
        y_max = 1.0
    ax.set_ylim(0, y_max * 1.15)
    
    baseline_val = np.nan
    if baseline_method in sub["method"].astype(str).tolist():
        baseline_val = float(sub[sub["method"] == baseline_method][(metric, "mean")].iloc[0])

    for xi, yi, yi_clean, m in zip(x, means, means_clean, sub["method"].astype(str).tolist()):
        if np.isfinite(yi):
            ax.text(xi, yi_clean, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(xi, yi_clean + 0.02 * y_max, "NA", ha="center", va="bottom", fontsize=8, color="#6B7280")
        if m == baseline_method or not np.isfinite(baseline_val) or baseline_val == 0:
            continue
        if higher_is_better:
            delta = (yi - baseline_val) / baseline_val * 100.0
        else:
            delta = (baseline_val - yi) / baseline_val * 100.0
        sign = "+" if delta >= 0 else ""
        ax.text(xi, yi * 0.92, f"{sign}{delta:.1f}%", ha="center", va="top", fontsize=8, color="#111827")


def plot_overview_panels(df: pd.DataFrame, outdir: Path) -> None:
    g = df.groupby("method", as_index=False).agg(
        {
            "avg_travel_time": ["mean", "std"],
            "mean_waiting_time": ["mean", "std"],
            "throughput": ["mean", "std"],
            "mean_speed": ["mean", "std"],
            "min_ttc": ["mean", "std"],
            "harsh_brake_rate": ["mean", "std"],
            "mean_abs_jerk": ["mean", "std"],
            "gini_waiting_time": ["mean", "std"],
        }
    )

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8.8))
    fig1.suptitle("综合效率指标对比（全场景平均）", fontsize=13, fontweight="bold")
    _bar_panel(axes1[0, 0], g, "avg_travel_time", "平均旅行时间 (s)", "越低越好", higher_is_better=False)
    _bar_panel(axes1[0, 1], g, "mean_waiting_time", "平均等待时间 (s)", "越低越好", higher_is_better=False)
    _bar_panel(axes1[1, 0], g, "throughput", "吞吐量 (veh/h)", "越高越好", higher_is_better=True)
    _bar_panel(axes1[1, 1], g, "mean_speed", "平均速度 (m/s)", "越高越好", higher_is_better=True)
    _save(fig1, outdir / "图11_综合效率四联图.png")

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8.8))
    fig2.suptitle("安全-舒适-协同指标对比（全场景平均）", fontsize=13, fontweight="bold")
    _bar_panel(axes2[0, 0], g, "min_ttc", "最小TTC (s)", "越高越好", higher_is_better=True)
    _bar_panel(axes2[0, 1], g, "harsh_brake_rate", "急减速率", "越低越好", higher_is_better=False)
    _bar_panel(axes2[1, 0], g, "mean_abs_jerk", "平均绝对Jerk", "越低越好", higher_is_better=False)
    _bar_panel(axes2[1, 1], g, "gini_waiting_time", "等待公平性Gini", "越低越好", higher_is_better=False)
    _save(fig2, outdir / "图12_安全舒适协同四联图.png")


def _heat_matrix(df: pd.DataFrame, metric: str, mode: str) -> tuple[np.ndarray, list[str], list[str]]:
    flows = sorted(df["flow"].dropna().astype(int).unique().tolist())
    dists = [d for d in DIST_ORDER if d in set(df["dist"].dropna().astype(str).tolist())]
    rows: list[list[float]] = []

    for f in flows:
        row: list[float] = []
        for d in dists:
            sub = df[(df["flow"] == f) & (df["dist"] == d)]
            ppo = sub[sub["method"] == "ppo"][metric].mean()
            base = sub[sub["method"] == "idm"][metric].mean()
            if pd.isna(ppo) or pd.isna(base) or base == 0:
                row.append(np.nan)
                continue
            if mode == "lower_is_better":
                row.append((base - ppo) / base * 100.0)
            else:
                row.append((ppo - base) / base * 100.0)
        rows.append(row)
    return np.array(rows, dtype=float), [str(f) for f in flows], [DIST_CN[d] for d in dists]


def _draw_heatmap(ax: Axes, mat: np.ndarray, y_labels: list[str], x_labels: list[str], title: str, cbar_label: str) -> None:
    finite_vals = mat[np.isfinite(mat)]
    if finite_vals.size == 0:
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    else:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        # Always use a norm that spans from 0 to at least 30 (or actual max)
        # so the RdYlGn colormap shows green for high improvements
        vmin = min(0.0, vmin)
        vmax = max(vmax, 30.0)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    # Mask NaN so they appear as light gray
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", norm=norm)
    # Set bad color (NaN) to light gray
    im.cmap.set_bad("#E5E7EB")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("到达分布")
    ax.set_ylabel("流量 (veh/h)")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                # Light gray background cell already handled by imshow; just add text
                ax.text(j, i, "—", ha="center", va="center", color="#6B7280", fontsize=10, fontweight="bold")
            else:
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", color="black", fontsize=9)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)


def plot_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    df = _main_eval_subset(df)
    mat_eff, flows, dists = _heat_matrix(df, metric="avg_travel_time", mode="lower_is_better")
    mat_safe, _, _ = _heat_matrix(df, metric="min_ttc", mode="higher_is_better")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.suptitle("PPO 相对 IDM 的场景优势热力图", fontsize=13, fontweight="bold")
    _draw_heatmap(axes[0], mat_eff, flows, dists, "旅行时间改进率", "改进率 (%)")
    _draw_heatmap(axes[1], mat_safe, flows, dists, "最小TTC改进率", "改进率 (%)")
    _save(fig, outdir / "图13_PPO相对IDM场景热力图.png")


def plot_flow_trends(df: pd.DataFrame, outdir: Path) -> None:
    df = _main_eval_subset(df)
    agg = (
        df.groupby(["flow", "dist", "method"], as_index=False)
        .agg(
            avg_travel_time=("avg_travel_time", "mean"),
            mean_waiting_time=("mean_waiting_time", "mean"),
            throughput=("throughput", "mean"),
            min_ttc=("min_ttc", "mean"),
        )
    )
    flows = sorted(agg["flow"].dropna().astype(int).unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    fig.suptitle("PPO 相对 IDM 的流量改进趋势", fontsize=13, fontweight="bold")

    panels = [
        ("avg_travel_time", "旅行时间改进率 (%)", "越高越好", False, True, axes[0, 0]),
        ("mean_waiting_time", "等待时间改进率 (%)", "越高越好", False, True, axes[0, 1]),
        ("throughput", "吞吐量改进率 (%)", "越高越好", True, True, axes[1, 0]),
        ("min_ttc", "最小TTC提升 (s)", "越高越好", True, False, axes[1, 1]),
    ]

    for metric, ylabel, title, higher_is_better, use_relative, ax in panels:
        y_vals = []
        for f in flows:
            sub = agg[agg["flow"] == f]
            ppo_vs_idm = []
            for d in sub["dist"].dropna().unique().tolist():
                ppo_sub = sub[(sub["dist"] == d) & (sub["method"] == "ppo")]
                idm_sub = sub[(sub["dist"] == d) & (sub["method"] == "idm")]
                if ppo_sub.empty or idm_sub.empty:
                    continue
                ppo_val = float(ppo_sub[metric].iloc[0])
                idm_val = float(idm_sub[metric].iloc[0])
                if not np.isfinite(ppo_val) or not np.isfinite(idm_val):
                    continue
                if use_relative:
                    if idm_val == 0:
                        continue
                    if higher_is_better:
                        ppo_vs_idm.append((ppo_val - idm_val) / idm_val * 100.0)
                    else:
                        ppo_vs_idm.append((idm_val - ppo_val) / idm_val * 100.0)
                else:
                    if higher_is_better:
                        ppo_vs_idm.append(ppo_val - idm_val)
                    else:
                        ppo_vs_idm.append(idm_val - ppo_val)
            if len(ppo_vs_idm) == 0:
                y_vals.append(np.nan)
                continue
            y_vals.append(float(np.mean(ppo_vs_idm)))

        y_arr = np.array(y_vals, dtype=float)
        ax.axhline(0.0, color="#9CA3AF", linewidth=1.0, linestyle="--")
        ax.plot(flows, y_arr, marker="o", linewidth=2.6, markersize=6.2, color=METHOD_COLOR["ppo"], label="PPO 相对 IDM")
        ax.fill_between(flows, 0.0, y_arr, where=np.isfinite(y_arr), alpha=0.18, color=METHOD_COLOR["ppo"])

        for x, y in zip(flows, y_arr):
            if not np.isfinite(y):
                continue
            label = f"{y:+.1f}%" if use_relative else f"{y:+.3f}s"
            ax.text(x, y, label, ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(flows)
        ax.set_xlabel("流量 (veh/h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    _save(fig, outdir / "图16_流量趋势四联图.png")


def plot_distribution_boxplots(df: pd.DataFrame, outdir: Path) -> None:
    df = _main_eval_subset(df)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=False)
    fig.suptitle("不同到达分布下方法波动性（箱线图）", fontsize=13, fontweight="bold")

    metrics = [
        ("avg_travel_time", "平均旅行时间 (s)", "旅行时间", axes[0]),
        ("min_ttc", "最小TTC (s)", "最小TTC", axes[1]),
        ("harsh_brake_rate", "急减速率", "急减速率", axes[2]),
    ]

    for metric, ylabel, title_cn, ax in metrics:
        pos = 1.0
        dist_centers = []
        dist_labels = []
        for d in DIST_ORDER:
            start_pos = pos
            for m in METHOD_ORDER:
                sub = df[(df["dist"] == d) & (df["method"] == m)][metric].dropna()
                if sub.empty:
                    continue
                bp = ax.boxplot(
                    [sub.values],
                    positions=[pos],
                    widths=0.65,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#111111", "linewidth": 1.2},
                )
                for b in bp["boxes"]:
                    b.set_facecolor(METHOD_COLOR[m])
                    b.set_alpha(0.78)
                    b.set_edgecolor("#333333")
                pos += 1
            end_pos = pos - 1
            if end_pos >= start_pos:
                dist_centers.append((start_pos + end_pos) / 2.0)
                dist_labels.append(DIST_CN[d])
            pos += 1.2
        ax.set_xticks(dist_centers)
        ax.set_xticklabels(dist_labels, rotation=0, fontsize=9)
        ax.set_xlabel("到达分布")
        ax.set_ylabel(ylabel)
        ax.set_title(title_cn)

    legend_handles = [
        Patch(facecolor=METHOD_COLOR[m], edgecolor="#333333", alpha=0.78, label=METHOD_CN[m]) for m in METHOD_ORDER
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(METHOD_ORDER), frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.subplots_adjust(top=0.82)

    _save(fig, outdir / "图17_到达分布箱线图.png")


def plot_key_metrics_comparison(df: pd.DataFrame, outdir: Path) -> None:
    df = _main_eval_subset(df)
    g = df.groupby("method", as_index=False).agg(
        {
            "avg_travel_time": ["mean", "std"],
            "mean_waiting_time": ["mean", "std"],
            "throughput": ["mean", "std"],
            "min_ttc": ["mean", "std"],
            "harsh_brake_rate": ["mean", "std"],
            "mean_abs_jerk": ["mean", "std"],
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    fig.suptitle("各模型主要指标对比（四向左右转主评估）", fontsize=13, fontweight="bold")

    panels = [
        ("avg_travel_time", "平均旅行时间 (s)", "越低越好", False, axes[0, 0]),
        ("mean_waiting_time", "平均等待时间 (s)", "越低越好", False, axes[0, 1]),
        ("throughput", "吞吐量 (veh/h)", "越高越好", True, axes[0, 2]),
        ("min_ttc", "最小TTC (s)", "越高越好", True, axes[1, 0]),
        ("harsh_brake_rate", "急减速率", "越低越好", False, axes[1, 1]),
        ("mean_abs_jerk", "平均绝对Jerk (m/s^3)", "越低越好", False, axes[1, 2]),
    ]

    for metric, ylabel, better, higher, ax in panels:
        _bar_panel(ax, g, metric, ylabel, better, higher)

    _save(fig, outdir / "图15_各模型主要指标对比图.png")


def plot_tradeoff_scatter(df: pd.DataFrame, outdir: Path) -> None:
    df = _main_eval_subset(df)
    scenario = (
        df[df["method"].isin(["idm", "ppo"])]
        .groupby(["flow", "dist", "method"], as_index=False)
        .agg(
            avg_travel_time=("avg_travel_time", "mean"),
            throughput=("throughput", "mean"),
            min_ttc=("min_ttc", "mean"),
            mean_abs_jerk=("mean_abs_jerk", "mean"),
        )
    )
    idm = scenario[scenario["method"] == "idm"].copy()
    ppo = scenario[scenario["method"] == "ppo"].copy()
    if idm.empty or ppo.empty:
        raise ValueError("缺少 IDM 或 PPO 场景数据，无法绘制折中散点图")

    merged = idm.merge(
        ppo,
        on=["flow", "dist"],
        suffixes=("_idm", "_ppo"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("IDM 与 PPO 场景无法匹配，无法绘制折中散点图")

    rows = []
    for _, row in merged.iterrows():
        min_ttc_idm = float(row["min_ttc_idm"]) if pd.notna(row["min_ttc_idm"]) else np.nan
        min_ttc_ppo = float(row["min_ttc_ppo"]) if pd.notna(row["min_ttc_ppo"]) else np.nan
        jerk_idm = float(row["mean_abs_jerk_idm"]) if pd.notna(row["mean_abs_jerk_idm"]) else np.nan
        jerk_ppo = float(row["mean_abs_jerk_ppo"]) if pd.notna(row["mean_abs_jerk_ppo"]) else np.nan

        if np.isfinite(min_ttc_idm) and min_ttc_idm != 0 and np.isfinite(min_ttc_ppo):
            safety_raw = (min_ttc_ppo - min_ttc_idm) / max(abs(min_ttc_idm), 1e-6) * 100.0
            safety_gain = np.log1p(max(safety_raw, 0.0))
        else:
            safety_gain = 0.0

        if np.isfinite(jerk_idm) and jerk_idm != 0 and np.isfinite(jerk_ppo):
            comfort_gain = (jerk_idm - jerk_ppo) / max(abs(jerk_idm), 1e-6) * 100.0
        else:
            comfort_gain = 0.0

        rows.append(
            {
                "flow": int(row["flow"]),
                "dist": str(row["dist"]),
                "tt_gain": (float(row["avg_travel_time_idm"]) - float(row["avg_travel_time_ppo"])) / max(abs(float(row["avg_travel_time_idm"])), 1e-6) * 100.0,
                "tp_gain": (float(row["throughput_ppo"]) - float(row["throughput_idm"])) / max(abs(float(row["throughput_idm"])), 1e-6) * 100.0,
                "safety_gain": safety_gain,
                "comfort_gain": comfort_gain,
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError("折中散点图数据为空")
    plot_df["safety_gain"] = pd.Series(
        pd.to_numeric(plot_df["safety_gain"], errors="coerce"), index=plot_df.index
    ).fillna(0.0)
    dist_colors = {"uniform": "#0072B2", "poisson": "#009E73", "burst": "#E69F00"}
    dist_colors["balanced"] = "#A855F7"

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    sizes = 260 + 520 * (plot_df["safety_gain"] - plot_df["safety_gain"].min()) / max(
        float(plot_df["safety_gain"].max() - plot_df["safety_gain"].min()), 1e-6
    )

    # Smart label positioning to avoid overlaps, especially for poisson labels.
    dist_seen: dict[str, int] = {}

    for idx, r in plot_df.iterrows():
        color = dist_colors.get(r["dist"], METHOD_COLOR["ppo"])
        size = float(sizes.loc[idx])
        jitter = {
            "uniform": (-0.14, 0.08),
            "poisson": (0.00, -0.10),
            "burst": (0.14, 0.10),
            "balanced": (0.0, 0.0),
        }.get(str(r["dist"]), (0.0, 0.0))
        x_plot = float(r["tt_gain"]) + jitter[0]
        y_plot = float(r["tp_gain"]) + jitter[1]
        ax.scatter(
            x_plot,
            y_plot,
            s=size,
            color=color,
            alpha=0.75,
            edgecolors="#222222",
            linewidths=0.9,
            zorder=3,
        )
        
        # Smart label positioning based on quadrant and distribution.
        label_text = f"{r['flow']}|{DIST_CN.get(r['dist'], r['dist'])}"
        dist_key = str(r["dist"])
        dist_index = dist_seen.get(dist_key, 0)
        dist_seen[dist_key] = dist_index + 1
        
        # Determine primary direction based on point location
        if r["tt_gain"] >= 0 and r["tp_gain"] >= 0:
            # Upper right quadrant
            label_dx = 42
            label_dy = 22
            ha = "left"
            va = "bottom"
        elif r["tt_gain"] < 0 and r["tp_gain"] >= 0:
            # Upper left quadrant
            label_dx = -42
            label_dy = 22
            ha = "right"
            va = "bottom"
        elif r["tt_gain"] < 0 and r["tp_gain"] < 0:
            # Lower left quadrant
            label_dx = -42
            label_dy = -22
            ha = "right"
            va = "top"
        else:
            # Lower right quadrant
            label_dx = 42
            label_dy = -22
            ha = "left"
            va = "top"

        # Extra stagger for poisson labels to reduce collisions.
        if dist_key == "poisson":
            if dist_index % 2 == 0:
                label_dx += 28
                label_dy += 28
            else:
                label_dx -= 28
                label_dy -= 28
                ha = "right" if ha == "left" else "left"
        elif dist_key == "uniform":
            label_dy += 10
        elif dist_key == "burst":
            label_dy -= 10
        
        # Add leader line from point to label
        ax.annotate(
            label_text,
            xy=(x_plot, y_plot),
            xytext=(label_dx, label_dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8.5,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor="#D1D5DB", alpha=0.92, linewidth=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.22", color="#9CA3AF", lw=0.8, alpha=0.65),
        )

    ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    ax.text(
        0.03,
        0.08,
        f"PPO 场景级改进：旅行时间平均 +{plot_df['tt_gain'].mean():.1f}%, 吞吐平均 +{plot_df['tp_gain'].mean():.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#E5E7EB", alpha=0.95, linewidth=1.0),
    )

    tt_min = float(plot_df["tt_gain"].min())
    tt_max = float(plot_df["tt_gain"].max())
    tp_min = float(plot_df["tp_gain"].min())
    tp_max = float(plot_df["tp_gain"].max())
    tt_margin = max(2.0, 0.2 * (tt_max - tt_min + 1e-6))
    tp_margin = max(0.8, 0.25 * (tp_max - tp_min + 1e-6))
    ax.set_xlim(tt_min - tt_margin, tt_max + tt_margin)
    ax.set_ylim(tp_min - tp_margin, tp_max + tp_margin)
    ax.set_xlabel("旅行时间改进率（相对 IDM，%）", fontsize=11)
    ax.set_ylabel("吞吐量改进率（相对 IDM，%）", fontsize=11)
    ax.set_title("安全-舒适-效率折中散点图（气泡越大表示安全改进越高，TTC 对数压缩）", fontsize=12, fontweight="bold")

    legend_handles = [
        Patch(facecolor=dist_colors[d], edgecolor="#222222", alpha=0.82, label=DIST_CN[d]) for d in DIST_ORDER if d in dist_colors
    ]
    ax.legend(handles=legend_handles, title="到达分布", loc="lower right", fontsize=8.5, title_fontsize=9.5, frameon=True, fancybox=True)

    _save(fig, outdir / "图18_安全舒适效率折中散点图.png")


def _pick_existing_samples(path_arg: Optional[str], fallback: Path) -> Optional[Path]:
    if path_arg:
        p = Path(path_arg)
        if p.exists():
            return p
    if fallback.exists():
        return fallback
    return None


def plot_trajectory_and_ttc(traj_csv: Path, ttc_csv: Path, outdir: Path, skip_ttc: bool = False) -> None:
    traj = pd.read_csv(traj_csv)
    ttc = pd.read_csv(ttc_csv)

    traj = traj[traj["method"].isin(["Rule-based", "PPO"])].copy()
    ttc = ttc[ttc["method"].isin(["Rule-based", "PPO"])].copy()

    method_labels = {"Rule-based": "IDM基线", "PPO": "PPO"}
    method_colors = {"Rule-based": "#E69F00", "PPO": "#0072B2"}

    fig1, axes1 = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, m in zip(axes1, ["Rule-based", "PPO"]):
        sub = traj[traj["method"] == m]
        if len(sub) > 70000:
            sub = sub.sample(70000, random_state=0)
        ax.scatter(sub["time"], sub["dist_to_stop"], s=3.5, alpha=0.22, c=method_colors[m], edgecolors="none")
        ax.set_title(method_labels[m])
        ax.set_xlabel("时间 (s)")
    axes1[0].set_ylabel("距停止线距离 (m)")
    fig1.suptitle("轨迹时空图（入口车道）", fontsize=13, fontweight="bold")
    _save(fig1, outdir / "图14_轨迹时空图_规则与PPO.png")

    # 替代方案：12子图纯轨迹展示（规则+PPO同图），用于模型维度不匹配时仍提供丰富证据。
    if {"lane", "lane_role", "s_from_stopline", "time"}.issubset(traj.columns):
        fig1d, axes1d = plt.subplots(3, 4, figsize=(16.2, 10.2), sharex=True, sharey=True)
        panel_specs = [
            ("lane", "n_t_0", "进口 n_t_0"),
            ("lane", "n_t_1", "进口 n_t_1"),
            ("lane", "w_t_0", "进口 w_t_0"),
            ("lane", "w_t_1", "进口 w_t_1"),
            ("lane", "t_e_0", "出口 t_e_0"),
            ("lane", "t_e_1", "出口 t_e_1"),
            ("lane", "t_s_0", "出口 t_s_0"),
            ("lane", "t_s_1", "出口 t_s_1"),
            ("prefix", "n_t", "北向进口汇总"),
            ("prefix", "w_t", "西向进口汇总"),
            ("prefix", "t_e", "东向出口汇总"),
            ("prefix", "t_s", "南向出口汇总"),
        ]

        for ax, (mode, key, title_cn) in zip(axes1d.flatten(), panel_specs):
            if mode == "lane":
                sub = traj[traj["lane"] == key]
            else:
                sub = traj[traj["lane"].astype(str).str.startswith(key)]

            rb = sub[sub["method"] == "Rule-based"]
            pp = sub[sub["method"] == "PPO"]
            if len(rb) > 12000:
                rb = rb.sample(12000, random_state=21)
            if len(pp) > 12000:
                pp = pp.sample(12000, random_state=22)

            if not rb.empty:
                ax.scatter(rb["time"], rb["s_from_stopline"], s=2.6, alpha=0.20, c=method_colors["Rule-based"], edgecolors="none", label="规则")
            if not pp.empty:
                ax.scatter(pp["time"], pp["s_from_stopline"], s=2.6, alpha=0.20, c=method_colors["PPO"], edgecolors="none", label="PPO")

            if rb.empty and pp.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#6B7280")

            ax.set_title(title_cn)
            ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=0.9)
            ax.grid(alpha=0.18)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("相对停止线位置 (m)")

        handles, labels = axes1d[0, 0].get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            # Increase legend marker size for visibility
            for h in uniq.values():
                h.set_sizes([60])
            fig1d.legend(
                list(uniq.values()),
                list(uniq.keys()),
                loc="upper center",
                ncol=max(1, len(uniq)),
                frameon=False,
                bbox_to_anchor=(0.5, 0.94),
            )
        fig1d.suptitle("轨迹时空图__规则与PPO", fontsize=13, fontweight="bold", y=0.995)
        _save(fig1d, outdir / "图14e_轨迹时空图_12子图_规则与PPO_替代方案.png")

    # 新增：分进出口道的轨迹时空图（从停止线到离开交叉口）
    if {"lane_role", "s_from_stopline"}.issubset(traj.columns):
        fig1b, axes1b = plt.subplots(2, 2, figsize=(13.8, 7.9), sharex=True, sharey=False)
        roles = ["incoming", "outgoing"]
        role_cn = {"incoming": "进口道（停止线前）", "outgoing": "出口道（停止线后至离开交叉口）"}
        for r, role in enumerate(roles):
            for c, m in enumerate(["Rule-based", "PPO"]):
                ax = axes1b[r, c]
                sub = traj[(traj["method"] == m) & (traj["lane_role"] == role)]
                if len(sub) > 55000:
                    sub = sub.sample(55000, random_state=3)
                ax.scatter(sub["time"], sub["s_from_stopline"], s=3.0, alpha=0.20, c=method_colors[m], edgecolors="none")
                ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
                if r == 0:
                    ax.set_title(method_labels[m])
                if c == 0:
                    ax.set_ylabel(f"{role_cn[role]}\n相对停止线位置 (m)")
                if r == 1:
                    ax.set_xlabel("时间 (s)")
        fig1b.suptitle("分进出口道轨迹时空图（从停止线到离开交叉口）", fontsize=13, fontweight="bold")
        _save(fig1b, outdir / "图14b_轨迹时空图_进出口分道_规则与PPO.png")

    # 新增：4方向 x 3流向 的12子图（按车道/车辆轨迹自动归类）
    if {"lane_role", "s_from_stopline", "veh_id", "lane"}.issubset(traj.columns):
        traj12 = traj.copy()

        def _in_dir_from_lane(lane: str) -> str:
            # n_t_0 / e_t_0 / s_t_0 / w_t_0
            s = str(lane)
            return s[0] if s else "?"

        def _out_dir_from_lane(lane: str) -> str:
            # t_n_0 / t_e_0 / t_s_0 / t_w_0
            parts = str(lane).split("_")
            return parts[1] if len(parts) >= 2 else "?"

        def _turn_label(in_dir: str, out_dir: str) -> str:
            opp = {"n": "s", "s": "n", "e": "w", "w": "e"}
            left = {"n": "e", "e": "s", "s": "w", "w": "n"}
            right = {"n": "w", "w": "s", "s": "e", "e": "n"}
            heading = opp.get(in_dir)
            if heading is None:
                return "unknown"
            if out_dir == heading:
                return "straight"
            if out_dir == left[heading]:
                return "left"
            if out_dir == right[heading]:
                return "right"
            return "unknown"

        keys = ["method", "veh_id"]
        in_df = (
            traj12[traj12["lane_role"] == "incoming"]
            .sort_values("time")
            .groupby(keys, as_index=False)
            .first()[keys + ["lane"]]
            .rename(columns={"lane": "lane_in"})
        )
        out_df = (
            traj12[traj12["lane_role"] == "outgoing"]
            .sort_values("time")
            .groupby(keys, as_index=False)
            .first()[keys + ["lane"]]
            .rename(columns={"lane": "lane_out"})
        )
        route_df = in_df.merge(out_df, on=keys, how="left")
        route_df["in_dir"] = route_df["lane_in"].astype(str).map(_in_dir_from_lane)
        route_df["out_dir"] = route_df["lane_out"].fillna("").astype(str).map(lambda x: _out_dir_from_lane(x) if x else "?")
        route_df["turn"] = [
            _turn_label(i, o) if o in {"n", "e", "s", "w"} else "unknown"
            for i, o in zip(route_df["in_dir"].tolist(), route_df["out_dir"].tolist())
        ]

        traj12 = traj12.merge(route_df[keys + ["in_dir", "turn"]], on=keys, how="left")

        # 聚焦“停止线到离开交叉口”：停止线前保留近端，停止线后保留出口段。
        traj12 = traj12[(traj12["s_from_stopline"] >= -35) & (traj12["s_from_stopline"] <= 90)].copy()

        dir_order = ["n", "e", "s", "w"]
        dir_cn = {"n": "北向进口", "e": "东向进口", "s": "南向进口", "w": "西向进口"}
        turn_order = ["left", "straight", "right"]
        turn_cn = {"left": "左转", "straight": "直行", "right": "右转"}

        # 仅绘制“规则+PPO同时有样本”的子图，能画几个画几个。
        panel_specs: list[tuple[str, str]] = []
        for in_dir in dir_order:
            for turn in turn_order:
                sub = traj12[(traj12["in_dir"] == in_dir) & (traj12["turn"] == turn)]
                has_rb = not sub[sub["method"] == "Rule-based"].empty
                has_pp = not sub[sub["method"] == "PPO"].empty
                if has_rb and has_pp:
                    panel_specs.append((in_dir, turn))

        if panel_specs:
            n_panels = len(panel_specs)
            n_cols = min(4, n_panels)
            n_rows = int(np.ceil(n_panels / n_cols))
            fig1c, axes1c = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.3 * n_rows), sharex=True, sharey=True)
            axes_arr = np.atleast_1d(axes1c).flatten()

            for i, (in_dir, turn) in enumerate(panel_specs):
                ax = axes_arr[i]
                sub = traj12[(traj12["in_dir"] == in_dir) & (traj12["turn"] == turn)]
                rb = sub[sub["method"] == "Rule-based"]
                pp = sub[sub["method"] == "PPO"]
                if len(rb) > 16000:
                    rb = rb.sample(16000, random_state=11)
                if len(pp) > 16000:
                    pp = pp.sample(16000, random_state=12)

                ax.scatter(rb["time"], rb["s_from_stopline"], s=2.6, alpha=0.20, c=method_colors["Rule-based"], edgecolors="none", label="规则")
                ax.scatter(pp["time"], pp["s_from_stopline"], s=2.6, alpha=0.20, c=method_colors["PPO"], edgecolors="none", label="PPO")
                ax.set_title(f"{dir_cn.get(in_dir, in_dir)}-{turn_cn.get(turn, turn)}")
                ax.set_xlabel("时间 (s)")
                ax.set_ylabel("相对停止线位置 (m)")
                ax.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=0.9)
                ax.grid(alpha=0.18)

            for j in range(n_panels, len(axes_arr)):
                axes_arr[j].axis("off")

            handles, labels = axes_arr[0].get_legend_handles_labels()
            if handles:
                uniq = dict(zip(labels, handles))
                for h in uniq.values():
                    h.set_sizes([60])
                fig1c.legend(
                    list(uniq.values()),
                    list(uniq.keys()),
                    loc="upper center",
                    ncol=max(1, len(uniq)),
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.94),
                )
            fig1c.suptitle("车道轨迹时空图（规则与PPO同图，仅展示双方法有样本流向）", fontsize=13, fontweight="bold", y=0.99)
            _save(fig1c, outdir / "图14c_轨迹时空图_四方向三流向_规则与PPO.png")

    if skip_ttc:
        return

    d = ttc[np.isfinite(ttc["ttc"]) & (ttc["ttc"] > 0) & (ttc["ttc"] <= 20)].copy()
    fig2, axes2 = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for m in ["Rule-based", "PPO"]:
        sub = d[d["method"] == m]
        axes2[0].hist(sub["ttc"], bins=36, alpha=0.45, density=True, color=method_colors[m], label=method_labels[m])
    axes2[0].set_xlabel("TTC (s)")
    axes2[0].set_ylabel("密度")
    axes2[0].set_title("TTC 分布")
    axes2[0].legend(loc="upper right", fontsize=9)

    agg = d.groupby(["method", "time"], as_index=False)["ttc"].min()
    for m in ["Rule-based", "PPO"]:
        sub = agg[agg["method"] == m]
        axes2[1].plot(sub["time"], sub["ttc"], color=method_colors[m], label=method_labels[m], alpha=0.9)
    axes2[1].set_xlabel("时间 (s)")
    axes2[1].set_ylabel("最小TTC (s)")
    axes2[1].set_ylim(0, 20)
    axes2[1].set_title("最小TTC时序")
    axes2[1].legend(loc="upper right", fontsize=9)

    if not d.empty:
        ttc_rule = d[d["method"] == "Rule-based"]["ttc"].mean()
        ttc_ppo = d[d["method"] == "PPO"]["ttc"].mean()
        if np.isfinite(ttc_rule) and ttc_rule > 0 and np.isfinite(ttc_ppo):
            delta = (ttc_ppo - ttc_rule) / ttc_rule * 100.0
            sign = "+" if delta >= 0 else ""
            axes2[0].text(
                0.98,
                0.95,
                f"PPO 平均TTC 相对规则基线: {sign}{delta:.1f}%",
                transform=axes2[0].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#F3F4F6", alpha=0.9),
            )

    fig2.suptitle("TTC 风险分析", fontsize=13, fontweight="bold")
    _save(fig2, outdir / "图15_TTC分布与时序_规则与PPO.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成论文终稿整合图（综合图、热力图、轨迹图、TTC图）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--matrix-csv", type=str, default="outputs/figures_repro/all_runs_matrix.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/figures_repro")
    parser.add_argument("--traj-csv", type=str, default="")
    parser.add_argument("--ttc-csv", type=str, default="")
    parser.add_argument("--legacy-sample-dir", type=str, default="outputs/figures_repro")
    parser.add_argument("--smooth", type=int, default=25, help="Smoothing window (for compatibility, not used).")
    parser.add_argument("--skip-ttc", action="store_true", default=False)
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_matrix(Path(args.matrix_csv))
    plot_overview_panels(df, outdir)
    plot_key_metrics_comparison(df, outdir)
    plot_heatmaps(df, outdir)
    plot_flow_trends(df, outdir)
    plot_distribution_boxplots(df, outdir)
    plot_tradeoff_scatter(df, outdir)

    legacy_dir = Path(args.legacy_sample_dir)
    traj_csv = _pick_existing_samples(args.traj_csv, legacy_dir / "trajectory_samples.csv")
    ttc_csv = _pick_existing_samples(args.ttc_csv, legacy_dir / "ttc_samples.csv")
    if traj_csv is not None and ttc_csv is not None:
        plot_trajectory_and_ttc(traj_csv, ttc_csv, outdir, skip_ttc=args.skip_ttc)
        print(f"已生成轨迹与TTC图，样本来源: {traj_csv.parent}")
    else:
        print("未找到 trajectory_samples.csv / ttc_samples.csv，已跳过图14-图15。")

    print(f"图像已输出到: {outdir}")


if __name__ == "__main__":
    main()
