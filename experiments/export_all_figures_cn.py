#!/usr/bin/env python3
"""
导出所有请求的中文图表，基于最新的多seed评估数据
图11-18: 从generate_thesis_figures.py生成
图19: PPO与DQN训练收敛曲线（3个seed对比）
图20: 消融实验总览
"""
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 极端强制字体初始化
def _force_chinese_font():
    """Aggressively set up Chinese font support."""
    # Method 1: Try to find actual font files on Windows
    windows_font_paths = [
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simsun.ttc",
        "C:\\Program Files\\Microsoft Office\\root\\Office16\\Library\\Fonts\\msyh.ttc",
    ]
    
    for font_path in windows_font_paths:
        if os.path.exists(font_path):
            try:
                prop = fm.FontProperties(fname=font_path)
                matplotlib.rcParams["font.sans-serif"] = [prop.get_name()]
                matplotlib.rcParams["font.family"] = prop.get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                print(f"[OK] Loaded font from file: {font_path}", flush=True)
                return
            except Exception as e:
                print(f"Warning loading {font_path}: {e}", flush=True)
    
    # Method 2: Try system font names
    font_names = ["SimHei", "Microsoft YaHei", "SimSun", "FangSong", "Kaiti", "Noto Sans CJK SC"]
    for fname in font_names:
        try:
            # Create a property with the font name
            prop = fm.FontProperties(family=fname)
            # Try to find it
            found_path = fm.findfont(prop)
            # Check if it's not a fallback
            if found_path and "DejaVu" not in found_path:
                matplotlib.rcParams["font.sans-serif"] = [fname]
                matplotlib.rcParams["font.family"] = "sans-serif"
                matplotlib.rcParams["axes.unicode_minus"] = False
                print(f"[OK] Found system font: {fname}", flush=True)
                return
        except Exception as e:
            pass
    
    # Method 3: At least prevent DejaVu from being used
    print("WARNING: No Chinese font found. Installing fallback CJK font...", flush=True)
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    

_force_chinese_font()
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["figure.autolayout"] = True


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] {out_path.name}")


def _collect_latest_monitors_by_seed(monitor_dir: Path) -> dict[int, Path]:
    """Collect latest monitor files for each seed."""
    pattern = re.compile(r"seed(\d+)")
    latest: dict[int, Path] = {}
    
    if not monitor_dir.exists():
        return latest
    
    for p in sorted(monitor_dir.glob("*.monitor.csv"), key=lambda x: x.stat().st_mtime):
        m = pattern.search(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        latest[seed] = p
    
    return latest


def _load_monitor(path: Path) -> pd.DataFrame:
    """Load monitor CSV and prepare for plotting."""
    df = pd.read_csv(path, comment="#")
    if "r" not in df.columns:
        raise ValueError(f"Monitor file missing 'r' column: {path}")
    out = df.copy()
    out["episode"] = np.arange(1, len(out) + 1)
    return out


def _rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Apply rolling mean smoothing."""
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=max(1, window // 3)).mean()


def plot_figure19(
    ppo_monitor_dir: Path,
    dqn_monitor_dir: Path,
    out_path: Path,
    smooth: int = 25,
) -> None:
    """图19: PPO与DQN训练收敛曲线（3个seed对比）"""
    print("Generating figure 19 (PPO & DQN convergence)...")
    
    ppo_monitors = _collect_latest_monitors_by_seed(ppo_monitor_dir)
    dqn_monitors = _collect_latest_monitors_by_seed(dqn_monitor_dir)
    
    if len(ppo_monitors) < 3 or len(dqn_monitors) < 3:
        print(f"  Warning: Expected 3 seeds each, got PPO={len(ppo_monitors)}, DQN={len(dqn_monitors)}")
    
    # Keep moderate smoothing to preserve real fluctuations.
    effective_smooth = max(3, min(int(smooth), 15))

    # Load and smooth all runs.
    ppo_runs = []
    for seed in sorted(ppo_monitors.keys()):
        df = _load_monitor(ppo_monitors[seed])
        df["smooth"] = _rolling_mean(df["r"], effective_smooth)
        ppo_runs.append((seed, df))
    
    dqn_runs = []
    for seed in sorted(dqn_monitors.keys()):
        df = _load_monitor(dqn_monitors[seed])
        df["smooth"] = _rolling_mean(df["r"], effective_smooth)
        dqn_runs.append((seed, df))
    
    # Compute and plot mean and std across seeds
    def aggregate_runs(runs: list[tuple[int, pd.DataFrame]]) -> pd.DataFrame:
        dfs = [r[1] for r in runs]
        min_len = min(len(r) for r in dfs)
        aligned = [r.iloc[:min_len][["episode", "smooth"]].reset_index(drop=True) for r in dfs]
        stacked = pd.concat(aligned, axis=1)
        # Group by episode (even indices are episodes)
        episode_col = stacked.iloc[:, 0]
        smooth_cols = [stacked.iloc[:, i] for i in range(1, len(stacked.columns), 2)]
        mean_val = pd.concat(smooth_cols, axis=1).mean(axis=1)
        std_val = pd.concat(smooth_cols, axis=1).std(axis=1)
        return pd.DataFrame({"episode": episode_col, "mean": mean_val, "std": std_val})
    
    ppo_agg = aggregate_runs(ppo_runs)
    dqn_agg = aggregate_runs(dqn_runs)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=False)
    ppo_color = "#1f77b4"
    dqn_color = "#d95f02"

    def draw_panel(ax: plt.Axes, runs: list[tuple[int, pd.DataFrame]], agg: pd.DataFrame, color: str, title: str, legend_prefix: str) -> None:
        for seed, run_df in runs:
            ax.plot(
                run_df["episode"],
                run_df["smooth"],
                color=color,
                alpha=0.22,
                linewidth=0.9,
                zorder=1,
            )

        ax.plot(
            agg["episode"],
            agg["mean"],
            color=color,
            linewidth=1.9,
            label=f"{legend_prefix} 平均",
            zorder=3,
        )
        ax.fill_between(
            agg["episode"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=color,
            alpha=0.16,
            label="均值±标准差",
            zorder=2,
        )

        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("平滑回报")
        ax.grid(True, alpha=0.28)
        ax.legend(loc="lower right", fontsize=8)

    draw_panel(axes[0], ppo_runs, ppo_agg, ppo_color, "PPO 训练收敛曲线", "PPO")
    draw_panel(axes[1], dqn_runs, dqn_agg, dqn_color, "DQN 训练收敛曲线", "DQN")

    _save(fig, out_path)


def _stamp_has_variation(stamp_dirs: list[Path], pat: re.Pattern) -> bool:
    """Check whether a stamp has meaningful variation across settings."""
    values: dict[str, list[float]] = {"avg_travel_time": [], "mean_waiting_time": [], "throughput": [], "collisions": []}
    for setting_dir in stamp_dirs:
        m = pat.match(setting_dir.name)
        if m is None:
            continue
        eval_summary = setting_dir / "eval" / "eval_summary.csv"
        if not eval_summary.exists():
            return False
        try:
            df = pd.read_csv(eval_summary)
            if df.empty:
                return False
            for k in values:
                if k in df.columns:
                    values[k].append(float(df[k].mean()))
        except Exception:
            return False
    # Require at least one metric with non-identical values across settings
    for vals in values.values():
        if len(vals) >= 2 and any(abs(v - vals[0]) > 1e-9 for v in vals[1:]):
            return True
    return False


def plot_figure20(ablation_dir: Path, out_path: Path) -> None:
    """图20: 消融实验总览（从各消融设置的评估结果读取真实数据）"""
    print("Generating figure 20 (Ablation overview)...")
    
    # Look for ablation subdirectories directly in the provided path
    ablation_base = ablation_dir
    
    if not ablation_base.exists():
        print(f"  Warning: Ablation directory not found: {ablation_base}")
        return
    
    # Collect evaluation results from the latest complete ablation stamp only.
    setting_cn = {
        ("full", "full"): "完整奖励\n完整状态",
        ("full", "phase_only"): "完整奖励\n相位状态",
        ("default", "full"): "默认奖励\n完整状态",
        ("default", "phase_only"): "默认奖励\n相位状态",
    }
    
    pat = re.compile(r"^(\d{8}_\d{6})_(full|default)_(full|phase_only)$")
    stamp_to_settings: dict[str, set[str]] = {}
    stamp_to_dirs: dict[str, list[Path]] = {}
    for setting_dir in sorted(ablation_base.glob("*_*_*")):
        if not setting_dir.is_dir():
            continue
        m = pat.match(setting_dir.name)
        if m is None:
            continue
        stamp = m.group(1)
        setting_key = f"{m.group(2)}_{m.group(3)}"
        stamp_to_settings.setdefault(stamp, set()).add(setting_key)
        stamp_to_dirs.setdefault(stamp, []).append(setting_dir)

    complete = [s for s, keys in stamp_to_settings.items() if keys == {"full_full", "full_phase_only", "default_full", "default_phase_only"}]
    if not complete:
        print("  Warning: No complete ablation stamp found")
        return
    
    # Prefer the latest stamp that actually has variation; warn if latest is flat.
    complete_sorted = sorted(complete)
    latest_stamp = complete_sorted[-1]
    if not _stamp_has_variation(stamp_to_dirs[latest_stamp], pat):
        print(f"  Warning: Latest stamp {latest_stamp} has identical values across settings (suspicious).")
        # Fall back to the most recent stamp with variation
        varied_stamps = [s for s in complete_sorted if _stamp_has_variation(stamp_to_dirs[s], pat)]
        if varied_stamps:
            latest_stamp = varied_stamps[-1]
            print(f"  Falling back to earlier stamp with variation: {latest_stamp}")
        else:
            print(f"  Warning: No stamp with variation found; using latest anyway.")

    setting_data = {}
    for setting_dir in sorted(stamp_to_dirs[latest_stamp]):
        m = pat.match(setting_dir.name)
        if m is None:
            continue
        reward_mode = m.group(2)
        obs_mode = m.group(3)
        
        # Look for eval_summary.csv in the eval subdirectory
        eval_summary = setting_dir / "eval" / "eval_summary.csv"
        if not eval_summary.exists():
            print(f"  Note: No eval_summary in {setting_dir.name}")
            continue
        
        try:
            df = pd.read_csv(eval_summary)
            if not df.empty:
                # Aggregate all rows
                setting_key_tuple = (reward_mode, obs_mode)
                setting_name = setting_cn.get(setting_key_tuple, f"{reward_mode}\n{obs_mode}")
                
                setting_data[setting_name] = {
                    "avg_travel_time": df["avg_travel_time"].mean(),
                    "mean_waiting_time": df["mean_waiting_time"].mean(),
                    "throughput": df["throughput"].mean(),
                    "collisions": df["collisions"].mean(),
                }
        except Exception as e:
            print(f"  Warning: Error reading {eval_summary}: {e}")
    
    if not setting_data:
        print("  Warning: No ablation setting data found")
        return
    
    # Convert to DataFrame in consistent order
    setting_order = [
        "完整奖励\n完整状态",
        "完整奖励\n相位状态",
        "默认奖励\n完整状态",
        "默认奖励\n相位状态",
    ]
    
    data_list = []
    for setting in setting_order:
        if setting in setting_data:
            row = {"setting": setting}
            row.update(setting_data[setting])
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    if df.empty:
        print("  Warning: No valid ablation data")
        return
    
    colors = {
        "完整奖励\n完整状态": "#0072B2",
        "完整奖励\n相位状态": "#56B4E9",
        "默认奖励\n完整状态": "#D55E00",
        "默认奖励\n相位状态": "#E69F00",
    }
    
    # Compute relative changes vs baseline (完整奖励/完整状态)
    baseline = df[df["setting"] == "完整奖励\n完整状态"].iloc[0]
    rel_df = df.copy()
    for metric in ["avg_travel_time", "mean_waiting_time", "throughput", "collisions"]:
        if metric in baseline:
            base_val = float(baseline[metric])
            if base_val != 0:
                rel_df[f"{metric}_rel"] = (df[metric] - base_val) / abs(base_val) * 100.0
            else:
                rel_df[f"{metric}_rel"] = 0.0
    
    # Check if data has meaningful variation
    max_rel_change = 0.0
    for metric in ["avg_travel_time", "mean_waiting_time", "throughput", "collisions"]:
        if metric in df.columns:
            vals = df[metric].to_numpy()
            if len(vals) > 1:
                max_rel_change = max(max_rel_change, abs(np.max(vals) - np.min(vals)) / (abs(np.mean(vals)) + 1e-6) * 100.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    if max_rel_change < 1.0:
        fig.suptitle(f"消融实验总览（四种设置对比）\n注：各设置间相对变化最大仅 {max_rel_change:.2f}%，差异极小", fontsize=13, fontweight="bold")
    else:
        fig.suptitle("消融实验总览（四种设置对比）", fontsize=14, fontweight="bold")
    
    metrics = [
        ("avg_travel_time", "平均旅行时间(秒)"),
        ("mean_waiting_time", "平均等待时间(秒)"),
        ("throughput", "通行量(辆/小时)"),
        ("collisions", "碰撞次数"),
    ]
    
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        if metric not in df.columns:
            ax.text(0.5, 0.5, f"缺少列: {metric}", ha="center", va="center")
            continue
        
        x = np.arange(len(df))
        y = df[metric].to_numpy()
        bar_colors = [colors.get(str(s), "#999999") for s in df["setting"]]
        
        ax.bar(x, y, color=bar_colors, edgecolor="#333333", linewidth=1.0, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["setting"].astype(str), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel}对比", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Add value labels on bars
        for xi, yi in zip(x, y):
            ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
        
        # Add relative change annotation if meaningful
        rel_col = f"{metric}_rel"
        if rel_col in rel_df.columns:
            for xi, ri in zip(x, rel_df[rel_col].to_numpy()):
                if abs(ri) > 0.01:
                    sign = "+" if ri >= 0 else ""
                    ax.text(xi, y.min() + 0.02 * (y.max() - y.min()), f"{sign}{ri:.2f}%", ha="center", va="bottom", fontsize=7.5, color="#374151", alpha=0.8)
    
    # Add diagnostic text box
    diag_text = ""
    for metric, ylabel in metrics:
        if metric in df.columns:
            vals = df[metric].to_numpy()
            if len(vals) >= 2:
                rel_range = (np.max(vals) - np.min(vals)) / (abs(np.mean(vals)) + 1e-6) * 100.0
                diag_text += f"{ylabel}: 极差={np.max(vals)-np.min(vals):.3f} ({rel_range:.2f}%)\n"
    if diag_text:
        fig.text(0.5, 0.01, diag_text.strip(), ha="center", va="bottom", fontsize=8, color="#6B7280", bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9FAFB", edgecolor="#E5E7EB"))
    
    plt.tight_layout(pad=2.0, rect=[0.02, 0.06, 1.0, 0.96])
    _save(fig, out_path)


def run_multiseed_eval(
    route_file: str,
    model_dir: Path,
    seconds: int,
    runs: int,
    matrix_out: Path,
) -> Path:
    """Run multi-seed evaluation if matrix doesn't exist."""
    if matrix_out.exists():
        print(f"Using existing matrix: {matrix_out}")
        return matrix_out
    
    print("Running multi-seed evaluation...")
    cmd = [
        sys.executable,
        "experiments/eval_4way_multiseed.py",
        "--route",
        route_file,
        "--seconds",
        str(seconds),
        "--runs",
        str(runs),
        "--model-dir",
        str(model_dir),
        "--matrix-out",
        str(matrix_out),
    ]
    
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode != 0:
        print("  Warning: Multi-seed evaluation had issues")
    
    if not matrix_out.exists():
        print(f"  Warning: Matrix file not created: {matrix_out}")
    
    return matrix_out


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export all requested Chinese figures (11-20) based on latest 3-seed models.",
    )
    p.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
        help="Route file for evaluation.",
    )
    p.add_argument("--seconds", type=int, default=600, help="Simulation duration (s).")
    p.add_argument("--eval-runs", type=int, default=3, help="Number of evaluation runs per model.")
    p.add_argument("--model-dir", type=str, default="models/4way-single-intersection", help="Model directory.")
    p.add_argument(
        "--matrix-csv",
        type=str,
        default="outputs/all_runs_matrix.csv",
        help="Output matrix CSV (will run multi-seed eval if not exist).",
    )
    p.add_argument(
        "--ppo-monitor-dir",
        type=str,
        default="logs/4way_single_intersection/ppo/monitor",
        help="PPO monitor logs directory.",
    )
    p.add_argument(
        "--dqn-monitor-dir",
        type=str,
        default="logs/4way_single_intersection/dqn/monitor",
        help="DQN monitor logs directory.",
    )
    p.add_argument(
        "--ablation-dir",
        type=str,
        default="outputs/4way-single-intersection/ablation",
        help="Ablation results directory.",
    )
    p.add_argument("--smooth", type=int, default=25, help="Smoothing window for convergence curves.")
    p.add_argument("--out-dir", type=str, default="", help="Output directory (auto-timestamped if empty).")
    p.add_argument("--skip-thesis", action="store_true", help="Skip thesis figures 11-18.")
    
    args = p.parse_args()
    
    # Create output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"figures_cn_{stamp}"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    
    # Run multi-seed evaluation if needed
    matrix_path = Path(args.matrix_csv)
    if not args.skip_thesis:
        matrix_path = run_multiseed_eval(
            route_file=args.route,
            model_dir=Path(args.model_dir),
            seconds=args.seconds,
            runs=args.eval_runs,
            matrix_out=matrix_path,
        )
    
    # Generate thesis figures 11-18
    if not args.skip_thesis:
        print("\nGenerating thesis figures 11-18...")
        cmd = [
            sys.executable,
            "experiments/generate_thesis_figures.py",
            "--matrix-csv",
            str(matrix_path),
            "--out-dir",
            str(out_dir),
            "--smooth",
            str(args.smooth),
            "--skip-ttc",
        ]
        result = subprocess.run(cmd, cwd=Path.cwd())
        if result.returncode != 0:
            print("  Warning: Thesis figure generation had issues")
    
    # Generate figure 19
    print("\nGenerating figure 19...")
    try:
        plot_figure19(
            ppo_monitor_dir=Path(args.ppo_monitor_dir),
            dqn_monitor_dir=Path(args.dqn_monitor_dir),
            out_path=out_dir / "图19_PPO与dqn训练收敛曲线.png",
            smooth=args.smooth,
        )
    except Exception as e:
        print(f"  Error: {e}")
    
    # Generate figure 20
    print("\nGenerating figure 20...")
    try:
        plot_figure20(
            ablation_dir=Path(args.ablation_dir),
            out_path=out_dir / "图20消融实验总览图.png",
        )
    except Exception as e:
        print(f"  Error: {e}")
    
    print(f"\n[OK] All figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
