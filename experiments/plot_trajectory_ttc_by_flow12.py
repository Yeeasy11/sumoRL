#!/usr/bin/env python3
"""Generate lane-movement trajectory and TTC figures (4 directions x 3 turns)."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment


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
    "DQN": "#DC2626",
}
METHOD_LABELS = {
    "Rule-based": "IDM基线",
    "PPO": "PPO",
    "DQN": "DQN",
}
DIR_CN = {"n": "北向", "e": "东向", "s": "南向", "w": "西向"}
TURN_CN = {"left": "左转", "straight": "直行", "right": "右转"}
DIR_ORDER = ["n", "e", "s", "w"]
TURN_ORDER = ["left", "straight", "right"]


def _load_model(method: str, model_path: Optional[str]):
    if not model_path:
        return None
    if method == "PPO":
        from stable_baselines3 import PPO

        return PPO.load(model_path)
    if method == "DQN":
        from stable_baselines3 import DQN

        return DQN.load(model_path)
    return None


def _parse_route(route_id: str) -> tuple[str, str]:
    # Expected pattern: route_n_left, route_w_straight, ...
    parts = str(route_id).split("_")
    if len(parts) >= 3 and parts[0] == "route":
        in_dir = parts[1]
        turn = parts[2]
        if in_dir in DIR_ORDER and turn in TURN_ORDER:
            return in_dir, turn
    return "?", "unknown"


def _collect_method(
    method: str,
    route_file: str,
    net_file: str,
    seconds: int,
    delta_time: int,
    seed: int,
    model_path: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_ts = method == "Rule-based"
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=None,
        use_gui=False,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=0 if delta_time <= 2 else 2,
        single_agent=True,
        fixed_ts=fixed_ts,
        sumo_seed=seed,
    )

    model = None if fixed_ts else _load_model(method, model_path)
    obs, _ = env.reset()

    ts_id = env.ts_ids[0]
    in_lanes = list(env.traffic_signals[ts_id].lanes)
    out_lanes: set[str] = set()
    for group in env.sumo.trafficlight.getControlledLinks(ts_id):
        if not group:
            continue
        for link in group:
            if len(link) >= 2 and link[1]:
                out_lanes.add(str(link[1]))

    lanes = sorted(set(in_lanes) | out_lanes)
    lane_len = {ln: env.sumo.lane.getLength(ln) for ln in lanes}

    traj_rows: list[dict] = []
    ttc_rows: list[dict] = []

    terminated, truncated = False, False
    while not (terminated or truncated):
        sim_t = float(env.sim_step)

        for lane in lanes:
            veh_ids = env.sumo.lane.getLastStepVehicleIDs(lane)
            ll = lane_len[lane]
            lane_role = "incoming" if lane in in_lanes else "outgoing"
            for vid in veh_ids:
                pos = float(env.sumo.vehicle.getLanePosition(vid))
                speed = float(env.sumo.vehicle.getSpeed(vid))
                dist_to_stop = max(0.0, ll - pos)
                if lane_role == "incoming":
                    s_from_stopline = -dist_to_stop
                else:
                    s_from_stopline = max(0.0, pos)

                route_id = env.sumo.vehicle.getRouteID(vid)
                in_dir, turn = _parse_route(route_id)

                traj_rows.append(
                    {
                        "method": method,
                        "time": sim_t,
                        "veh_id": vid,
                        "lane": lane,
                        "lane_role": lane_role,
                        "route_id": route_id,
                        "in_dir": in_dir,
                        "turn": turn,
                        "s_from_stopline": s_from_stopline,
                        "speed": speed,
                    }
                )

                lead = env.sumo.vehicle.getLeader(vid)
                if lead is None:
                    continue
                lead_id, gap = lead
                if gap is None:
                    continue
                v_f = speed
                v_l = float(env.sumo.vehicle.getSpeed(lead_id))
                rel = v_f - v_l
                ttc = np.inf if rel <= 1e-6 else float(gap) / rel

                ttc_rows.append(
                    {
                        "method": method,
                        "time": sim_t,
                        "veh_id": vid,
                        "lane": lane,
                        "route_id": route_id,
                        "in_dir": in_dir,
                        "turn": turn,
                        "ttc": ttc,
                    }
                )

        if fixed_ts:
            action = {}
        else:
            if model is None:
                raise ValueError(f"{method} requires model path")
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)

    env.close()
    return pd.DataFrame(traj_rows), pd.DataFrame(ttc_rows)


def _plot_trajectory_by_direction(
    traj: pd.DataFrame,
    outdir: Path,
    methods: list[str],
    pair_title: str,
    file_prefix: str,
) -> None:
    for in_dir in DIR_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), sharex=True, sharey=True)

        for ax, turn in zip(axes, TURN_ORDER):
            sub = traj[(traj["in_dir"] == in_dir) & (traj["turn"] == turn)]
            for method in methods:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                if len(msub) > 14000:
                    msub = msub.sample(14000, random_state=7)
                marker_size = 0.7 if turn == "left" else 2.1
                marker_alpha = 0.06 if turn == "left" else 0.18
                ax.scatter(
                    msub["time"],
                    msub["s_from_stopline"],
                    s=marker_size,
                    alpha=marker_alpha,
                    c=METHOD_COLORS[method],
                    edgecolors="none",
                    label=METHOD_LABELS[method],
                )

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
            for h in uniq.values():
                h.set_sizes([170])
            fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.85))
        fig.suptitle(f"轨迹时空图：{DIR_CN[in_dir]}进口三流向（{pair_title}）", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.78])
        fig.savefig(outdir / f"图21_{file_prefix}轨迹时空图_{DIR_CN[in_dir]}三流向.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_ttc_by_direction(
    ttc: pd.DataFrame,
    outdir: Path,
    methods: list[str],
    pair_title: str,
    file_prefix: str,
) -> None:
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
            ax.set_ylim(0, 20)
            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]}")
            ax.set_xlabel("时间 (s)")
            ax.grid(alpha=0.20)

        axes[0].set_ylabel("最小TTC (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.85))
        fig.suptitle(f"TTC时序图：{DIR_CN[in_dir]}进口三流向（{pair_title}）", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.78])
        fig.savefig(outdir / f"图22_{file_prefix}TTC时序图_{DIR_CN[in_dir]}三流向.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_ttc_per_movement(ttc: pd.DataFrame, outdir: Path, methods: list[str], file_prefix: str, pair_title: str) -> None:
    d = ttc[np.isfinite(ttc["ttc"]) & (ttc["ttc"] > 0) & (ttc["ttc"] <= 20.0)].copy()
    for in_dir in DIR_ORDER:
        for turn in TURN_ORDER:
            fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.4))
            sub = d[(d["in_dir"] == in_dir) & (d["turn"] == turn)]

            for method in methods:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                s = msub.groupby("time", as_index=False)["ttc"].min().sort_values("time")
                s["ttc_smooth"] = s["ttc"].rolling(window=9, min_periods=1).mean()
                ax.plot(s["time"], s["ttc_smooth"], color=METHOD_COLORS[method], linewidth=1.8, alpha=0.92, label=METHOD_LABELS[method])

            if sub.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=10)

            ax.axhline(3.0, color="#DC2626", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.set_ylim(0, 20)
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("最小TTC (s)")
            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]} TTC时序（{pair_title}）")
            ax.grid(alpha=0.22)
            ax.legend(loc="upper right", frameon=False, fontsize=9)

            fig.tight_layout(pad=1.2)
            fig.savefig(outdir / f"图22a_{file_prefix}TTC时序图_{DIR_CN[in_dir]}_{TURN_CN[turn]}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def _plot_ttc_grouped_by_turn(
    ttc: pd.DataFrame,
    outdir: Path,
    methods: list[str],
    pair_title: str,
    file_prefix: str,
) -> None:
    d = ttc[np.isfinite(ttc["ttc"]) & (ttc["ttc"] > 0) & (ttc["ttc"] <= 20.0)].copy()
    for turn in TURN_ORDER:
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for ax, in_dir in zip(axes_flat, DIR_ORDER):
            sub = d[(d["in_dir"] == in_dir) & (d["turn"] == turn)]
            for method in methods:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                s = msub.groupby("time", as_index=False)["ttc"].min().sort_values("time")
                s["ttc_smooth"] = s["ttc"].rolling(window=9, min_periods=1).mean()
                ax.plot(s["time"], s["ttc_smooth"], color=METHOD_COLORS[method], linewidth=1.7, alpha=0.90, label=METHOD_LABELS[method])

            if sub.empty:
                ax.text(0.5, 0.5, "无样本", transform=ax.transAxes, ha="center", va="center", fontsize=10)

            ax.axhline(3.0, color="#DC2626", linestyle="--", linewidth=0.9, alpha=0.8)
            ax.set_title(f"{DIR_CN[in_dir]}-{TURN_CN[turn]}")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("最小TTC (s)")
            ax.set_ylim(0, 20)
            ax.grid(alpha=0.22)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.85))
        fig.suptitle(f"TTC时序图分组：{TURN_CN[turn]}（四方向，{pair_title}）", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout(pad=1.1, rect=[0.02, 0.03, 0.98, 0.78])
        fig.savefig(outdir / f"图22b_{file_prefix}TTC时序图_{TURN_CN[turn]}_四方向.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_trajectory_4x3(
    traj: pd.DataFrame,
    outdir: Path,
    methods: list[str],
    pair_title: str,
    file_prefix: str,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 14.5), sharex=True, sharey=True)
    for r, in_dir in enumerate(DIR_ORDER):
        for c, turn in enumerate(TURN_ORDER):
            ax = axes[r, c]
            sub = traj[(traj["in_dir"] == in_dir) & (traj["turn"] == turn)]
            for method in methods:
                msub = sub[sub["method"] == method]
                if msub.empty:
                    continue
                if len(msub) > 12000:
                    msub = msub.sample(12000, random_state=7)
                marker_size = 0.7 if turn == "left" else 2.1
                marker_alpha = 0.06 if turn == "left" else 0.18
                ax.scatter(
                    msub["time"],
                    msub["s_from_stopline"],
                    s=marker_size,
                    alpha=marker_alpha,
                    c=METHOD_COLORS[method],
                    edgecolors="none",
                    label=METHOD_LABELS[method],
                )
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
        for h in uniq.values():
            h.set_sizes([120])
        fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=11)
    fig.suptitle(f"轨迹时空图 4×3 全流向汇总（{pair_title}）", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.94])
    fig.savefig(outdir / f"图21c_{file_prefix}轨迹时空图_4x3汇总.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ttc_4x3(
    ttc: pd.DataFrame,
    outdir: Path,
    methods: list[str],
    pair_title: str,
    file_prefix: str,
) -> None:
    d = ttc[np.isfinite(ttc["ttc"]) & (ttc["ttc"] > 0) & (ttc["ttc"] <= 20.0)].copy()
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 14.5), sharex=True, sharey=True)
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
        fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", ncol=len(uniq), frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=11)
    fig.suptitle(f"TTC时序图 4×3 全流向汇总（{pair_title}）", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(pad=1.2, rect=[0.02, 0.03, 0.98, 0.94])
    fig.savefig(outdir / f"图22c_{file_prefix}TTC时序图_4x3汇总.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate trajectory/TTC figures by 12 movements (4x3).")
    p.add_argument("--route", type=str, default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml")
    p.add_argument("--net", type=str, default="sumo_rl/nets/4way-single-intersection/4way-single-intersection.net.xml")
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--delta-time", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ppo-model", type=str, default="models/4way-single-intersection/ppo_4way_turns_seed1.zip")
    p.add_argument("--dqn-model", type=str, default="models/4way-single-intersection/dqn_4way_turns_seed1.zip")
    p.add_argument("--skip-rule", action="store_true", help="Skip Rule-based collection")
    p.add_argument("--skip-ppo", action="store_true", help="Skip PPO collection")
    p.add_argument("--skip-dqn", action="store_true", help="Skip DQN collection")
    p.add_argument("--outdir", type=str, default="outputs/figures")
    p.add_argument("--traj-csv", type=str, default="", help="Optional existing trajectory CSV path.")
    p.add_argument("--ttc-csv", type=str, default="", help="Optional existing TTC CSV path.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    traj_all: list[pd.DataFrame] = []
    ttc_all: list[pd.DataFrame] = []

    if args.traj_csv and args.ttc_csv:
        traj_df = pd.read_csv(args.traj_csv)
        ttc_df = pd.read_csv(args.ttc_csv)
        pairs = [
            (["Rule-based", "PPO"], "IDM基线与PPO", ""),
            (["PPO", "DQN"], "PPO与DQN", "PPO与DQN_"),
        ]
        for methods, pair_title, file_prefix in pairs:
            t_sub = traj_df[traj_df["method"].isin(methods)].copy()
            c_sub = ttc_df[ttc_df["method"].isin(methods)].copy()
            _plot_trajectory_by_direction(t_sub, outdir, methods, pair_title, file_prefix)
            _plot_ttc_by_direction(c_sub, outdir, methods, pair_title, file_prefix)
            _plot_ttc_per_movement(c_sub, outdir, methods, file_prefix, pair_title)
            _plot_ttc_grouped_by_turn(c_sub, outdir, methods, pair_title, file_prefix)
        print(f"Loaded trajectory csv: {args.traj_csv}")
        print(f"Loaded ttc csv: {args.ttc_csv}")
        print("Saved trajectory/TTC figures including 12 movement TTC plots and grouped TTC views.")
        return

    if not args.skip_rule:
        print("Collecting Rule-based trajectory/TTC...")
        traj, ttc = _collect_method("Rule-based", args.route, args.net, args.seconds, args.delta_time, args.seed, None)
        traj_all.append(traj)
        ttc_all.append(ttc)

    if not args.skip_ppo:
        print("Collecting PPO trajectory/TTC...")
        traj, ttc = _collect_method("PPO", args.route, args.net, args.seconds, args.delta_time, args.seed, args.ppo_model)
        traj_all.append(traj)
        ttc_all.append(ttc)

    if not args.skip_dqn:
        print("Collecting DQN trajectory/TTC...")
        traj, ttc = _collect_method("DQN", args.route, args.net, args.seconds, args.delta_time, args.seed, args.dqn_model)
        traj_all.append(traj)
        ttc_all.append(ttc)

    if not traj_all or not ttc_all:
        raise RuntimeError("No method selected; nothing to plot")

    traj_df = pd.concat(traj_all, ignore_index=True)
    ttc_df = pd.concat(ttc_all, ignore_index=True)

    traj_df.to_csv(outdir / "trajectory_samples_by_flow12.csv", index=False, encoding="utf-8-sig")
    ttc_df.to_csv(outdir / "ttc_samples_by_flow12.csv", index=False, encoding="utf-8-sig")

    pairs = [
        (["Rule-based", "PPO"], "IDM基线与PPO", ""),
        (["PPO", "DQN"], "PPO与DQN", "PPO与DQN_"),
    ]
    for methods, pair_title, file_prefix in pairs:
        t_sub = traj_df[traj_df["method"].isin(methods)].copy()
        c_sub = ttc_df[ttc_df["method"].isin(methods)].copy()
        _plot_trajectory_by_direction(t_sub, outdir, methods, pair_title, file_prefix)
        _plot_ttc_by_direction(c_sub, outdir, methods, pair_title, file_prefix)
        _plot_ttc_per_movement(c_sub, outdir, methods, file_prefix, pair_title)
        _plot_ttc_grouped_by_turn(c_sub, outdir, methods, pair_title, file_prefix)
        _plot_trajectory_4x3(t_sub, outdir, methods, pair_title, file_prefix)
        _plot_ttc_4x3(c_sub, outdir, methods, pair_title, file_prefix)

    print(f"Saved trajectory csv: {outdir / 'trajectory_samples_by_flow12.csv'}")
    print(f"Saved ttc csv: {outdir / 'ttc_samples_by_flow12.csv'}")
    print("Saved trajectory/TTC figures including 12 movement TTC plots and grouped TTC views.")


if __name__ == "__main__":
    main()
