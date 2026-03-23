import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

import matplotlib.pyplot as plt

from sumo_rl import SumoEnvironment


def _run_collect(
    method: str,
    route_file: str,
    net_file: str,
    seconds: int,
    delta_time: int,
    seed: int,
    ppo_model: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    model = None
    if not fixed_ts:
        if not ppo_model:
            raise ValueError("PPO mode requires --ppo-model")
        try:
            from stable_baselines3 import PPO
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "stable_baselines3 is required for PPO runs. Install stable-baselines3 and torch."
            ) from e
        model = PPO.load(ppo_model)

    obs, _ = env.reset()

    # Controlled incoming lanes at the single intersection
    ts_id = env.ts_ids[0]
    lanes = env.traffic_signals[ts_id].lanes
    lane_len = {ln: env.sumo.lane.getLength(ln) for ln in lanes}

    traj_rows: List[Dict] = []
    ttc_rows: List[Dict] = []

    terminated, truncated = False, False
    while not (terminated or truncated):
        sim_t = float(env.sim_step)

        # Collect per-vehicle trajectory samples on incoming lanes
        for lane in lanes:
            veh_ids = env.sumo.lane.getLastStepVehicleIDs(lane)
            ll = lane_len[lane]
            for vid in veh_ids:
                pos = float(env.sumo.vehicle.getLanePosition(vid))
                spd = float(env.sumo.vehicle.getSpeed(vid))
                dist_to_stop = max(0.0, ll - pos)
                traj_rows.append(
                    {
                        "method": method,
                        "time": sim_t,
                        "veh_id": vid,
                        "lane": lane,
                        "lane_pos": pos,
                        "dist_to_stop": dist_to_stop,
                        "speed": spd,
                    }
                )

                # TTC w.r.t lane leader
                lead = env.sumo.vehicle.getLeader(vid)
                if lead is None:
                    continue
                lead_id, gap = lead
                if gap is None:
                    continue
                v_f = spd
                v_l = float(env.sumo.vehicle.getSpeed(lead_id))
                rel = v_f - v_l
                if rel <= 1e-6:
                    ttc = np.inf
                else:
                    ttc = float(gap) / rel
                ttc_rows.append(
                    {
                        "method": method,
                        "time": sim_t,
                        "veh_id": vid,
                        "leader_id": lead_id,
                        "gap": float(gap),
                        "v_f": v_f,
                        "v_l": v_l,
                        "rel_speed": rel,
                        "ttc": ttc,
                    }
                )

        if fixed_ts:
            action = {}
        else:
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)

    env.close()
    return pd.DataFrame(traj_rows), pd.DataFrame(ttc_rows)


def _plot_spacetime(df: pd.DataFrame, out_path: Path, max_points: int = 60000) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    methods = ["Rule-based", "PPO"]
    colors = {"Rule-based": "#e76f51", "PPO": "#2a9d8f"}

    for ax, m in zip(axes, methods):
        sub = df[df["method"] == m]
        if len(sub) > max_points:
            sub = sub.sample(max_points, random_state=0)
        ax.scatter(
            sub["time"],
            sub["dist_to_stop"],
            s=4,
            alpha=0.25,
            c=colors[m],
            edgecolors="none",
        )
        ax.set_title(m)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Distance to Stop Line (m)")
    fig.suptitle("Space-Time Trajectory Diagram (Incoming Lanes)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_ttc(df: pd.DataFrame, out_path: Path) -> None:
    d = df[np.isfinite(df["ttc"]) & (df["ttc"] > 0)].copy()
    d = d[d["ttc"] <= 20.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for m, c in [("Rule-based", "#e76f51"), ("PPO", "#2a9d8f")]:
        sub = d[d["method"] == m]
        axes[0].hist(sub["ttc"], bins=40, alpha=0.45, color=c, label=m, density=True)

    axes[0].set_xlabel("TTC (s)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("TTC Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # min TTC per time step
    agg = d.groupby(["method", "time"], as_index=False)["ttc"].min()
    for m, c in [("Rule-based", "#e76f51"), ("PPO", "#2a9d8f")]:
        sub = agg[agg["method"] == m]
        axes[1].plot(sub["time"], sub["ttc"], color=c, alpha=0.8, label=m)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Min TTC (s)")
    axes[1].set_title("Minimum TTC Over Time")
    axes[1].set_ylim(0, 20)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("TTC Analysis (Leader-Follower on Incoming Lanes)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate space-time trajectory and TTC figures for thesis.")
    p.add_argument("--route", type=str, default="sumo_rl/nets/single-intersection/single-intersection.rou.xml")
    p.add_argument("--net", type=str, default="sumo_rl/nets/single-intersection/single-intersection.net.xml")
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--delta-time", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ppo-model", type=str, default="models/ppo_final.zip")
    p.add_argument("--outdir", type=str, default="outputs/thesis_figures")
    args = p.parse_args()

    outdir = Path(args.outdir)

    print("Collecting Rule-based trajectory and TTC samples...")
    traj_rule, ttc_rule = _run_collect(
        method="Rule-based",
        route_file=args.route,
        net_file=args.net,
        seconds=args.seconds,
        delta_time=args.delta_time,
        seed=args.seed,
        ppo_model=None,
    )

    print("Collecting PPO trajectory and TTC samples...")
    traj_ppo, ttc_ppo = _run_collect(
        method="PPO",
        route_file=args.route,
        net_file=args.net,
        seconds=args.seconds,
        delta_time=args.delta_time,
        seed=args.seed,
        ppo_model=args.ppo_model,
    )

    traj = pd.concat([traj_rule, traj_ppo], ignore_index=True)
    ttc = pd.concat([ttc_rule, ttc_ppo], ignore_index=True)

    outdir.mkdir(parents=True, exist_ok=True)
    traj.to_csv(outdir / "trajectory_samples.csv", index=False)
    ttc.to_csv(outdir / "ttc_samples.csv", index=False)

    _plot_spacetime(traj, outdir / "trajectory_spacetime.png")
    _plot_ttc(ttc, outdir / "ttc_analysis.png")

    print(f"Saved: {outdir / 'trajectory_spacetime.png'}")
    print(f"Saved: {outdir / 'ttc_analysis.png'}")
    print(f"Saved: {outdir / 'trajectory_samples.csv'}")
    print(f"Saved: {outdir / 'ttc_samples.csv'}")


if __name__ == "__main__":
    main()
