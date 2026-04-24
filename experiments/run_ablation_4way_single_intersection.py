import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction


NET_FILE = "sumo_rl/nets/2way-single-intersection/single-intersection-fourdir-straight.net.xml"


class PhaseOnlyObservation(ObservationFunction):
    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        return np.array(phase_id + min_green, dtype=np.float32)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1, dtype=np.float32),
        )


def make_additional_sumo_cmd(tripinfo_path: Path, collision_path: Path) -> str:
    tripinfo_path.parent.mkdir(parents=True, exist_ok=True)
    collision_path.parent.mkdir(parents=True, exist_ok=True)
    return f"--tripinfo-output {tripinfo_path.resolve()} --collision-output {collision_path.resolve()}"


def parse_tripinfo_avg_duration(tripinfo_path: Path) -> float:
    if not tripinfo_path.exists():
        return float("nan")
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()
    durations: list[float] = []
    for ti in root.iter("tripinfo"):
        duration = ti.get("duration")
        if duration is not None:
            try:
                durations.append(float(duration))
            except ValueError:
                pass
    return float(sum(durations) / len(durations)) if durations else float("nan")


def parse_collision_count(collision_path: Path) -> int:
    if not collision_path.exists():
        return 0
    try:
        tree = ET.parse(collision_path)
    except ET.ParseError:
        return 0
    root = tree.getroot()
    return sum(1 for _ in root.iter("collision"))


def resolve_sumo_rl_csv(prefix: str, episode: int) -> Path:
    p = Path(prefix)
    pattern = f"{p.name}_conn*_ep{episode}.csv"
    matches = sorted(p.parent.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No SUMO-RL CSV found for prefix={prefix}, episode={episode}")
    return matches[0]


def env_metrics_from_csv(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)

    def last_or(col: str, default: float) -> float:
        if col not in df.columns:
            return float(default)
        v = df[col].iloc[-1]
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    return {
        "total_arrived": int(last_or("system_total_arrived", 0.0)),
        "mean_waiting_time": float(last_or("system_mean_waiting_time", float("nan"))),
        "mean_speed": float(last_or("system_mean_speed", float("nan"))),
    }


def build_env(
    net_file: str,
    route_file: str,
    out_csv: str,
    seconds: int,
    delta_time: int,
    seed: int,
    reward_mode: str,
    obs_mode: str,
    collision_penalty: float,
    monitor_file: Optional[str],
    additional_sumo_cmd: str = "",
) -> Any:
    yellow_time = 0 if delta_time <= 2 else 2
    observation_class = PhaseOnlyObservation if obs_mode == "phase_only" else None

    if reward_mode == "full":
        reward_fn = "pressure"
    elif reward_mode == "default":
        reward_fn = "diff-waiting-time"
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    kwargs = dict(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        single_agent=True,
        fixed_ts=False,
        sumo_seed=seed,
        reward_fn=reward_fn,
    )
    if additional_sumo_cmd:
        kwargs["additional_sumo_cmd"] = additional_sumo_cmd
    if observation_class is not None:
        kwargs["observation_class"] = observation_class

    env = SumoEnvironment(**kwargs)
    if monitor_file:
        return Monitor(env, filename=monitor_file)
    return env


def train_one_setting(
    route_file: str,
    seconds: int,
    delta_time: int,
    timesteps: int,
    lr: float,
    gamma: float,
    n_steps: int,
    n_epochs: int,
    batch_size: int,
    hidden: list[int],
    seed: int,
    collision_penalty: float,
    reward_mode: str,
    obs_mode: str,
    run_dir: Path,
    model_root: Path,
    allow_collisions: bool,
    collision_action: str,
    experiment_time: str,
) -> Path:
    tag = f"reward{reward_mode}_obs{obs_mode}_seed{seed}"
    train_csv_prefix = str(run_dir / "train_csv" / tag)
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = str(monitor_dir / f"{tag}.monitor.csv")

    extra_cmd = f"--collision.action {collision_action}" if allow_collisions else ""
    env = build_env(
        net_file=NET_FILE,
        route_file=route_file,
        out_csv=train_csv_prefix,
        seconds=seconds,
        delta_time=delta_time,
        seed=seed,
        reward_mode=reward_mode,
        obs_mode=obs_mode,
        collision_penalty=collision_penalty,
        monitor_file=monitor_file,
        additional_sumo_cmd=extra_cmd,
    )

    effective_n_steps = min(max(2, n_steps), timesteps)
    effective_batch_size = min(batch_size, effective_n_steps)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        n_steps=effective_n_steps,
        n_epochs=n_epochs,
        batch_size=effective_batch_size,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        seed=seed,
    )
    print(
        "[INFO] Training setting "
        f"reward={reward_mode}, obs={obs_mode}, timesteps={timesteps}, "
        f"n_steps={effective_n_steps}, batch_size={effective_batch_size}"
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)

    model_save_root = model_root / experiment_time
    model_save_root.mkdir(parents=True, exist_ok=True)
    model_path = model_save_root / f"ppo_ablation_4way_{tag}.zip"
    model.save(str(model_path))
    env.close()
    return model_path


def evaluate_one_run(
    model_path: Path,
    route_file: str,
    seconds: int,
    delta_time: int,
    seed: int,
    collision_penalty: float,
    reward_mode: str,
    obs_mode: str,
    run_id: int,
    eval_dir: Path,
    allow_collisions: bool,
    collision_action: str,
) -> dict[str, Any]:
    run_tag = f"reward{reward_mode}_obs{obs_mode}_seed{seed}_run{run_id}"
    tripinfo_path = eval_dir / "tripinfo" / f"{run_tag}.xml"
    collision_path = eval_dir / "collisions" / f"{run_tag}.xml"
    sumo_cmd = make_additional_sumo_cmd(tripinfo_path, collision_path)
    if allow_collisions:
        sumo_cmd = f"{sumo_cmd} --collision.action {collision_action}".strip()
    eval_csv_prefix = str(eval_dir / "csv" / run_tag)

    env = build_env(
        net_file=NET_FILE,
        route_file=route_file,
        out_csv=eval_csv_prefix,
        seconds=seconds,
        delta_time=delta_time,
        seed=seed,
        reward_mode=reward_mode,
        obs_mode=obs_mode,
        collision_penalty=collision_penalty,
        monitor_file=None,
        additional_sumo_cmd=sumo_cmd,
    )
    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    env.save_csv(eval_csv_prefix, run_id)
    env.close()

    csv_path = resolve_sumo_rl_csv(eval_csv_prefix, run_id)
    metrics = env_metrics_from_csv(csv_path)

    return {
        "reward_mode": reward_mode,
        "obs_mode": obs_mode,
        "run": run_id,
        "seed": seed,
        "total_arrived": int(metrics["total_arrived"]),
        "mean_waiting_time": float(metrics["mean_waiting_time"]),
        "mean_speed": float(metrics["mean_speed"]),
        "avg_travel_time": float(parse_tripinfo_avg_duration(tripinfo_path)),
        "collisions": int(parse_collision_count(collision_path)),
        "model_path": str(model_path),
        "eval_csv": str(csv_path),
        "tripinfo_path": str(tripinfo_path),
        "collision_path": str(collision_path),
    }


def plot_ablation(summary_csv: Path, out_dir: Path, seconds: int) -> None:
    df = pd.read_csv(summary_csv)
    df["throughput"] = (df["total_arrived"] / float(seconds)) * 3600.0
    df["setting"] = df.apply(lambda r: f"{r['reward_mode']} / {r['obs_mode']}", axis=1)

    order = ["full / full", "full / phase_only", "default / full", "default / phase_only"]
    df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)
    df = df.sort_values("setting")

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fig.suptitle("4way single intersection PPO ablation", fontsize=14, fontweight="bold")
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

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "4way_ablation_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train and evaluate PPO ablations on the 4-direction single-lane intersection.",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
    )
    prs.add_argument("--seconds", type=int, default=1200)
    prs.add_argument("--delta-time", type=int, default=1)
    prs.add_argument("--eval-delta-time", type=int, default=3)
    prs.add_argument("--timesteps", type=int, default=250_000)
    prs.add_argument("--lr", type=float, default=3e-4)
    prs.add_argument("--gamma", type=float, default=0.99)
    prs.add_argument("--n-steps", type=int, default=2048)
    prs.add_argument("--n-epochs", type=int, default=10)
    prs.add_argument("--batch-size", type=int, default=64)
    prs.add_argument("--hidden", type=int, nargs=2, default=[128, 128])
    prs.add_argument("--seed", type=int, default=0)
    prs.add_argument("--runs", type=int, default=5)
    prs.add_argument("--collision-penalty", type=float, default=100.0)
    prs.add_argument("--allow-collisions", action="store_true", help="Enable collision handling mode in SUMO.")
    prs.add_argument(
        "--collision-action",
        type=str,
        choices=["none", "warn", "teleport", "remove"],
        default="warn",
        help="SUMO collision action when --allow-collisions is enabled.",
    )
    prs.add_argument("--outdir", type=str, default="outputs/4way-single-intersection/ablation")
    prs.add_argument("--model-root", type=str, default="models/4way-single-intersection/ablation")
    prs.add_argument("--fig-dir", type=str, default="outputs/figures")
    args = prs.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)

    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    settings = [
        ("full", "full"),
        ("full", "phase_only"),
        ("default", "full"),
        ("default", "phase_only"),
    ]

    train_rows = []
    for reward_mode, obs_mode in settings:
        tag_dir = out_dir / f"{experiment_time}_{reward_mode}_{obs_mode}"
        model_path = train_one_setting(
            route_file=args.route,
            seconds=args.seconds,
            delta_time=args.delta_time,
            timesteps=args.timesteps,
            lr=args.lr,
            gamma=args.gamma,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            hidden=args.hidden,
            seed=args.seed,
            collision_penalty=args.collision_penalty,
            reward_mode=reward_mode,
            obs_mode=obs_mode,
            run_dir=tag_dir,
            model_root=model_root,
            allow_collisions=args.allow_collisions,
            collision_action=args.collision_action,
            experiment_time=experiment_time,
        )

        eval_dir = tag_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for run_id in range(1, args.runs + 1):
            sumo_seed = args.seed + run_id
            rows.append(
                evaluate_one_run(
                    model_path=model_path,
                    route_file=args.route,
                    seconds=args.seconds,
                    delta_time=args.eval_delta_time,
                    seed=sumo_seed,
                    collision_penalty=args.collision_penalty,
                    reward_mode=reward_mode,
                    obs_mode=obs_mode,
                    run_id=run_id,
                    eval_dir=eval_dir,
                    allow_collisions=args.allow_collisions,
                    collision_action=args.collision_action,
                )
            )

        eval_df = pd.DataFrame(rows)
        eval_df["throughput"] = (eval_df["total_arrived"] / float(args.seconds)) * 3600.0
        eval_summary = (
            eval_df.groupby(["reward_mode", "obs_mode"], as_index=False)[
                ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions", "throughput"]
            ]
            .mean()
        )
        eval_df.to_csv(eval_dir / "eval_runs.csv", index=False)
        eval_summary.to_csv(eval_dir / "eval_summary.csv", index=False)
        train_rows.extend(eval_summary.to_dict(orient="records"))

    summary_df = pd.DataFrame(train_rows)
    summary_path = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_ablation(summary_path, Path(args.fig_dir), seconds=args.seconds)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {Path(args.fig_dir) / '4way_ablation_summary.png'}")


if __name__ == "__main__":
    main()