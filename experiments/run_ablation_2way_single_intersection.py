from typing import Optional
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction
from plot_font_cn import configure_chinese_font


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


def reward_full_with_collision_penalty(ts, collision_penalty: float = 100.0) -> float:
    base = ts._diff_waiting_time_reward()
    try:
        num = ts.env.sumo.simulation.getCollidingVehiclesNumber()
    except Exception:
        num = 0
    return float(base - collision_penalty * float(num))


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
        reward_fn = lambda ts: reward_full_with_collision_penalty(ts, collision_penalty=collision_penalty)  # noqa: E731
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
    net_file: str,
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
) -> Path:
    tag = f"reward{reward_mode}_obs{obs_mode}_seed{seed}"
    train_csv_prefix = str(run_dir / "train_csv" / tag)
    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = str(monitor_dir / f"{tag}.monitor.csv")

    env = build_env(
        net_file=net_file,
        route_file=route_file,
        out_csv=train_csv_prefix,
        seconds=seconds,
        delta_time=delta_time,
        seed=seed,
        reward_mode=reward_mode,
        obs_mode=obs_mode,
        collision_penalty=collision_penalty,
        monitor_file=monitor_file,
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
        policy_kwargs={"net_arch": list(hidden)},
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

    model_dir = model_root
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"ppo_ablation_2way_{tag}.zip"
    model.save(str(model_path))
    env.close()
    return model_path


def evaluate_one_run(
    model_path: Path,
    net_file: str,
    route_file: str,
    seconds: int,
    delta_time: int,
    seed: int,
    collision_penalty: float,
    reward_mode: str,
    obs_mode: str,
    run_id: int,
    eval_dir: Path,
) -> dict[str, Any]:
    run_tag = f"reward{reward_mode}_obs{obs_mode}_seed{seed}_run{run_id}"
    tripinfo_path = eval_dir / "tripinfo" / f"{run_tag}.xml"
    collision_path = eval_dir / "collisions" / f"{run_tag}.xml"
    sumo_cmd = make_additional_sumo_cmd(tripinfo_path, collision_path)
    eval_csv_prefix = str(eval_dir / "csv" / run_tag)

    env = build_env(
        net_file=net_file,
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


def _setting_cn_label(reward_mode: str, obs_mode: str) -> str:
    mapping = {
        ("full", "full"): "完整奖励 + 完整状态",
        ("full", "phase_only"): "完整奖励 + 相位状态",
    }
    return mapping.get((reward_mode, obs_mode), f"{reward_mode} + {obs_mode}")


def plot_ablation(summary_csv: Path, out_dir: Path, seconds: int, figures_repro_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    font_name = configure_chinese_font()
    print(f"[INFO] Matplotlib 中文字体: {font_name}")

    df = pd.read_csv(summary_csv)
    df["setting"] = df.apply(lambda r: _setting_cn_label(str(r["reward_mode"]), str(r["obs_mode"])), axis=1)
    df["throughput"] = (df["total_arrived"] / float(seconds)) * 3600.0

    metrics = [
        ("avg_travel_time", "平均旅行时间 (s)"),
        ("mean_waiting_time", "平均等待时间 (s)"),
        ("throughput", "吞吐量 (veh/h)"),
        ("mean_speed", "平均速度 (m/s)"),
    ]
    colors = ["#4C72B0", "#55A868"]
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_repro_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    axes = axes.flatten()
    x = np.arange(len(df))

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        ax.bar(x, df[col].values, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df["setting"].tolist(), rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [Patch(color=colors[i % len(colors)], label=df["setting"].iloc[i]) for i in range(len(df))]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=True, title="消融设置")
    fig.suptitle("2way-single-intersection 消融实验总览", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    combined_out = out_dir / "ablation_2way_combined_cn.png"
    plt.savefig(combined_out, dpi=240)
    print(f"Wrote: {combined_out}")

    repro_out = figures_repro_dir / "ablation_2way_combined_cn.png"
    plt.savefig(repro_out, dpi=240)
    print(f"Wrote: {repro_out}")
    plt.close(fig)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run PPO ablation experiment on 2way-single-intersection and export CSV + charts.",
    )
    prs.add_argument("--net-file", type=str, default="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml")
    prs.add_argument(
        "--route-file",
        type=str,
        default="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    )
    prs.add_argument("--seconds", type=int, default=600)
    prs.add_argument("--delta-time", type=int, default=3)
    prs.add_argument("--timesteps", type=int, default=20_000)
    prs.add_argument("--eval-runs", type=int, default=3)
    prs.add_argument("--seed", type=int, default=0)

    prs.add_argument("--lr", type=float, default=3e-4)
    prs.add_argument("--gamma", type=float, default=0.99)
    prs.add_argument("--n-steps", type=int, default=2048)
    prs.add_argument("--n-epochs", type=int, default=10)
    prs.add_argument("--batch-size", type=int, default=64)
    prs.add_argument("--hidden", type=int, nargs=2, default=[64, 64])
    prs.add_argument("--collision-penalty", type=float, default=100.0)

    prs.add_argument(
        "--outdir",
        type=str,
        default="outputs/eval_repro/2way-single-intersection/ablation",
        help="Root directory for train/eval data and figures.",
    )
    prs.add_argument(
        "--model-root",
        type=str,
        default="models/repro/2way_ablation",
        help="Directory for trained ablation models.",
    )
    prs.add_argument(
        "--figures-repro-dir",
        type=str,
        default="outputs/figures_repro",
        help="Centralized figure directory for reproducible reports.",
    )
    args = prs.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.outdir) / stamp
    root.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    figures_repro_dir = Path(args.figures_repro_dir)
    figures_repro_dir.mkdir(parents=True, exist_ok=True)

    settings = [
        ("full", "full"),
        ("full", "phase_only"),
    ]

    model_map: dict[tuple[str, str], Path] = {}
    for i, (reward_mode, obs_mode) in enumerate(settings):
        setting_seed = args.seed + i
        setting_dir = root / f"reward_{reward_mode}__obs_{obs_mode}"
        setting_dir.mkdir(parents=True, exist_ok=True)
        model_path = train_one_setting(
            net_file=args.net_file,
            route_file=args.route_file,
            seconds=args.seconds,
            delta_time=args.delta_time,
            timesteps=args.timesteps,
            lr=args.lr,
            gamma=args.gamma,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            hidden=args.hidden,
            seed=setting_seed,
            collision_penalty=args.collision_penalty,
            reward_mode=reward_mode,
            obs_mode=obs_mode,
            run_dir=setting_dir,
            model_root=model_root,
        )
        model_map[(reward_mode, obs_mode)] = model_path
        print(f"Saved model for setting ({reward_mode}, {obs_mode}): {model_path}")

    rows: list[dict[str, Any]] = []
    for i, (reward_mode, obs_mode) in enumerate(settings):
        base_seed = args.seed + 100 + i * 100
        eval_dir = root / f"reward_{reward_mode}__obs_{obs_mode}" / "eval"
        for run_id in range(1, args.eval_runs + 1):
            row = evaluate_one_run(
                model_path=model_map[(reward_mode, obs_mode)],
                net_file=args.net_file,
                route_file=args.route_file,
                seconds=args.seconds,
                delta_time=args.delta_time,
                seed=base_seed + run_id,
                collision_penalty=args.collision_penalty,
                reward_mode=reward_mode,
                obs_mode=obs_mode,
                run_id=run_id,
                eval_dir=eval_dir,
            )
            rows.append(row)

    df_runs = pd.DataFrame(rows)
    runs_csv = root / "eval_runs.csv"
    df_runs.to_csv(runs_csv, index=False)

    summary = (
        df_runs.groupby(["reward_mode", "obs_mode"], as_index=False)[
            ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions"]
        ]
        .mean()
        .sort_values(["reward_mode", "obs_mode"])
    )
    summary_csv = root / "eval_summary.csv"
    summary.to_csv(summary_csv, index=False)

    table_md = root / "ablation_table.md"
    md = []
    md.append("| Setting | Travel Time | Waiting Time | Throughput (veh/h) | Mean Speed (m/s) |")
    md.append("|---|---:|---:|---:|---:|")
    for _, r in summary.iterrows():
        throughput = (float(r["total_arrived"]) / float(args.seconds)) * 3600.0
        label = f"reward={r['reward_mode']}, obs={r['obs_mode']}"
        md.append(
            f"| {label} | {float(r['avg_travel_time']):.3f} | {float(r['mean_waiting_time']):.3f} | {throughput:.1f} | {float(r['mean_speed']):.3f} |"
        )
    table_md.write_text("\n".join(md), encoding="utf-8")

    fig_dir = root / "figures"
    plot_ablation(
        summary_csv=summary_csv,
        out_dir=fig_dir,
        seconds=args.seconds,
        figures_repro_dir=figures_repro_dir,
    )

    print(f"Wrote: {runs_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {table_md}")
    print(f"Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()


