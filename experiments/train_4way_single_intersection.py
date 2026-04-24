import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction


NET_FILE = "sumo_rl/nets/2way-single-intersection/single-intersection-fourdir-straight.net.xml"


class PhaseOnlyObservation(ObservationFunction):
    def __call__(self):
        import numpy as np

        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        return np.array(phase_id + min_green, dtype=np.float32)

    def observation_space(self):
        import numpy as np

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


def build_env(
    route_file: str,
    out_csv: str,
    use_gui: bool,
    seconds: int,
    delta_time: int,
    seed: int,
    reward_mode: str,
    obs_mode: str,
    collision_penalty: float,
    fixed_ts: bool,
    additional_sumo_cmd: Optional[str],
    monitor_filename: Optional[str],
):
    yellow_time = 0 if delta_time <= 2 else 2
    observation_class = PhaseOnlyObservation if obs_mode == "phase_only" else None

    if reward_mode == "full":
        reward_fn = lambda ts: reward_full_with_collision_penalty(ts, collision_penalty=collision_penalty)  # noqa: E731
    elif reward_mode == "default":
        reward_fn = "diff-waiting-time"
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    env_kwargs = dict(
        net_file=NET_FILE,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=True,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        single_agent=True,
        fixed_ts=fixed_ts,
        sumo_seed=seed,
        reward_fn=reward_fn,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    if observation_class is not None:
        env_kwargs["observation_class"] = observation_class
    env = SumoEnvironment(**env_kwargs)
    return Monitor(env, filename=monitor_filename) if monitor_filename else env


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train PPO or DQN on a 4-direction single-lane intersection with turn movements.",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
        help="Route definition xml file.",
    )
    prs.add_argument("--algo", type=str, choices=["ppo", "dqn"], default="ppo")
    prs.add_argument("--seconds", type=int, default=1200, help="Simulation duration (s).")
    prs.add_argument("--delta-time", type=int, default=1, help="SUMO seconds per action step.")
    prs.add_argument("--timesteps", type=int, default=300_000, help="Total training timesteps.")
    prs.add_argument("--lr", type=float, default=0.0, help="Learning rate. If <=0, use algo default from thesis setup.")
    prs.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    prs.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    prs.add_argument("--hidden", type=int, nargs=2, default=[], help="MLP hidden sizes. Empty means algo default.")
    prs.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length.")
    prs.add_argument("--n-epochs", type=int, default=10, help="PPO epochs.")
    prs.add_argument("--buffer-size", type=int, default=100_000, help="DQN replay buffer size.")
    prs.add_argument("--target-update-interval", type=int, default=1000, help="DQN target update interval.")
    prs.add_argument("--seed", type=int, default=0, help="Random seed.")
    prs.add_argument("--gui", action="store_true", default=False, help="Run SUMO with GUI.")
    prs.add_argument("--reward-mode", type=str, choices=["default", "full"], default="full")
    prs.add_argument("--obs-mode", type=str, choices=["full", "phase_only"], default="full")
    prs.add_argument("--collision-penalty", type=float, default=100.0)
    prs.add_argument(
        "--logdir",
        type=str,
        default="logs/4way_single_intersection",
        help="Tensorboard log dir.",
    )
    prs.add_argument(
        "--monitor-dir",
        type=str,
        default="logs/4way_single_intersection/monitor",
        help="Directory to write Monitor logs.",
    )
    prs.add_argument(
        "--outdir",
        type=str,
        default="outputs/4way-single-intersection/train",
        help="Training CSV output directory.",
    )
    prs.add_argument(
        "--model-out",
        type=str,
        default="",
        help="Where to save the trained model. If empty, a default path under models/ is used.",
    )
    prs.add_argument(
        "--tripinfo-out",
        type=str,
        default="",
        help="Optional SUMO tripinfo output path (xml). Leave empty to disable.",
    )
    prs.add_argument(
        "--collision-out",
        type=str,
        default="",
        help="Optional SUMO collision output path (xml). Leave empty to disable.",
    )
    prs.add_argument("--allow-collisions", action="store_true", help="Enable collision handling mode in SUMO instead of teleport-based conflict removal.")
    prs.add_argument(
        "--collision-action",
        type=str,
        choices=["none", "warn", "teleport", "remove"],
        default="warn",
        help="SUMO collision action when --allow-collisions is enabled.",
    )
    args = prs.parse_args()

    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{experiment_time}_{args.algo}_reward{args.reward_mode}_obs{args.obs_mode}_seed{args.seed}"
    out_csv = str(Path(args.outdir) / tag)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    monitor_dir = Path(args.monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_filename = str(monitor_dir / f"monitor_{tag}.csv")

    extra_cmd_parts: list[str] = []
    if args.tripinfo_out:
        Path(args.tripinfo_out).parent.mkdir(parents=True, exist_ok=True)
        extra_cmd_parts += ["--tripinfo-output", args.tripinfo_out]
    if args.collision_out:
        Path(args.collision_out).parent.mkdir(parents=True, exist_ok=True)
        extra_cmd_parts += ["--collision-output", args.collision_out]
    if args.allow_collisions:
        extra_cmd_parts += ["--collision.action", args.collision_action]
    additional_sumo_cmd = " ".join(extra_cmd_parts) if extra_cmd_parts else None

    env = build_env(
        route_file=args.route,
        out_csv=out_csv,
        use_gui=args.gui,
        seconds=args.seconds,
        delta_time=args.delta_time,
        seed=args.seed,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        collision_penalty=args.collision_penalty,
        fixed_ts=False,
        additional_sumo_cmd=additional_sumo_cmd,
        monitor_filename=monitor_filename,
    )

    if args.algo == "ppo":
        hidden = list(args.hidden) if args.hidden else [64, 64]
        lr = args.lr if args.lr and args.lr > 0 else 3e-4
    else:
        hidden = list(args.hidden) if args.hidden else [128, 128]
        lr = args.lr if args.lr and args.lr > 0 else 1e-3

    policy_kwargs = dict(net_arch=hidden)
    if args.algo == "ppo":
        effective_n_steps = min(max(2, args.n_steps), args.timesteps)
        effective_batch_size = min(args.batch_size, effective_n_steps)
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=args.gamma,
            n_steps=effective_n_steps,
            n_epochs=args.n_epochs,
            batch_size=effective_batch_size,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            target_update_interval=args.target_update_interval,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
        )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    if args.model_out:
        model_out = Path(args.model_out)
    else:
        model_out = Path("models/4way-single-intersection") / f"{args.algo}_{tag}.zip"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))

    env.close()
    print(f"Saved model: {model_out}")
    print(f"Training CSV prefix: {out_csv}")
    print(f"Monitor log: {monitor_filename}")


if __name__ == "__main__":
    main()