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


import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction


class PhaseOnlyObservation(ObservationFunction):
    """
    Ablation observation: keep only signal phase (one-hot) + min_green flag.

    This removes lane density/queue terms (i.e., removes traffic-state / "neighbor" info in the TSC setting).
    """

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
    """
    Full reward = diff-waiting-time - collision_penalty * (#colliding vehicles this step)

    Notes:
    - diff-waiting-time is the SUMO-RL default, already tracked in ts.last_ts_waiting_time.
    - Collisions are read from TraCI simulation if available.
    """
    base = ts._diff_waiting_time_reward()
    try:
        num = ts.env.sumo.simulation.getCollidingVehiclesNumber()
    except Exception:
        num = 0
    return float(base - collision_penalty * float(num))


def make_env(
    route_file: str,
    out_csv: str,
    seconds: int,
    delta_time: int,
    monitor_filename: str,
    reward_mode: str,
    obs_mode: str,
    collision_penalty: float,
    seed: int,
):
    yellow_time = 0 if delta_time <= 2 else 2

    if obs_mode == "phase_only":
        observation_class = PhaseOnlyObservation
    else:
        observation_class = None  # default inside SumoEnvironment

    if reward_mode == "full":
        reward_fn = lambda ts: reward_full_with_collision_penalty(ts, collision_penalty=collision_penalty)  # noqa: E731
    elif reward_mode == "no_collision":
        reward_fn = "diff-waiting-time"
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    kwargs = dict(
        net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
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
    if observation_class is not None:
        kwargs["observation_class"] = observation_class

    env = SumoEnvironment(**kwargs)
    return Monitor(env, filename=monitor_filename)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PPO ablation training on single-intersection (reward/state ablations).",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
        help="Route definition xml file.",
    )
    prs.add_argument("--seconds", type=int, default=600)
    prs.add_argument("--delta-time", type=int, default=1)
    prs.add_argument("--timesteps", type=int, default=200_000)
    prs.add_argument("--lr", type=float, default=3e-4)
    prs.add_argument("--gamma", type=float, default=0.99)
    prs.add_argument("--batch-size", type=int, default=64)
    prs.add_argument("--hidden", type=int, nargs=2, default=[64, 64])
    prs.add_argument("--seed", type=int, default=0)
    prs.add_argument("--logdir", type=str, default="logs/ppo_ablation")
    prs.add_argument("--monitor-dir", type=str, default="logs/ppo_ablation/monitor")
    prs.add_argument("--outdir", type=str, default="outputs/single-intersection/ablation_train")
    prs.add_argument(
        "--reward-mode",
        type=str,
        choices=["full", "no_collision"],
        default="full",
        help="Reward ablation: full includes collision penalty; no_collision uses default diff-waiting-time.",
    )
    prs.add_argument(
        "--obs-mode",
        type=str,
        choices=["full", "phase_only"],
        default="full",
        help="State ablation: phase_only removes lane density/queue terms.",
    )
    prs.add_argument("--collision-penalty", type=float, default=100.0)
    prs.add_argument("--model-out", type=str, default="")
    args = prs.parse_args()

    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{experiment_time}_reward{args.reward_mode}_obs{args.obs_mode}_seed{args.seed}"

    out_csv = str(Path(args.outdir) / run_tag)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    monitor_dir = Path(args.monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_filename = str(monitor_dir / f"monitor_{run_tag}.csv")

    env = make_env(
        route_file=args.route,
        out_csv=out_csv,
        seconds=args.seconds,
        delta_time=args.delta_time,
        monitor_filename=monitor_filename,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        collision_penalty=args.collision_penalty,
        seed=args.seed,
    )

    policy_kwargs = dict(net_arch=list(args.hidden))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    if args.model_out:
        model_path = Path(args.model_out)
    else:
        model_path = Path("models") / f"ppo_ablation_{run_tag}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    env.close()
    print(f"Saved model: {model_path}")
    print(f"Monitor log: {monitor_filename}")
    print(f"Ablation tag: reward={args.reward_mode}, obs={args.obs_mode}")


if __name__ == "__main__":
    main()

