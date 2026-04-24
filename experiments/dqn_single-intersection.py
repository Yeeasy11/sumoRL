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

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment


def make_env(
    route_file: str,
    out_csv: str,
    use_gui: bool,
    seconds: int,
    delta_time: int,
    additional_sumo_cmd: Optional[str],
    monitor_filename: Optional[str],
):
    yellow_time = 0 if delta_time <= 2 else 2
    env = SumoEnvironment(
        net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        single_agent=True,
        fixed_ts=False,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    return Monitor(env, filename=monitor_filename)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="DQN training on SUMO-RL single-intersection (traffic signal control).",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/single-intersection/single-intersection.rou.gen.xml",
        help="Route definition xml file.",
    )
    prs.add_argument("--seconds", type=int, default=600, help="Simulation duration (s).")
    prs.add_argument("--delta-time", type=int, default=1, help="SUMO seconds per action step.")
    prs.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps.")
    prs.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    prs.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    prs.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    prs.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer size.")
    prs.add_argument("--hidden", type=int, nargs=2, default=[128, 128], help="MLP hidden sizes.")
    prs.add_argument("--seed", type=int, default=0, help="Random seed.")
    prs.add_argument("--gui", action="store_true", default=False, help="Run SUMO with GUI.")
    prs.add_argument(
        "--logdir",
        type=str,
        default="logs/dqn_single_intersection",
        help="Tensorboard log dir.",
    )
    prs.add_argument(
        "--monitor-dir",
        type=str,
        default="logs/dqn_single_intersection/monitor",
        help="Directory to write Monitor logs.",
    )
    prs.add_argument(
        "--model-out",
        type=str,
        default="models/dqn_single_intersection.zip",
        help="Where to save the trained model.",
    )
    args = prs.parse_args()

    experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"outputs/single-intersection/dqn_train/{experiment_time}"
    monitor_dir = Path(args.monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_filename = str(monitor_dir / f"monitor_{experiment_time}.csv")

    env = make_env(
        route_file=args.route,
        out_csv=out_csv,
        use_gui=args.gui,
        seconds=args.seconds,
        delta_time=args.delta_time,
        additional_sumo_cmd=None,
        monitor_filename=monitor_filename,
    )

    policy_kwargs = dict(net_arch=list(args.hidden))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))

    env.close()
    print(f"Saved model: {model_out}")
    print(f"Training CSV prefix: {out_csv}")
    print(f"Monitor log: {monitor_filename}")


if __name__ == "__main__":
    main()
