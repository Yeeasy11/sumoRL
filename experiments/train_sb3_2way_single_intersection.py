import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from sumo_rl import SumoEnvironment


def make_env(net_file: str, route_file: str, out_csv: str, use_gui: bool, seconds: int, delta_time: int, monitor_file: str):
    yellow_time = 0 if delta_time <= 2 else 2
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        out_csv_name=out_csv,
        use_gui=use_gui,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        single_agent=True,
        fixed_ts=False,
    )
    return Monitor(env, filename=monitor_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train PPO/DQN on 2way single-intersection (4-way entries/exits with turns).",
    )
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], required=True)
    parser.add_argument(
        "--net-file",
        type=str,
        default="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
    )
    parser.add_argument(
        "--route-file",
        type=str,
        default="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
    )
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--delta-time", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default="logs/repro/2way")
    parser.add_argument("--monitor-dir", type=str, default="logs/repro/2way/monitor")
    parser.add_argument("--model-out", type=str, required=True)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"outputs/eval_repro/2way-single-intersection/{args.algo}_train/{stamp}"
    monitor_dir = Path(args.monitor_dir)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    monitor_file = str(monitor_dir / f"{args.algo}_{stamp}.monitor.csv")

    env = make_env(
        net_file=args.net_file,
        route_file=args.route_file,
        out_csv=out_csv,
        use_gui=args.gui,
        seconds=args.seconds,
        delta_time=args.delta_time,
        monitor_file=monitor_file,
    )

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            buffer_size=100000,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
        )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))

    env.close()
    print(f"Saved model: {model_out}")
    print(f"Training CSV prefix: {out_csv}")
    print(f"Monitor log: {monitor_file}")


if __name__ == "__main__":
    main()
