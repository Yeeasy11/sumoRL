import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


import pandas as pd

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction


def _load_rl_model_for_eval(method: str, model_path: str, env: SumoEnvironment):
    """Load RL model robustly for inference across SB3 version differences."""
    if method == "PPO":
        from stable_baselines3 import PPO as Algo
    elif method == "DQN":
        from stable_baselines3 import DQN as Algo
    else:
        raise ValueError(f"Unsupported RL method: {method}")

    try:
        return Algo.load(model_path)
    except Exception as exc:
        print(f"Warning: standard {method} load failed, fallback to policy-only load: {exc}")

    from stable_baselines3.common.save_util import load_from_zip_file

    data, params, _ = load_from_zip_file(model_path, device="cpu")
    policy_kwargs = data.get("policy_kwargs", None)
    if not isinstance(policy_kwargs, dict):
        policy_kwargs = None

    if method == "PPO":
        model = Algo(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            gamma=0.99,
            n_steps=2048,
            n_epochs=10,
            batch_size=64,
            policy_kwargs=policy_kwargs or {"net_arch": [64, 64]},
            verbose=0,
        )
    else:
        model = Algo(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            buffer_size=100_000,
            policy_kwargs=policy_kwargs or {"net_arch": [128, 128]},
            verbose=0,
        )

    if "policy" not in params:
        raise KeyError(f"No policy weights found in {model_path}")
    model.policy.load_state_dict(params["policy"], strict=True)
    return model


NET_FILE = "sumo_rl/nets/2way-single-intersection/single-intersection-fourdir-straight.net.xml"


@dataclass
class EvalResult:
    run: int
    method: str
    flow_n: int
    flow_e: int
    flow_s: int
    flow_w: int
    sim_seconds: int
    delta_time: int
    total_arrived: int
    mean_waiting_time: float
    mean_speed: float
    avg_travel_time: float
    collisions: int
    out_csv_prefix: str
    tripinfo_path: str
    collision_path: str


class PhaseOnlyObservation(ObservationFunction):
    def __call__(self):
        import numpy as np

        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        return np.array(phase_id + min_green, dtype=np.float32)

    def observation_space(self):
        import numpy as np
        from gymnasium import spaces

        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1, dtype=np.float32),
        )


def parse_tripinfo_avg_duration(tripinfo_path: Path) -> float:
    if not tripinfo_path.exists():
        return float("nan")
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()
    durs: list[float] = []
    for ti in root.iter("tripinfo"):
        dur = ti.get("duration")
        if dur is not None:
            try:
                durs.append(float(dur))
            except ValueError:
                pass
    return float(sum(durs) / len(durs)) if durs else float("nan")


def parse_collision_count(collision_path: Path) -> int:
    if not collision_path.exists():
        return 0
    try:
        tree = ET.parse(collision_path)
    except ET.ParseError:
        return 0
    root = tree.getroot()
    return sum(1 for _ in root.iter("collision"))


def make_additional_sumo_cmd(tripinfo_path: Path, collision_path: Path) -> str:
    tripinfo_path.parent.mkdir(parents=True, exist_ok=True)
    collision_path.parent.mkdir(parents=True, exist_ok=True)
    trip_abs = tripinfo_path.resolve()
    col_abs = collision_path.resolve()
    return f"--tripinfo-output {trip_abs} --collision-output {col_abs}"


def append_collision_cmd(base_cmd: str, allow_collisions: bool, collision_action: str) -> str:
    if not allow_collisions:
        return base_cmd
    return f"{base_cmd} --collision.action {collision_action}".strip()


def env_metrics_from_outcsv(out_csv_prefix: str) -> dict[str, Any]:
    df = pd.read_csv(out_csv_prefix)

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


def resolve_sumo_rl_csv(prefix: str, episode: int) -> Path:
    p = Path(prefix)
    pattern = f"{p.name}_conn*_ep{episode}.csv"
    candidates = sorted(p.parent.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No SUMO-RL CSV found for prefix={prefix}, episode={episode} (pattern={pattern})")
    return candidates[0]


def load_model(method: str, model_path: Optional[str], env: SumoEnvironment):
    if not model_path:
        return None
    try:
        if method in {"PPO", "DQN"}:
            return _load_rl_model_for_eval(method, model_path, env)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "stable_baselines3 is required for PPO/DQN evaluation.\n"
            'Install with: pip install "stable-baselines3[extra]" torch tensorboard'
        ) from e
    raise ValueError(f"Unsupported method: {method}")


def build_env(
    route_file: str,
    out_csv_prefix: str,
    seconds: int,
    delta_time: int,
    seed: int,
    obs_mode: str,
    fixed_ts: bool,
    additional_sumo_cmd: str,
):
    yellow_time = 0 if delta_time <= 2 else 2
    kwargs = dict(
        net_file=NET_FILE,
        route_file=route_file,
        out_csv_name=out_csv_prefix,
        use_gui=False,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        single_agent=True,
        fixed_ts=fixed_ts,
        sumo_seed=seed,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    if obs_mode == "phase_only":
        kwargs["observation_class"] = PhaseOnlyObservation
    return SumoEnvironment(**kwargs)


def run_episode(
    method: str,
    route_file: str,
    seconds: int,
    delta_time: int,
    run_id: int,
    out_dir: Path,
    model_path: Optional[str],
    sumo_seed: int,
    obs_mode: str,
    allow_collisions: bool,
    collision_action: str,
) -> EvalResult:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{method}_{timestamp}_seed{sumo_seed}_run{run_id}"

    tripinfo_path = out_dir / "tripinfo" / f"{run_tag}.xml"
    collision_path = out_dir / "collisions" / f"{run_tag}.xml"
    additional_sumo_cmd = make_additional_sumo_cmd(tripinfo_path, collision_path)
    additional_sumo_cmd = append_collision_cmd(additional_sumo_cmd, allow_collisions, collision_action)

    out_csv_prefix = (out_dir / "csv" / run_tag).as_posix()
    fixed_ts = method == "Rule-based"

    env = build_env(
        route_file=route_file,
        out_csv_prefix=out_csv_prefix,
        seconds=seconds,
        delta_time=delta_time,
        seed=sumo_seed,
        obs_mode=obs_mode,
        fixed_ts=fixed_ts,
        additional_sumo_cmd=additional_sumo_cmd,
    )

    model = None if fixed_ts else load_model(method, model_path, env)

    obs, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        if fixed_ts:
            action = {}
        else:
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

    env.save_csv(out_csv_prefix, run_id)
    env.close()

    csv_path = resolve_sumo_rl_csv(out_csv_prefix, run_id)
    m = env_metrics_from_outcsv(str(csv_path))

    return EvalResult(
        run=run_id,
        method=method,
        flow_n=-1,
        flow_e=-1,
        flow_s=-1,
        flow_w=-1,
        sim_seconds=seconds,
        delta_time=delta_time,
        total_arrived=int(m["total_arrived"]),
        mean_waiting_time=float(m["mean_waiting_time"]),
        mean_speed=float(m["mean_speed"]),
        avg_travel_time=float(parse_tripinfo_avg_duration(tripinfo_path)),
        collisions=int(parse_collision_count(collision_path)),
        out_csv_prefix=out_csv_prefix,
        tripinfo_path=str(tripinfo_path),
        collision_path=str(collision_path),
    )


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate Rule-based vs PPO vs DQN on the 4-direction single-lane intersection.",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
        help="Route file to evaluate.",
    )
    prs.add_argument("--flow-n", type=int, default=600, help="North incoming flow (veh/h).")
    prs.add_argument("--flow-e", type=int, default=600, help="East incoming flow (veh/h).")
    prs.add_argument("--flow-s", type=int, default=600, help="South incoming flow (veh/h).")
    prs.add_argument("--flow-w", type=int, default=600, help="West incoming flow (veh/h).")
    prs.add_argument("--seconds", type=int, default=1200, help="Simulation duration (s).")
    prs.add_argument("--delta-time", type=int, default=1, help="SUMO seconds per step.")
    prs.add_argument("--runs", type=int, default=10, help="Number of evaluation runs per method.")
    prs.add_argument("--seed", type=int, default=0, help="Base seed (seed+i used per run).")
    prs.add_argument("--obs-mode", type=str, choices=["full", "phase_only"], default="full")
    prs.add_argument("--allow-collisions", action="store_true", help="Enable collision handling mode in SUMO.")
    prs.add_argument(
        "--collision-action",
        type=str,
        choices=["none", "warn", "teleport", "remove"],
        default="warn",
        help="SUMO collision action when --allow-collisions is enabled.",
    )
    prs.add_argument("--ppo-model", type=str, default="", help="Path to trained PPO model.")
    prs.add_argument("--dqn-model", type=str, default="", help="Path to trained DQN model.")
    prs.add_argument("--outdir", type=str, default="outputs/4way-single-intersection/eval")
    args = prs.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["Rule-based"]
    if args.ppo_model:
        methods.append("PPO")
    if args.dqn_model:
        methods.append("DQN")

    model_lookup = {"PPO": args.ppo_model, "DQN": args.dqn_model}

    results: list[EvalResult] = []
    for method in methods:
        for i in range(1, args.runs + 1):
            sumo_seed = args.seed + i
            r = run_episode(
                method=method,
                route_file=args.route,
                seconds=args.seconds,
                delta_time=args.delta_time,
                run_id=i,
                out_dir=out_dir,
                model_path=model_lookup.get(method),
                sumo_seed=sumo_seed,
                obs_mode=args.obs_mode,
                allow_collisions=args.allow_collisions,
                collision_action=args.collision_action,
            )
            r.flow_n = args.flow_n
            r.flow_e = args.flow_e
            r.flow_s = args.flow_s
            r.flow_w = args.flow_w
            results.append(r)

    df = pd.DataFrame([asdict(r) for r in results])
    raw_path = out_dir / "eval_runs.csv"
    df.to_csv(raw_path, index=False)

    df["throughput"] = (df["total_arrived"] / float(args.seconds)) * 3600.0
    summary = (
        df.groupby(["method", "flow_n", "flow_e", "flow_s", "flow_w"], as_index=False)[
            ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions", "throughput"]
        ]
        .mean()
        .sort_values(["flow_n", "flow_e", "flow_s", "flow_w", "method"])
    )
    summary_path = out_dir / "eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()