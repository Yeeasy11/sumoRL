import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
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


@dataclass
class EvalResult:
    run: int
    method: str
    flow_ns: int
    flow_we: int
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
    """Evaluation-time observation to match ablation models trained with phase_only."""

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
    # SumoEnvironment appends additional_sumo_cmd via `.split()`.
    # Do NOT add quotes here (quotes would become part of the filename on Windows).
    trip_abs = tripinfo_path.resolve()
    col_abs = collision_path.resolve()
    return f"--tripinfo-output {trip_abs} --collision-output {col_abs}"


def env_metrics_from_outcsv(out_csv_prefix: str) -> dict[str, Any]:
    """
    SUMO-RL writes metrics to {out_csv_prefix}_conn{label}_ep{episode}.csv via env.save_csv(prefix, episode).
    We compute system-level means from that CSV.
    """
    df = pd.read_csv(out_csv_prefix)
    # Common columns in SUMO-RL outputs
    # - system_total_stopped
    # - system_total_waiting_time
    # - system_mean_waiting_time
    # - system_mean_speed
    # - system_total_arrived
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
    """
    Resolve SUMO-RL metrics CSV written as: {prefix}_conn{label}_ep{episode}.csv
    There may be different conn labels depending on how many envs were created.
    """
    p = Path(prefix)
    pattern = f"{p.name}_conn*_ep{episode}.csv"
    candidates = sorted(p.parent.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No SUMO-RL CSV found for prefix={prefix}, episode={episode} (pattern={pattern})")
    return candidates[0]


def run_episode(
    method: str,
    algo: str,
    route_file: str,
    seconds: int,
    delta_time: int,
    run_id: int,
    out_dir: Path,
    model_path: Optional[str],
    sumo_seed: int,
    obs_mode: str,
) -> EvalResult:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{method}_{timestamp}_seed{sumo_seed}_run{run_id}"

    tripinfo_path = out_dir / "tripinfo" / f"{run_tag}.xml"
    collision_path = out_dir / "collisions" / f"{run_tag}.xml"
    additional_sumo_cmd = make_additional_sumo_cmd(tripinfo_path, collision_path)

    # Store SUMO-RL CSV per run
    out_csv_prefix = (out_dir / "csv" / run_tag).as_posix()

    fixed_ts = method.lower() in {"rule", "rule-based", "fixed"}
    yellow_time = 0 if delta_time <= 2 else 2
    obs_class = PhaseOnlyObservation if obs_mode == "phase_only" else None

    env_kwargs = dict(
        net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
        route_file=route_file,
        out_csv_name=out_csv_prefix,
        use_gui=False,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=yellow_time,
        single_agent=True,
        fixed_ts=fixed_ts,
        sumo_seed=sumo_seed,
        additional_sumo_cmd=additional_sumo_cmd,
    )
    if obs_class is not None:
        env_kwargs["observation_class"] = obs_class
    env = SumoEnvironment(**env_kwargs)

    model = None
    if model_path and not fixed_ts:
        try:
            if algo.lower() == "ppo":
                from stable_baselines3 import PPO  # lazy import
                model = PPO.load(model_path)
            elif algo.lower() == "dqn":
                from stable_baselines3 import DQN  # lazy import
                model = DQN.load(model_path)
            else:
                raise ValueError(f"Unsupported algo: {algo}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "stable_baselines3 is required for model evaluation.\n"
                'Install with: pip install "stable-baselines3[extra]" torch tensorboard\n'
                "Then re-run with --ppo-model ..."
            ) from e

    obs, info = env.reset()
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

    avg_tt = parse_tripinfo_avg_duration(tripinfo_path)
    col = parse_collision_count(collision_path)

    # flow numbers are not encoded in route_file name; caller should fill them in
    return EvalResult(
        run=run_id,
        method=method,
        flow_ns=-1,
        flow_we=-1,
        sim_seconds=seconds,
        delta_time=delta_time,
        total_arrived=int(m["total_arrived"]),
        mean_waiting_time=float(m["mean_waiting_time"]),
        mean_speed=float(m["mean_speed"]),
        avg_travel_time=float(avg_tt),
        collisions=int(col),
        out_csv_prefix=out_csv_prefix,
        tripinfo_path=str(tripinfo_path),
        collision_path=str(collision_path),
    )


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate Rule-based (fixed TS) vs PPO on single-intersection with aggregated metrics.",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/single-intersection/single-intersection.rou.gen.xml",
        help="Route file to evaluate.",
    )
    prs.add_argument("--flow-ns", type=int, default=600, help="For logging only (veh/h).")
    prs.add_argument("--flow-we", type=int, default=600, help="For logging only (veh/h).")
    prs.add_argument("--seconds", type=int, default=600, help="Simulation duration (s).")
    prs.add_argument("--delta-time", type=int, default=1, help="SUMO seconds per step.")
    prs.add_argument("--runs", type=int, default=10, help="Number of evaluation runs per method.")
    prs.add_argument("--seed", type=int, default=0, help="Base seed (seed+i used per run).")
    prs.add_argument(
        "--obs-mode",
        type=str,
        choices=["full", "phase_only"],
        default="full",
        help="Observation mode for evaluation (must match the PPO model).",
    )
    prs.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "dqn"],
        default="ppo",
        help="RL algorithm type for --ppo-model loading.",
    )
    prs.add_argument(
        "--ppo-model",
        type=str,
        default="",
        help="Path to trained PPO model. If empty, only Rule-based is evaluated.",
    )
    prs.add_argument(
        "--outdir",
        type=str,
        default="outputs/single-intersection/eval",
        help="Evaluation output directory.",
    )
    args = prs.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["Rule-based"]
    if args.ppo_model:
        methods.append(args.algo.upper())

    results: list[EvalResult] = []
    for method in methods:
        for i in range(1, args.runs + 1):
            sumo_seed = args.seed + i
            model_path = None if method == "Rule-based" else args.ppo_model
            r = run_episode(
                method=method,
                algo=args.algo,
                route_file=args.route,
                seconds=args.seconds,
                delta_time=args.delta_time,
                run_id=i,
                out_dir=out_dir,
                model_path=model_path,
                sumo_seed=sumo_seed,
                obs_mode=args.obs_mode,
            )
            r.flow_ns = args.flow_ns
            r.flow_we = args.flow_we
            results.append(r)

    df = pd.DataFrame([asdict(r) for r in results])
    raw_path = out_dir / "eval_runs.csv"
    df.to_csv(raw_path, index=False)

    summary = (
        df.groupby(["method", "flow_ns", "flow_we"], as_index=False)[
            ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions"]
        ]
        .mean()
        .sort_values(["flow_ns", "flow_we", "method"])
    )
    summary_path = out_dir / "eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()

