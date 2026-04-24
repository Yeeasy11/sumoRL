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


def _load_rl_model_for_eval(method: str, model_path: str, env: SumoEnvironment):
    """Load RL model robustly for inference across SB3 version differences."""
    if method == "ppo":
        from stable_baselines3 import PPO as Algo
    elif method == "dqn":
        from stable_baselines3 import DQN as Algo
    else:
        raise ValueError(f"Unsupported RL method: {method}")

    try:
        return Algo.load(model_path)
    except Exception as exc:
        print(f"Warning: standard {method.upper()} load failed, fallback to policy-only load: {exc}")

    from stable_baselines3.common.save_util import load_from_zip_file

    data, params, _ = load_from_zip_file(model_path, device="cpu")
    policy_kwargs = data.get("policy_kwargs", None)
    if not isinstance(policy_kwargs, dict):
        policy_kwargs = None

    if method == "ppo":
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
    method_cn: str
    flow_ns: int
    flow_we: int
    arrival_dist: str
    sim_seconds: int
    delta_time: int
    total_arrived: int
    mean_waiting_time: float
    mean_speed: float
    avg_travel_time: float
    collisions: int
    min_ttc: float
    harsh_brake_rate: float
    mean_abs_jerk: float
    gini_waiting_time: float
    out_csv_prefix: str
    tripinfo_path: str
    collision_path: str


METHOD_CN = {
    "idm": "IDM默认模型",
    "fixed_speed": "固定目标速度",
    "yield": "礼让规则",
    "ppo": "PPO",
    "dqn": "DQN",
}


def parse_tripinfo_stats(tripinfo_path: Path) -> tuple[float, list[float]]:
    if not tripinfo_path.exists():
        return float("nan"), []
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()
    durs: list[float] = []
    waits: list[float] = []
    for ti in root.iter("tripinfo"):
        dur = ti.get("duration")
        if dur is not None:
            try:
                durs.append(float(dur))
            except ValueError:
                pass
        wt = ti.get("waitingTime")
        if wt is not None:
            try:
                waits.append(float(wt))
            except ValueError:
                pass
    avg_dur = float(sum(durs) / len(durs)) if durs else float("nan")
    return avg_dur, waits


def parse_collision_count(collision_path: Path) -> int:
    if not collision_path.exists():
        return 0
    try:
        tree = ET.parse(collision_path)
    except ET.ParseError:
        return 0
    root = tree.getroot()
    return sum(1 for _ in root.iter("collision"))


def gini(values: list[float]) -> float:
    if not values:
        return float("nan")
    xs = sorted(float(v) for v in values)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s <= 0:
        return 0.0
    acc = 0.0
    for i, x in enumerate(xs, start=1):
        acc += i * x
    return (2.0 * acc) / (n * s) - (n + 1.0) / n


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


def env_metrics_from_outcsv(out_csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(out_csv_path)

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
        raise FileNotFoundError(f"No SUMO-RL CSV found for prefix={prefix}, episode={episode}")
    return candidates[0]


def apply_fixed_speed_control(env: SumoEnvironment, target_speed: float) -> None:
    for vid in env.sumo.vehicle.getIDList():
        try:
            env.sumo.vehicle.setSpeed(vid, target_speed)
        except Exception:
            continue


def apply_yield_control(env: SumoEnvironment, target_speed: float, yield_dist: float) -> None:
    ts_id = env.ts_ids[0]
    lanes = env.traffic_signals[ts_id].lanes
    lane_len = {ln: env.sumo.lane.getLength(ln) for ln in lanes}

    lane_front: dict[str, tuple[str, float]] = {}
    for ln in lanes:
        ids = env.sumo.lane.getLastStepVehicleIDs(ln)
        best: Optional[tuple[str, float]] = None
        for vid in ids:
            pos = env.sumo.vehicle.getLanePosition(vid)
            dist = max(0.0, lane_len[ln] - pos)
            if best is None or dist < best[1]:
                best = (vid, dist)
        if best is not None:
            lane_front[ln] = best

    near = [(vid, dist) for vid, dist in lane_front.values() if dist <= yield_dist]
    allow_vid: Optional[str] = None
    if near:
        near_sorted = sorted(near, key=lambda x: (x[1], x[0]))
        allow_vid = near_sorted[0][0]

    for vid in env.sumo.vehicle.getIDList():
        try:
            lane_id = env.sumo.vehicle.getLaneID(vid)
            pos = env.sumo.vehicle.getLanePosition(vid)
            if lane_id in lane_len:
                dist = max(0.0, lane_len[lane_id] - pos)
                if dist <= yield_dist and allow_vid is not None and vid != allow_vid:
                    env.sumo.vehicle.setSpeed(vid, 0.0)
                else:
                    env.sumo.vehicle.setSpeed(vid, target_speed)
            else:
                env.sumo.vehicle.setSpeed(vid, target_speed)
        except Exception:
            continue


def collect_vehicle_metrics(
    env: SumoEnvironment,
    prev_acc: dict[str, float],
    min_ttc: float,
    harsh_brake_events: int,
    acc_samples: int,
    jerk_sum: float,
    jerk_count: int,
    delta_time: int,
    harsh_threshold: float,
) -> tuple[float, int, int, float, int]:
    veh_ids = set(env.sumo.vehicle.getIDList())
    for vid in list(prev_acc.keys()):
        if vid not in veh_ids:
            del prev_acc[vid]

    dt = max(float(delta_time), 1e-6)

    for vid in veh_ids:
        try:
            v_f = float(env.sumo.vehicle.getSpeed(vid))
            acc = float(env.sumo.vehicle.getAcceleration(vid))
        except Exception:
            continue

        acc_samples += 1
        if acc < harsh_threshold:
            harsh_brake_events += 1

        if vid in prev_acc:
            jerk_sum += abs(acc - prev_acc[vid]) / dt
            jerk_count += 1
        prev_acc[vid] = acc

        try:
            lead = env.sumo.vehicle.getLeader(vid)
        except Exception:
            lead = None

        if lead is None:
            continue

        lead_id, gap = lead
        if gap is None:
            continue

        try:
            v_l = float(env.sumo.vehicle.getSpeed(lead_id))
        except Exception:
            continue

        rel = v_f - v_l
        if rel > 1e-6:
            ttc = float(gap) / rel
            if ttc > 0 and ttc < min_ttc:
                min_ttc = ttc

    return min_ttc, harsh_brake_events, acc_samples, jerk_sum, jerk_count


def run_episode(
    method: str,
    route_file: str,
    seconds: int,
    delta_time: int,
    run_id: int,
    out_dir: Path,
    model_path: Optional[str],
    sumo_seed: int,
    flow_ns: int,
    flow_we: int,
    arrival_dist: str,
    fixed_speed: float,
    yield_dist: float,
    harsh_threshold: float,
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

    fixed_ts = method in {"idm", "fixed_speed", "yield"}

    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=route_file,
        out_csv_name=out_csv_prefix,
        use_gui=False,
        num_seconds=seconds,
        delta_time=delta_time,
        yellow_time=0 if delta_time <= 2 else 2,
        single_agent=True,
        fixed_ts=fixed_ts,
        sumo_seed=sumo_seed,
        additional_sumo_cmd=additional_sumo_cmd,
    )

    model = None
    if method in {"ppo", "dqn"}:
        if not model_path:
            raise ValueError(f"method={method} requires model path")
        try:
            model = _load_rl_model_for_eval(method, model_path, env)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "stable_baselines3 is required for PPO/DQN evaluation."
            ) from e

    obs, _ = env.reset()
    terminated, truncated = False, False

    prev_acc: dict[str, float] = {}
    min_ttc = float("inf")
    harsh_brake_events = 0
    acc_samples = 0
    jerk_sum = 0.0
    jerk_count = 0

    while not (terminated or truncated):
        if method == "fixed_speed":
            apply_fixed_speed_control(env, fixed_speed)
        elif method == "yield":
            apply_yield_control(env, fixed_speed, yield_dist)

        if method in {"idm", "fixed_speed", "yield"}:
            action = {}
        else:
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)

        min_ttc, harsh_brake_events, acc_samples, jerk_sum, jerk_count = collect_vehicle_metrics(
            env=env,
            prev_acc=prev_acc,
            min_ttc=min_ttc,
            harsh_brake_events=harsh_brake_events,
            acc_samples=acc_samples,
            jerk_sum=jerk_sum,
            jerk_count=jerk_count,
            delta_time=delta_time,
            harsh_threshold=harsh_threshold,
        )

    env.save_csv(out_csv_prefix, run_id)
    env.close()

    csv_path = resolve_sumo_rl_csv(out_csv_prefix, run_id)
    m = env_metrics_from_outcsv(csv_path)
    avg_tt, waits = parse_tripinfo_stats(tripinfo_path)
    col = parse_collision_count(collision_path)

    harsh_rate = float(harsh_brake_events) / float(max(acc_samples, 1))
    mean_abs_jerk = jerk_sum / float(max(jerk_count, 1))
    min_ttc_value = float("nan") if min_ttc == float("inf") else float(min_ttc)
    gini_wait = gini(waits)

    return EvalResult(
        run=run_id,
        method=method,
        method_cn=METHOD_CN.get(method, method),
        flow_ns=flow_ns,
        flow_we=flow_we,
        arrival_dist=arrival_dist,
        sim_seconds=seconds,
        delta_time=delta_time,
        total_arrived=int(m["total_arrived"]),
        mean_waiting_time=float(m["mean_waiting_time"]),
        mean_speed=float(m["mean_speed"]),
        avg_travel_time=float(avg_tt),
        collisions=int(col),
        min_ttc=min_ttc_value,
        harsh_brake_rate=harsh_rate,
        mean_abs_jerk=mean_abs_jerk,
        gini_waiting_time=float(gini_wait),
        out_csv_prefix=out_csv_prefix,
        tripinfo_path=str(tripinfo_path),
        collision_path=str(collision_path),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="统一评估入口：IDM/固定目标速度/礼让/PPO/DQN，并提取车辆级指标。",
    )
    p.add_argument("--method", type=str, choices=["idm", "fixed_speed", "yield", "ppo", "dqn"], required=True)
    p.add_argument("--route", type=str, required=True)
    p.add_argument("--flow-ns", type=int, default=600)
    p.add_argument("--flow-we", type=int, default=600)
    p.add_argument("--arrival-dist", type=str, default="poisson")
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--delta-time", type=int, default=3)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ppo-model", type=str, default="")
    p.add_argument("--dqn-model", type=str, default="")
    p.add_argument("--fixed-speed", type=float, default=10.0)
    p.add_argument("--yield-dist", type=float, default=20.0)
    p.add_argument("--harsh-threshold", type=float, default=-3.0)
    p.add_argument("--allow-collisions", action="store_true", help="Enable collision handling mode in SUMO.")
    p.add_argument(
        "--collision-action",
        type=str,
        choices=["none", "warn", "teleport", "remove"],
        default="warn",
        help="SUMO collision action when --allow-collisions is enabled.",
    )
    p.add_argument("--outdir", type=str, required=True)
    args = p.parse_args()

    model_path = None
    if args.method == "ppo":
        model_path = args.ppo_model
    elif args.method == "dqn":
        model_path = args.dqn_model

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    for i in range(1, args.runs + 1):
        sumo_seed = args.seed + i
        r = run_episode(
            method=args.method,
            route_file=args.route,
            seconds=args.seconds,
            delta_time=args.delta_time,
            run_id=i,
            out_dir=out_dir,
            model_path=model_path,
            sumo_seed=sumo_seed,
            flow_ns=args.flow_ns,
            flow_we=args.flow_we,
            arrival_dist=args.arrival_dist,
            fixed_speed=args.fixed_speed,
            yield_dist=args.yield_dist,
            harsh_threshold=args.harsh_threshold,
            allow_collisions=args.allow_collisions,
            collision_action=args.collision_action,
        )
        results.append(r)

    df = pd.DataFrame([asdict(r) for r in results])
    raw_path = out_dir / "eval_runs.csv"
    df.to_csv(raw_path, index=False)

    summary = (
        df.groupby(["method", "method_cn", "flow_ns", "flow_we", "arrival_dist"], as_index=False)[
            [
                "total_arrived",
                "mean_waiting_time",
                "mean_speed",
                "avg_travel_time",
                "collisions",
                "min_ttc",
                "harsh_brake_rate",
                "mean_abs_jerk",
                "gini_waiting_time",
            ]
        ]
        .mean()
        .sort_values(["flow_ns", "flow_we", "arrival_dist", "method"])
    )
    summary_path = out_dir / "eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
