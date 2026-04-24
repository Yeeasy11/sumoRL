#!/usr/bin/env python3
"""Run 4-way evaluation matrix with 3-seed RL aggregation and rule baselines."""

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_eval_runs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_runs.csv: {path}")
    return pd.read_csv(path)


def _aggregate_method(eval_runs_list: list[pd.DataFrame], method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat = pd.concat(eval_runs_list, ignore_index=True)
    cat["method"] = method
    summary = (
        cat.groupby(["method", "method_cn", "flow_ns", "flow_we", "arrival_dist"], as_index=False)[
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
    return cat, summary


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate flow x dist matrix for idm/fixed_speed/yield and 3-seed PPO/DQN aggregation.",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--eval-script", type=str, default="experiments/eval_unified_cn.py")
    p.add_argument("--route-dir", type=str, default="sumo_rl/nets/4way-single-intersection/routes")
    p.add_argument("--out-root", type=str, default="outputs/eval_repro")
    p.add_argument("--flows", nargs="+", type=int, default=[300, 600, 900])
    p.add_argument("--dists", nargs="+", type=str, default=["uniform", "poisson", "burst"])
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--delta-time", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-collisions", action="store_true")
    p.add_argument("--collision-action", type=str, default="warn")
    p.add_argument("--ppo-models", nargs="+", required=True)
    p.add_argument("--dqn-models", nargs="+", required=True)
    p.add_argument("--fixed-speed", type=float, default=10.0)
    p.add_argument("--yield-dist", type=float, default=20.0)
    args = p.parse_args()

    py = args.python
    eval_script = args.eval_script
    route_dir = Path(args.route_dir)
    out_root = Path(args.out_root)

    for flow in args.flows:
        for dist in args.dists:
            route = route_dir / f"4way_{flow}_{dist}.rou.xml"
            if not route.exists():
                raise FileNotFoundError(f"Route not found: {route}")

            scenario_dir = out_root / f"flow{flow}_{dist}"
            scenario_dir.mkdir(parents=True, exist_ok=True)

            # Rule baselines: idm/fixed_speed/yield
            for method in ["idm", "fixed_speed", "yield"]:
                outdir = scenario_dir / method
                cmd = [
                    py,
                    eval_script,
                    "--method",
                    method,
                    "--route",
                    str(route),
                    "--flow-ns",
                    str(flow),
                    "--flow-we",
                    str(flow),
                    "--arrival-dist",
                    dist,
                    "--seconds",
                    str(args.seconds),
                    "--delta-time",
                    str(args.delta_time),
                    "--runs",
                    str(args.runs),
                    "--seed",
                    str(args.seed),
                    "--fixed-speed",
                    str(args.fixed_speed),
                    "--yield-dist",
                    str(args.yield_dist),
                    "--outdir",
                    str(outdir),
                ]
                if args.allow_collisions:
                    cmd.extend(["--allow-collisions", "--collision-action", args.collision_action])
                _run(cmd)

            # PPO aggregated across 3 seed-models.
            ppo_seed_runs: list[pd.DataFrame] = []
            for model_idx, model_path in enumerate(args.ppo_models):
                outdir = scenario_dir / f"ppo_seed{model_idx}"
                cmd = [
                    py,
                    eval_script,
                    "--method",
                    "ppo",
                    "--route",
                    str(route),
                    "--flow-ns",
                    str(flow),
                    "--flow-we",
                    str(flow),
                    "--arrival-dist",
                    dist,
                    "--seconds",
                    str(args.seconds),
                    "--delta-time",
                    str(args.delta_time),
                    "--runs",
                    str(args.runs),
                    "--seed",
                    str(args.seed + model_idx * 1000),
                    "--ppo-model",
                    str(model_path),
                    "--outdir",
                    str(outdir),
                ]
                if args.allow_collisions:
                    cmd.extend(["--allow-collisions", "--collision-action", args.collision_action])
                _run(cmd)
                seed_df = _load_eval_runs(outdir / "eval_runs.csv")
                seed_df["model_seed_idx"] = model_idx
                ppo_seed_runs.append(seed_df)

            ppo_cat, ppo_summary = _aggregate_method(ppo_seed_runs, "ppo")
            ppo_final_dir = scenario_dir / "ppo"
            ppo_final_dir.mkdir(parents=True, exist_ok=True)
            ppo_cat.to_csv(ppo_final_dir / "eval_runs.csv", index=False)
            ppo_summary.to_csv(ppo_final_dir / "eval_summary.csv", index=False)

            # DQN aggregated across 3 seed-models.
            dqn_seed_runs: list[pd.DataFrame] = []
            for model_idx, model_path in enumerate(args.dqn_models):
                outdir = scenario_dir / f"dqn_seed{model_idx}"
                cmd = [
                    py,
                    eval_script,
                    "--method",
                    "dqn",
                    "--route",
                    str(route),
                    "--flow-ns",
                    str(flow),
                    "--flow-we",
                    str(flow),
                    "--arrival-dist",
                    dist,
                    "--seconds",
                    str(args.seconds),
                    "--delta-time",
                    str(args.delta_time),
                    "--runs",
                    str(args.runs),
                    "--seed",
                    str(args.seed + model_idx * 1000),
                    "--dqn-model",
                    str(model_path),
                    "--outdir",
                    str(outdir),
                ]
                if args.allow_collisions:
                    cmd.extend(["--allow-collisions", "--collision-action", args.collision_action])
                _run(cmd)
                seed_df = _load_eval_runs(outdir / "eval_runs.csv")
                seed_df["model_seed_idx"] = model_idx
                dqn_seed_runs.append(seed_df)

            dqn_cat, dqn_summary = _aggregate_method(dqn_seed_runs, "dqn")
            dqn_final_dir = scenario_dir / "dqn"
            dqn_final_dir.mkdir(parents=True, exist_ok=True)
            dqn_cat.to_csv(dqn_final_dir / "eval_runs.csv", index=False)
            dqn_summary.to_csv(dqn_final_dir / "eval_summary.csv", index=False)

    print("Done matrix evaluation with multiseed PPO/DQN aggregation.")


if __name__ == "__main__":
    main()
