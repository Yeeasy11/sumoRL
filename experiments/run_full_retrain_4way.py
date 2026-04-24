#!/usr/bin/env python3
"""Full retrain/eval/ablation pipeline for 4-way single intersection.

Covers:
- 3-seed PPO/DQN retraining (overwrite model files)
- 3 flow levels x 3 arrival distributions evaluation, 20 runs each (overwrite outputs)
- PPO ablation retrain/eval (overwrite outputs)
- Composite summary + all_runs_matrix rebuild
- Figure regeneration (overwrite outputs/figures)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}")


def write_4way_route(out_path: Path, flow: int, dist: str, begin: int = 0, end: int = 600) -> None:
    base_rates = {
        "left": int(round(flow * 0.2)),
        "straight": int(round(flow * 0.6)),
        "right": int(round(flow * 0.2)),
    }

    root = ET.Element("routes")

    directions = ["n", "e", "s", "w"]
    route_map = {
        "n": {"left": "t_e", "straight": "t_s", "right": "t_w"},
        "e": {"left": "t_s", "straight": "t_w", "right": "t_n"},
        "s": {"left": "t_w", "straight": "t_n", "right": "t_e"},
        "w": {"left": "t_n", "straight": "t_e", "right": "t_s"},
    }

    for d in directions:
        for turn in ["left", "straight", "right"]:
            rid = f"route_{d}_{turn}"
            edges = f"{d}_t {route_map[d][turn]}"
            ET.SubElement(root, "route", id=rid, edges=edges)

    def add_uniform_flow(fid: str, rid: str, rate: int) -> None:
        period = max(0.1, 3600.0 / float(max(rate, 1)))
        ET.SubElement(
            root,
            "flow",
            id=fid,
            route=rid,
            begin=str(begin),
            end=str(end),
            period=f"{period:.4f}",
            departSpeed="max",
            departPos="base",
            departLane="best",
        )

    def add_poisson_flow(fid: str, rid: str, rate: int) -> None:
        ET.SubElement(
            root,
            "flow",
            id=fid,
            route=rid,
            begin=str(begin),
            end=str(end),
            vehsPerHour=str(max(rate, 1)),
            departSpeed="max",
            departPos="base",
            departLane="best",
        )

    def add_burst_flow(fid_prefix: str, rid: str, rate: int) -> None:
        span = max(3, end - begin)
        seg = span // 3
        low = max(1, int(rate * 0.5))
        high = max(1, int(rate * 1.8))
        pieces = [
            (f"{fid_prefix}_1", begin, begin + seg, low),
            (f"{fid_prefix}_2", begin + seg, begin + 2 * seg, high),
            (f"{fid_prefix}_3", begin + 2 * seg, end, low),
        ]
        for fid, b, e, vph in pieces:
            ET.SubElement(
                root,
                "flow",
                id=fid,
                route=rid,
                begin=str(b),
                end=str(e),
                vehsPerHour=str(max(vph, 1)),
                departSpeed="max",
                departPos="base",
                departLane="best",
            )

    for d in directions:
        for turn in ["left", "straight", "right"]:
            rid = f"route_{d}_{turn}"
            fid = f"flow_{d}_{turn}_{dist}"
            rate = base_rates[turn]
            if dist == "uniform":
                add_uniform_flow(fid, rid, rate)
            elif dist == "poisson":
                add_poisson_flow(fid, rid, rate)
            elif dist == "burst":
                add_burst_flow(fid, rid, rate)
            else:
                raise ValueError(f"Unknown dist: {dist}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def retrain_models(python_exec: str, allow_collisions: bool, collision_action: str) -> None:
    model_dir = ROOT / "models" / "4way-single-intersection"
    monitor_ppo = ROOT / "logs" / "4way_single_intersection" / "ppo" / "monitor"
    monitor_dqn = ROOT / "logs" / "4way_single_intersection" / "dqn" / "monitor"
    model_dir.mkdir(parents=True, exist_ok=True)
    monitor_ppo.mkdir(parents=True, exist_ok=True)
    monitor_dqn.mkdir(parents=True, exist_ok=True)

    for seed in [0, 1, 2]:
        # PPO
        cmd_ppo = [
            python_exec,
            "experiments/train_4way_single_intersection.py",
            "--algo",
            "ppo",
            "--route",
            "sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
            "--seconds",
            "600",
            "--delta-time",
            "1",
            "--timesteps",
            "200000",
            "--seed",
            str(seed),
            "--gamma",
            "0.99",
            "--batch-size",
            "64",
            "--model-out",
            f"models/4way-single-intersection/ppo_4way_turns_seed{seed}.zip",
            "--monitor-dir",
            "logs/4way_single_intersection/ppo/monitor",
            "--reward-mode",
            "full",
            "--obs-mode",
            "full",
            "--collision-penalty",
            "100.0",
        ]
        if allow_collisions:
            cmd_ppo += ["--allow-collisions", "--collision-action", collision_action]
        run_cmd(cmd_ppo)

        # DQN
        cmd_dqn = [
            python_exec,
            "experiments/train_4way_single_intersection.py",
            "--algo",
            "dqn",
            "--route",
            "sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
            "--seconds",
            "600",
            "--delta-time",
            "1",
            "--timesteps",
            "200000",
            "--seed",
            str(seed),
            "--gamma",
            "0.99",
            "--batch-size",
            "64",
            "--buffer-size",
            "100000",
            "--model-out",
            f"models/4way-single-intersection/dqn_4way_turns_seed{seed}.zip",
            "--monitor-dir",
            "logs/4way_single_intersection/dqn/monitor",
            "--reward-mode",
            "full",
            "--obs-mode",
            "full",
            "--collision-penalty",
            "100.0",
        ]
        if allow_collisions:
            cmd_dqn += ["--allow-collisions", "--collision-action", collision_action]
        run_cmd(cmd_dqn)


def run_eval_matrix(python_exec: str, allow_collisions: bool, collision_action: str) -> None:
    route_root = ROOT / "outputs" / "4way-single-intersection" / "routes"
    eval_root = ROOT / "outputs" / "4way-single-intersection" / "eval_matrix"
    route_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for flow in [300, 600, 900]:
        for dist in ["uniform", "poisson", "burst"]:
            route = route_root / f"route_4way_flow{flow}_{dist}.rou.xml"
            write_4way_route(route, flow=flow, dist=dist, begin=0, end=600)

            # Baselines once
            for method in ["idm", "fixed_speed", "yield"]:
                out_dir = eval_root / f"flow{flow}_{dist}" / method
                cmd = [
                    python_exec,
                    "experiments/eval_unified_cn.py",
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
                    "600",
                    "--delta-time",
                    "3",
                    "--runs",
                    "20",
                    "--seed",
                    "0",
                    "--fixed-speed",
                    "10.0",
                    "--yield-dist",
                    "20.0",
                    "--outdir",
                    str(out_dir),
                ]
                if allow_collisions:
                    cmd += ["--allow-collisions", "--collision-action", collision_action]
                run_cmd(cmd)

            # PPO/DQN for 3 seeds
            for method in ["ppo", "dqn"]:
                for seed in [0, 1, 2]:
                    out_dir = eval_root / f"flow{flow}_{dist}" / f"{method}_seed{seed}"
                    cmd = [
                        python_exec,
                        "experiments/eval_unified_cn.py",
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
                        "600",
                        "--delta-time",
                        "3",
                        "--runs",
                        "20",
                        "--seed",
                        str(seed),
                        "--outdir",
                        str(out_dir),
                    ]
                    if method == "ppo":
                        cmd += ["--ppo-model", f"models/4way-single-intersection/ppo_4way_turns_seed{seed}.zip"]
                    else:
                        cmd += ["--dqn-model", f"models/4way-single-intersection/dqn_4way_turns_seed{seed}.zip"]
                    if allow_collisions:
                        cmd += ["--allow-collisions", "--collision-action", collision_action]
                    run_cmd(cmd)

            # collect all eval runs for this scenario
            for fp in sorted((eval_root / f"flow{flow}_{dist}").rglob("eval_runs.csv")):
                df = pd.read_csv(fp)
                df["flow"] = flow
                df["dist"] = dist
                # normalize method name for seed models
                if "method" in df.columns:
                    df["method"] = df["method"].astype(str).str.lower()
                rows.append(df)

    if not rows:
        raise RuntimeError("No eval runs collected")

    all_runs = pd.concat(rows, ignore_index=True)
    # normalize names to thesis expected identifiers
    all_runs["method"] = all_runs["method"].replace(
        {
            "rule-based": "idm",
            "fixed_speed": "fixed_speed",
            "yield": "yield",
            "ppo": "ppo",
            "dqn": "dqn",
        }
    )

    # overwrite matrix files
    out_matrix = ROOT / "outputs" / "all_runs_matrix.csv"
    out_matrix_fig = ROOT / "outputs" / "figures" / "all_runs_matrix.csv"
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    out_matrix_fig.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(out_matrix, index=False, encoding="utf-8-sig")
    all_runs.to_csv(out_matrix_fig, index=False, encoding="utf-8-sig")


def run_ablation(python_exec: str, allow_collisions: bool, collision_action: str) -> None:
    cmd = [
        python_exec,
        "experiments/run_ablation_4way_single_intersection.py",
        "--route",
        "sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
        "--seconds",
        "600",
        "--delta-time",
        "1",
        "--eval-delta-time",
        "3",
        "--timesteps",
        "200000",
        "--runs",
        "20",
        "--seed",
        "0",
        "--gamma",
        "0.99",
        "--batch-size",
        "64",
        "--outdir",
        "outputs/4way-single-intersection/ablation",
        "--model-root",
        "models/4way-single-intersection/ablation",
        "--fig-dir",
        "outputs/figures",
        "--collision-penalty",
        "100.0",
    ]
    if allow_collisions:
        cmd += ["--allow-collisions", "--collision-action", collision_action]
    run_cmd(cmd)


def regenerate_figures(python_exec: str) -> None:
    run_cmd([python_exec, "experiments/generate_composite_summary.py"])
    run_cmd([
        python_exec,
        "experiments/generate_thesis_figures.py",
        "--matrix-csv",
        "outputs/all_runs_matrix.csv",
        "--out-dir",
        "outputs/figures",
        "--smooth",
        "9",
        "--skip-ttc",
    ])
    run_cmd([
        python_exec,
        "experiments/export_all_figures_cn.py",
        "--skip-thesis",
        "--out-dir",
        "outputs/figures",
        "--smooth",
        "9",
        "--ablation-dir",
        "outputs/4way-single-intersection/ablation",
    ])


def main() -> None:
    p = argparse.ArgumentParser(description="Run full retrain/eval/ablation pipeline and overwrite outputs.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable path")
    p.add_argument("--allow-collisions", action="store_true", help="Enable SUMO collision-action mode")
    p.add_argument(
        "--collision-action",
        type=str,
        choices=["none", "warn", "teleport", "remove"],
        default="warn",
        help="SUMO collision action when --allow-collisions is set",
    )
    args = p.parse_args()

    retrain_models(args.python, allow_collisions=args.allow_collisions, collision_action=args.collision_action)
    run_eval_matrix(args.python, allow_collisions=args.allow_collisions, collision_action=args.collision_action)
    run_ablation(args.python, allow_collisions=args.allow_collisions, collision_action=args.collision_action)
    regenerate_figures(args.python)

    print("Done: models/results/figures overwritten.")


if __name__ == "__main__":
    main()
