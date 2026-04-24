#!/usr/bin/env python3
"""
多seed评估脚本：评估PPO和DQN的所有seed模型，生成聚合评估矩阵
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def find_seed_models(model_dir: Path, algo: str) -> dict[int, Path]:
    """Find all seed models for given algorithm."""
    pattern = re.compile(rf"{algo.lower()}_.*_seed(\d+)\.zip")
    models = {}
    if not model_dir.exists():
        return models
    for f in model_dir.glob(f"{algo.lower()}_*.zip"):
        match = pattern.match(f.name)
        if match:
            seed = int(match.group(1))
            models[seed] = f
    return models


def run_eval_for_seed(
    seed: int,
    ppo_model: Optional[Path],
    dqn_model: Optional[Path],
    route_file: str,
    seconds: int,
    runs: int,
    outdir: Path,
) -> pd.DataFrame:
    """Run evaluation for single seed and return results."""
    seed_outdir = outdir / f"seed{seed}"
    seed_outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "experiments/eval_4way_single_intersection.py",
        "--route",
        route_file,
        "--seconds",
        str(seconds),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--outdir",
        str(seed_outdir),
    ]

    if ppo_model:
        cmd.extend(["--ppo-model", str(ppo_model)])
    if dqn_model:
        cmd.extend(["--dqn-model", str(dqn_model)])

    print(f"Running seed {seed} evaluation...")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed for seed {seed}")

    summary_csv = seed_outdir / "eval_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"eval_summary.csv not found at {summary_csv}")

    df = pd.read_csv(summary_csv)
    df["seed"] = seed
    return df


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Multi-seed evaluation for 4-way intersection with PPO/DQN.",
    )
    prs.add_argument(
        "--route",
        type=str,
        default="sumo_rl/nets/4way-single-intersection/4way-turns-balanced.rou.xml",
        help="Route file.",
    )
    prs.add_argument("--seconds", type=int, default=600, help="Simulation duration (s).")
    prs.add_argument("--runs", type=int, default=3, help="Number of eval runs per seed.")
    prs.add_argument("--model-dir", type=str, default="models/4way-single-intersection", help="Model directory.")
    prs.add_argument("--outdir", type=str, default="outputs/4way-single-intersection/multiseed_eval")
    prs.add_argument(
        "--matrix-out",
        type=str,
        default="outputs/figures/all_runs_matrix_3seed.csv",
        help="Output matrix file for figure generation.",
    )
    args = prs.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all seed models
    ppo_models = find_seed_models(model_dir, "ppo")
    dqn_models = find_seed_models(model_dir, "dqn")

    if not ppo_models and not dqn_models:
        print("No models found!", file=sys.stderr)
        sys.exit(1)

    seeds = sorted(set(ppo_models.keys()) | set(dqn_models.keys()))
    print(f"Found seeds: {seeds}")
    print(f"PPO models: {ppo_models}")
    print(f"DQN models: {dqn_models}")

    # Run evaluation for each seed
    all_results = []
    for seed in seeds:
        ppo_path = ppo_models.get(seed)
        dqn_path = dqn_models.get(seed)

        df = run_eval_for_seed(
            seed=seed,
            ppo_model=ppo_path,
            dqn_model=dqn_path,
            route_file=args.route,
            seconds=args.seconds,
            runs=args.runs,
            outdir=out_dir,
        )
        all_results.append(df)

    # Combine results from all seeds
    combined = pd.concat(all_results, ignore_index=True)

    # Also include Rule-based baseline (same for all seeds)
    if len(all_results) > 0:
        baseline_seed = all_results[0]
        baseline = baseline_seed[baseline_seed["method"] == "Rule-based"].copy()
        for seed in seeds[1:]:
            baseline_copy = baseline.copy()
            baseline_copy["seed"] = seed
            combined = pd.concat([combined, baseline_copy], ignore_index=True)

    # Generate matrix for figures: aggregate across seeds for each method
    matrix_rows = []
    for method in combined["method"].unique():
        method_data = combined[combined["method"] == method]
        for metric in ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions", "throughput"]:
            if metric in method_data.columns:
                mean_val = method_data[metric].mean()
                std_val = method_data[metric].std()
                matrix_rows.append(
                    {
                        "method": method,
                        f"{metric}_mean": mean_val,
                        f"{metric}_std": std_val,
                    }
                )

    # Restructure for figure generation
    matrix = pd.DataFrame(matrix_rows).groupby("method").first().reset_index()

    # Use consistent metric naming for figure generation
    matrix["flow_n"] = 600
    matrix["flow_e"] = 600
    matrix["flow_s"] = 600
    matrix["flow_w"] = 600

    # Add multi-index for _bar_panel expectations
    final_matrix = []
    for _, row in matrix.iterrows():
        for metric in ["total_arrived", "mean_waiting_time", "mean_speed", "avg_travel_time", "collisions", "throughput"]:
            final_matrix.append(
                {
                    "method": row["method"],
                    "flow_n": 600,
                    "flow_e": 600,
                    "flow_s": 600,
                    "flow_w": 600,
                    "metric": metric,
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                }
            )

    final_df = pd.DataFrame(final_matrix)

    # Pivot to match expected format
    pivot_matrix = final_df.pivot_table(
        index=["method", "flow_n", "flow_e", "flow_s", "flow_w"],
        columns="metric",
        values=["mean", "std"],
        aggfunc="first",
    )
    pivot_matrix = pivot_matrix.reset_index()

    # Save
    matrix_path = Path(args.matrix_out)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    pivot_matrix.to_csv(matrix_path, index=False)
    print(f"Wrote aggregated matrix: {matrix_path}")

    # Also save individual seed results
    combined_path = out_dir / "all_seeds_eval.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Wrote seed-wise results: {combined_path}")


if __name__ == "__main__":
    main()
