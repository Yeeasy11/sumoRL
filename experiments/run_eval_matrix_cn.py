import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="统一实验矩阵评估入口（流量×到达分布×方法）。",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--eval-script", type=str, default="experiments/eval_unified_cn.py")
    p.add_argument("--out-root", type=str, default="outputs/eval_repro")
    p.add_argument("--flows", nargs="+", type=int, default=[300, 600, 900])
    p.add_argument("--dists", nargs="+", type=str, default=["uniform", "poisson", "burst"])
    p.add_argument("--methods", nargs="+", type=str, default=["idm", "fixed_speed", "yield", "ppo", "dqn"])
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--delta-time", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ppo-model", type=str, default="models/repro/ppo_seed0.zip")
    p.add_argument("--dqn-model", type=str, default="models/repro/dqn_seed0.zip")
    p.add_argument("--fixed-speed", type=float, default=10.0)
    p.add_argument("--yield-dist", type=float, default=20.0)
    args = p.parse_args()

    py = args.python
    eval_script = args.eval_script

    total = 0
    for flow in args.flows:
        for dist in args.dists:
            route = f"sumo_rl/nets/single-intersection/routes/single_{flow}_{dist}.rou.xml"
            for method in args.methods:
                outdir = Path(args.out_root) / f"flow{flow}_{dist}" / method
                outdir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    py,
                    eval_script,
                    "--method",
                    method,
                    "--route",
                    route,
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
                    "--ppo-model",
                    args.ppo_model,
                    "--dqn-model",
                    args.dqn_model,
                    "--fixed-speed",
                    str(args.fixed_speed),
                    "--yield-dist",
                    str(args.yield_dist),
                    "--outdir",
                    str(outdir),
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                total += 1

    print(f"Done matrix evaluation tasks: {total}")


if __name__ == "__main__":
    main()
