import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _load_runs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        fp = Path(p)
        if fp.is_dir():
            fp = fp / "eval_runs.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Missing eval_runs.csv: {fp}")
        df = pd.read_csv(fp)
        df["source"] = str(fp)
        frames.append(df)
    if not frames:
        raise ValueError("No eval runs provided")
    return pd.concat(frames, ignore_index=True)


def _add_metrics(df: pd.DataFrame, seconds: int = 600) -> pd.DataFrame:
    out = df.copy()
    out["吞吐量_veh_h"] = out["total_arrived"] / float(seconds) * 3600.0
    return out


def _summary_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "flow_ns", "flow_we"], as_index=False).agg(
        平均旅行时间均值=("avg_travel_time", "mean"),
        平均旅行时间标准差=("avg_travel_time", "std"),
        平均等待时间均值=("mean_waiting_time", "mean"),
        平均等待时间标准差=("mean_waiting_time", "std"),
        平均速度均值=("mean_speed", "mean"),
        平均速度标准差=("mean_speed", "std"),
        吞吐量均值=("吞吐量_veh_h", "mean"),
        吞吐量标准差=("吞吐量_veh_h", "std"),
        碰撞次数均值=("collisions", "mean"),
        碰撞次数标准差=("collisions", "std"),
        最小TTC均值=("min_ttc", "mean"),
        最小TTC标准差=("min_ttc", "std"),
        急减速率均值=("harsh_brake_rate", "mean"),
        急减速率标准差=("harsh_brake_rate", "std"),
        平均绝对Jerk均值=("mean_abs_jerk", "mean"),
        平均绝对Jerk标准差=("mean_abs_jerk", "std"),
        等待公平性Gini均值=("gini_waiting_time", "mean"),
        等待公平性Gini标准差=("gini_waiting_time", "std"),
        样本数=("run", "count"),
    )
    return g.sort_values(["flow_ns", "flow_we", "method"]).reset_index(drop=True)


def _significance(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy import stats
    except Exception:
        return pd.DataFrame([
            {"说明": "未安装scipy，跳过显著性检验。可安装 scipy 后重跑。"}
        ])

    rows = []
    metrics = [
        "avg_travel_time",
        "mean_waiting_time",
        "mean_speed",
        "吞吐量_veh_h",
        "collisions",
        "min_ttc",
        "harsh_brake_rate",
        "mean_abs_jerk",
        "gini_waiting_time",
    ]
    for (fns, fwe), sub in df.groupby(["flow_ns", "flow_we"]):
        methods = sorted(sub["method"].unique())
        if len(methods) < 2:
            continue
        if "Rule-based" in methods and ("PPO" in methods or "DQN" in methods):
            baseline = "Rule-based"
            others = [m for m in methods if m != baseline]
            for m in others:
                a = sub[sub["method"] == baseline]
                b = sub[sub["method"] == m]
                for metric in metrics:
                    x = a[metric].dropna().values
                    y = b[metric].dropna().values
                    if len(x) < 2 or len(y) < 2:
                        continue
                    t_p = stats.ttest_ind(x, y, equal_var=False).pvalue
                    mw_p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                    rows.append(
                        {
                            "flow_ns": int(fns),
                            "flow_we": int(fwe),
                            "对比": f"{baseline} vs {m}",
                            "指标": metric,
                            "t检验p值": float(t_p),
                            "MannWhitney_p值": float(mw_p),
                        }
                    )
    if not rows:
        return pd.DataFrame([{"说明": "样本不足或方法不足，未生成显著性检验结果。"}])
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "method", "flow_ns", "flow_we",
        "平均旅行时间均值", "平均旅行时间标准差",
        "平均等待时间均值", "平均等待时间标准差",
        "吞吐量均值", "吞吐量标准差",
        "平均速度均值", "平均速度标准差",
        "碰撞次数均值", "碰撞次数标准差", "样本数",
        "最小TTC均值", "最小TTC标准差",
        "急减速率均值", "急减速率标准差",
        "平均绝对Jerk均值", "平均绝对Jerk标准差",
        "等待公平性Gini均值", "等待公平性Gini标准差",
    ]
    use = df[cols].copy()
    headers = [
        "方法" if c == "method" else "南北流量" if c == "flow_ns" else "东西流量" if c == "flow_we" else c
        for c in cols
    ]
    rename_map = dict(zip(cols, headers))
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, r in use.iterrows():
        vals = []
        for original_col in cols:
            v = r[original_col]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="生成中文实验汇总表（均值±标准差）与显著性检验结果。",
    )
    p.add_argument("--runs", nargs="+", required=True, help="eval_runs.csv 文件或目录列表")
    p.add_argument("--seconds", type=int, default=600)
    p.add_argument("--outdir", type=str, default="outputs/reports_repro")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = _load_runs(args.runs)
    df = _add_metrics(df, seconds=args.seconds)

    summary = _summary_mean_std(df)
    sig = _significance(df)

    summary_csv = outdir / "实验汇总_均值标准差.csv"
    sig_csv = outdir / "显著性检验.csv"
    md_path = outdir / "实验汇总表.md"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    sig.to_csv(sig_csv, index=False, encoding="utf-8-sig")

    md = "# 实验结果汇总（中文）\n\n"
    md += "## 1. 均值±标准差\n\n"
    md += _markdown_table(summary)
    md += "\n\n## 2. 显著性检验\n\n"
    sig_headers = list(sig.columns)
    md += "| " + " | ".join(sig_headers) + " |\n"
    md += "|" + "|".join(["---"] * len(sig_headers)) + "|\n"
    for _, r in sig.iterrows():
        vals = []
        for h in sig_headers:
            v = r[h]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        md += "| " + " | ".join(vals) + " |\n"
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {sig_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
