import argparse
import itertools
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def collect_eval_runs(eval_root: Path, include_legacy_ablation: bool) -> list[Path]:
    files = list(eval_root.glob("flow*_*/*/eval_runs.csv"))
    files += list(eval_root.glob("2way-single-intersection/ablation/*/eval_runs.csv"))
    if include_legacy_ablation:
        files += list(eval_root.glob("2way-single-intersection/ablation_legacy/*/eval_runs.csv"))
    files = sorted(set(files))
    return files


def normalize_runs(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    tag = fp.parent.parent.name

    if tag.startswith("flow") and "_" in tag:
        flow_str, dist = tag.split("_", 1)
        flow = int(flow_str.replace("flow", ""))
        df["flow"] = flow
        df["dist"] = dist
    else:
        if "flow" not in df.columns:
            df["flow"] = 600
        if "dist" not in df.columns:
            df["dist"] = "ablation_2way"

    if "method" not in df.columns:
        if {"reward_mode", "obs_mode"}.issubset(df.columns):
            df["method"] = df.apply(
                lambda r: f"ablation_{str(r['reward_mode'])}_{str(r['obs_mode'])}",
                axis=1,
            )
        else:
            df["method"] = "unknown_method"

    if "flow_ns" not in df.columns:
        df["flow_ns"] = (pd.to_numeric(df["flow"], errors="coerce").fillna(600).astype(int) // 2)
    if "flow_we" not in df.columns:
        df["flow_we"] = (pd.to_numeric(df["flow"], errors="coerce").fillna(600).astype(int) // 2)

    # Keep file provenance to trace sources after merge.
    df["source_eval_runs"] = str(fp)
    return df


def build_all_runs_matrix(eval_root: Path, include_legacy_ablation: bool) -> pd.DataFrame:
    files = collect_eval_runs(eval_root, include_legacy_ablation=include_legacy_ablation)
    if not files:
        raise FileNotFoundError(f"No eval_runs.csv found under: {eval_root}")

    frames = [normalize_runs(fp) for fp in files]
    out = pd.concat(frames, ignore_index=True)
    # Drop deprecated ablation setting that removes collision penalty.
    if "reward_mode" in out.columns:
        out = out[out["reward_mode"].astype(str) != "no_collision"].copy()
    out = out[~out["method"].astype(str).str.contains("no_collision", na=False)].copy()
    out["throughput"] = out["total_arrived"] / 600.0 * 3600.0
    return out


def summarize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "avg_travel_time",
        "mean_waiting_time",
        "mean_speed",
        "throughput",
        "collisions",
        "min_ttc",
        "harsh_brake_rate",
        "mean_abs_jerk",
        "gini_waiting_time",
    ]
    present = [m for m in metrics if m in df.columns]
    aggs: dict[str, list[str]] = {m: ["mean", "std"] for m in present}
    out = (
        df.groupby(["flow", "dist", "flow_ns", "flow_we", "method"], as_index=False)
        .agg(aggs)
        .sort_values(["flow", "dist", "method"])  # type: ignore[arg-type]
    )
    out.columns = [
        c if isinstance(c, str) else (f"{c[0]}_{c[1]}" if c[1] else c[0])
        for c in out.columns
    ]
    return out


def run_significance(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy import stats
    except Exception:
        return pd.DataFrame([{"info": "scipy not installed, significance skipped"}])

    metrics = [
        "avg_travel_time",
        "mean_waiting_time",
        "mean_speed",
        "throughput",
        "collisions",
        "min_ttc",
        "harsh_brake_rate",
        "mean_abs_jerk",
        "gini_waiting_time",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows: list[dict[str, Any]] = []
    for (flow, dist), sub in df.groupby(["flow", "dist"]):
        methods = sorted(set(sub["method"].astype(str).tolist()))
        if len(methods) < 2:
            continue

        for m1, m2 in itertools.combinations(methods, 2):
            a = sub[sub["method"] == m1]
            b = sub[sub["method"] == m2]
            for metric in metrics:
                x = pd.to_numeric(a[metric], errors="coerce").dropna().values
                y = pd.to_numeric(b[metric], errors="coerce").dropna().values
                if len(x) < 2 or len(y) < 2:
                    continue
                t_p = stats.ttest_ind(x, y, equal_var=False).pvalue
                mw_p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                rows.append(
                    {
                        "flow": int(flow),
                        "dist": str(dist),
                        "method_a": m1,
                        "method_b": m2,
                        "metric": metric,
                        "t_test_p": float(t_p),
                        "mannwhitney_p": float(mw_p),
                    }
                )

    if not rows:
        return pd.DataFrame([{"info": "insufficient samples for significance"}])
    return pd.DataFrame(rows)


def cleanup_legacy_dirs(root: Path, apply_delete: bool) -> list[tuple[Path, str]]:
    candidates = [
        root / "outputs" / "2way-single-intersection" / "ablation",
        root / "outputs" / "eval_repro" / "2way-single-intersection" / "ablation_legacy",
        root / "logs" / "thesis_cn_2way",
        root / "logs" / "repro" / "thesis_cn_2way_legacy",
    ]

    actions: list[tuple[Path, str]] = []
    for p in candidates:
        if not p.exists():
            continue
        if apply_delete:
            shutil.rmtree(p)
            actions.append((p, "deleted"))
        else:
            actions.append((p, "would_delete"))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rebuild all_runs_matrix (including ablation), run significance, and optionally clean legacy folders.",
    )
    parser.add_argument("--eval-root", type=str, default="outputs/eval_repro")
    parser.add_argument("--matrix-out", type=str, default="outputs/figures_repro/all_runs_matrix.csv")
    parser.add_argument("--summary-out", type=str, default="outputs/reports_repro/实验汇总_均值标准差_all_runs_matrix.csv")
    parser.add_argument("--sig-out", type=str, default="outputs/reports_repro/显著性检验_all_runs_matrix.csv")
    parser.add_argument("--include-legacy-ablation", action="store_true")
    parser.add_argument("--cleanup-legacy", action="store_true")
    parser.add_argument("--apply-delete", action="store_true", help="Actually delete legacy folders when --cleanup-legacy is set")
    args = parser.parse_args()

    eval_root = ROOT / args.eval_root
    matrix_out = ROOT / args.matrix_out
    summary_out = ROOT / args.summary_out
    sig_out = ROOT / args.sig_out

    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    sig_out.parent.mkdir(parents=True, exist_ok=True)

    matrix_df = build_all_runs_matrix(eval_root, include_legacy_ablation=args.include_legacy_ablation)
    matrix_df.to_csv(matrix_out, index=False, encoding="utf-8-sig")

    summary_df = summarize_matrix(matrix_df)
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")

    sig_df = run_significance(matrix_df)
    sig_df.to_csv(sig_out, index=False, encoding="utf-8-sig")

    print(f"Wrote matrix: {matrix_out}")
    print(f"Wrote summary: {summary_out}")
    print(f"Wrote significance: {sig_out}")
    print(f"Rows in matrix: {len(matrix_df)}")

    if args.cleanup_legacy:
        actions = cleanup_legacy_dirs(ROOT, apply_delete=args.apply_delete)
        if not actions:
            print("No legacy folders found for cleanup.")
        else:
            for p, status in actions:
                print(f"{status}: {p}")


if __name__ == "__main__":
    main()
