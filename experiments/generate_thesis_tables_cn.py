#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


METHOD_ORDER = ["dqn", "ppo", "idm", "fixed_speed", "yield"]
METHOD_CN = {
    "dqn": "DQN",
    "ppo": "PPO",
    "idm": "IDM",
    "fixed_speed": "固定速度",
    "yield": "礼让规则",
}


def _main_eval_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "experiment_group" in out.columns:
        out = out[out["experiment_group"] == "main_eval"]
    if "dist" in out.columns:
        out["dist"] = out["dist"].astype(str).str.lower()
    if "arrival_dist" in out.columns:
        ad = out["arrival_dist"].astype(str).str.lower().replace({"nan": np.nan})
        out["dist"] = out.get("dist", pd.Series([np.nan] * len(out))).replace({"nan": np.nan}).fillna(ad)
    if "flow" in out.columns:
        out["flow"] = pd.to_numeric(out["flow"], errors="coerce")
    return out


def _load_monitor_series(monitor_dir: Path) -> pd.Series:
    if not monitor_dir.exists():
        return pd.Series(dtype=float)
    files = sorted(monitor_dir.glob("*.monitor.csv"))
    if not files:
        return pd.Series(dtype=float)
    arr = []
    for fp in files:
        try:
            df = pd.read_csv(fp, comment="#")
            if "r" in df.columns:
                arr.append(pd.to_numeric(df["r"], errors="coerce").dropna())
        except Exception:
            continue
    if not arr:
        return pd.Series(dtype=float)
    return pd.concat(arr, ignore_index=True)


def _stability_label(std_val: float) -> str:
    if not np.isfinite(std_val):
        return "数据不足"
    if std_val <= 10:
        return "稳定"
    if std_val <= 25:
        return "较稳定"
    return "波动较大"


def make_table_5_1(log_root: Path) -> pd.DataFrame:
    rows = []
    for algo, sub in [("PPO", "ppo"), ("DQN", "dqn")]:
        monitor_dir = log_root / sub / "monitor"
        rewards = _load_monitor_series(monitor_dir)
        if rewards.empty:
            rows.append(
                {
                    "算法": algo,
                    "训练步数": np.nan,
                    "末段平均回报": np.nan,
                    "回报标准差": np.nan,
                    "峰值平滑回报": np.nan,
                    "稳定性评价": "数据不足",
                }
            )
            continue

        n = len(rewards)
        tail_n = max(10, int(n * 0.2))
        tail = rewards.iloc[-tail_n:]
        smooth = rewards.rolling(window=25, min_periods=5).mean()

        rows.append(
            {
                "算法": algo,
                "训练步数": int(n),
                "末段平均回报": float(tail.mean()),
                "回报标准差": float(tail.std(ddof=0)),
                "峰值平滑回报": float(smooth.max()),
                "稳定性评价": _stability_label(float(tail.std(ddof=0))),
            }
        )

    return pd.DataFrame(rows)


def make_table_5_2(df_main: pd.DataFrame) -> pd.DataFrame:
    d = df_main[df_main["flow"] == 600].copy()
    grp = (
        d.groupby("method", as_index=False)
        .agg(
            avg_travel_time=("avg_travel_time", "mean"),
            mean_waiting_time=("mean_waiting_time", "mean"),
            throughput=("throughput", "mean"),
            min_ttc=("min_ttc", "mean"),
            harsh_brake_rate=("harsh_brake_rate", "mean"),
            mean_abs_jerk=("mean_abs_jerk", "mean"),
            gini_waiting_time=("gini_waiting_time", "mean"),
        )
    )

    rows = []
    for m in METHOD_ORDER:
        sub = grp[grp["method"] == m]
        if sub.empty:
            rows.append(
                {
                    "模型": METHOD_CN[m],
                    "平均旅行时间 (s)": np.nan,
                    "平均等待时间 (s)": np.nan,
                    "吞吐量(veh/h)": np.nan,
                    "最小TTC (s)": np.nan,
                    "急减速率": np.nan,
                    "平均绝对 Jerk (m/s^3)": np.nan,
                    "等待公平性 Gini": np.nan,
                }
            )
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "模型": METHOD_CN[m],
                "平均旅行时间 (s)": float(r["avg_travel_time"]),
                "平均等待时间 (s)": float(r["mean_waiting_time"]),
                "吞吐量(veh/h)": float(r["throughput"]),
                "最小TTC (s)": float(r["min_ttc"]),
                "急减速率": float(r["harsh_brake_rate"]),
                "平均绝对 Jerk (m/s^3)": float(r["mean_abs_jerk"]),
                "等待公平性 Gini": float(r["gini_waiting_time"]),
            }
        )

    return pd.DataFrame(rows)


def make_table_5_3(df_main: pd.DataFrame) -> pd.DataFrame:
    d = df_main[df_main["flow"].isin([300, 600, 900]) & df_main["method"].isin(["ppo", "idm"])].copy()
    grp = (
        d.groupby(["flow", "method"], as_index=False)
        .agg(
            avg_travel_time=("avg_travel_time", "mean"),
            mean_waiting_time=("mean_waiting_time", "mean"),
            throughput=("throughput", "mean"),
        )
    )

    rows = []
    for flow in [300, 600, 900]:
        idm_row = grp[(grp["flow"] == flow) & (grp["method"] == "idm")]
        base_tt = float(idm_row.iloc[0]["avg_travel_time"]) if not idm_row.empty else np.nan

        for method in ["ppo", "idm"]:
            row = grp[(grp["flow"] == flow) & (grp["method"] == method)]
            if row.empty:
                rows.append(
                    {
                        "流量(veh/h)": flow,
                        "模型": METHOD_CN[method],
                        "旅行时间(s)": np.nan,
                        "等待时间(s)": np.nan,
                        "吞吐量(veh/h)": np.nan,
                        "效率增益(vs.IDM)": np.nan,
                    }
                )
                continue
            rr = row.iloc[0]
            tt = float(rr["avg_travel_time"])
            gain = (base_tt - tt) / max(abs(base_tt), 1e-6) * 100.0 if np.isfinite(base_tt) else np.nan
            rows.append(
                {
                    "流量(veh/h)": flow,
                    "模型": METHOD_CN[method],
                    "旅行时间(s)": tt,
                    "等待时间(s)": float(rr["mean_waiting_time"]),
                    "吞吐量(veh/h)": float(rr["throughput"]),
                    "效率增益(vs.IDM)": gain,
                }
            )

    return pd.DataFrame(rows)


def make_table_5_4(df_main: pd.DataFrame) -> pd.DataFrame:
    d = df_main[(df_main["flow"] == 600) & df_main["method"].isin(["ppo", "idm"])].copy()
    grp = (
        d.groupby(["dist", "method"], as_index=False)
        .agg(
            avg_travel_time=("avg_travel_time", "mean"),
            min_ttc=("min_ttc", "mean"),
            mean_abs_jerk=("mean_abs_jerk", "mean"),
        )
    )

    dist_cn = {"uniform": "均匀分布", "poisson": "泊松分布", "burst": "突发分布", "balanced": "平衡分布"}
    # Only include distributions that actually have data
    available_dists = [dist for dist in ["uniform", "poisson", "burst", "balanced"] if dist in grp["dist"].values]

    rows = []
    for dist in available_dists:
        idm_row = grp[(grp["dist"] == dist) & (grp["method"] == "idm")]
        base_tt = float(idm_row.iloc[0]["avg_travel_time"]) if not idm_row.empty else np.nan

        for method in ["ppo", "idm"]:
            row = grp[(grp["dist"] == dist) & (grp["method"] == method)]
            if row.empty:
                rows.append(
                    {
                        "分布": dist_cn.get(dist, dist),
                        "模型": METHOD_CN[method],
                        "旅行时间(s)": np.nan,
                        "最小TTC(s)": np.nan,
                        "急减速率Jerk": np.nan,
                        "效率增益(vs.IDM)": np.nan,
                    }
                )
                continue
            rr = row.iloc[0]
            tt = float(rr["avg_travel_time"])
            gain = (base_tt - tt) / max(abs(base_tt), 1e-6) * 100.0 if np.isfinite(base_tt) else np.nan
            rows.append(
                {
                    "分布": dist_cn.get(dist, dist),
                    "模型": METHOD_CN[method],
                    "旅行时间(s)": tt,
                    "最小TTC(s)": float(rr["min_ttc"]),
                    "急减速率Jerk": float(rr["mean_abs_jerk"]),
                    "效率增益(vs.IDM)": gain,
                }
            )

    return pd.DataFrame(rows)


def _find_latest_complete_ablation(ablation_root: Path) -> tuple[Optional[str], list[Path]]:
    pat = re.compile(r"^(\d{8}_\d{6})_(full|default)_(full|phase_only)$")
    need = {"full_full", "full_phase_only", "default_full", "default_phase_only"}
    stamp_to_keys: dict[str, set[str]] = {}
    stamp_to_dirs: dict[str, list[Path]] = {}

    for d in sorted(ablation_root.glob("20*_*") ):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m is None:
            continue
        stamp = m.group(1)
        key = f"{m.group(2)}_{m.group(3)}"
        stamp_to_keys.setdefault(stamp, set()).add(key)
        stamp_to_dirs.setdefault(stamp, []).append(d)

    complete = [s for s, ks in stamp_to_keys.items() if ks == need]
    if not complete:
        return None, []

    def _has_variation(dirs: list[Path]) -> bool:
        vals = {"avg_travel_time": [], "mean_waiting_time": [], "throughput": [], "collisions": []}
        for d in dirs:
            f = d / "eval" / "eval_summary.csv"
            if not f.exists():
                return False
            df = pd.read_csv(f)
            if df.empty:
                return False
            for m in vals:
                if m in df.columns:
                    vals[m].append(float(pd.to_numeric(df[m], errors="coerce").mean()))
        for arr in vals.values():
            if len(arr) >= 2 and any(abs(v - arr[0]) > 1e-9 for v in arr[1:]):
                return True
        return False

    complete_sorted = sorted(complete)
    latest = complete_sorted[-1]
    latest_dirs = sorted(stamp_to_dirs[latest])
    if _has_variation(latest_dirs):
        return latest, latest_dirs

    varied = [s for s in complete_sorted if _has_variation(sorted(stamp_to_dirs[s]))]
    if varied:
        chosen = varied[-1]
        return chosen, sorted(stamp_to_dirs[chosen])
    return latest, latest_dirs


def make_ablation_table(ablation_root: Path) -> pd.DataFrame:
    setting_cn = {
        ("full", "full"): "压力奖励/完整状态",
        ("full", "phase_only"): "压力奖励/相位状态",
        ("default", "full"): "等待差奖励/完整状态",
        ("default", "phase_only"): "等待差奖励/相位状态",
    }
    pat = re.compile(r"^(\d{8}_\d{6})_(full|default)_(full|phase_only)$")
    stamp, dirs = _find_latest_complete_ablation(ablation_root)
    rows = []
    for d in dirs:
        m = pat.match(d.name)
        if m is None:
            continue
        reward_mode, obs_mode = m.group(2), m.group(3)
        f = d / "eval" / "eval_summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        rows.append(
            {
                "消融设置": setting_cn.get((reward_mode, obs_mode), f"{reward_mode}/{obs_mode}"),
                "平均旅行时间 (s)": float(df["avg_travel_time"].mean()),
                "平均等待时间 (s)": float(df["mean_waiting_time"].mean()),
                "吞吐量(veh/h)": float(df["throughput"].mean()),
                "碰撞数": float(df["collisions"].mean()),
                "ablation_stamp": stamp,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate thesis tables 5.1/5.2/5.3 and ablation table.")
    p.add_argument("--matrix-csv", type=str, default="outputs/all_runs_matrix.csv")
    p.add_argument("--log-root", type=str, default="logs/4way_single_intersection")
    p.add_argument("--ablation-root", type=str, default="outputs/4way-single-intersection/ablation")
    p.add_argument("--out-dir", type=str, default="outputs/tables")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.matrix_csv)
    main_df = _main_eval_subset(df)

    t51 = make_table_5_1(Path(args.log_root))
    t52 = make_table_5_2(main_df)
    t53 = make_table_5_3(main_df)
    t54 = make_table_5_4(main_df)
    tablation = make_ablation_table(Path(args.ablation_root))

    t51.to_csv(out_dir / "表5.1_强化学习模型训练收敛性统计.csv", index=False, encoding="utf-8-sig")
    t52.to_csv(out_dir / "表5.2_标准流量600下各模型性能指标对比.csv", index=False, encoding="utf-8-sig")
    t53.to_csv(out_dir / "表5.3_不同交通负载下的性能表现对照.csv", index=False, encoding="utf-8-sig")
    t54.to_csv(out_dir / "表5.4_不同到达分布下的性能表现对照.csv", index=False, encoding="utf-8-sig")
    tablation.to_csv(out_dir / "表5.3_消融实验性能对照.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_dir / "thesis_tables_5x.xlsx", engine="openpyxl") as writer:
        t51.to_excel(writer, sheet_name="表5.1_训练收敛", index=False)
        t52.to_excel(writer, sheet_name="表5.2_600场景对比", index=False)
        t53.to_excel(writer, sheet_name="表5.3_负载对照", index=False)
        t54.to_excel(writer, sheet_name="表5.4_分布对照", index=False)
        tablation.to_excel(writer, sheet_name="表5.3_消融对照", index=False)

    print(f"Wrote tables to: {out_dir}")


if __name__ == "__main__":
    main()
