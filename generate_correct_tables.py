#!/usr/bin/env python3
"""从 eval_repro 实际实验数据提取并生成一致的表5.2/5.3/5.4"""

import pandas as pd
from pathlib import Path

EVAL_ROOT = Path("outputs/eval_repro")
OUT_DIR = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["ppo", "dqn", "idm", "fixed_speed", "yield"]
METHOD_CN = {
    "ppo": "PPO",
    "dqn": "DQN",
    "idm": "IDM",
    "fixed_speed": "固定速度",
    "yield": "礼让规则",
}


def read_eval_summary(flow_dir: Path, method: str) -> dict:
    f = flow_dir / method / "eval_summary.csv"
    df = pd.read_csv(f)
    r = df.iloc[0]
    return {
        "avg_travel_time": float(r["avg_travel_time"]),
        "mean_waiting_time": float(r["mean_waiting_time"]),
        "total_arrived": float(r["total_arrived"]),
        "min_ttc": float(r["min_ttc"]),
        "harsh_brake_rate": float(r["harsh_brake_rate"]),
        "mean_abs_jerk": float(r["mean_abs_jerk"]),
        "gini_waiting_time": float(r["gini_waiting_time"]),
    }


# ============================================================
# Table 5.2: 标准流量场景 600 veh/h (poisson)
# ============================================================
rows_52 = []
flow_dir = EVAL_ROOT / "flow600_poisson"
for m in METHOD_ORDER:
    d = read_eval_summary(flow_dir, m)
    rows_52.append({
        "模型": METHOD_CN[m],
        "平均旅行时间 (s)": round(d["avg_travel_time"], 2),
        "平均等待时间 (s)": round(d["mean_waiting_time"], 2),
        "吞吐量(veh/h)": round(d["total_arrived"] / 600 * 3600, 1),
        "最小TTC (s)": round(d["min_ttc"], 4),
        "急减速率": round(d["harsh_brake_rate"], 4),
        "平均绝对 Jerk (m/s^3)": round(d["mean_abs_jerk"], 2),
        "等待公平性 Gini": round(d["gini_waiting_time"], 2),
    })
table_5_2 = pd.DataFrame(rows_52)

# ============================================================
# Table 5.3: 不同交通负载 (300/600/900, poisson)
# ============================================================
rows_53 = []
for flow_val, flow_name in [(300, "flow300_poisson"), (600, "flow600_poisson"), (900, "flow900_poisson")]:
    flow_dir = EVAL_ROOT / flow_name
    for m in METHOD_ORDER:
        d = read_eval_summary(flow_dir, m)
        rows_53.append({
            "流量(veh/h)": flow_val,
            "模型": METHOD_CN[m],
            "旅行时间(s)": round(d["avg_travel_time"], 2),
            "等待时间(s)": round(d["mean_waiting_time"], 2),
            "吞吐量(veh/h)": round(d["total_arrived"] / 600 * 3600, 1),
        })
table_5_3 = pd.DataFrame(rows_53)

# ============================================================
# Table 5.4: 300 veh/h 下不同到达分布
# ============================================================
dist_map = {"flow300_burst": "突发", "flow300_poisson": "泊松", "flow300_uniform": "均匀"}
rows_54 = []
for flow_name, dist_cn in dist_map.items():
    flow_dir = EVAL_ROOT / flow_name
    for m in METHOD_ORDER:
        d = read_eval_summary(flow_dir, m)
        rows_54.append({
            "分布": dist_cn,
            "模型": METHOD_CN[m],
            "旅行时间(s)": round(d["avg_travel_time"], 2),
            "等待时间(s)": round(d["mean_waiting_time"], 2),
            "吞吐量(veh/h)": round(d["total_arrived"] / 600 * 3600, 1),
            "最小TTC(s)": round(d["min_ttc"], 4),
            "急减速率": round(d["harsh_brake_rate"], 4),
            "平均绝对Jerk(m/s^3)": round(d["mean_abs_jerk"], 2),
            "等待公平性Gini": round(d["gini_waiting_time"], 2),
        })
table_5_4 = pd.DataFrame(rows_54)

# Write CSVs
table_5_2.to_csv(OUT_DIR / "表5.2_标准流量600下各模型性能指标对比.csv", index=False, encoding="utf-8-sig")
table_5_3.to_csv(OUT_DIR / "表5.3_不同交通负载下的性能表现对照.csv", index=False, encoding="utf-8-sig")
table_5_4.to_csv(OUT_DIR / "表5.4_不同到达分布下的性能表现对照.csv", index=False, encoding="utf-8-sig")

# Write Excel (preserving 5.1 if already present)
excel_path = OUT_DIR / "thesis_tables_5x.xlsx"
try:
    existing_51 = pd.read_excel(excel_path, sheet_name="表5.1_训练收敛")
except Exception:
    existing_51 = None

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    if existing_51 is not None:
        existing_51.to_excel(writer, sheet_name="表5.1_训练收敛", index=False)
    table_5_2.to_excel(writer, sheet_name="表5.2_600场景对比", index=False)
    table_5_3.to_excel(writer, sheet_name="表5.3_负载对照", index=False)
    table_5_4.to_excel(writer, sheet_name="表5.4_分布对照", index=False)

print("已生成一致的表格文件到 outputs/tables/")
print("\n=== 表5.2 (600 veh/h 标准场景) ===")
print(table_5_2.to_string(index=False))
print("\n=== 表5.3 (不同交通负载) ===")
print(table_5_3.to_string(index=False))
print("\n=== 表5.4 (300 veh/h 不同到达分布) ===")
print(table_5_4.to_string(index=False))
