#!/usr/bin/env python3
"""
Quick TTC statistics from trajectory_ttc.py output.
"""
import sys
import pandas as pd
import numpy as np

ttc_df = pd.read_csv("outputs/figures_repro/ttc_samples.csv")

for method in ["Rule-based", "PPO"]:
    sub = ttc_df[ttc_df["method"] == method]
    # Remove inf and negative TTC
    sub = sub[(np.isfinite(sub["ttc"])) & (sub["ttc"] > 0)]
    
    print(f"\n{method}:")
    print(f"  Samples: {len(sub)}")
    print(f"  Mean TTC: {sub['ttc'].mean():.3f} s")
    print(f"  Median TTC: {sub['ttc'].median():.3f} s")
    print(f"  Min TTC: {sub['ttc'].min():.3f} s")
    print(f"  P5 TTC: {sub['ttc'].quantile(0.05):.3f} s")
    print(f"  P10 TTC: {sub['ttc'].quantile(0.10):.3f} s")
    
    # Critical TTC threshold
    critical_ttc = 2.5
    pct_critical = (sub['ttc'] < critical_ttc).mean() * 100
    print(f"  % TTC < {critical_ttc}s (critical): {pct_critical:.2f}%")
    
    # Collision risk
    collision_ttc = 1.0
    pct_collision = (sub['ttc'] < collision_ttc).mean() * 100
    print(f"  % TTC < {collision_ttc}s (collision risk): {pct_collision:.4f}%")
