#!/usr/bin/env python3
"""Compute TTC statistics for thesis."""
import pandas as pd
import numpy as np

ttc_df = pd.read_csv('outputs/figures_repro/ttc_samples.csv')

for method in ['Rule-based', 'PPO']:
    sub = ttc_df[ttc_df['method'] == method]
    # Remove inf and negative TTC
    sub_valid = sub[(np.isfinite(sub['ttc'])) & (sub['ttc'] > 0)]
    
    print(f'\n=== {method} ===')
    print(f'Total samples: {len(sub)}')
    print(f'Valid samples (no inf, > 0): {len(sub_valid)}')
    print(f'Mean TTC: {sub_valid["ttc"].mean():.3f} s')
    print(f'Median TTC: {sub_valid["ttc"].median():.3f} s')
    print(f'Min TTC: {sub_valid["ttc"].min():.3f} s')
    print(f'P5: {sub_valid["ttc"].quantile(0.05):.3f} s')
    print(f'P25: {sub_valid["ttc"].quantile(0.25):.3f} s')
    print(f'P50: {sub_valid["ttc"].quantile(0.50):.3f} s')
    print(f'P75: {sub_valid["ttc"].quantile(0.75):.3f} s')
    print(f'P95: {sub_valid["ttc"].quantile(0.95):.3f} s')
    print(f'Std Dev: {sub_valid["ttc"].std():.3f} s')
    
    # Dangerous events (TTC < 2.5s)
    pct_2_5 = (sub_valid['ttc'] < 2.5).mean() * 100
    print(f'% TTC < 2.5s (dangerous): {pct_2_5:.2f}%')
    
    # Collision risk
    pct_1_0 = (sub_valid['ttc'] < 1.0).mean() * 100
    print(f'% TTC < 1.0s (collision risk): {pct_1_0:.2f}%')
