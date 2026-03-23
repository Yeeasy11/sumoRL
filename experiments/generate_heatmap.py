#!/usr/bin/env python3
"""
Generate heatmap visualizations comparing Rule-based vs PPO methods.
Shows traffic congestion distribution at the intersection.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import xml.etree.ElementTree as ET
from scipy.stats import gaussian_kde

# Publication-quality defaults
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['axes.grid'] = False
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10


def parse_tripinfo_xml(xml_path: Path) -> pd.DataFrame:
    """
    Parse SUMO tripinfo XML file to extract vehicle trajectory data.
    
    Returns DataFrame with columns:
    - id: vehicle id
    - depart: departure time
    - arrival: arrival time (when left network)
    - duration: total trip time
    - waitingTime: total waiting time
    - waitingCount: number of stops
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = []
    for tripinfo in root.findall('tripinfo'):
        data.append({
            'id': tripinfo.get('id'),
            'depart': float(tripinfo.get('depart')),
            'arrival': float(tripinfo.get('arrival')),
            'duration': float(tripinfo.get('duration')),
            'waitingTime': float(tripinfo.get('waitingTime', 0)),
            'waitingCount': int(tripinfo.get('waitingCount', 0)),
            'timeLoss': float(tripinfo.get('timeLoss', 0)),
        })
    
    return pd.DataFrame(data)


def load_method_data(eval_dir: Path, method: str) -> pd.DataFrame:
    """
    Load all runs for a given method and combine.
    """
    tripinfo_dir = eval_dir / "tripinfo"
    if not tripinfo_dir.exists():
        print(f"⚠️  No tripinfo directory in {eval_dir}")
        return pd.DataFrame()
    
    all_data = []
    for xml_file in sorted(tripinfo_dir.glob(f"{method}*.xml")):
        df = parse_tripinfo_xml(xml_file)
        all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def create_heatmap_comparison(eval_600_dir: Path, out_dir: Path) -> None:
    """
    Create heatmap comparing Rule-based vs PPO traffic distribution.
    Shows: waiting time distribution, trip duration distribution, traffic density.
    """
    print("\n📊 Loading tripinfo data...")
    
    # Load both methods
    df_rule = load_method_data(eval_600_dir, "Rule-based")
    df_ppo = load_method_data(eval_600_dir, "PPO")
    
    if df_rule.empty or df_ppo.empty:
        print("⚠️  Could not load tripinfo data")
        return
    
    print(f"  Rule-based: {len(df_rule)} vehicles")
    print(f"  PPO: {len(df_ppo)} vehicles")
    
    # Setup figure with 3 heatmaps (2D distributions)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Traffic Congestion Heatmaps: Rule-based vs PPO", 
                fontsize=14, fontweight='bold', y=0.995)
    
    # ========================================================================
    # Row 1: Rule-based method
    # ========================================================================
    
    # 1.1: Waiting Time vs Duration (Rule-based)
    ax = axes[0, 0]
    if len(df_rule) > 2:
        try:
            x = df_rule['waitingTime'].values
            y = df_rule['duration'].values
            # Create 2D histogram
            h = ax.hist2d(x, y, bins=30, cmap='YlOrRd', cmin=1)
            ax.set_xlabel('Waiting Time (s)', fontweight='bold')
            ax.set_ylabel('Trip Duration (s)', fontweight='bold')
            ax.set_title('Rule-based: Waiting vs Duration', fontweight='bold', fontsize=11)
            plt.colorbar(h[3], ax=ax, label='Count')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    # 1.2: Arrival Time Distribution (Rule-based)
    ax = axes[0, 1]
    if len(df_rule) > 2:
        try:
            # Bin arrival times into 1-minute buckets
            arrival_binned = pd.cut(df_rule['arrival'], bins=60, labels=False)
            wait_by_time = df_rule.groupby(arrival_binned)['waitingTime'].agg(['mean', 'count'])
            x_time = np.arange(len(wait_by_time)) * 10  # 10s per bin
            colors_intensity = wait_by_time['count'].values / wait_by_time['count'].max()
            scatter = ax.scatter(x_time, wait_by_time['mean'].values, 
                               c=colors_intensity, s=100, cmap='YlOrRd', alpha=0.7, edgecolors='black')
            ax.set_xlabel('Arrival Time (s)', fontweight='bold')
            ax.set_ylabel('Mean Waiting Time (s)', fontweight='bold')
            ax.set_title('Rule-based: Traffic Evolution', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Vehicle Count (normalized)')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    # 1.3: Waiting Count Distribution (Rule-based)
    ax = axes[0, 2]
    if len(df_rule) > 0:
        try:
            wait_count_dist = df_rule['waitingCount'].value_counts().sort_index()
            colors_rb = ['#FF6B6B' if i > 5 else '#FFB3B3' for i in wait_count_dist.index]
            ax.bar(wait_count_dist.index, wait_count_dist.values, color=colors_rb, 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Number of Stops', fontweight='bold')
            ax.set_ylabel('Vehicle Count', fontweight='bold')
            ax.set_title('Rule-based: Stop Frequency', fontweight='bold', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    # ========================================================================
    # Row 2: PPO method
    # ========================================================================
    
    # 2.1: Waiting Time vs Duration (PPO)
    ax = axes[1, 0]
    if len(df_ppo) > 2:
        try:
            x = df_ppo['waitingTime'].values
            y = df_ppo['duration'].values
            h = ax.hist2d(x, y, bins=30, cmap='Blues', cmin=1)
            ax.set_xlabel('Waiting Time (s)', fontweight='bold')
            ax.set_ylabel('Trip Duration (s)', fontweight='bold')
            ax.set_title('PPO: Waiting vs Duration', fontweight='bold', fontsize=11)
            plt.colorbar(h[3], ax=ax, label='Count')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    # 2.2: Arrival Time Distribution (PPO)
    ax = axes[1, 1]
    if len(df_ppo) > 2:
        try:
            arrival_binned = pd.cut(df_ppo['arrival'], bins=60, labels=False)
            wait_by_time = df_ppo.groupby(arrival_binned)['waitingTime'].agg(['mean', 'count'])
            x_time = np.arange(len(wait_by_time)) * 10
            colors_intensity = wait_by_time['count'].values / wait_by_time['count'].max()
            scatter = ax.scatter(x_time, wait_by_time['mean'].values,
                               c=colors_intensity, s=100, cmap='Blues', alpha=0.7, edgecolors='darkblue')
            ax.set_xlabel('Arrival Time (s)', fontweight='bold')
            ax.set_ylabel('Mean Waiting Time (s)', fontweight='bold')
            ax.set_title('PPO: Traffic Evolution', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Vehicle Count (normalized)')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    # 2.3: Waiting Count Distribution (PPO)
    ax = axes[1, 2]
    if len(df_ppo) > 0:
        try:
            wait_count_dist = df_ppo['waitingCount'].value_counts().sort_index()
            colors_ppo = ['#4ECDC4' if i > 5 else '#A8E6E1' for i in wait_count_dist.index]
            ax.bar(wait_count_dist.index, wait_count_dist.values, color=colors_ppo,
                  edgecolor='darkblue', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('Number of Stops', fontweight='bold')
            ax.set_ylabel('Vehicle Count', fontweight='bold')
            ax.set_title('PPO: Stop Frequency', fontweight='bold', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center')
    
    plt.tight_layout()
    out_path = out_dir / "congestion_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {out_path}")
    plt.close()


def create_waiting_time_comparison(eval_600_dir: Path, out_dir: Path) -> None:
    """
    Create detailed comparison of waiting time metrics.
    """
    df_rule = load_method_data(eval_600_dir, "Rule-based")
    df_ppo = load_method_data(eval_600_dir, "PPO")
    
    if df_rule.empty or df_ppo.empty:
        return
    
    # Create figure with distribution comparisons
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Waiting Time Analysis: Rule-based vs PPO", 
                fontsize=14, fontweight='bold')
    
    # 1. Waiting time distribution (histograms)
    ax = axes[0, 0]
    ax.hist(df_rule['waitingTime'], bins=40, alpha=0.6, label='Rule-based', 
           color='#FF6B6B', edgecolor='black', linewidth=1)
    ax.hist(df_ppo['waitingTime'], bins=40, alpha=0.6, label='PPO', 
           color='#4ECDC4', edgecolor='black', linewidth=1)
    ax.set_xlabel('Waiting Time (s)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Waiting Time Distribution', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Cumulative distribution
    ax = axes[0, 1]
    sorted_rule = np.sort(df_rule['waitingTime'])
    sorted_ppo = np.sort(df_ppo['waitingTime'])
    ax.plot(sorted_rule, np.linspace(0, 1, len(sorted_rule)), 
           label='Rule-based', color='#FF6B6B', linewidth=2.5)
    ax.plot(sorted_ppo, np.linspace(0, 1, len(sorted_ppo)), 
           label='PPO', color='#4ECDC4', linewidth=2.5)
    ax.set_xlabel('Waiting Time (s)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontweight='bold')
    ax.set_title('CDF: Waiting Time', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Time Loss comparison
    ax = axes[1, 0]
    data_to_plot = [df_rule['timeLoss'].values, df_ppo['timeLoss'].values]
    bp = ax.boxplot(data_to_plot, labels=['Rule-based', 'PPO'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][1].set_facecolor('#4ECDC4')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Time Loss (s)', fontweight='bold')
    ax.set_title('Time Loss Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats = {
        'Metric': [
            'Mean Waiting Time',
            'Median Waiting Time',
            'Max Waiting Time',
            'Std Dev',
            'Mean Time Loss',
            'Mean Trip Duration',
            'Mean Stops'
        ],
        'Rule-based': [
            f"{df_rule['waitingTime'].mean():.3f}",
            f"{df_rule['waitingTime'].median():.3f}",
            f"{df_rule['waitingTime'].max():.3f}",
            f"{df_rule['waitingTime'].std():.3f}",
            f"{df_rule['timeLoss'].mean():.3f}",
            f"{df_rule['duration'].mean():.3f}",
            f"{df_rule['waitingCount'].mean():.2f}",
        ],
        'PPO': [
            f"{df_ppo['waitingTime'].mean():.3f}",
            f"{df_ppo['waitingTime'].median():.3f}",
            f"{df_ppo['waitingTime'].max():.3f}",
            f"{df_ppo['waitingTime'].std():.3f}",
            f"{df_ppo['timeLoss'].mean():.3f}",
            f"{df_ppo['duration'].mean():.3f}",
            f"{df_ppo['waitingCount'].mean():.2f}",
        ],
        'Improvement (%)': []
    }
    
    # Calculate improvements
    for i in range(len(stats['Rule-based'])):
        try:
            rb_val = float(stats['Rule-based'][i])
            ppo_val = float(stats['PPO'][i])
            if rb_val > 0:
                improvement = (rb_val - ppo_val) / rb_val * 100
                stats['Improvement (%)'].append(f"{improvement:+.1f}%")
            else:
                stats['Improvement (%)'].append("N/A")
        except:
            stats['Improvement (%)'].append("N/A")
    
    df_stats = pd.DataFrame(stats)
    
    table = ax.table(cellText=df_stats.values, colLabels=df_stats.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1],
                    colWidths=[0.35, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(df_stats.columns)):
        table[(0, i)].set_facecolor("#34495E")
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df_stats) + 1):
        for j in range(len(df_stats.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ECF0F1")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")
            
            # Highlight improvements
            if j == 3 and i > 0:  # Improvement column
                cell_text = table[(i, j)].get_text().get_text()
                if '+' in cell_text:  # PPO is better
                    table[(i, j)].set_facecolor("#D5F4E6")
    
    plt.tight_layout()
    out_path = out_dir / "waiting_time_analysis.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmap visualizations for traffic analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/single-intersection",
        help="Base directory containing evaluation results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/thesis_figures",
        help="Output directory for generated figures.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("🎨 GENERATING TRAFFIC HEATMAPS".center(70))
    print("="*70)

    eval_600_dir = base_dir / "eval_600"

    print("\n📊 Congestion Heatmap...")
    create_heatmap_comparison(eval_600_dir, out_dir)

    print("\n📊 Waiting Time Analysis...")
    create_waiting_time_comparison(eval_600_dir, out_dir)

    print("\n" + "="*70)
    print(f"✅ Heatmaps generated in: {out_dir}".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
