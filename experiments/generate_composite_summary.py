#!/usr/bin/env python3
"""
生成综合数据总表：包含所有seed、所有实验、所有分布的评估和消融结果
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np


def collect_eval_summaries() -> pd.DataFrame:
    """收集所有评估结果汇总表"""
    eval_files = list(Path("outputs").rglob("*eval*summary*.csv"))
    
    all_data = []
    for csv_file in eval_files:
        try:
            df = pd.read_csv(csv_file)
            # 添加来源信息
            df['source'] = str(csv_file)
            df['experiment'] = 'evaluation'
            all_data.append(df)
            print(f"✓ 加载: {csv_file}")
        except Exception as e:
            print(f"✗ 错误 {csv_file}: {e}")
    
    if not all_data:
        print("警告: 未找到评估结果")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def collect_ablation_summaries() -> pd.DataFrame:
    """收集所有消融结果汇总表"""
    ablation_files = list(Path("outputs").rglob("*ablation*summary*.csv"))
    
    all_data = []
    for csv_file in ablation_files:
        try:
            df = pd.read_csv(csv_file)
            # 添加来源信息
            df['source'] = str(csv_file)
            df['experiment'] = 'ablation'
            all_data.append(df)
            print(f"✓ 加载: {csv_file}")
        except Exception as e:
            print(f"✗ 错误 {csv_file}: {e}")
    
    if not all_data:
        print("警告: 未找到消融结果")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def parse_source_info(source: str) -> Dict[str, str]:
    """从文件路径解析信息"""
    parts = source.lower().split(os.sep)
    
    info = {
        'seed': 'unknown',
        'distribution': 'unknown',
        'intersection': 'single'
    }
    
    for part in parts:
        if 'seed' in part:
            for word in part.split('_'):
                if word.isdigit():
                    info['seed'] = word
                    break
        if '4way' in part or 'intersection' in part:
            info['intersection'] = '4way'
        if 'uniform' in part:
            info['distribution'] = 'uniform'
        elif 'poisson' in part:
            info['distribution'] = 'poisson'
        elif 'burst' in part:
            info['distribution'] = 'burst'
    
    return info


def enrich_with_source_info(df: pd.DataFrame) -> pd.DataFrame:
    """使用源信息丰富数据"""
    if 'source' not in df.columns:
        return df
    
    source_info_list = []
    for source in df['source']:
        info = parse_source_info(source)
        source_info_list.append(info)
    
    source_df = pd.DataFrame(source_info_list)
    for col in source_df.columns:
        df[f'parsed_{col}'] = source_df[col]
    
    return df


def generate_composite_summary(eval_df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    """生成综合摘要表"""
    summary_rows = []
    
    # 评估结果摘要
    if not eval_df.empty:
        print("\n=== 评估结果摘要 ===")
        if 'method' in eval_df.columns:
            for method in eval_df['method'].unique():
                method_data = eval_df[eval_df['method'] == method]
                
                metrics = {}
                metrics['experiment_type'] = 'evaluation'
                metrics['method'] = str(method)
                
                # 计算主要指标的统计
                numeric_cols = method_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in method_data.columns:
                        valid_data = pd.to_numeric(method_data[col], errors='coerce').dropna()
                        if len(valid_data) > 0:
                            metrics[f'{col}_mean'] = valid_data.mean()
                            metrics[f'{col}_std'] = valid_data.std()
                            metrics[f'{col}_count'] = len(valid_data)
                
                summary_rows.append(metrics)
                print(f"  {method}: {len(valid_data)} 次评估")
    
    # 消融结果摘要
    if not ablation_df.empty:
        print("\n=== 消融结果摘要 ===")
        if 'reward_mode' in ablation_df.columns and 'obs_mode' in ablation_df.columns:
            for reward_mode in ablation_df['reward_mode'].unique():
                for obs_mode in ablation_df['obs_mode'].unique():
                    setting_data = ablation_df[
                        (ablation_df['reward_mode'] == reward_mode) & 
                        (ablation_df['obs_mode'] == obs_mode)
                    ]
                    
                    if not setting_data.empty:
                        metrics = {}
                        metrics['experiment_type'] = 'ablation'
                        metrics['reward_mode'] = str(reward_mode)
                        metrics['obs_mode'] = str(obs_mode)
                        
                        numeric_cols = setting_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            valid_data = pd.to_numeric(setting_data[col], errors='coerce').dropna()
                            if len(valid_data) > 0:
                                metrics[f'{col}_mean'] = valid_data.mean()
                                metrics[f'{col}_std'] = valid_data.std()
                                metrics[f'{col}_count'] = len(valid_data)
                        
                        summary_rows.append(metrics)
                        print(f"  {reward_mode}+{obs_mode}: {len(valid_data)} 次结果")
    
    if summary_rows:
        return pd.DataFrame(summary_rows)
    else:
        return pd.DataFrame()


def main() -> None:
    print("=" * 60)
    print("生成综合数据总表")
    print("=" * 60)
    
    # 收集数据
    print("\n【第1步】收集评估结果...")
    eval_df = collect_eval_summaries()
    
    print("\n【第2步】收集消融结果...")
    ablation_df = collect_ablation_summaries()
    
    # 丰富信息
    if not eval_df.empty:
        print("\n【第3步】解析评估数据来源信息...")
        eval_df = enrich_with_source_info(eval_df)
    
    if not ablation_df.empty:
        print("\n【第4步】解析消融数据来源信息...")
        ablation_df = enrich_with_source_info(ablation_df)
    
    # 生成综合摘要
    print("\n【第5步】生成综合摘要...")
    composite_summary = generate_composite_summary(eval_df, ablation_df)
    
    # 保存输出
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n【第6步】保存综合总表...")
    
    # 1. 完整评估表
    if not eval_df.empty:
        eval_output = output_dir / "all_evaluation_results.csv"
        eval_df.to_csv(eval_output, index=False)
        print(f"✓ 已保存: {eval_output}")
    
    # 2. 完整消融表
    if not ablation_df.empty:
        ablation_output = output_dir / "all_ablation_results.csv"
        ablation_df.to_csv(ablation_output, index=False)
        print(f"✓ 已保存: {ablation_output}")
    
    # 3. 综合摘要表
    if not composite_summary.empty:
        summary_output = output_dir / "composite_summary.csv"
        composite_summary.to_csv(summary_output, index=False)
        print(f"✓ 已保存: {summary_output}")
        
        print("\n" + "=" * 60)
        print("综合摘要表预览:")
        print("=" * 60)
        print(composite_summary.to_string())
    
    # 4. 合并所有数据的总表
    if not eval_df.empty and not ablation_df.empty:
        # 对齐列
        common_cols = set(eval_df.columns) & set(ablation_df.columns)
        
        eval_aligned = eval_df[[col for col in eval_df.columns if col in common_cols or col in ['method', 'experiment']]].copy()
        ablation_aligned = ablation_df[[col for col in ablation_df.columns if col in common_cols or col in ['reward_mode', 'obs_mode', 'experiment']]].copy()
        
        # 添加实验标识
        if 'experiment' not in eval_aligned.columns:
            eval_aligned['experiment'] = 'evaluation'
        if 'experiment' not in ablation_aligned.columns:
            ablation_aligned['experiment'] = 'ablation'
        
        master_table = pd.concat([eval_aligned, ablation_aligned], ignore_index=True, sort=False)
        master_output = output_dir / "master_data_table.csv"
        master_table.to_csv(master_output, index=False)
        print(f"✓ 已保存: {master_output}")
    
    print("\n" + "=" * 60)
    print("✅ 综合数据总表生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
