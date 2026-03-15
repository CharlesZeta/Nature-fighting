"""
============================================================
结果导出与可视化模块
- 输出结构化 CSV 表
- 为 MATLAB 提供数据接口
============================================================
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional

def export_price_table(df_results: pd.DataFrame, output_path: str):
    """
    导出价格结果表
    """
    price_columns = [
        'model', 'H', 'vol_type', 'jump_type',
        'S0', 'K', 'T', 'moneyness', 'r',
        'american_price', 'european_price', 'american_premium',
        'early_exercise_ratio', 'mean_itm_samples',
    ]
    
    # 选择存在的列
    available_cols = [c for c in price_columns if c in df_results.columns]
    df_price = df_results[available_cols].copy()
    
    df_price.to_csv(output_path, index=False)
    print(f"价格表已保存: {output_path}")
    
    return df_price


def export_smile_table(df_results: pd.DataFrame, output_path: str):
    """
    导出 Smile 相关表
    """
    smile_columns = [
        'model', 'H', 'T', 'moneyness', 'K',
        'implied_vol_proxy',
    ]
    
    available_cols = [c for c in smile_columns if c in df_results.columns]
    df_smile = df_results[available_cols].copy()
    
    df_smile.to_csv(output_path, index=False)
    print(f"Smile 表已保存: {output_path}")
    
    return df_smile


def export_error_table(df_results: pd.DataFrame, output_path: str):
    """
    导出误差分析表
    """
    error_columns = [
        'model', 'H', 'T', 'moneyness',
        'american_price', 'reference_price',
        'abs_error', 'rel_error',
    ]
    
    available_cols = [c for c in error_columns if c in df_results.columns]
    df_error = df_results[available_cols].copy()
    
    # 计算 RMSE
    if 'abs_error' in df_error.columns and not df_error['abs_error'].isna().all():
        rmse = np.sqrt(np.nanmean(df_error['abs_error']**2))
        print(f"Overall RMSE: {rmse:.6f}")
    
    df_error.to_csv(output_path, index=False)
    print(f"误差表已保存: {output_path}")
    
    return df_error


def export_jump_stats(df_results: pd.DataFrame, output_path: str):
    """
    导出跳跃统计表
    """
    jump_columns = [
        'model', 'H', 'T', 'moneyness',
        'mean_jump_count_S', 'mean_jump_count_V',
        'mean_lambda_S', 'mean_lambda_V',
    ]
    
    available_cols = [c for c in jump_columns if c in df_results.columns]
    df_jump = df_results[available_cols].copy()
    
    df_jump.to_csv(output_path, index=False)
    print(f"跳跃统计表已保存: {output_path}")
    
    return df_jump


def export_boundary_table(df_results: pd.DataFrame, output_path: str):
    """
    导出提前行权边界表
    """
    boundary_columns = [
        'model', 'H', 'T', 'K', 'moneyness',
        'exercise_boundary_T01', 'exercise_boundary_T05',
    ]
    
    available_cols = [c for c in boundary_columns if c in df_results.columns]
    df_boundary = df_results[available_cols].copy()
    
    df_boundary.to_csv(output_path, index=False)
    print(f"行权边界表已保存: {output_path}")
    
    return df_boundary


def generate_matlab_data(df_results: pd.DataFrame, output_dir: str):
    """
    生成 MATLAB 可用的数据文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 按模型分组保存
    for model in df_results['model'].unique():
        df_model = df_results[df_results['model'] == model]
        
        # 按 H 和 T 分组
        for (H, T), group in df_model.groupby(['H', 'T']):
            # Smile 数据: moneyness vs IV
            df_smile = group[['moneyness', 'K', 'american_price', 'european_price', 
                             'american_premium', 'implied_vol_proxy']].copy()
            
            filename = f"matlab_{model}_H{H:.2f}_T{T:.2f}.csv"
            df_smile.to_csv(f"{output_dir}/{filename}", index=False)
    
    # 2. 汇总表: ATM 数据
    df_atm = df_results[np.abs(df_results['moneyness'] - 1.0) < 0.05].copy()
    df_atm = df_atm[['model', 'H', 'T', 'american_price', 'european_price', 
                    'american_premium', 'implied_vol_proxy']].copy()
    df_atm.to_csv(f"{output_dir}/matlab_atm_summary.csv", index=False)
    
    # 3. Term structure 数据
    df_term = df_atm.pivot_table(values='implied_vol_proxy', 
                                  index=['model', 'H'], 
                                  columns='T').reset_index()
    df_term.to_csv(f"{output_dir}/matlab_term_structure.csv", index=False)
    
    print(f"MATLAB 数据已保存: {output_dir}/")


def create_summary_statistics(df_results: pd.DataFrame, output_dir: str):
    """
    创建汇总统计表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary = []
    
    # 按模型汇总
    for model in df_results['model'].unique():
        df_model = df_results[df_results['model'] == model]
        
        # 过滤掉错误行
        if 'error' in df_model.columns:
            df_model = df_model[df_model['error'].isna()]
        
        if len(df_model) == 0:
            continue
        
        summary.append({
            'model': model,
            'n_experiments': len(df_model),
            'mean_american_price': df_model['american_price'].mean(),
            'mean_european_price': df_model['european_price'].mean(),
            'mean_premium': df_model['american_premium'].mean(),
            'mean_iv': df_model['implied_vol_proxy'].mean(),
            'std_premium': df_model['american_premium'].std(),
            'std_iv': df_model['implied_vol_proxy'].std(),
        })
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(f"{output_dir}/summary_statistics.csv", index=False)
    print(f"汇总统计已保存: {output_dir}/summary_statistics.csv")
    
    return df_summary


def export_all(df_results: pd.DataFrame, output_dir: str = 'results第二轮'):
    """
    导出所有结果表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 价格表
    export_price_table(df_results, f"{output_dir}/price_table.csv")
    
    # Smile 表
    export_smile_table(df_results, f"{output_dir}/smile_table.csv")
    
    # 误差表 (如果有)
    if 'abs_error' in df_results.columns:
        export_error_table(df_results, f"{output_dir}/error_table.csv")
    
    # 跳跃统计表
    if 'mean_jump_count_S' in df_results.columns:
        export_jump_stats(df_results, f"{output_dir}/jump_stats.csv")
    
    # 边界表
    if 'exercise_boundary_T01' in df_results.columns:
        export_boundary_table(df_results, f"{output_dir}/boundary_table.csv")
    
    # MATLAB 数据
    generate_matlab_data(df_results, f"{output_dir}/matlab")
    
    # 汇总统计
    create_summary_statistics(df_results, output_dir)
    
    print(f"\n所有结果已导出到: {output_dir}/")


def load_results(input_path: str) -> pd.DataFrame:
    """
    加载结果文件
    """
    return pd.read_csv(input_path)


def compare_models(df_results: pd.DataFrame, metric: str = 'american_price') -> pd.DataFrame:
    """
    模型对比分析
    """
    # 按 model, H, T 分组计算均值
    comparison = df_results.groupby(['model', 'H', 'T'])[metric].mean().unstack(level='T')
    
    return comparison


def plot_convergence(df_results: pd.DataFrame, output_dir: str = 'results第二轮'):
    """
    绘制收敛性分析 (需要 matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
        
        # 路径数 vs 误差
        # 需要多次不同路径数的实验，这里只是框架
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Premium 分布
        ax = axes[0, 0]
        for model in df_results['model'].unique():
            df_model = df_results[df_results['model'] == model]
            if 'error' in df_model.columns:
                df_model = df_model[df_model['error'].isna()]
            ax.hist(df_model['american_premium'].dropna(), alpha=0.5, label=model)
        ax.set_xlabel('American Premium')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title('Premium Distribution')
        
        # 2. IV vs Moneyness
        ax = axes[0, 1]
        for model in ['M2', 'M5']:
            df_model = df_results[(df_results['model'] == model) & (df_results['H'] == 0.05)]
            if len(df_model) > 0:
                ax.plot(df_model['moneyness'], df_model['implied_vol_proxy'], 
                       'o-', label=model)
        ax.set_xlabel('Moneyness (K/S0)')
        ax.set_ylabel('Implied Volatility')
        ax.legend()
        ax.set_title('IV Smile')
        
        # 3. Premium vs Maturity
        ax = axes[1, 0]
        df_atm = df_results[np.abs(df_results['moneyness'] - 1.0) < 0.05]
        for model in ['M2', 'M5']:
            df_model = df_atm[df_atm['model'] == model]
            if len(df_model) > 0:
                ax.plot(df_model['T'], df_model['american_premium'], 
                       'o-', label=model)
        ax.set_xlabel('Maturity')
        ax.set_ylabel('American Premium')
        ax.legend()
        ax.set_title('Premium vs Maturity')
        
        # 4. Early Exercise Ratio
        ax = axes[1, 1]
        for model in ['M2', 'M5']:
            df_model = df_atm[df_atm['model'] == model]
            if len(df_model) > 0:
                ax.plot(df_model['T'], df_model['early_exercise_ratio'], 
                       'o-', label=model)
        ax.set_xlabel('Maturity')
        ax.set_ylabel('Early Exercise Ratio')
        ax.legend()
        ax.set_title('Early Exercise vs Maturity')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_plots.png", dpi=150)
        plt.close()
        
        print(f"分析图表已保存: {output_dir}/analysis_plots.png")
        
    except ImportError:
        print("matplotlib 未安装，跳过绘图")


if __name__ == '__main__':
    # 测试导出功能
    test_data = {
        'model': ['M2', 'M5', 'M2', 'M5'],
        'H': [0.05, 0.05, 0.10, 0.10],
        'T': [0.10, 0.10, 1.00, 1.00],
        'moneyness': [0.90, 0.90, 1.00, 1.00],
        'K': [90, 90, 100, 100],
        'american_price': [8.5, 8.7, 10.2, 10.4],
        'european_price': [8.2, 8.3, 9.8, 9.9],
        'american_premium': [0.3, 0.4, 0.4, 0.5],
        'implied_vol_proxy': [0.25, 0.27, 0.22, 0.24],
    }
    
    df_test = pd.DataFrame(test_data)
    
    # 测试导出
    export_all(df_test, 'test_output')
