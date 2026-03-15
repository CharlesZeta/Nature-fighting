"""
============================================================
统计指标模块 - 用于多 seed 分析、收敛实验、误差计算
============================================================
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def compute_ci95(values: np.ndarray) -> Tuple[float, float, float]:
    """
    计算 95% 置信区间
    
    参数:
    -----
    values : ndarray
        多次实验的结果数组
        
    返回:
    -----
    (mean, std, ci95) : tuple
        均值、标准差、95% 置信区间半宽
    """
    n = len(values)
    if n < 2:
        return np.mean(values), 0.0, 0.0
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # 样本标准差
    ci95 = 1.96 * std / np.sqrt(n)
    
    return mean, std, ci95


def compute_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    计算 RMSE
    
    参数:
    -----
    pred : ndarray
        预测值
    ref : ndarray
        参考值
        
    返回:
    -----
    rmse : float
    """
    return np.sqrt(np.mean((pred - ref) ** 2))


def compute_rel_rmse(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算相对 RMSE (百分比)
    
    rel_rmse = RMSE / mean(|ref|) * 100%
    """
    rmse = compute_rmse(pred, ref)
    mean_ref = np.mean(np.abs(ref))
    if mean_ref < eps:
        return np.nan
    return rmse / mean_ref * 100


def compute_ivrmse(iv_pred: np.ndarray, iv_ref: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算 IV-RMSE (隐含波动率 RMSE)
    """
    valid_mask = ~np.isnan(iv_pred) & ~np.isnan(iv_ref)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    pred_valid = iv_pred[valid_mask]
    ref_valid = iv_ref[valid_mask]
    
    rmse = np.sqrt(np.mean((pred_valid - ref_valid) ** 2))
    mean_ref = np.mean(np.abs(ref_valid))
    
    if mean_ref < eps:
        return np.nan
    return rmse / mean_ref * 100


def aggregate_multi_seed(results_list: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    对多次 seed 的结果进行汇总统计
    
    参数:
    -----
    results_list : list of dict
        每次实验的结果字典列表
        
    返回:
    -----
    aggregated : dict
        每个指标的 {mean, std, ci95} 字典
    """
    if not results_list:
        return {}
    
    # 确定要聚合的数值字段
    numeric_keys = set()
    for r in results_list:
        for k, v in r.items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)):
                numeric_keys.add(k)
    
    aggregated = {}
    
    for key in numeric_keys:
        values = []
        for r in results_list:
            v = r.get(key)
            if v is not None and not np.isnan(v):
                values.append(v)
        
        if len(values) >= 2:
            mean, std, ci95 = compute_ci95(np.array(values))
            aggregated[key] = {
                'mean': mean,
                'std': std,
                'ci95': ci95,
                'n': len(values)
            }
        elif len(values) == 1:
            aggregated[key] = {
                'mean': values[0],
                'std': 0.0,
                'ci95': 0.0,
                'n': 1
            }
    
    return aggregated


def aggregate_by_group(df: pd.DataFrame, 
                      group_cols: List[str], 
                      value_cols: List[str]) -> pd.DataFrame:
    """
    按组聚合多 seed 结果
    
    参数:
    -----
    df : DataFrame
        包含多次实验结果的 DataFrame
    group_cols : list
        分组列（如 ['model', 'H', 'T', 'moneyness']）
    value_cols : list
        需要聚合的数值列
        
    返回:
    -----
    aggregated_df : DataFrame
        聚合后的 DataFrame，包含 mean, std, ci95 列
    """
    result_rows = []
    
    for (group_vals), group_df in df.groupby(group_cols):
        if isinstance(group_vals, str):
            group_vals = (group_vals,)
        
        row = {}
        for i, col in enumerate(group_cols):
            row[col] = group_vals[i]
        
        for col in value_cols:
            values = group_df[col].dropna().values
            if len(values) >= 2:
                mean, std, ci95 = compute_ci95(values)
                row[f'{col}_mean'] = mean
                row[f'{col}_std'] = std
                row[f'{col}_ci95'] = ci95
            elif len(values) == 1:
                row[f'{col}_mean'] = values[0]
                row[f'{col}_std'] = 0.0
                row[f'{col}_ci95'] = 0.0
            else:
                row[f'{col}_mean'] = np.nan
                row[f'{col}_std'] = np.nan
                row[f'{col}_ci95'] = np.nan
        
        result_rows.append(row)
    
    return pd.DataFrame(result_rows)


def compute_decomposition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 early exercise 分解统计
    
    参数:
    -----
    df : DataFrame
        包含 strict_early, maturity, total, gap 的 DataFrame
        
    返回:
    -----
    stats : DataFrame
        分解一致性统计
    """
    stats = []
    
    for (model, H, T), group in df.groupby(['model', 'H', 'T']):
        gap = group['decomposition_gap'].dropna()
        
        stats.append({
            'model': model,
            'H': H,
            'T': T,
            'gap_mean': gap.mean() if len(gap) > 0 else np.nan,
            'gap_std': gap.std() if len(gap) > 0 else np.nan,
            'gap_max_abs': gap.abs().max() if len(gap) > 0 else np.nan,
            'gap_within_tolerance': (gap.abs() < 1e-10).all() if len(gap) > 0 else np.nan,
        })
    
    return pd.DataFrame(stats)


def format_summary_table(df: pd.DataFrame, 
                        metrics: List[str] = None) -> str:
    """
    格式化 summary table 为可读字符串
    
    参数:
    -----
    df : DataFrame
        汇总表
    metrics : list, optional
        要显示的指标列
        
    返回:
    -----
    table_str : str
        格式化的表格字符串
    """
    if metrics is None:
        metrics = ['model', 'H', 'T', 'american_price_mean', 'european_price_mean', 
                   'american_premium_mean', 'delta_iv_mean', 'strict_early_exercise_ratio_mean']
    
    # 保留有效列
    cols = [c for c in metrics if c in df.columns]
    df_display = df[cols].copy()
    
    # 数值列保留 4 位小数
    for col in df_display.columns:
        if col != 'model':
            df_display[col] = df_display[col].round(4)
    
    return df_display.to_string(index=False)


if __name__ == '__main__':
    # 测试
    # 模拟 5 次 seed 的结果
    test_results = [
        {'american_premium': 0.065, 'delta_iv': 0.012, 'strict_early': 0.42},
        {'american_premium': 0.068, 'delta_iv': 0.011, 'strict_early': 0.44},
        {'american_premium': 0.063, 'delta_iv': 0.013, 'strict_early': 0.41},
        {'american_premium': 0.067, 'delta_iv': 0.012, 'strict_early': 0.43},
        {'american_premium': 0.066, 'delta_iv': 0.012, 'strict_early': 0.42},
    ]
    
    agg = aggregate_multi_seed(test_results)
    print("多 seed 聚合结果:")
    for k, v in agg.items():
        print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, ci95=±{v['ci95']:.4f}")