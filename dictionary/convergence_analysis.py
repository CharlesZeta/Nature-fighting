"""
============================================================
收敛与误差分析实验
- 路径数收敛 (N_paths)
- 时间步收敛 (N_steps)  
- 多 seed 误差分析 (CI)
- 量化对比表生成
============================================================
"""

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import config
from simulation import PathSimulator
from lsm_american import AmericanLSM
from implied_vol import implied_volatility
from metrics import compute_ci95, aggregate_multi_seed, aggregate_by_group, compute_decomposition_stats


def run_convergence_experiment(
    model_name='M2',
    H=0.1,
    vol_type='rough',
    jump_type='none',
    S0=100,
    K=100,
    T=0.5,
    n_steps=100,
    use_aggressive_jump=False,
    output_dir='results第二轮',
    seed_base=42
):
    """
    运行收敛性实验：测试不同路径数下的定价稳定性
    
    参数:
    -----
    model_name : str
        模型名称 (M0-M5)
    H : float
        Hurst 指数
    vol_type : str
        'markovian' 或 'rough'
    jump_type : str
        'none', 'poisson', 'double exponential'
    output_dir : str
        输出目录
    seed_base : int
        基础随机种子
        
    返回:
    -----
    results : DataFrame
        包含不同路径数的定价结果
    """
    
    # 生成参数字典
    params = config.generate_param_dict(
        H=H, 
        vol_type=vol_type, 
        jump_type=jump_type,
        use_aggressive_jump=use_aggressive_jump
    )
    
    # 测试不同的路径数
    path_counts = [5000, 10000, 20000, 50000, 100000]
    results = []
    
    print(f"\n{'='*60}")
    print(f"收敛实验: {model_name}, H={H}, T={T}, K/S0={K/S0}")
    print(f"{'='*60}")
    
    for n_paths in path_counts:
        print(f"  路径数: {n_paths}")
        
        # 多次 seed 运行以计算误差
        n_seeds = 5
        prices = []
        premiums = []
        ivs = []
        
        for s in range(n_seeds):
            seed = seed_base + s * 1000 + n_paths
            
            # 仿真路径
            np.random.seed(seed)
            simulator = PathSimulator(params)
            sim_results = simulator.simulate(T, n_paths, n_steps, seed=seed, antithetic=True)
            S_paths = sim_results['S']
            
            # 定价
            pricer = AmericanLSM(S0, K, params['r'], T, 
                               option_type='put',
                               poly_degree=config.POLY_DEGREE,
                               ridge_alpha=config.RIDGE_ALPHA)
            
            result = pricer.price_american(S_paths, return_boundary=False)
            
            prices.append(result['american_price'])
            premiums.append(result['american_premium'])
            
            # IV
            try:
                iv = implied_volatility(result['american_price'], S0, K, T, params['r'], 'put')
            except:
                iv = np.nan
            ivs.append(iv)
        
        # 统计
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        mean_premium = np.mean(premiums)
        std_premium = np.std(premiums)
        mean_iv = np.nanmean(ivs)
        std_iv = np.nanstd(ivs)
        
        # 95% CI
        ci_price = 1.96 * std_price / np.sqrt(n_seeds)
        ci_premium = 1.96 * std_premium / np.sqrt(n_seeds)
        
        results.append({
            'model': model_name,
            'H': H,
            'T': T,
            'moneyness': K/S0,
            'n_paths': n_paths,
            'n_seeds': n_seeds,
            'mean_american_price': mean_price,
            'std_american_price': std_price,
            'ci95_american_price': ci_price,
            'mean_european_price': result['european_price'],
            'mean_american_premium': mean_premium,
            'std_american_premium': std_premium,
            'ci95_american_premium': ci_premium,
            'mean_iv_american': mean_iv,
            'std_iv_american': std_iv,
        })
        
        print(f"    价格: {mean_price:.4f} ± {std_price:.4f} (95% CI: ±{ci_price:.4f})")
        print(f"    Premium: {mean_premium:.4f} ± {std_premium:.4f}")
    
    df = pd.DataFrame(results)
    
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/convergence_paths_{model_name}_H{H}_T{T}_{timestamp}.csv'
    df.to_csv(filename, index=False)
    
    print(f"\n收敛实验结果已保存到: {filename}")
    
    return df


def run_time_step_convergence(
    model_name='M2',
    H=0.1,
    vol_type='rough',
    jump_type='none',
    S0=100,
    K=100,
    T=0.5,
    n_paths=20000,
    use_aggressive_jump=False,
    output_dir='results第二轮',
    seed_base=42
):
    """
    时间步收敛实验
    """
    params = config.generate_param_dict(
        H=H, 
        vol_type=vol_type, 
        jump_type=jump_type,
        use_aggressive_jump=use_aggressive_jump
    )
    
    step_counts = [20, 50, 100, 200, 500]
    results = []
    
    print(f"\n{'='*60}")
    print(f"时间步收敛实验: {model_name}, H={H}, T={T}")
    print(f"{'='*60}")
    
    for n_steps in step_counts:
        print(f"  时间步: {n_steps}")
        
        np.random.seed(seed_base)
        simulator = PathSimulator(params)
        sim_results = simulator.simulate(T, n_paths, n_steps, seed=seed_base, antithetic=True)
        S_paths = sim_results['S']
        
        pricer = AmericanLSM(S0, K, params['r'], T, 
                           option_type='put',
                           poly_degree=config.POLY_DEGREE,
                           ridge_alpha=config.RIDGE_ALPHA)
        
        result = pricer.price_american(S_paths, return_boundary=False)
        
        results.append({
            'model': model_name,
            'H': H,
            'T': T,
            'moneyness': K/S0,
            'n_steps': n_steps,
            'american_price': result['american_price'],
            'european_price': result['european_price'],
            'american_premium': result['american_premium'],
        })
        
        print(f"    价格: {result['american_price']:.4f}, Premium: {result['american_premium']:.4f}")
    
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/convergence_steps_{model_name}_H{H}_T{T}_{timestamp}.csv'
    df.to_csv(filename, index=False)
    
    return df


def run_multi_seed_ci_experiment(
    models_to_test=None,
    output_dir='results第二轮',
    seed_base=42,
    n_seeds=5,
    n_paths=20000,
    n_steps=100
):
    """
    多 seed 误差分析实验：计算各模型的 95% 置信区间
    
    返回:
    -----
    ci_results : DataFrame
        包含各模型在不同 moneyness 下的均值、标准差、95% CI
    """
    
    if models_to_test is None:
        models_to_test = ['M0', 'M2', 'M5']
    
    # 固定测试参数
    T = 0.5
    moneyness_grid = [0.80, 0.90, 1.00, 1.10, 1.20]
    
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"多 seed 误差分析实验 (n_seeds={n_seeds})")
    print(f"{'='*60}")
    
    for model_name in models_to_test:
        model_cfg = config.MODEL_CONFIG[model_name]
        vol_type = model_cfg['vol_type']
        jump_type = model_cfg['jump_type']
        actual_H = 0.5 if vol_type == 'markovian' else 0.1
        
        params = config.generate_param_dict(
            H=actual_H, 
            vol_type=vol_type, 
            jump_type=jump_type,
            use_aggressive_jump=False
        )
        
        for moneyness in moneyness_grid:
            K = config.S0 * moneyness
            
            prices = []
            premiums = []
            early_ratios = []
            
            for seed_idx in range(n_seeds):
                seed = seed_base + seed_idx * 100
                
                np.random.seed(seed)
                simulator = PathSimulator(params)
                sim_results = simulator.simulate(T, n_paths, n_steps, seed=seed, antithetic=True)
                S_paths = sim_results['S']
                
                pricer = AmericanLSM(config.S0, K, params['r'], T, 
                                   option_type='put',
                                   poly_degree=config.POLY_DEGREE,
                                   ridge_alpha=config.RIDGE_ALPHA)
                
                result = pricer.price_american(S_paths, return_boundary=False)
                
                prices.append(result['american_price'])
                premiums.append(result['american_premium'])
                early_ratios.append(result.get('strict_early_exercise_ratio', np.nan))
            
            # 计算统计量
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            ci95_price = 1.96 * std_price / np.sqrt(n_seeds)
            
            mean_premium = np.mean(premiums)
            std_premium = np.std(premiums)
            ci95_premium = 1.96 * std_premium / np.sqrt(n_seeds)
            
            mean_early = np.mean(early_ratios)
            std_early = np.std(early_ratios)
            ci95_early = 1.96 * std_early / np.sqrt(n_seeds)
            
            all_results.append({
                'model': model_name,
                'H': actual_H,
                'T': T,
                'moneyness': moneyness,
                'n_seeds': n_seeds,
                'mean_price': mean_price,
                'std_price': std_price,
                'ci95_price': ci95_price,
                'mean_premium': mean_premium,
                'std_premium': std_premium,
                'ci95_premium': ci95_premium,
                'mean_early_ratio': mean_early,
                'std_early_ratio': std_early,
                'ci95_early_ratio': ci95_early,
            })
            
            print(f"{model_name}, K/S0={moneyness:.2f}: Price={mean_price:.4f}±{ci95_price:.4f}, Premium={mean_premium:.4f}±{ci95_premium:.4f}")
    
    df = pd.DataFrame(all_results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{output_dir}/ci_analysis_{timestamp}.csv'
    df.to_csv(filename, index=False)
    
    # 同时保存 latest 版本
    df.to_csv(f'{output_dir}/ci_analysis_latest.csv', index=False)
    
    print(f"\n误差分析结果已保存到: {filename}")
    
    return df


def generate_summary_table_from_main(df_main, output_dir='results第二轮'):
    """
    从主实验结果生成量化对比表
    """
    return generate_summary_table(df_main, output_dir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='收敛与误差分析实验')
    parser.add_argument('--mode', type=str, default='ci', 
                       choices=['paths', 'steps', 'ci', 'summary'],
                       help='实验模式: paths(路径收敛), steps(时间步收敛), ci(置信区间), summary(汇总表)')
    parser.add_argument('--model', type=str, default='M2', help='模型名称')
    parser.add_argument('--H', type=float, default=0.1, help='Hurst 指数')
    parser.add_argument('--T', type=float, default=0.5, help='到期时间')
    parser.add_argument('--output', type=str, default='results第二轮', help='输出目录')
    parser.add_argument('--seeds', type=int, default=5, help='seed 数量')
    parser.add_argument('--paths', type=int, default=20000, help='路径数')
    parser.add_argument('--steps', type=int, default=100, help='时间步数')
    
    args = parser.parse_args()
    
    if args.mode == 'paths':
        run_convergence_experiment(
            model_name=args.model,
            H=args.H,
            T=args.T,
            output_dir=args.output,
            seed_base=42
        )
    elif args.mode == 'steps':
        run_time_step_convergence(
            model_name=args.model,
            H=args.H,
            T=args.T,
            n_paths=args.paths,
            output_dir=args.output,
            seed_base=42
        )
    elif args.mode == 'ci':
        run_multi_seed_ci_experiment(
            models_to_test=['M0', 'M2', 'M5'],
            output_dir=args.output,
            seed_base=42,
            n_seeds=args.seeds,
            n_paths=args.paths,
            n_steps=args.steps
        )
    elif args.mode == 'summary':
        # 读取主结果文件
        df_main = pd.read_csv(f'{args.output}/price_results_latest.csv')
        generate_summary_table(df_main, args.output)