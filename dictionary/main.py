# main.py
"""
主程序 - DST 数值实验平台

Usage:
    python main.py --experiment iv_strike
    python main.py --experiment iv_maturity
    python main.py --experiment robustness_hawkes
    python main.py --experiment all
"""

import argparse
import time
import sys
from pathlib import Path

# 确保可以导入模块
sys.path.insert(0, str(Path(__file__).parent))

from experiments.exp_iv_strike import run_iv_strike_experiment, plot_iv_strike
from experiments.exp_iv_maturity import run_iv_maturity_experiment, plot_iv_maturity
from experiments.exp_iv_jump_sensitivity import run_jump_sensitivity_experiment, plot_jump_sensitivity
from experiments.exp_mean_variance_scatter import run_mean_variance_experiment, plot_mean_variance
from experiments.exp_robustness_hawkes import run_robustness_hawkes_experiment, plot_robustness_hawkes
from experiments.exp_robustness_hurst import run_robustness_hurst_experiment, plot_robustness_hurst
from experiments.exp_mc_convergence import run_mc_convergence_experiment, plot_mc_convergence
from experiments.exp_nexp_convergence import run_nexp_convergence_experiment, plot_nexp_convergence
from experiments.exp_boundary_comparison import run_boundary_comparison, plot_boundary_comparison


# 默认配置
DEFAULT_CONFIG = {
    # 市场参数
    'S0': 100.0,
    'r': 0.02,
    'q': 0.0,
    
    # Heston 参数
    'v0': 0.04,
    'kappa': 2.0,
    'theta': 0.04,
    'xi': 0.3,
    'rho': -0.7,
    
    # Rough Heston 参数
    'H': 0.1,
    'N_exp': 5,
    
    # Merton 跳跃参数
    'lambda_M': 0.5,
    'muJ': -0.1,
    'sigmaJ': 0.3,
    
    # Hawkes 参数
    'lambdaS0': 0.5,
    'lambdaV0': 0.3,
    'betaS': 3.0,
    'betaV': 2.0,
    'alphaSS': 2.0,
    'alphaSV': 0.5,
    'alphaVS': 0.3,
    'alphaVV': 1.5,
    
    # 方差跳跃参数
    'eta': 0.1,
    'muV': -0.5,
    'sigmaV': 0.3,
    
    # 仿真参数
    'T': 1.0,
    'n_steps': 100,
    'n_paths': 20000,
    
    # 实验网格
    'T_values': [0.1, 0.5],
    'strike_grid': [0.8, 0.9, 1.0, 1.1, 1.2],
    'maturity_grid': [0.1, 0.25, 0.5, 1.0, 2.0],
    'moneyness_grid': [0.9, 1.0, 1.1],
    'H_grid': [0.05, 0.1, 0.2, 0.5],
    'alpha_S_grid': [0.5, 1.0, 1.5, 2.0, 2.5],
    'sigmaJ_grid': [0.1, 0.2, 0.3, 0.4, 0.5],
    'N_exp_grid': [2, 3, 5, 7, 10],
    'n_paths_grid': [5000, 10000, 20000, 50000],
    
    # LSM 参数
    'basis_type': 'polynomial',
    'basis_degree': 2,
    
    # 随机种子
    'seed': 42
}


def run_experiment(experiment_name: str, config: dict = None):
    """运行单个实验"""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    print("="*60)
    print(f"Running experiment: {experiment_name}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        if experiment_name == "iv_strike":
            df = run_iv_strike_experiment(config)
            plot_iv_strike(df, config)
            
        elif experiment_name == "iv_maturity":
            df = run_iv_maturity_experiment(config)
            plot_iv_maturity(df, config)
            
        elif experiment_name == "jump_sensitivity":
            df = run_jump_sensitivity_experiment(config)
            plot_jump_sensitivity(df, config)
            
        elif experiment_name == "mean_variance":
            df, scatter_df = run_mean_variance_experiment(config)
            plot_mean_variance(df, scatter_df, config)
            
        elif experiment_name == "robustness_hawkes":
            df = run_robustness_hawkes_experiment(config)
            plot_robustness_hawkes(df, config)
            
        elif experiment_name == "robustness_hurst":
            df = run_robustness_hurst_experiment(config)
            plot_robustness_hurst(df, config)
            
        elif experiment_name == "mc_convergence":
            df = run_mc_convergence_experiment(config)
            plot_mc_convergence(df, config)
            
        elif experiment_name == "nexp_convergence":
            df = run_nexp_convergence_experiment(config)
            plot_nexp_convergence(df, config)
            
        elif experiment_name == "boundary_comparison":
            boundaries = run_boundary_comparison(config)
            plot_boundary_comparison(boundaries, config)
            
        else:
            print(f"Unknown experiment: {experiment_name}")
            return False
        
        elapsed = time.time() - start_time
        print(f"\n{experiment_name} completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error running {experiment_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_experiments(config: dict = None):
    """运行所有实验"""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    experiments = [
        "iv_strike",
        "iv_maturity", 
        "jump_sensitivity",
        "mean_variance",
        "robustness_hawkes",
        "robustness_hurst",
        "mc_convergence",
        "nexp_convergence",
        "boundary_comparison"
    ]
    
    print("="*60)
    print("Running ALL experiments")
    print("="*60)
    
    results = {}
    total_start = time.time()
    
    for exp in experiments:
        success = run_experiment(exp, config)
        results[exp] = "SUCCESS" if success else "FAILED"
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for exp, status in results.items():
        print(f"  {exp}: {status}")
    print(f"\nTotal time: {total_elapsed:.2f} seconds")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="DST Numerical Experiment Platform")
    parser.add_argument("--experiment", type=str, default="all",
                       help="Experiment to run: iv_strike, iv_maturity, jump_sensitivity, "
                           "mean_variance, robustness_hawkes, robustness_hurst, "
                           "mc_convergence, nexp_convergence, boundary_comparison, all")
    parser.add_argument("--n_paths", type=int, default=None,
                       help="Number of Monte Carlo paths")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--T", type=float, default=None,
                       help="Maturity")
    
    args = parser.parse_args()
    
    # 合并配置
    config = DEFAULT_CONFIG.copy()
    if args.n_paths is not None:
        config['n_paths'] = args.n_paths
    if args.seed is not None:
        config['seed'] = args.seed
    if args.T is not None:
        config['T'] = args.T
    
    # 创建必要的目录
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # 运行实验
    if args.experiment == "all":
        run_all_experiments(config)
    else:
        run_experiment(args.experiment, config)


if __name__ == "__main__":
    main()
