"""
============================================================
多方法/多模型对比模块
============================================================
功能：
1. DST vs 其他定价方法 (Heston, Rough Heston, SINC, etc.)
2. DST vs 复合 Hawkes 跳跃过程对比
3. 生成 LaTeX 表格和高对比度图表
============================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import config
from simulation import PathSimulator
from lsm_american import AmericanLSM
from implied_vol import implied_volatility


# Benchmark 参数 (Heston 原始论文)
BENCHMARK_PARAMS = {
    'v0': 0.0175,
    'kappa': 1.5768,
    'theta': 0.0398,
    'xi': 0.5751,
    'rho': -0.5711,
}


class BenchmarkComparison:
    """多方法对比器"""
    
    def __init__(self, S0: float = 100, K: float = 100, T: float = 0.5, 
                 r: float = 0.05, option_type: str = 'put'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.option_type = option_type
        
    def price_dst(self, model: str = 'M2', H: float = 0.1,
                  n_paths: int = 20000, n_steps: int = 100,
                  seed: int = 42) -> Dict:
        """使用 DST/Rough Heston 定价"""
        np.random.seed(seed)
        
        vol_type = 'rough' if H < 0.5 else 'markovian'
        jump_type = 'none'
        
        if model in ['M3', 'M4', 'M5']:
            jump_type = 'compound_poisson'
        
        params = config.generate_param_dict(
            H=H, vol_type=vol_type, jump_type=jump_type
        )
        
        simulator = PathSimulator(params)
        sim_results = simulator.simulate(self.T, n_paths, n_steps, seed=seed, antithetic=True)
        S_paths = sim_results['S']
        
        pricer = AmericanLSM(self.S0, self.K, self.r, self.T,
                           option_type=self.option_type,
                           poly_degree=config.POLY_DEGREE,
                           ridge_alpha=config.RIDGE_ALPHA)
        
        result = pricer.price_american(S_paths, return_boundary=False)
        
        return {
            'american_price': result['american_price'],
            'european_price': result['european_price'],
            'american_premium': result['american_premium'],
            'strict_early_ratio': result.get('strict_early_exercise_ratio', np.nan),
            'maturity_ratio': result.get('exercise_at_maturity_ratio', np.nan),
            'total_ratio': result.get('total_exercise_ratio', np.nan),
        }
    
    def price_heston_analytic(self, S0: float = None, K: float = None,
                            T: float = None, r: float = None) -> float:
        """
        Heston 解析解 (Characteristic Function)
        使用 Carr-Madan 方法
        """
        from math import exp, sqrt, log
        
        S0 = S0 or self.S0
        K = K or self.K
        T = T or self.T
        r = r or self.r
        
        v0 = BENCHMARK_PARAMS['v0']
        kappa = BENCHMARK_PARAMS['kappa']
        theta = BENCHMARK_PARAMS['theta']
        xi = BENCHMARK_PARAMS['xi']
        rho = BENCHMARK_PARAMS['rho']
        
        # 简化：使用 Monte Carlo 作为近似
        # (完整的解析实现需要复杂复数运算)
        n_paths = 50000
        n_steps = 100
        dt = T / n_steps
        
        S = np.zeros(n_paths)
        v = np.zeros(n_paths)
        S[:] = S0
        v[:] = v0
        
        for t in range(n_steps):
            Z1 = np.random.randn(n_paths)
            Z2 = np.random.randn(n_paths)
            W1 = Z1
            W2 = rho * Z1 + sqrt(1 - rho**2) * Z2
            
            v = np.maximum(
                v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * W2,
                0.001
            )
            S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * W1)
        
        payoff = np.maximum(K - S, 0)
        price = np.mean(payoff) * exp(-r * T)
        
        return price
    
    def price_bs(self) -> float:
        """Black-Scholes 解析解"""
        from math import exp, sqrt, log
        from scipy.stats import norm
        
        S0, K, T, r = self.S0, self.K, self.T, self.r
        sigma = 0.2  # 假设波动率
        
        d1 = (log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        
        if self.option_type == 'put':
            price = K * exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        else:
            price = S0 * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
        
        return price
    
    def run_comparison(self,
                      models: List[str] = None,
                      moneyness_grid: List[float] = None,
                      T_values: List[float] = None,
                      n_paths: int = 20000,
                      n_seeds: int = 5,
                      output_dir: str = 'results第三轮') -> pd.DataFrame:
        """
        运行多方法对比实验
        
        参数:
        -----
        models : list
            模型列表，如 ['M0', 'M2', 'M5', 'Heston', 'BS']
        moneyness_grid : list
            moneyness 网格
        T_values : list
            到期时间列表
        n_paths : int
            蒙特卡洛路径数
        n_seeds : int
            seed 数量（用于计算误差）
        output_dir : str
            输出目录
        """
        if models is None:
            models = ['M0', 'M2', 'M5']
        
        if moneyness_grid is None:
            moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        if T_values is None:
            T_values = [0.25, 0.5, 1.0]
        
        print("="*60)
        print("多方法对比实验")
        print("="*60)
        
        results = []
        
        for T in T_values:
            for moneyness in moneyness_grid:
                K = self.S0 * moneyness
                
                # 计算 Benchmark (高精度)
                print(f"\n计算: T={T:.2f}, K/S0={moneyness:.2f}")
                
                # BS 基准
                bs_price = self.price_bs()
                
                # Heston MC 基准
                heston_price = self.price_heston_analytic(T=T, K=K)
                
                # DST 模型
                for model in models:
                    prices = []
                    premiums = []
                    early_ratios = []
                    
                    for seed_idx in range(n_seeds):
                        seed = 42 + seed_idx * 100
                        
                        # 确定 H
                        if model in ['M0', 'M1']:
                            H = 0.5
                        elif model == 'M2':
                            H = 0.1
                        elif model in ['M3', 'M4', 'M5']:
                            H = 0.1
                        else:
                            H = 0.1
                        
                        result = self.price_dst(
                            model=model, H=H,
                            n_paths=n_paths, n_steps=100,
                            seed=seed
                        )
                        
                        prices.append(result['american_price'])
                        premiums.append(result['american_premium'])
                        early_ratios.append(result.get('strict_early_ratio', np.nan))
                    
                    # 统计
                    mean_price = np.mean(prices)
                    std_price = np.std(prices)
                    ci95_price = 1.96 * std_price / np.sqrt(n_seeds)
                    mean_premium = np.mean(premiums)
                    mean_early = np.mean(early_ratios)
                    
                    # 相对误差 (vs Heston)
                    rel_error = (mean_price - heston_price) / heston_price * 100 if heston_price > 0 else np.nan
                    
                    results.append({
                        'model': model,
                        'T': T,
                        'moneyness': moneyness,
                        'K': K,
                        'bs_price': bs_price,
                        'heston_price': heston_price,
                        'dst_price': mean_price,
                        'dst_std': std_price,
                        'dst_ci95': ci95_price,
                        'dst_premium': mean_premium,
                        'early_ratio': mean_early,
                        'rel_error_vs_heston': rel_error,
                        'n_paths': n_paths,
                        'n_seeds': n_seeds,
                    })
                    
                    print(f"  {model}: price={mean_price:.4f}±{ci95_price:.4f}, premium={mean_premium:.4f}")
        
        df = pd.DataFrame(results)
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df.to_csv(f'{output_dir}/benchmark_comparison_{timestamp}.csv', index=False)
        df.to_csv(f'{output_dir}/benchmark_comparison_latest.csv', index=False)
        
        print(f"\n对比结果已保存: {output_dir}/benchmark_comparison_*.csv")
        
        return df
    
    def export_latex_table(self, df: pd.DataFrame, 
                          output_path: str,
                          caption: str = "Model Comparison",
                          label: str = "tab:comparison"):
        """
        导出 LaTeX 格式对比表
        """
        # 筛选特定 T 和 moneyness
        df_sub = df[(df['T'] == 0.5) & (df['moneyness'].isin([0.8, 1.0, 1.2]))]
        
        lines = []
        lines.append(f"\\begin{{table}}[h]")
        lines.append(f"\\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabular}}{{lcccccc}}")
        lines.append(f"\\hline")
        lines.append(f"Method & $K/S_0$ & Price & Std & 95\\% CI & Premium & Rel.Err.\\\\")
        lines.append(f"\\hline")
        
        for _, row in df_sub.iterrows():
            method = row['model']
            moneyness = row['moneyness']
            price = row['dst_price']
            std = row['dst_std']
            ci95 = row['dst_ci95']
            premium = row['dst_premium']
            rel_err = row['rel_error_vs_heston']
            
            lines.append(f"{method:8s} & {moneyness:.2f} & {price:.4f} & {std:.4f} & ±{ci95:.4f} & {premium:.4f} & {rel_err:+.2f}\\%\\\\")
        
        lines.append(f"\\hline")
        lines.append(f"\\end{tabular}")
        lines.append(f"\\end{table}")
        
        latex_code = '\n'.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"LaTeX 表格已保存: {output_path}")
        
        return latex_code


class DstHawkesComparison:
    """DST vs 复合 Hawkes 对比"""
    
    def __init__(self, S0: float = 100, r: float = 0.05):
        self.S0 = S0
        self.r = r
        
    def simulate_hawkes(self, T: float, n_paths: int, n_steps: int,
                       mu: float = 0.05, alpha: float = 0.3, 
                       beta: float = 1.0, seed: int = 42) -> np.ndarray:
        """
        模拟 Hawkes 跳跃过程
        
        dN_t = λ(t) dt
        λ(t) = μ + α ∫_0^t β e^{-β(t-s)} dN_s
        
        简化版本：使用复合 Poisson 近似
        """
        np.random.seed(seed)
        
        dt = T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        
        for path in range(n_paths):
            # 生成跳跃时间 (非齐次 Poisson)
            jump_times = []
            t = 0
            while t < T:
                # 强度 λ(t) = mu + alpha * past_jumps contribution
                intensity = mu
                # 简化：使用常数强度
                u = np.random.rand()
                inter_arrival = -np.log(u) / intensity
                t += inter_arrival
                if t < T:
                    jump_times.append(t)
            
            # 跳跃幅度
            jump_sizes = np.random.randn(len(jump_times)) * 0.1
            
            # 路径演化
            for i, jump_t in enumerate(jump_times):
                step_idx = int(jump_t / dt)
                if step_idx < n_steps:
                    S[path, step_idx + 1:] *= (1 + jump_sizes[i])
        
        return S
    
    def price_with_hawkes(self, K: float, T: float,
                         params_hawkes: Dict,
                         n_paths: int = 20000, n_steps: int = 100,
                         seed: int = 42) -> Dict:
        """
        使用 Hawkes 跳跃定价
        """
        S_paths = self.simulate_hawkes(
            T, n_paths, n_steps,
            mu=params_hawkes.get('mu', 0.05),
            alpha=params_hawkes.get('alpha', 0.3),
            beta=params_hawkes.get('beta', 1.0),
            seed=seed
        )
        
        pricer = AmericanLSM(self.S0, K, self.r, T,
                           option_type='put',
                           poly_degree=config.POLY_DEGREE,
                           ridge_alpha=config.RIDGE_ALPHA)
        
        result = pricer.price_american(S_paths, return_boundary=False)
        
        # 跳跃统计
        n_jumps_per_path = np.sum(np.diff(S_paths, axis=1) != 0, axis=1)
        
        return {
            'american_price': result['american_price'],
            'european_price': result['european_price'],
            'american_premium': result['american_premium'],
            'mean_jumps': np.mean(n_jumps_per_path),
            'std_jumps': np.std(n_jumps_per_path),
        }
    
    def run_comparison(self,
                      models: List[str] = None,
                      hawkes_configs: List[Dict] = None,
                      T_values: List[float] = None,
                      moneyness: float = 1.0,
                      n_paths: int = 20000,
                      output_dir: str = 'results第三轮') -> pd.DataFrame:
        """
        运行 DST vs Hawkes 对比
        """
        if models is None:
            models = ['M2', 'M5']
        
        if hawkes_configs is None:
            hawkes_configs = [
                {'name': 'Hawkes-Low', 'mu': 0.02, 'alpha': 0.2, 'beta': 1.0},
                {'name': 'Hawkes-Med', 'mu': 0.05, 'alpha': 0.3, 'beta': 1.0},
                {'name': 'Hawkes-High', 'mu': 0.1, 'alpha': 0.4, 'beta': 1.0},
            ]
        
        if T_values is None:
            T_values = [0.25, 0.5, 1.0]
        
        print("="*60)
        print("DST vs 复合 Hawkes 对比实验")
        print("="*60)
        
        K = self.S0 * moneyness
        results = []
        
        # DST 模型
        for T in T_values:
            for model in models:
                H = 0.1 if model in ['M2', 'M3', 'M4', 'M5'] else 0.5
                
                comp = BenchmarkComparison(self.S0, K, T, self.r)
                result = comp.price_dst(model=model, H=H, n_paths=n_paths)
                
                results.append({
                    'type': 'DST',
                    'model': model,
                    'T': T,
                    'moneyness': moneyness,
                    'price': result['american_price'],
                    'premium': result['american_premium'],
                    'early_ratio': result.get('strict_early_ratio', np.nan),
                })
                
                print(f"DST {model}, T={T}: price={result['american_price']:.4f}")
        
        # Hawkes 模型
        for T in T_values:
            for hw in hawkes_configs:
                result = self.price_with_hawkes(K, T, hw, n_paths=n_paths)
                
                results.append({
                    'type': 'Hawkes',
                    'model': hw['name'],
                    'T': T,
                    'moneyness': moneyness,
                    'price': result['american_price'],
                    'premium': result['american_premium'],
                    'mean_jumps': result['mean_jumps'],
                })
                
                print(f"Hawkes {hw['name']}, T={T}: price={result['american_price']:.4f}, jumps={result['mean_jumps']:.2f}")
        
        df = pd.DataFrame(results)
        
        # 保存
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df.to_csv(f'{output_dir}/dst_vs_hawkes_{timestamp}.csv', index=False)
        df.to_csv(f'{output_dir}/dst_vs_hawkes_latest.csv', index=False)
        
        print(f"\n对比结果已保存: {output_dir}/dst_vs_hawkes_*.csv")
        
        return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多方法对比实验')
    parser.add_argument('--mode', type=str, default='benchmark',
                       choices=['benchmark', 'hawkes', 'both'],
                       help='实验模式')
    parser.add_argument('--S0', type=float, default=100, help='标的价格')
    parser.add_argument('--r', type=float, default=0.05, help='无风险利率')
    parser.add_argument('--paths', type=int, default=20000, help='路径数')
    parser.add_argument('--seeds', type=int, default=5, help='seed 数量')
    parser.add_argument('--output', type=str, default='results第三轮',
                       help='输出目录')
    
    args = parser.parse_args()
    
    comp = BenchmarkComparison(S0=args.S0, r=args.r)
    
    if args.mode in ['benchmark', 'both']:
        df = comp.run_comparison(
            models=['M0', 'M2', 'M5'],
            moneyness_grid=[0.8, 0.9, 1.0, 1.1, 1.2],
            T_values=[0.25, 0.5, 1.0],
            n_paths=args.paths,
            n_seeds=args.seeds,
            output_dir=args.output
        )
        
        # 导出 LaTeX
        comp.export_latex_table(
            df, 
            f'{args.output}/benchmark_comparison_table.tex'
        )
    
    if args.mode in ['hawkes', 'both']:
        hawkes_comp = DstHawkesComparison(S0=args.S0, r=args.r)
        df_hawkes = hawkes_comp.run_comparison(
            models=['M2', 'M5'],
            T_values=[0.25, 0.5, 1.0],
            n_paths=args.paths,
            output_dir=args.output
        )