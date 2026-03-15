"""
============================================================
隐含波动率曲面校准模块
============================================================
功能：
1. 使用 DST/Rough Heston 模型校准隐含波动率曲面
2. 生成 synthetic market data 进行验证
3. 输出校准参数、误差曲面、收敛曲线
============================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, minimize_scalar
from scipy.interpolate import griddata
from typing import Tuple, Dict, List, Optional
import time
import os
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
import config
from simulation import PathSimulator
from lsm_american import AmericanLSM
from implied_vol import implied_volatility


class IVSurfaceCalibrator:
    """隐含波动率曲面校准器"""
    
    def __init__(self, S0: float, r: float, 
                 market_iv: np.ndarray, 
                 T_grid: np.ndarray, 
                 K_grid: np.ndarray,
                 option_type: str = 'put'):
        """
        参数:
        -----
        S0 : float
            初始标的价格
        r : float
            无风险利率
        market_iv : ndarray
            市场隐含波动率网格 (n_T, n_K)
        T_grid : ndarray
            到期时间网格 (n_T,)
        K_grid : ndarray
            行权价网格 (n_K,)
        option_type : str
            期权类型 'put' 或 'call'
        """
        self.S0 = S0
        self.r = r
        self.market_iv = market_iv
        self.T_grid = T_grid
        self.K_grid = K_grid
        self.option_type = option_type
        
        # 网格维度
        self.n_T = len(T_grid)
        self.n_K = len(K_grid)
        
    def objective(self, params: np.ndarray) -> float:
        """
        校准目标函数：相对误差加权求和
        
        f(θ) = Σ_ij |IV_model - IV_market| / IV_market
        """
        # 解包参数: [v0, kappa, theta, xi, rho, H, lambda_jump]
        v0, kappa, theta, xi, rho, H, lam_jump = params
        
        # 快速检查参数有效性
        if v0 <= 0 or theta <= 0 or xi <= 0 or kappa <= 0:
            return 1e10
        if abs(rho) >= 1:
            return 1e10
        if H <= 0 or H > 1:
            return 1e10
        if lam_jump < 0:
            return 1e10
            
        # 构建参数字典
        vol_type = 'rough' if H < 0.5 else 'markovian'
        params_dict = {
            'S0': self.S0,
            'v0': v0,
            'kappa': kappa,
            'theta': theta,
            'xi': xi,
            'rho': rho,
            'r': self.r,
            'q': 0.0,
            'vol_type': vol_type,  # 添加 vol_type
            'jump_type': 'poisson' if lam_jump > 0 else 'none',
            'lambda_jump': lam_jump,
            'jump_mu': -0.1,
            'jump_sigma': 0.1,
        }
        
        # 计算模型 IV 曲面
        try:
            model_iv = self._compute_model_iv_fast(params_dict)
        except Exception as e:
            return 1e10
        
        # 计算相对误差 (避免除零 + 避免 NaN 让优化失效)
        valid_mask = (self.market_iv > 1e-6) & (~np.isnan(model_iv))
        if not np.any(valid_mask):
            return 1e10
            
        rel_error = np.abs(model_iv[valid_mask] - self.market_iv[valid_mask]) / self.market_iv[valid_mask]
        
        return np.sum(rel_error)
    
    def _compute_model_iv_fast(self, params_dict: Dict, n_paths: int = 5000, n_steps: int = 50) -> np.ndarray:
        """
        快速计算模型隐含波动率曲面
        使用有限数量的路径和简化计算
        """
        model_iv = np.zeros((self.n_T, self.n_K))
        
        for i, T in enumerate(self.T_grid):
            for j, K in enumerate(self.K_grid):
                # 计算模型价格
                try:
                    np.random.seed(42)
                    simulator = PathSimulator(params_dict)
                    sim_results = simulator.simulate(T, n_paths, n_steps, seed=42, antithetic=True)
                    S_paths = sim_results['S']
                    
                    pricer = AmericanLSM(self.S0, K, self.r, T,
                                       option_type=self.option_type,
                                       poly_degree=2,
                                       ridge_alpha=1.0)
                    
                    result = pricer.price_american(S_paths, return_boundary=False)
                    american_price = result['american_price']
                    
                    # 反推 IV
                    iv = implied_volatility(american_price, self.S0, K, T, self.r, self.option_type)
                    model_iv[i, j] = iv
                    
                except:
                    model_iv[i, j] = np.nan
        
        return model_iv
    
    def calibrate(self, method: str = 'de', 
                  initial_params: Optional[np.ndarray] = None,
                  bounds: Optional[List[Tuple]] = None,
                  workers: Optional[int] = None,
                  maxiter: Optional[int] = None,
                  fast_mode: bool = False) -> Dict:
        """
        执行校准
        
        参数:
        -----
        method : str
            优化方法: 'de' (差分进化), 'lbfgsb', 'cg', 'dual_annealing'
        initial_params : ndarray, optional
            初始参数
        bounds : list, optional
            参数边界
            
        返回:
        -----
        result : dict
            最优参数、误差、迭代历史
        """
        if bounds is None:
            bounds = [
                (0.01, 0.2),    # v0
                (0.5, 5.0),     # kappa
                (0.01, 0.3),    # theta
                (0.1, 1.0),     # xi
                (-0.9, 0.0),    # rho
                (0.01, 0.5),    # H
                (0.0, 2.0),     # lambda_jump
            ]
        
        if initial_params is None:
            # 默认初始参数
            initial_params = np.array([0.04, 2.0, 0.04, 0.3, -0.5, 0.1, 0.0])
        
        if workers is None:
            # Windows 上过多进程会显著拖慢；默认留 1 个核心给系统
            workers = max(1, mp.cpu_count() - 1)

        if maxiter is None:
            maxiter = 60 if fast_mode else 200

        print(f"\n开始校准 (方法: {method})...")
        print(f"初始参数: {initial_params}")
        print(f"workers={workers}, maxiter={maxiter}, fast_mode={fast_mode}")
        
        start_time = time.time()
        
        if method == 'de':
            # 差分进化 - 全局优化
            result_opt = differential_evolution(
                self.objective, 
                bounds, 
                maxiter=maxiter, 
                tol=1e-4,
                seed=42, 
                workers=workers,
                polish=True,
                updating='deferred',
                disp=True
            )
            final_params = result_opt.x
            final_error = result_opt.fun
            nit = result_opt.nit
            
        elif method == 'lbfgsb':
            # L-BFGS-B - 局部优化
            from scipy.optimize import Bounds
            bounds_arr = np.array(bounds)
            lb = bounds_arr[:, 0]
            ub = bounds_arr[:, 1]
            
            result_opt = minimize(
                self.objective,
                initial_params,
                method='L-BFGS-B',
                bounds=Bounds(lb, ub),
                options={'maxiter': 500, 'disp': True}
            )
            final_params = result_opt.x
            final_error = result_opt.fun
            nit = result_opt.nit
            
        elif method == 'dual_annealing':
            # 双重退火 - 全局优化
            from scipy.optimize import dual_annealing
            
            result_opt = dual_annealing(
                self.objective,
                bounds=bounds,
                maxiter=300,
                seed=42,
                initial_temp=5230,
                restart_temp_ratio=2e-5,
                visit=2.62,
                accept=-5.0,
                no_local_search=False,
                disp=True
            )
            final_params = result_opt.x
            final_error = result_opt.fun
            nit = result_opt.nit
            
        else:
            raise ValueError(f"未知优化方法: {method}")
        
        elapsed = time.time() - start_time
        
        # 解包最优参数
        v0, kappa, theta, xi, rho, H, lam = final_params
        
        result = {
            'params': {
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'xi': xi,
                'rho': rho,
                'H': H,
                'lambda_jump': lam,
            },
            'objective_value': final_error,
            'n_iterations': nit,
            'elapsed_time': elapsed,
            'success': result_opt.success if hasattr(result_opt, 'success') else True,
            'method': method,
        }
        
        print(f"\n校准完成!")
        print(f"最优参数: {result['params']}")
        print(f"目标函数值: {final_error:.6f}")
        print(f"迭代次数: {nit}")
        print(f"耗时: {elapsed:.2f}秒")
        
        return result
    
    def compute_calibrated_surface(self, params: Dict, 
                                   n_paths: int = 10000,
                                   n_steps: int = 100) -> np.ndarray:
        """
        使用校准后的参数计算完整的 IV 曲面
        
        参数:
        -----
        params : dict
            校准后的参数
        n_paths : int
            蒙特卡洛路径数
        n_steps : int
            时间步数
            
        返回:
        -----
        calibrated_iv : ndarray
            校准后的 IV 曲面
        """
        vol_type = 'rough' if params['H'] < 0.5 else 'markovian'
        params_dict = {
            'S0': self.S0,
            'v0': params['v0'],
            'kappa': params['kappa'],
            'theta': params['theta'],
            'xi': params['xi'],
            'rho': params['rho'],
            'r': self.r,
            'q': 0.0,
            'vol_type': vol_type,  # 添加 vol_type
            'jump_type': 'poisson' if params['lambda_jump'] > 0 else 'none',
            'lambda_jump': params['lambda_jump'],
            'jump_mu': -0.1,
            'jump_sigma': 0.1,
        }
        
        calibrated_iv = np.zeros((self.n_T, self.n_K))
        
        print("\n计算校准后的 IV 曲面...")
        for i, T in enumerate(self.T_grid):
            for j, K in enumerate(self.K_grid):
                np.random.seed(123)
                simulator = PathSimulator(params_dict)
                sim_results = simulator.simulate(T, n_paths, n_steps, seed=123, antithetic=True)
                S_paths = sim_results['S']
                
                pricer = AmericanLSM(self.S0, K, self.r, T,
                                   option_type=self.option_type,
                                   poly_degree=config.POLY_DEGREE,
                                   ridge_alpha=config.RIDGE_ALPHA)
                
                result = pricer.price_american(S_paths, return_boundary=False)
                
                try:
                    iv = implied_volatility(result['american_price'], self.S0, K, T, self.r, self.option_type)
                    calibrated_iv[i, j] = iv
                except:
                    calibrated_iv[i, j] = np.nan
                    
                if (j + 1) % 5 == 0:
                    print(f"  T={T:.2f}: {j+1}/{self.n_K} K values done")
        
        return calibrated_iv
    
    def compute_parameter_sensitivity(self, params: Dict,
                                      param_name: str = 'xi',
                                      delta: float = 0.1) -> np.ndarray:
        """
        计算参数敏感度 ∂IV/∂param
        
        使用数值差分
        """
        base_params = params.copy()
        
        # 计算基准曲面
        base_iv = self.compute_calibrated_surface(base_params, n_paths=5000, n_steps=50)
        
        # 微调参数
        perturbed_params = base_params.copy()
        perturbed_params[param_name] *= (1 + delta)
        
        # 计算扰动后曲面
        perturbed_iv = self.compute_calibrated_surface(perturbed_params, n_paths=5000, n_steps=50)
        
        # 敏感度
        sensitivity = (perturbed_iv - base_iv) / (delta * base_params[param_name])
        
        return sensitivity
    
    def export_results(self, result: Dict, calibrated_iv: np.ndarray,
                       output_dir: str = 'results第三轮'):
        """
        导出校准结果
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存参数
        params_df = pd.DataFrame([{
            'timestamp': timestamp,
            'method': result['method'],
            'v0': result['params']['v0'],
            'kappa': result['params']['kappa'],
            'theta': result['params']['theta'],
            'xi': result['params']['xi'],
            'rho': result['params']['rho'],
            'H': result['params']['H'],
            'lambda_jump': result['params']['lambda_jump'],
            'objective_value': result['objective_value'],
            'n_iterations': result['n_iterations'],
            'elapsed_time': result['elapsed_time'],
        }])
        
        params_path = f'{output_dir}/iv_calibration_params_{timestamp}.csv'
        params_df.to_csv(params_path, index=False)
        
        # 保存 IV 曲面
        iv_df = pd.DataFrame(calibrated_iv, 
                            index=[f'T={t:.2f}' for t in self.T_grid],
                            columns=[f'K={k:.2f}' for k in self.K_grid])
        iv_path = f'{output_dir}/iv_calibration_surface_{timestamp}.csv'
        iv_df.to_csv(iv_path)
        
        # 同时保存 latest 版本
        params_df.to_csv(f'{output_dir}/iv_calibration_params_latest.csv', index=False)
        
        print(f"\n校准结果已保存:")
        print(f"  参数: {params_path}")
        print(f"  曲面: {iv_path}")
        
        return params_path, iv_path


def generate_synthetic_market_data(S0: float = 100,
                                   T_range: List[float] = [0.1, 0.25, 0.5, 1.0],
                                   K_range: List[float] = [0.8, 0.9, 1.0, 1.1, 1.2],
                                   r: float = 0.05,
                                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 Synthetic Market Data
    使用高精度 Heston 模型作为 "真实市场"
    """
    np.random.seed(seed)
    
    # 高精度参数 (Heston 原始论文)
    true_params = {
        'v0': 0.0175,
        'kappa': 1.5768,
        'theta': 0.0398,
        'xi': 0.5751,
        'rho': -0.5711,
    }
    
    n_T = len(T_range)
    n_K = len(K_range)
    market_iv = np.zeros((n_T, n_K))
    
    print("生成 Synthetic Market Data (高精度 Heston)...")
    print(f"真实参数: {true_params}")
    
    for i, T in enumerate(T_range):
        for j, K in enumerate(K_range):
            # 高精度蒙特卡洛
            n_paths = 50000
            n_steps = 200
            
            # 简化的 Heston 模拟 (使用 Euler)
            # dS = rS dt + sqrt(v) S dW1
            # dv = kappa(theta - v) dt + xi sqrt(v) dW2
            # dW1, dW2 = rho dt
            
            dt = T / n_steps
            S = np.zeros((n_paths, n_steps + 1))
            v = np.zeros((n_paths, n_steps + 1))
            S[:, 0] = S0
            v[:, 0] = true_params['v0']
            
            for t in range(n_steps):
                Z1 = np.random.randn(n_paths)
                Z2 = np.random.randn(n_paths)
                W2 = Z2
                rho = true_params['rho']
                W1 = rho * Z2 + np.sqrt(1 - rho**2) * Z1
                
                v[:, t+1] = np.maximum(
                    v[:, t] + true_params['kappa'] * (true_params['theta'] - v[:, t]) * dt 
                    + true_params['xi'] * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * W2,
                    0.001
                )
                
                S[:, t+1] = S[:, t] * np.exp(
                    (r - 0.5 * v[:, t]) * dt + np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * W1
                )
            
            S_T = S[:, -1]
            
            # 计算欧式期权价格
            payoff = np.maximum(K - S_T, 0)
            price = np.mean(payoff) * np.exp(-r * T)
            
            # 反推 IV
            try:
                iv = implied_volatility(price, S0, K, T, r, 'put')
                market_iv[i, j] = iv
            except:
                # 如果失败，使用 BS 近似
                iv_approx = np.sqrt(2 * np.pi / T) * price / K
                market_iv[i, j] = max(iv_approx, 0.01)
            
            print(f"  T={T:.2f}, K={K:.0f}: IV={market_iv[i,j]:.4f}")
    
    return market_iv, np.array(T_range), np.array(K_range)


def run_calibration_experiment(
    S0: float = 100,
    r: float = 0.05,
    T_range: List[float] = [0.1, 0.25, 0.5, 1.0],
    K_range: List[float] = [0.8, 0.9, 1.0, 1.1, 1.2],
    use_synthetic: bool = True,
    output_dir: str = 'results第三轮',
    calibration_method: str = 'de',
    workers: Optional[int] = None,
    fast_mode: bool = False
):
    """
    运行完整校准实验
    """
    print("="*60)
    print("隐含波动率曲面校准实验")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if use_synthetic:
        # 生成 synthetic market data
        market_iv, T_grid, K_grid = generate_synthetic_market_data(
            S0=S0, T_range=T_range, K_range=K_range, r=r
        )
        
        # 保存 market data
        market_df = pd.DataFrame(
            market_iv,
            index=[f'T={t:.2f}' for t in T_grid],
            columns=[f'K={k:.2f}' for k in K_grid]
        )
        market_df.to_csv(f'{output_dir}/market_iv_synthetic.csv')
    else:
        # 使用预设的 market data
        raise NotImplementedError("真实市场数据加载尚未实现")
    
    # 初始化校准器
    calibrator = IVSurfaceCalibrator(
        S0=S0,
        r=r,
        market_iv=market_iv,
        T_grid=T_grid,
        K_grid=K_grid,
        option_type='put'
    )
    
    # 执行校准
    result = calibrator.calibrate(method=calibration_method, workers=workers, fast_mode=fast_mode)
    
    # 计算校准后的曲面
    calibrated_iv = calibrator.compute_calibrated_surface(
        result['params'],
        n_paths=20000,
        n_steps=100
    )
    
    # 导出结果
    calibrator.export_results(result, calibrated_iv, output_dir)
    
    # 计算误差统计
    valid_mask = market_iv > 1e-6
    rmse = np.sqrt(np.mean((calibrated_iv[valid_mask] - market_iv[valid_mask])**2) )
    mae = np.mean(np.abs(calibrated_iv[valid_mask] - market_iv[valid_mask]))
    mape = np.mean(np.abs((calibrated_iv[valid_mask] - market_iv[valid_mask]) / market_iv[valid_mask])) * 100
    
    print(f"\n校准误差统计:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return result, calibrated_iv, market_iv


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='隐含波动率曲面校准')
    parser.add_argument('--S0', type=float, default=100, help='标的价格')
    parser.add_argument('--r', type=float, default=0.05, help='无风险利率')
    parser.add_argument('--method', type=str, default='de', 
                       choices=['de', 'lbfgsb', 'dual_annealing'],
                       help='优化方法')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数（默认: CPU核数-1）')
    parser.add_argument('--fast', action='store_true',
                       help='快速模式：降低 maxiter，用于先跑通流程')
    parser.add_argument('--output', type=str, default='results第三轮',
                       help='输出目录')
    parser.add_argument('--T', type=str, default='0.1,0.25,0.5,1.0',
                       help='到期时间网格 (逗号分隔)')
    parser.add_argument('--K', type=str, default='80,90,100,110,120',
                       help='行权价网格 (逗号分隔)')
    
    args = parser.parse_args()
    
    T_range = [float(t) for t in args.T.split(',')]
    K_range = [float(k) / 100 * args.S0 for k in args.K.split(',')]  # 转换为绝对价格
    
    run_calibration_experiment(
        S0=args.S0,
        r=args.r,
        T_range=T_range,
        K_range=K_range,
        use_synthetic=True,
        output_dir=args.output,
        calibration_method=args.method,
        workers=args.workers,
        fast_mode=args.fast
    )