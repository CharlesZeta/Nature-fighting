"""
============================================================
LSM 美式期权定价模块
- Longstaff-Schwartz 算法
- Ridge 正则化回归
- Early exercise boundary 输出
============================================================
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import config


class AmericanLSM:
    """
    Longstaff-Schwartz 美式期权定价器
    """
    
    def __init__(self, S0: float, K: float, r: float, T: float,
                 option_type: str = 'put',
                 poly_degree: int = 3,
                 ridge_alpha: float = 0.01):
        """
        参数:
        -----
        S0 : float
            初始标的价格
        K : float
            行权价
        r : float
            无风险利率
        T : float
            到期时间
        option_type : str
            'call' 或 'put'
        poly_degree : int
            多项式阶数
        ridge_alpha : float
            Ridge 正则化系数
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        self.poly_degree = poly_degree
        self.ridge_alpha = ridge_alpha
    
    def compute_intrinsic(self, S: np.ndarray) -> np.ndarray:
        """
        计算期权内在价值
        """
        if self.option_type == 'call':
            return np.maximum(S - self.K, 0)
        else:  # put
            return np.maximum(self.K - S, 0)
    
    def price_european(self, S_paths: np.ndarray) -> float:
        """
        计算 European 期权价格 (蒙特卡洛)
        
        S_paths: shape (n_paths, n_steps+1)
        """
        n_paths = S_paths.shape[0]
        
        # 到期日payoff
        S_T = S_paths[:, -1]
        payoff = self.compute_intrinsic(S_T)
        
        # 折现
        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoff)
        
        return price
    
    def price_american(self, S_paths: np.ndarray, 
                       return_boundary: bool = False) -> Dict:
        """
        LSM 美式期权定价
        
        参数:
        -----
        S_paths : ndarray, shape (n_paths, n_steps+1)
            标的资产价格路径
        return_boundary : bool
            是否返回提前行权边界
            
        返回:
        -----
        results : Dict
            - american_price: 美式期权价格
            - european_price: European 价格 (同路径)
            - american_premium: American - European
            - early_exercise_ratio: 总行权比例 (包含到期日)
            - early_exercise_ratio_before_maturity: 严格到期前行权比例
            - exercise_at_maturity_ratio: 到期日行权比例
            - itm_ratio: 到期日 ITM 比例 (S_T < K for put)
            - exercise_boundary: 行权边界中位数 (如果 return_boundary=True)
            - boundary_median, boundary_q25, boundary_q75: 边界分位数
            - exercise_count_per_time: 每个时间步行权样本数
            - boundary_nan_ratio: 边界 NaN 比例
            - itm_sample_counts: 每个时间步的 ITM 样本数
        """
        n_paths, n_steps_plus_1 = S_paths.shape
        n_steps = n_steps_plus_1 - 1
        dt = self.T / n_steps
        
        # 1. 计算每个时间步的 payoff
        intrinsic_values = np.zeros((n_paths, n_steps_plus_1))
        for t in range(n_steps_plus_1):
            intrinsic_values[:, t] = self.compute_intrinsic(S_paths[:, t])
        
        # 2. 初始化: 到期日
        cash_flow = intrinsic_values[:, -1].copy()
        exercised = np.zeros(n_paths, dtype=bool)
        exercised_before_maturity = np.zeros(n_paths, dtype=bool)  # 严格在 t < T 时行权
        boundary_median_list = [] if return_boundary else None
        boundary_q25_list = [] if return_boundary else None
        boundary_q75_list = [] if return_boundary else None
        exercise_count_per_time = [] if return_boundary else None
        itm_sample_counts = []
        
        # 3. 逐时间步回溯
        for t in range(n_steps - 1, -1, -1):
            # 当前时间点的标的价格 (仅 ITM 路径)
            S_t = S_paths[~exercised, t]
            itm_count = len(S_t)
            itm_sample_counts.append(itm_count)
            
            # 如果没有 ITM 路径，跳过
            if itm_count < config.MIN_ITM_SAMPLES:
                if return_boundary:
                    boundary_median_list.append(np.nan)
                    boundary_q25_list.append(np.nan)
                    boundary_q75_list.append(np.nan)
                    exercise_count_per_time.append(0)
                continue
            
            # 当前未行权路径的 continuation value
            future_cash_flow = cash_flow[~exercised]
            discounted_cf = future_cash_flow * np.exp(-self.r * dt)
            
            # 构建回归特征 (多项式)
            poly = PolynomialFeatures(degree=self.poly_degree)
            X = poly.fit_transform(S_t.reshape(-1, 1))
            
            # Ridge 回归估计 continuation value
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(X, discounted_cf)
            
            # 预测 continuation value
            continuation = model.predict(X)
            
            # 计算 R² 用于回归诊断（仅在 return_boundary=True 时）
            if return_boundary:
                ss_res = np.sum((discounted_cf - continuation) ** 2)
                ss_tot = np.sum((discounted_cf - np.mean(discounted_cf)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else np.nan
                regression_r2_list.append(r2)
            
            # 当前时刻的内在价值
            intrinsic_t = intrinsic_values[~exercised, t]
            
            # 判断是否行权
            exercise_now = intrinsic_t > continuation
            
            # 更新 cash flow 和 exercised 状态
            not_exercised_mask = ~exercised
            exercised_now_mask = np.zeros(n_paths, dtype=bool)
            exercised_now_mask[not_exercised_mask] = exercise_now
            
            # 更新 cash flow: 行权的获得内在价值，未行权的保持
            cash_flow[exercised_now_mask] = intrinsic_t[exercise_now]
            exercised = exercised | exercised_now_mask
            
            # 记录严格在到期日之前（t < n_steps-1）行权的路径
            if t < n_steps - 1:
                exercised_before_maturity = exercised_before_maturity | exercised_now_mask
            
            # 行权边界：用该时点「被行权」路径的 S_t 分位数定义，样本不足则 NaN
            if return_boundary:
                n_exercised = int(np.sum(exercise_now))
                exercise_count_per_time.append(n_exercised)
                if n_exercised >= config.MIN_BOUNDARY_SAMPLES:
                    S_exercised = S_t[exercise_now]
                    boundary_median_list.append(float(np.median(S_exercised)))
                    boundary_q25_list.append(float(np.percentile(S_exercised, 25)))
                    boundary_q75_list.append(float(np.percentile(S_exercised, 75)))
                else:
                    boundary_median_list.append(np.nan)
                    boundary_q25_list.append(np.nan)
                    boundary_q75_list.append(np.nan)
        
        # 4. 折现到初始时刻
        discount = np.exp(-self.r * self.T)
        american_price = np.mean(cash_flow) * discount
        
        # 5. 计算 European 价格 (同一路径)
        european_price = self.price_european(S_paths)
        
        # 6. 提前行权比例（严格区分：ITM / 到期前 / 到期日行权 / 总行权）
        # 所有比例的分母 = all simulated paths (统一明确)
        # ITM：到期日 S_T < K（对于 put）
        itm_at_maturity = intrinsic_values[:, -1] > 0
        itm_ratio = np.mean(itm_at_maturity)
        # 到期前行权：严格在 t < T 时行权
        early_exercise_ratio_before_maturity = np.mean(exercised_before_maturity)
        # 到期日行权：到期日才被执行（到期前未行权 + 到期日 ITM）
        exercised_at_maturity = exercised & ~exercised_before_maturity
        exercise_at_maturity_ratio = np.mean(exercised_at_maturity)
        # 总行权比例（包含到期日）
        total_exercise_ratio = np.mean(exercised)
        
        # 分解一致性校验：decomposition_gap = total - (strict_early + maturity)
        # 这个 gap 应该接近 0（允许浮点误差）
        decomposition_gap = total_exercise_ratio - (early_exercise_ratio_before_maturity + exercise_at_maturity_ratio)
        
        # 7. 回归诊断：计算每时间步的 R²（用于诊断 continuation 拟合质量）
        regression_r2_list = [] if return_boundary else None
        
        # 7. 反转边界相关列表 (从 t=0 到 T)
        if return_boundary:
            boundary_median_list = boundary_median_list[::-1]
            boundary_q25_list = boundary_q25_list[::-1]
            boundary_q75_list = boundary_q75_list[::-1]
            exercise_count_per_time = exercise_count_per_time[::-1]
            n_boundary = len(boundary_median_list)
            boundary_nan_ratio = float(np.sum(np.isnan(boundary_median_list))) / max(n_boundary, 1)
        else:
            boundary_nan_ratio = np.nan
        
        return {
            'american_price': american_price,
            'european_price': european_price,
            'american_premium': american_price - european_price,
            'total_exercise_ratio': total_exercise_ratio,
            'strict_early_exercise_ratio': early_exercise_ratio_before_maturity,
            'exercise_at_maturity_ratio': exercise_at_maturity_ratio,
            'itm_ratio': itm_ratio,
            'decomposition_gap': decomposition_gap,  # 分解一致性校验，应接近 0
            'exercise_boundary': boundary_median_list if return_boundary else None,
            'boundary_median': boundary_median_list if return_boundary else None,
            'boundary_q25': boundary_q25_list if return_boundary else None,
            'boundary_q75': boundary_q75_list if return_boundary else None,
            'exercise_count_per_time': exercise_count_per_time if return_boundary else None,
            'boundary_nan_ratio': boundary_nan_ratio if return_boundary else np.nan,
            'itm_sample_counts': itm_sample_counts[::-1],
            'regression_r2_per_time': regression_r2_list[::-1] if return_boundary else None,  # 回归诊断 R²
        }
    
    def price_bermudan(self, S_paths: np.ndarray, n_exercise_dates: int = 20) -> Dict:
        """
        Bermudan 期权定价 (作为高精度 reference)
        
        参数:
        -----
        n_exercise_dates : int
            行权时点数量 (均匀分布)
        """
        n_paths, n_steps_plus_1 = S_paths.shape
        n_steps = n_steps_plus_1 - 1
        dt = self.T / n_steps
        
        # 行权时点索引
        exercise_indices = np.linspace(0, n_steps - 1, n_exercise_dates, dtype=int)
        
        # 每个时间步的 payoff
        intrinsic_values = np.zeros((n_paths, n_steps_plus_1))
        for t in range(n_steps_plus_1):
            intrinsic_values[:, t] = self.compute_intrinsic(S_paths[:, t])
        
        # 初始化
        cash_flow = intrinsic_values[:, -1].copy()
        exercised = np.zeros(n_paths, dtype=bool)
        
        # 逐行权时点回溯
        for exercise_idx in exercise_indices[:-1][::-1]:
            # 当前时间点
            S_t = S_paths[~exercised, exercise_idx]
            
            if len(S_t) < config.MIN_ITM_SAMPLES:
                continue
            
            future_cash_flow = cash_flow[~exercised]
            discounted_cf = future_cash_flow * np.exp(-self.r * dt)
            
            # 回归
            poly = PolynomialFeatures(degree=self.poly_degree)
            X = poly.fit_transform(S_t.reshape(-1, 1))
            
            model = Ridge(alpha=self.ridge_alpha)
            model.fit(X, discounted_cf)
            
            continuation = model.predict(X)
            intrinsic_t = intrinsic_values[~exercised, exercise_idx]
            
            exercise_now = intrinsic_t > continuation
            
            # 更新
            not_exercised_mask = ~exercised
            exercised_now_mask = np.zeros(n_paths, dtype=bool)
            exercised_now_mask[not_exercised_mask] = exercise_now
            
            cash_flow[exercised_now_mask] = intrinsic_t[exercise_now]
            exercised = exercised | exercised_now_mask
        
        # 折现
        discount = np.exp(-self.r * self.T)
        bermudan_price = np.mean(cash_flow) * discount
        
        return {
            'bermudan_price': bermudan_price,
            'early_exercise_ratio': np.mean(exercised),
        }


def price_american_option(S_paths: np.ndarray, K: float, r: float, T: float,
                          option_type: str = 'put',
                          poly_degree: int = 3,
                          ridge_alpha: float = 0.01,
                          return_boundary: bool = True) -> Dict:
    """
    便捷函数: 定价美式期权
    """
    S0 = S_paths[0, 0]
    pricer = AmericanLSM(S0, K, r, T, option_type, poly_degree, ridge_alpha)
    return pricer.price_american(S_paths, return_boundary)


def compute_early_exercise_boundary(S_paths: np.ndarray, K: float, r: float, T: float,
                                    option_type: str = 'put',
                                    poly_degree: int = 3,
                                    ridge_alpha: float = 0.01) -> np.ndarray:
    """
    计算美式期权的提前行权边界 S*(t)
    """
    result = price_american_option(S_paths, K, r, T, option_type, 
                                    poly_degree, ridge_alpha, return_boundary=True)
    return np.array(result['exercise_boundary'])
