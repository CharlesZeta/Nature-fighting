"""
============================================================
隐含波动率计算模块
- Black-Scholes 反推隐含波动率
- Smile 插值与偏斜度计算
============================================================
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional
import config


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes Call 价格
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes Put 价格
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes Vega
    """
    if T <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(market_price: float, S: float, K: float, 
                      T: float, r: float, 
                      option_type: str = 'put',
                      sigma_init: float = 0.2,
                      sigma_min: float = 0.001,
                      sigma_max: float = 5.0) -> float:
    """
    计算隐含波动率 (使用 Brent 方法)
    
    参数:
    -----
    market_price : float
        市场/模型价格
    S : float
        标的资产价格
    K : float
        行权价
    T : float
        到期时间
    r : float
        无风险利率
    option_type : str
        'call' 或 'put'
    sigma_init : float
        初始猜测
    sigma_min, sigma_max : float
        波动率搜索区间
        
    返回:
    -----
    iv : float
        隐含波动率
    """
    # 如果价格低于内在价值，返回 NaN
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
    
    if market_price < intrinsic * 0.999:
        return np.nan
    
    # 定义目标函数
    if option_type == 'call':
        def objective(sigma):
            return bs_call_price(S, K, T, r, sigma) - market_price
    else:
        def objective(sigma):
            return bs_put_price(S, K, T, r, sigma) - market_price
    
    # 检查边界
    try:
        f_min = objective(sigma_min)
        f_max = objective(sigma_max)
        
        # 如果边界值同号，尝试在更宽的区间搜索
        if f_min * f_max > 0:
            # 尝试使用初始猜测
            f_init = objective(sigma_init)
            if f_init * f_min < 0:
                sigma_max = sigma_init
            elif f_init * f_max < 0:
                sigma_min = sigma_init
            else:
                return np.nan
        
        # Brent 方法求解
        iv = brentq(objective, sigma_min, sigma_max, xtol=1e-6)
        return iv
        
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_proxy(market_price: float, S: float, K: float,
                     T: float, r: float,
                     option_type: str = 'put') -> float:
    """
    快速隐含波动率近似 (使用 At-the-money 近似)
    
    对于 ATM 期权，可以使用近似公式:
    σ ≈ √(2π/T) * (C / S)  (对于 call)
    
    这是一个粗糙的近似，适合作为初始猜测或快速估计
    """
    if T <= 0:
        return np.nan
    
    # 简单近似: IV ≈ σ_0 + adjustment
    # 这里我们使用 vega 加权的迭代
    sigma = 0.2  # 初始猜测
    
    for _ in range(10):
        if option_type == 'call':
            model_price = bs_call_price(S, K, T, r, sigma)
        else:
            model_price = bs_put_price(S, K, T, r, sigma)
        
        error = model_price - market_price
        
        if abs(error) < 1e-6:
            break
        
        # Vega 校正
        vega = bs_vega(S, K, T, r, sigma)
        if abs(vega) > 1e-10:
            sigma = sigma - error / vega * 0.5  # 步长减半以保证稳定
            sigma = max(0.001, min(5.0, sigma))
        else:
            break
    
    return sigma


def compute_iv_from_prices(prices: np.ndarray, S: float, strikes: np.ndarray,
                          T: float, r: float,
                          option_type: str = 'put') -> np.ndarray:
    """
    从价格数组计算隐含波动率数组
    
    参数:
    -----
    prices : ndarray
        期权价格数组
    S : float
        标的价格
    strikes : ndarray
        行权价数组
    T : float
        到期时间
    r : float
        无风险利率
    option_type : str
        期权类型
        
    返回:
    -----
    ivs : ndarray
        隐含波动率数组
    """
    ivs = np.zeros_like(prices)
    
    for i, (price, K) in enumerate(zip(prices, strikes)):
        ivs[i] = implied_volatility(price, S, K, T, r, option_type)
    
    return ivs


def compute_atm_iv(moneyness_ivs: np.ndarray, moneyness: np.ndarray) -> float:
    """
    计算 ATM 隐含波动率 (通过插值)
    
    参数:
    -----
    moneyness_ivs : ndarray
        不同 moneyness 对应的 IV
    moneyness : ndarray
        对应的 moneyness 值
        
    返回:
    -----
    atm_iv : float
        ATM (moneyness = 1.0) 的 IV
    """
    # 过滤掉 NaN
    valid = ~np.isnan(moneyness_ivs)
    if np.sum(valid) < 2:
        return np.nan
    
    moneyness_valid = moneyness[valid]
    ivs_valid = moneyness_ivs[valid]
    
    # 如果有 ATM 点，直接返回
    if 1.0 in moneyness_valid:
        return ivs_valid[moneyness_valid == 1.0][0]
    
    # 否则插值
    try:
        cs = CubicSpline(moneyness_valid, ivs_valid)
        return float(cs(1.0))
    except:
        # 简单线性插值
        return np.interp(1.0, moneyness_valid, ivs_valid)


def compute_smile_skew(moneyness_ivs: np.ndarray, moneyness: np.ndarray) -> Tuple[float, float]:
    """
    计算 Smile 偏斜度
    
    参数:
    -----
    moneyness_ivs : ndarray
        不同 moneyness 对应的 IV
    moneyness : ndarray
        对应的 moneyness 值
        
    返回:
    -----
    (skew_left, skew_right) : tuple
        左偏斜 (OTM side) 和 右偏斜 (ITM side)
        skew = IV(moneyness) - IV(ATM)
    """
    atm_iv = compute_atm_iv(moneyness_ivs, moneyness)
    
    if np.isnan(atm_iv):
        return np.nan, np.nan
    
    # 过滤 NaN
    valid = ~np.isnan(moneyness_ivs)
    moneyness_valid = moneyness[valid]
    ivs_valid = moneyness_ivs[valid]
    
    if len(moneyness_valid) < 3:
        return np.nan, np.nan
    
    # OTM 侧 (moneyness < 1.0): 通常用于 put
    otm_mask = moneyness_valid < 1.0
    if np.sum(otm_mask) > 0:
        skew_left = np.min(ivs_valid[otm_mask]) - atm_iv
    else:
        skew_left = np.nan
    
    # ITM 侧 (moneyness > 1.0): 通常用于 call
    itm_mask = moneyness_valid > 1.0
    if np.sum(itm_mask) > 0:
        skew_right = np.max(ivs_valid[itm_mask]) - atm_iv
    else:
        skew_right = np.nan
    
    return skew_left, skew_right


def compute_smile_curvature(moneyness_ivs: np.ndarray, moneyness: np.ndarray) -> float:
    """
    计算 Smile 曲率 (curvature / convexity)
    
    curvature = IV(0.9) + IV(1.1) - 2 * IV(1.0)
    """
    atm_iv = compute_atm_iv(moneyness_ivs, moneyness)
    
    if np.isnan(atm_iv):
        return np.nan
    
    # 过滤 NaN
    valid = ~np.isnan(moneyness_ivs)
    moneyness_valid = moneyness[valid]
    ivs_valid = moneyness_ivs[valid]
    
    # 插值得到 0.9 和 1.1 处的 IV
    try:
        iv_09 = np.interp(0.9, moneyness_valid, ivs_valid)
        iv_11 = np.interp(1.1, moneyness_valid, ivs_valid)
        curvature = iv_09 + iv_11 - 2 * atm_iv
        return curvature
    except:
        return np.nan


def compute_term_structure_iv(maturities: np.ndarray, atm_ivs: np.ndarray) -> np.ndarray:
    """
    计算 IV 期限结构 (用于绘图)
    """
    # 过滤 NaN
    valid = ~np.isnan(atm_ivs)
    return np.interp(maturities, maturities[valid], atm_ivs[valid])
