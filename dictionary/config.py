# config.py
"""
统一参数配置类 - 使用 dataclass 管理所有参数
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class MarketParams:
    """市场参数"""
    S0: float = 100.0       # 初始股票价格
    r: float = 0.02         # 无风险利率
    q: float = 0.0          # 股息收益率


@dataclass
class HestonParams:
    """Classic Heston 参数"""
    kappa: float = 2.0      # 方差均值回归速度
    theta: float = 0.04     # 方差长期均值
    xi: float = 0.3         # 方差波动率
    rho: float = -0.7       # 股票与方差的相关系数
    v0: float = 0.04        # 初始方差


@dataclass
class RoughHestonParams:
    """Rough Heston / Volterra Heston 参数"""
    H: float = 0.1          # Hurst 指数 (0 < H < 1)
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v0: float = 0.04


@dataclass
class ExpApproxParams:
    """指数和逼近参数 (用于 Markovian embedding)"""
    N_exp: int = 5           # 指数项数量
    lambda_min: float = 0.5  # 最小衰减率
    lambda_max: float = 20.0 # 最大衰减率
    method: str = "laguerre" # "laguerre" or "custom"

    def generate_exp_nodes_weights(self) -> tuple:
        """
        生成指数和逼近的节点和权重
        使用 Gauss-Laguerre 规则近似
        """
        if self.method == "laguerre":
            # 使用 Gauss-Laguerre 节点和权重
            from scipy.special import roots_laguerre
            
            # 生成 N_exp 个节点
            x, w = roots_laguerre(self.N_exp)
            
            # 转换到指数参数
            # λ_i = x_i / T (归一化到时间范围)
            # c_i = w_i * exp(x_i) / (T * L(T))
            lambda_i = x / 1.0  # 归一化时间尺度
            c_i = w * np.exp(x) / np.sum(w * np.exp(x))
            
            return lambda_i, c_i
        else:
            # 线性分布的 λ
            lambda_i = np.linspace(self.lambda_min, self.lambda_max, self.N_exp)
            c_i = np.ones(self.N_exp) / self.N_exp
            return lambda_i, c_i


@dataclass
class MertonJumpParams:
    """Merton 跳跃参数"""
    lambda_jump: float = 0.5  # 跳跃强度 (Poisson 强度)
    muJ: float = -0.1         # 跳跃幅度均值 (log 尺度)
    sigmaJ: float = 0.3       # 跳跃幅度标准差


@dataclass
class VarianceJumpParams:
    """方差跳跃参数 (独立 Poisson 过程)"""
    eta: float = 0.1          # 方差跳跃强度
    muV: float = -0.5         # 方差跳跃幅度均值
    sigmaV: float = 0.3       # 方差跳跃幅度标准差


@dataclass
class HawkesParams:
    """双变量 Hawkes 过程参数"""
    # 基础强度
    lambdaS0: float = 0.5    # 价格跳跃的基础强度
    lambdaV0: float = 0.3    # 方差跳跃的基础强度
    
    # 衰减率
    betaS: float = 3.0        # 价格跳跃衰减率
    betaV: float = 2.0        # 方差跳跃衰减率
    
    # 兴奋系数 (非爆炸性: alpha < beta)
    alphaSS: float = 2.0      # 价格 -> 价格 兴奋
    alphaSV: float = 0.5      # 方差 -> 价格 兴奋
    alphaVS: float = 0.3      # 价格 -> 方差 兴奋
    alphaVV: float = 1.5      # 方差 -> 方差 兴奋
    
    # 上限 (用于稳定性检查)
    lambdaS_inf: float = 5.0
    lambdaV_inf: float = 3.0


@dataclass
class SimulationParams:
    """仿真参数"""
    T: float = 1.0            # 到期时间 (年)
    n_steps: int = 100        # 时间步数
    n_paths: int = 50000      # 模拟路径数
    seed: Optional[int] = 42  # 随机种子
    dt: float = field(init=False)
    
    def __post_init__(self):
        self.dt = self.T / self.n_steps


@dataclass
class LSMParams:
    """Longstaff-Schwartz Monte Carlo 参数"""
    strike: float = 100.0     # 行权价
    option_type: str = "put"  # "put" or "call"
    basis_type: str = "polynomial"  # "polynomial" or "physics"
    basis_degree: int = 2     # 多项式次数
    ridge_alpha: float = 1e-6 # Ridge 正则化参数
    train_ratio: float = 0.5  # 训练集比例
    exercise_dates: Optional[List[int]] = None  # 可提前行权日期
    
    def __post_init__(self):
        if self.exercise_dates is None:
            # 默认每10个时间步可提前行权
            self.exercise_dates = list(range(10, 100, 10))


@dataclass
class ExperimentGrids:
    """实验参数网格"""
    # Strike 网格 (moneyness)
    strike_grid: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1, 1.2])
    
    # Maturity 网格
    maturity_grid: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0])
    
    # Hurst 指数网格
    H_grid: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5])
    
    # Hawkes 自激参数网格
    alpha_S_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5])
    alpha_V_grid: List[float] = field(default_factory=lambda: [0.3, 0.6, 1.0, 1.4])
    
    # Jump size 网格
    sigmaJ_grid: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    
    # N_exp 网格
    N_exp_grid: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 10])
    
    # MC 路径数网格
    n_paths_grid: List[int] = field(default_factory=lambda: [10000, 20000, 50000, 100000])


@dataclass
class PlottingParams:
    """绘图参数"""
    figsize: tuple = (8, 6)
    fontsize: int = 12
    legend_fontsize: int = 10
    title_fontsize: int = 14
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"  # 论文友好风格


@dataclass
class ModelConfig:
    """完整模型配置"""
    market: MarketParams = field(default_factory=MarketParams)
    heston: HestonParams = field(default_factory=HestonParams)
    rough_heston: RoughHestonParams = field(default_factory=RoughHestonParams)
    merton: MertonJumpParams = field(default_factory=MertonJumpParams)
    var_jump: VarianceJumpParams = field(default_factory=VarianceJumpParams)
    hawkes: HawkesParams = field(default_factory=HawkesParams)
    exp_approx: ExpApproxParams = field(default_factory=ExpApproxParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    lsm: LSMParams = field(default_factory=LSMParams)
    grids: ExperimentGrids = field(default_factory=ExperimentGrids)
    plotting: PlottingParams = field(default_factory=PlottingParams)
    
    # 模型选择
    model_type: str = "dst_main"  # "classic_heston", "rough_heston", "rough_poisson_merton", "rough_hawkes_merton", "dst_main"


# 默认配置实例
DEFAULT_CONFIG = ModelConfig()


def get_config(model_type: str = "dst_main") -> ModelConfig:
    """获取指定类型的默认配置"""
    config = ModelConfig()
    config.model_type = model_type
    return config
