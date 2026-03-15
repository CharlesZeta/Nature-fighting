# DST 期权隐含波动率曲面校准与多方法对比 - 程序修改 Prompt

---

## 背景说明

在现有 Rough Heston + DST 跳跃模型基础上，增加：
1. 隐含波动率关系函数图像与曲面校准
2. 多种方法/模型并列对比
3. 与复合 Hawkes 跳跃过程对比

---

## 任务 1：隐含波动率关系函数图像

### 1.1 需实现的函数图

| 图像 | 横轴 | 纵轴 | 说明 |
|------|------|------|------|
| IV vs T | 到期时间 T | BS隐含波动率 | 固定 moneyness，IV 随 T 变化 |
| IV vs K | 行权价 K | BS隐含波动率 | 固定 T，IV 随 K 变化（波动率微笑） |
| IV敏感性 | K/S0 | ∂IV/∂σ | 参数敏感度分析 |
| 跳跃统计-均值 | H/λ | 跳跃幅度均值 | 不同参数下跳跃强度 |
| 跳跃统计-聚类 | H | 跳跃聚类强度 | Hawkes 过程特征 |
| 对数收益分布 | Log-return bins | 频率 | 模拟路径的对数收益直方图 |
| 时间衰退函数 | t | φ(t) | 特征函数衰减 |

### 1.2 参考文献与公式

参考图1公式，搜索关键词：
- "rough volatility implied volatility term structure"
- "DST dispersion trading volatility smile"
- "Hawkes process jump clustering"
- "characteristic function rough Heston"
- "implied volatility surface calibration"

---

## 任务 2：隐含波动率曲面校准

### 2.1 校准目标函数

参考公式 (68)：
```
f(θ) = Σ_i Σ_j |IV_model(T_i, K_j) - IV_market(T_i, K_j)| / IV_market(T_i, K_j)
```

### 2.2 待校准参数

| 参数 | 含义 | 初始范围 |
|------|------|----------|
| v0 | 初始方差 | [0.01, 0.2] |
| κ | 均值回归速度 | [0.5, 5.0] |
| θ | 长期方差 | [0.01, 0.3] |
| ξ | 波动率方差 | [0.1, 1.0] |
| ρ | 相关系数 | [-0.9, 0.0] |
| H | Hurst指数 | [0.01, 0.5] |
| λ | 跳跃强度 | [0.0, 2.0] |

### 2.3 输出

- 校准前后 IV 曲面 3D/2D 对比图
- 目标函数迭代收敛曲线
- 最优参数表
- 过拟合检查（train vs validation 误差）

### 2.4 实现文件

新建 `calibrate_iv_surface.py`：
- `calibrate_surface(params_init, market_iv_grid)`：返回最优参数
- `compute_model_iv(params, T_grid, K_grid)`：计算模型 IV 曲面
- `plot_surface_comparison()`：绘制对比图

---

## 任务 3：多种方法/模型并列对比

### 3.1 对比方法

| 方法 | 说明 |
|------|------|
| DST (当前) | 分散跳跃阈值模型 |
| M0 | 标准 Heston |
| M1 | Markovian Heston |
| M2 | Rough Heston (H=0.1) |
| SINC | sinc 离散化方法 |
| Gauss-Laguerre | 高斯-拉盖尔积分 |
| Flat iFT | 逆傅里叶变换 |

### 3.2 对比指标

| 指标 | 说明 |
|------|------|
| Price | 美式/欧式期权价格 |
| Premium | American - European |
| ΔIV | 美式 IV - 欧式 IV |
| Runtime | 计算时间 |
| Relative Error | vs Benchmark |

### 3.3 输出表格 (LaTeX 格式)

参考 Table 4：
```
\begin{table}[h]
\caption{Model Comparison: European Put Option Pricing}
\label{tab:comparison}
\begin{tabular}{lcccccc}
\hline
Method & K/S0 & Price & Std & Premium & Rel.Error & Time(s)\\
\hline
DST & 0.80 & 1.2345 & 0.0012 & 0.1234 & - & 0.45\\
Heston & 0.80 & 1.2356 & 0.0013 & 0.1245 & 0.09\% & 0.38\\
Rough Heston & 0.80 & 1.2341 & 0.0011 & 0.1230 & 0.03\% & 0.52\\
\hline
\end{tabular}
\end{table}
```

### 3.4 DST vs 复合 Hawkes 对比图

- 同一坐标轴：DST vs Single Hawkes vs Double Exponential Hawkes
- 指标：price, premium, delta_IV, early exercise ratio
- 输出：对比折线图、箱线图

---

## 任务 4：程序模块分工

| 文件 | 职责 |
|------|------|
| `implied_vol.py` | 扩展 `batch_iv(moneyness_list, T_list)` 用于曲面 |
| `calibrate_iv_surface.py` | **新建**，最小化校准目标函数，输出最优参数与曲面 |
| `benchmark_comparison.py` | **新建**，多方法/多模型对比 |
| `dst_jump.py` | 保持，确保返回跳跃统计（mean jump, cluster intensity 等） |
| `run_experiments.py` | 调用 calibration 与 benchmark，导出结果 |
| `export_results.py` | 导出校准曲面、对比表、对比图数据 |

---

## 任务 5：输出文件清单

```
results第三轮/
├── iv_surface_calibration_results.csv    # 校准参数与误差
├── iv_surface_before_after.png            # 校准前后对比
├── calibration_objective_convergence.png # 目标函数收敛
├── benchmark_comparison_table.csv         # 多方法对比表
├── benchmark_comparison_latest.csv
├── dst_vs_hawkes_comparison.png           # DST vs Hawkes
├── model_params_comparison.csv            # 参数对比
├── jump_statistics.csv                    # 跳跃统计
├── iv_vs_maturity.png                     # IV vs T
├── iv_vs_strike.png                       # IV vs K (smile)
├── log_return_distribution.png            # 对数收益分布
└── time_decay_function.png                # 特征函数衰减
```

---

## 任务 6：关键代码框架

### 6.1 calibrate_iv_surface.py 框架

```python
"""
隐含波动率曲面校准模块
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Dict, List
import pandas as pd
import time

class IVSurfaceCalibrator:
    """隐含波动率曲面校准器"""
    
    def __init__(self, S0: float, r: float, market_iv: np.ndarray, 
                 T_grid: np.ndarray, K_grid: np.ndarray):
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
            到期时间网格
        K_grid : ndarray
            行权价网格
        """
        self.S0 = S0
        self.r = r
        self.market_iv = market_iv
        self.T_grid = T_grid
        self.K_grid = K_grid
        
    def objective(self, params: np.ndarray) -> float:
        """
        校准目标函数：相对误差加权求和
        
        f(θ) = Σ_ij |IV_model - IV_market| / IV_market
        """
        # 解包参数
        v0, kappa, theta, xi, rho, H, lam = params
        
        # 计算模型 IV 曲面
        model_iv = self.compute_model_iv(params)
        
        # 计算相对误差
        rel_error = np.abs(model_iv - self.market_iv) / (self.market_iv + 1e-8)
        
        return np.sum(rel_error)
    
    def compute_model_iv(self, params: np.ndarray) -> np.ndarray:
        """计算模型隐含波动率曲面"""
        # TODO: 实现模型 IV 计算
        pass
    
    def calibrate(self, method: str = 'de', 
                  bounds: List[Tuple] = None) -> Dict:
        """
        执行校准
        
        参数:
        -----
        method : str
            优化方法: 'de' (差分进化), 'lbfgsb', 'cg'
        bounds : list
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
                (0.01, 0.3),   # theta
                (0.1, 1.0),     # xi
                (-0.9, 0.0),    # rho
                (0.01, 0.5),    # H
                (0.0, 2.0),    # lambda
            ]
        
        if method == 'de':
            result = differential_evolution(
                self.objective, bounds, 
                maxiter=500, tol=1e-6,
                seed=42, workers=1,
                polish=True
            )
        
        return {
            'params': result.x,
            'error': result.fun,
            'nit': result.nit,
            'success': result.success,
        }
    
    def compute_greeks(self) -> Dict:
        """计算参数敏感度 ∂IV/∂σ"""
        # 数值差分
        pass
    
    def export_results(self, output_path: str):
        """导出校准结果"""
        pass


def run_calibration_experiment(
    S0: float = 100,
    T_range: List[float] = [0.1, 0.25, 0.5, 1.0],
    K_range: List[float] = [0.8, 0.9, 1.0, 1.1, 1.2],
    output_dir: str = 'results第三轮'
):
    """运行完整校准实验"""
    
    # 1. 生成 synthetic market data (或读取真实数据)
    # 2. 初始化校准器
    # 3. 执行校准
    # 4. 可视化结果
    
    pass
```

### 6.2 benchmark_comparison.py 框架

```python
"""
多方法/多模型对比模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import time
from pathlib import Path

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
    
    def __init__(self, S0: float, K: float, T: float, r: float):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        
    def price_with_method(self, method: str, **kwargs) -> Dict:
        """
        使用指定方法计算价格
        
        方法列表:
        - 'dst': 当前 DST 模型
        - 'heston': 标准 Heston
        - 'rough_heston': Rough Heston
        - 'sinc': SINC 方法
        - 'gauss_laguerre': 高斯-拉盖尔
        - 'flat_ift': 逆傅里叶变换
        """
        if method == 'dst':
            # 调用当前 DST 实现
            pass
        elif method == 'heston':
            # Heston 解析解
            pass
        # ... 其他方法
        
    def run_comparison(self, 
                      methods: List[str],
                      moneyness_grid: List[float]) -> pd.DataFrame:
        """
        运行对比实验
        
        返回:
        -----
        results : DataFrame
            包含各方法的 price, premium, runtime, error
        """
        results = []
        
        for K in moneyness_grid:
            for method in methods:
                start = time.time()
                result = self.price_with_method(method, K=K)
                elapsed = time.time() - start
                
                results.append({
                    'method': method,
                    'moneyness': K / self.S0,
                    'price': result['price'],
                    'premium': result.get('premium', np.nan),
                    'runtime': elapsed,
                    'error_vs_benchmark': result.get('error', np.nan),
                })
        
        return pd.DataFrame(results)
    
    def export_latex_table(self, df: pd.DataFrame, output_path: str):
        """导出 LaTeX 格式对比表"""
        # 生成 table 环境 latex 代码
        pass
    
    def plot_comparison(self, df: pd.DataFrame, output_dir: str):
        """绘制对比图"""
        # 折线图：各方法 price vs moneyness
        # 箱线图：多 seed 误差分布
        pass


class DstHawkesComparison:
    """DST vs 复合 Hawkes 对比"""
    
    def __init__(self):
        pass
    
    def compare(self, 
               params_dst: Dict,
               params_hawkes: Dict,
               T_range: List[float]) -> pd.DataFrame:
        """
        对比 DST 与复合 Hawkes 模型
        
        参数:
        -----
        params_dst : dict
            DST 模型参数
        params_hawkes : dict
            Hawkes 过程参数 (mu, alpha, beta, etc.)
        T_range : list
            到期时间范围
            
        返回:
        -----
        comparison : DataFrame
        """
        pass
```

---

## 任务 7：MATLAB 绘图扩展

在 `plot_results_v3.m` 基础上增加：

```matlab
%% 图 X：隐含波动率曲面校准
figure('Name', 'IV Surface Calibration');
subplot(1,2,1);
surf(T_grid, K_grid, market_iv);
title('Market IV Surface'); xlabel('T'); ylabel('K/S0'); zlabel('IV');

subplot(1,2,2);
surf(T_grid, K_grid, model_iv);
title('Calibrated Model IV Surface'); xlabel('T'); ylabel('K/S0'); zlabel('IV');

%% 图 X：DST vs Hawkes 对比
figure('Name', 'DST vs Hawkes');
plot(moneyness, dst_price, '-o', 'DisplayName', 'DST');
hold on;
plot(moneyness, hawkes_price, '-s', 'DisplayName', 'Hawkes');
xlabel('Moneyness K/S0'); ylabel('American Put Price');
legend('Location', 'best'); grid on;
```

---

## 任务 8：输出要求

1. **CSV 文件**：所有数值结果导出为 CSV，便于后续分析
2. **PNG 图像**：高分辨率图像（300 DPI）
3. **LaTeX 表格**：可直接插入论文的 table 环境
4. **日志文件**：记录校准迭代过程、参数变化

---

## 验收标准

- [ ] 隐含波动率曲面校准可运行，返回稳定参数
- [ ] 多方法对比表包含至少 5 种方法
- [ ] DST vs Hawkes 对比图清晰展示差异
- [ ] 所有图像分母/坐标轴标签清晰
- [ ] 代码注释完整，可复现

---

请依次实现上述模块。