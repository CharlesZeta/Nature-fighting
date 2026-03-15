# Rough Heston + DST 美式期权 LSM 定价 - 第二轮实验程序

## 目录结构

```
论文/
├── config.py              # 配置文件 (参数、网格、模型)
├── rough_heston.py       # Rough Heston 方差过程
├── dst_jump.py           # DST 双 Hawkes 跳跃过程
├── simulation.py          # 路径仿真器
├── lsm_american.py        # LSM 美式期权定价
├── implied_vol.py        # 隐含波动率计算
├── run_experiments.py    # 实验主程序
├── export_results.py      # 结果导出
├── README.md             # 说明文档
└── results第二轮/        # 结果输出目录
```

## 依赖

```bash
pip install numpy pandas scipy scikit-learn
```

## 快速开始

### 1. 快速测试 (小规模)

```bash
python run_experiments.py --mode quick
```

这将运行小规模测试：
- 模型: M2, M5
- Moneyness: 5 个点 (0.80 - 1.20)
- Maturity: 2 档 (0.1, 1.0)
- Hurst: 2 个值 (0.05, 0.10)
- 路径数: 5,000

### 2. 完整实验

```bash
python run_experiments.py --mode full
```

完整配置：
- 模型: M0 - M5 (6 个)
- Moneyness: 25 个点 (0.70 - 1.30)
- Maturity: 3 档 (0.1, 0.5, 1.0)
- Hurst: 6 个值 (0.03, 0.05, 0.07, 0.10, 0.15, 0.50)
- 路径数: 50,000 (测试) / 200,000 (参考)

总计: 6 × 25 × 3 × 6 = 2700 个实验

### 3. 自定义运行

```bash
python run_experiments.py --models M2 M5 --output my_results
```

## 输出文件

运行后会在 `results第二轮/` 目录生成：

| 文件 | 说明 |
|------|------|
| `price_results_latest.csv` | 价格结果表 |
| `smile_statistics.csv` | Smile 统计 (ATM IV, skew, curvature) |
| `price_table.csv` | 价格类汇总表 |
| `smile_table.csv` | Smile 数据表 |
| `jump_stats.csv` | 跳跃统计表 |
| `boundary_table.csv` | 行权边界表 |
| `summary_statistics.csv` | 汇总统计 |
| `matlab/` | MATLAB 数据接口 |

## 参数调整

在 `config.py` 中可以修改：

```python
# 网格配置
MONEYNESS_GRID = [...]   # Strike 网格
MATURITY_GRID = [...]    # Maturity 网格  
HURST_GRID = [...]       # Hurst 参数网格
MODEL_CONFIG = {...}     # 模型配置

# 路径数
N_TEST_PATHS = 50000     # 测试组路径数
N_REFERENCE_PATHS = 200000  # 参考组路径数

# LSM 参数
POLY_DEGREE = 3          # 多项式阶数
RIDGE_ALPHA = 0.01       # Ridge 正则化系数
```

## 模型说明

| 模型 | 波动率 | 跳跃机制 |
|------|--------|----------|
| M0 | Markovian Heston | Poisson |
| M1 | Markovian Heston | Single Hawkes |
| M2 | Rough Heston | 无 |
| M3 | Rough Heston | Poisson |
| M4 | Rough Heston | Single Hawkes |
| M5 | Rough Heston | DST (双重 Hawkes) |

## 结果字段

### 价格类
- `american_price`: 美式期权价格
- `european_price`: European 价格
- `american_premium`: 美式溢价
- `early_exercise_ratio`: 提前行权比例

### IV 类
- `implied_vol_proxy`: 隐含波动率
- `atm_iv`: ATM 隐含波动率

### 误差类
- `abs_error`: 绝对误差
- `rel_error`: 相对误差

### 跳跃统计
- `mean_jump_count_S`: 平均价格跳跃次数
- `mean_lambda_S`: 平均价格跳跃强度

## 注意事项

1. **路径数**: 第一轮只用 5000 路径是诊断级，第二轮需要 50000+ 才能得到论文级精度
2. **IV 过高的处理**: 程序中使用 Brent 方法求解隐含波动率，如果价格异常会返回 NaN
3. **Premium 为负**: 通过 Ridge 正则化可降低概率
4. **10^-6 精度**: 仅适用于子问题/局部精度，不能作为 American 最终价格的全局保证
