"""
============================================================
DST 双 Hawkes 跳跃过程模块
- 支持 Poisson, Single Hawkes, Dual Hawkes (DST) 三种跳跃机制
============================================================
"""

import numpy as np
from typing import Tuple, Optional

class JumpProcess:
    """
    跳跃过程基类
    """
    
    def __init__(self, lambda_poisson=0.1, mu_J=-0.05, sigma_J=0.10):
        """
        参数:
        -----
        lambda_poisson : float
            泊松强度 (对 Poisson 适用)
        mu_J : float
            跳幅均值 (对数收益率)
        sigma_J : float
            跳幅标准差
        """
        self.lambda_poisson = lambda_poisson
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        
        # 计算 E[exp(J) - 1] = exp(mu_J + 0.5*sigma_J^2) - 1
        self.m_J = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    
    def simulate(self, T, n_paths, n_steps, dt=None, seed=None) -> Tuple:
        """
        仿真跳跃路径 (基类方法，子类实现)
        
        返回:
        -----
        jump_times : list of ndarray
            每个路径的跳跃时间点
        jump_amplitudes : list of ndarray
            每个路径的跳跃幅度
        """
        raise NotImplementedError
    
    def generate_jump_amplitudes(self, n_jumps, seed=None):
        """生成跳幅 J ~ N(mu_J, sigma_J^2)"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(self.mu_J, self.sigma_J, n_jumps)


class PoissonJump(JumpProcess):
    """
    独立同分布 Poisson 跳跃
    
    N_t ~ Poisson(λ t)
    J_i ~ N(mu_J, sigma_J^2)
    """
    
    def __init__(self, lambda_poisson=0.1, mu_J=-0.05, sigma_J=0.10):
        super().__init__(lambda_poisson, mu_J, sigma_J)
    
    def simulate(self, T, n_paths, n_steps, dt=None, seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        仿真 Poisson 跳跃路径
        
        返回:
        -----
        jump_indicator : ndarray, shape (n_paths, n_steps)
            跳跃指示变量 (0 或 1)
        jump_times : list of ndarray
            每个路径的跳跃时间点
        jump_amplitudes_list : list of ndarray
            每个路径的跳跃幅度
        """
        if seed is not None:
            np.random.seed(seed)
        
        if dt is None:
            dt = T / n_steps
        
        # 每个路径的期望跳跃数
        expected_jumps = self.lambda_poisson * T
        
        # 初始化输出
        jump_indicator = np.zeros((n_paths, n_steps))
        jump_times_list = []
        jump_amplitudes_list = []
        
        for path_idx in range(n_paths):
            # 使用泊松采样确定跳跃次数
            n_jumps = np.random.poisson(expected_jumps)
            
            if n_jumps > 0:
                # 跳跃时间: 均匀分布在 [0, T]
                jump_times = np.sort(np.random.uniform(0, T, n_jumps))
                jump_amplitudes = self.generate_jump_amplitudes(n_jumps)
                
                # 将跳跃映射到离散时间步
                jump_steps = np.searchsorted(np.linspace(0, T, n_steps), jump_times)
                jump_steps = np.clip(jump_steps, 0, n_steps - 1)
                
                for step in jump_steps:
                    jump_indicator[path_idx, step] = 1
                
                jump_times_list.append(jump_times)
                jump_amplitudes_list.append(jump_amplitudes)
            else:
                jump_times_list.append(np.array([]))
                jump_amplitudes_list.append(np.array([]))
        
        return jump_indicator, jump_times_list, jump_amplitudes_list


class SingleHawkes(JumpProcess):
    """
    单变量自激 Hawkes 跳跃过程
    
    dλ_t = β(λ_∞ - λ_t) dt + η dN_t
    λ_t = λ_∞ + (λ_0 - λ_∞) e^{-β t} + η ∫_0^t e^{-β(t-s)} dN_s
    
    N_t 的强度为 λ_t
    """
    
    def __init__(self, lambda_inf=0.1, beta=1.0, eta=0.1, mu_J=-0.05, sigma_J=0.10):
        """
        参数:
        -----
        lambda_inf : float
            稳态强度
        beta : float
            衰减速率
        eta : float
            自激系数 (jump -> increase lambda)
        """
        super().__init__(lambda_poisson=lambda_inf, mu_J=sigma_J)
        self.lambda_inf = lambda_inf
        self.beta = beta
        self.eta = eta
        self.lambda_0 = lambda_inf  # 初始强度
    
    def simulate(self, T, n_paths, n_steps, dt=None, seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        仿真 Single Hawkes 跳跃路径
        
        返回:
        -----
        jump_indicator : ndarray, shape (n_paths, n_steps)
            跳跃指示变量
        lambda_path : ndarray, shape (n_paths, n_steps+1)
            强度路径 λ_t
        jump_times_list : list of ndarray
            每个路径的跳跃时间点
        jump_amplitudes_list : list of ndarray
            每个路径的跳跃幅度
        """
        if seed is not None:
            np.random.seed(seed)
        
        if dt is None:
            dt = T / n_steps
        
        t_grid = np.linspace(0, T, n_steps + 1)
        
        # 初始化
        jump_indicator = np.zeros((n_paths, n_steps))
        lambda_path = np.zeros((n_paths, n_steps + 1))
        lambda_path[:, 0] = self.lambda_0
        
        jump_times_list = []
        jump_amplitudes_list = []
        
        for path_idx in range(n_paths):
            jump_times = []
            jump_amplitudes = []
            lambda_current = self.lambda_0
            
            for n in range(n_steps):
                # 在当前强度下决定是否跳跃
                # 使用 thinning algorithm
                lambda_max = self.lambda_inf + self.eta * self.lambda_inf * 2  # 上界估计
                u = np.random.random()
                
                if u < lambda_current * dt / lambda_max:
                    # 发生跳跃
                    jump_indicator[path_idx, n] = 1
                    jump_times.append(t_grid[n])
                    jump_amplitudes.append(np.random.normal(self.mu_J, self.sigma_J))
                    
                    # 更新强度: λ_{t+} = λ_{t-} + η
                    lambda_current = min(lambda_current + self.eta, lambda_max)
                
                # 强度衰减: λ_{t+dt} = λ_t + β(λ_∞ - λ_t) dt
                lambda_current = lambda_current + self.beta * (self.lambda_inf - lambda_current) * dt
                
                lambda_path[path_idx, n + 1] = lambda_current
            
            jump_times_list.append(np.array(jump_times))
            jump_amplitudes_list.append(np.array(jump_amplitudes))
        
        return jump_indicator, lambda_path, jump_times_list, jump_amplitudes_list


class DualHawkes(JumpProcess):
    """
    双重 Hawkes (DST) 跳跃过程 - 主模型
    
    dλ_t^S = β_S (λ_∞^S - λ_t^S) dt + η_SS dN_t^S + η_SV dN_t^V
    dλ_t^V = β_V (λ_∞^V - λ_t^V) dt + η_VV dN_t^V + η_VS dN_t^S
    
    N_t^S: 价格跳跃过程 (强度 λ_t^S)
    N_t^V: 波动率跳跃过程 (强度 λ_t^V)
    """
    
    def __init__(
        self,
        lambda_inf_S=0.1, beta_S=1.0, eta_SS=0.1, eta_SV=0.05,
        lambda_inf_V=0.05, beta_V=1.0, eta_VV=0.05, eta_VS=0.05,
        mu_J=-0.05, sigma_J=0.10
    ):
        """
        参数:
        -----
        lambda_inf_S : float
            价格跳跃稳态强度
        beta_S : float
            价格跳跃衰减速率
        eta_SS : float
            价格跳跃自激系数
        eta_SV : float
            价格跳跃 -> 波动率跳跃强度
        lambda_inf_V : float
            波动率跳跃稳态强度
        beta_V : float
            波动率跳跃衰减速率
        eta_VV : float
            波动率跳跃自激系数
        eta_VS : float
            波动率跳跃 -> 价格跳跃强度
        """
        super().__init__(lambda_poisson=lambda_inf_S, mu_J=mu_J, sigma_J=sigma_J)
        
        self.lambda_inf_S = lambda_inf_S
        self.beta_S = beta_S
        self.eta_SS = eta_SS
        self.eta_SV = eta_SV
        
        self.lambda_inf_V = lambda_inf_V
        self.beta_V = beta_V
        self.eta_VV = eta_VV
        self.eta_VS = eta_VS
        
        # 初始强度
        self.lambda_0_S = lambda_inf_S
        self.lambda_0_V = lambda_inf_V
    
    def simulate(self, T, n_paths, n_steps, dt=None, seed=None) -> Tuple:
        """
        仿真 Dual Hawkes (DST) 跳跃路径
        
        返回:
        -----
        jump_indicator_S : ndarray
            价格跳跃指示变量
        jump_indicator_V : ndarray
            波动率跳跃指示变量
        lambda_S_path : ndarray
            价格跳跃强度路径
        lambda_V_path : ndarray
            波动率跳跃强度路径
        jump_times_S_list : list
            价格跳跃时间点
        jump_times_V_list : list
            波动率跳跃时间点
        jump_amplitudes_S_list : list
            价格跳跃幅度
        jump_amplitudes_V_list : list
            波动率跳跃幅度
        """
        if seed is not None:
            np.random.seed(seed)
        
        if dt is None:
            dt = T / n_steps
        
        t_grid = np.linspace(0, T, n_steps + 1)
        
        # 初始化
        jump_indicator_S = np.zeros((n_paths, n_steps))
        jump_indicator_V = np.zeros((n_paths, n_steps))
        lambda_S_path = np.zeros((n_paths, n_steps + 1))
        lambda_V_path = np.zeros((n_paths, n_steps + 1))
        
        lambda_S_path[:, 0] = self.lambda_0_S
        lambda_V_path[:, 0] = self.lambda_0_V
        
        jump_times_S_list = []
        jump_times_V_list = []
        jump_amplitudes_S_list = []
        jump_amplitudes_V_list = []
        
        for path_idx in range(n_paths):
            jump_times_S = []
            jump_times_V = []
            jump_amplitudes_S = []
            jump_amplitudes_V = []
            
            lambda_S = self.lambda_0_S
            lambda_V = self.lambda_0_V
            
            for n in range(n_steps):
                # 强度上界 (保守估计)
                lambda_max_S = self.lambda_inf_S + self.eta_SS * 2 + self.eta_VS * 2
                lambda_max_V = self.lambda_inf_V + self.eta_VV * 2 + self.eta_SV * 2
                
                # 判定价格跳跃
                u_S = np.random.random()
                if u_S < lambda_S * dt:
                    jump_indicator_S[path_idx, n] = 1
                    jump_times_S.append(t_grid[n])
                    jump_amplitudes_S.append(np.random.normal(self.mu_J, self.sigma_J))
                    # 互激: 价格跳跃增加波动率跳跃强度
                    lambda_V = lambda_V + self.eta_SV
                
                # 判定波动率跳跃
                u_V = np.random.random()
                if u_V < lambda_V * dt:
                    jump_indicator_V[path_idx, n] = 1
                    jump_times_V.append(t_grid[n])
                    # 波动率跳跃的幅度可以设为 0 或小随机值
                    jump_amplitudes_V.append(0.0)
                    # 互激: 波动率跳跃增加价格跳跃强度
                    lambda_S = lambda_S + self.eta_VS
                
                # 自激更新
                lambda_S = lambda_S + self.beta_S * (self.lambda_inf_S - lambda_S) * dt
                lambda_V = lambda_V + self.beta_V * (self.lambda_inf_V - lambda_V) * dt
                
                # 确保非负
                lambda_S = max(lambda_S, 0)
                lambda_V = max(lambda_V, 0)
                
                lambda_S_path[path_idx, n + 1] = lambda_S
                lambda_V_path[path_idx, n + 1] = lambda_V
            
            jump_times_S_list.append(np.array(jump_times_S))
            jump_times_V_list.append(np.array(jump_times_V))
            jump_amplitudes_S_list.append(np.array(jump_amplitudes_S))
            jump_amplitudes_V_list.append(np.array(jump_amplitudes_V))
        
        return (
            jump_indicator_S, jump_indicator_V,
            lambda_S_path, lambda_V_path,
            jump_times_S_list, jump_times_V_list,
            jump_amplitudes_S_list, jump_amplitudes_V_list
        )


def create_jump_process(jump_type='none', **kwargs):
    """
    工厂函数: 创建跳跃过程
    """
    if jump_type == 'none':
        return None
    elif jump_type == 'poisson':
        return PoissonJump(**kwargs)
    elif jump_type == 'single_hawkes':
        return SingleHawkes(**kwargs)
    elif jump_type == 'dst':
        return DualHawkes(**kwargs)
    else:
        raise ValueError(f"Unknown jump_type: {jump_type}")
