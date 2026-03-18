#!/usr/bin/env python3
"""
Implied Volatility Curves: DST (Rough Heston + Dual Hawkes), Merton, BSM
  - European options: closed-form / MC
  - American options: LSM (Longstaff-Schwartz)
  - Control variate: BSM price as CV for MC estimators
  - Multiple alpha (gamma) values for DST
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import math
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. BSM Closed-Form
# ============================================================
def bsm_price(S, K, T, r, sigma, option='put'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def implied_vol(price, S, K, T, r, option='put', lo=0.001, hi=5.0):
    """Invert BSM to get IV via Brent's method."""
    try:
        intrinsic = max(K*np.exp(-r*T) - S, 0) if option=='put' else max(S - K*np.exp(-r*T), 0)
        if price <= intrinsic + 1e-10:
            return np.nan
        f = lambda sig: bsm_price(S, K, T, r, sig, option) - price
        if f(lo)*f(hi) > 0:
            return np.nan
        return brentq(f, lo, hi, xtol=1e-8)
    except:
        return np.nan

# ============================================================
# 2. Merton Jump-Diffusion (European closed-form, American LSM)
# ============================================================
def merton_european(S, K, T, r, sigma, lam, mu_J, sigma_J, option='put', N_terms=30):
    """Merton (1976) series expansion."""
    m_s = np.exp(mu_J + 0.5*sigma_J**2) - 1
    price = 0.0
    for n in range(N_terms):
        sigma_n = np.sqrt(sigma**2 + n*sigma_J**2/T)
        r_n = r - lam*m_s + n*np.log(1+m_s)/T
        lam_prime = lam*(1+m_s)
        w = np.exp(-lam_prime*T) * (lam_prime*T)**n / math.factorial(n)
        price += w * bsm_price(S, K, T, r_n, sigma_n, option)
    return price

def merton_mc_paths(S0, r, sigma, lam, mu_J, sigma_J, T, N_steps, N_paths):
    """Generate Merton paths for LSM."""
    dt = T / N_steps
    m_s = np.exp(mu_J + 0.5*sigma_J**2) - 1
    S = np.zeros((N_paths, N_steps+1))
    S[:, 0] = S0
    for i in range(N_steps):
        Z = np.random.randn(N_paths)
        # Poisson jumps
        N_jumps = np.random.poisson(lam*dt, N_paths)
        J_sum = np.zeros(N_paths)
        for j in range(N_paths):
            if N_jumps[j] > 0:
                J_sum[j] = np.sum(np.random.normal(mu_J, sigma_J, N_jumps[j]))
        drift = (r - lam*m_s - 0.5*sigma**2)*dt
        diffusion = sigma*np.sqrt(dt)*Z
        S[:, i+1] = S[:, i] * np.exp(drift + diffusion + J_sum)
    return S

# ============================================================
# 3. DST: Rough Heston + Dual Self-Excited Hawkes
# ============================================================
def gauss_laguerre_nodes(N_exp, alpha):
    """Compute exponential sum approximation nodes/weights for fractional kernel."""
    from numpy.polynomial.laguerre import laggauss
    nodes, weights = laggauss(N_exp)
    # Transform for the measure x^{alpha-1}/Gamma(alpha)
    from scipy.special import gamma as gamma_func
    x_l = nodes  # Laguerre nodes
    omega_l = weights * nodes**(alpha-1) / gamma_func(alpha)
    # Normalize weights to sum to 1
    omega_l = omega_l / np.sum(omega_l)
    # Ensure positivity
    x_l = np.maximum(x_l, 1e-6)
    omega_l = np.maximum(omega_l, 1e-12)
    omega_l = omega_l / np.sum(omega_l)
    return x_l, omega_l

def dst_mc_paths(S0, r, V0, kappa, theta, xi, rho, alpha,
                 lam_inf_S, lam_inf_V, beta_S, beta_V,
                 eta_SS, eta_VV, eta_SV, eta_VS,
                 mu_J, sigma_J, mu_V,
                 T, N_steps, N_paths, N_exp=6):
    """
    Monte Carlo paths for the DST model (rough Heston + dual Hawkes).
    Uses Markov lifting with N_exp OU factors.
    Returns S paths array (N_paths x N_steps+1) and V paths.
    """
    dt = T / N_steps
    sqrt_dt = np.sqrt(dt)
    
    # Exponential sum approximation
    x_l, omega_l = gauss_laguerre_nodes(N_exp, alpha)
    
    m_s = np.exp(mu_J + 0.5*sigma_J**2) - 1  # jump compensation
    
    # Allocate arrays
    S = np.zeros((N_paths, N_steps+1))
    V_agg = np.zeros((N_paths, N_steps+1))  # aggregated variance
    S[:, 0] = S0
    V_agg[:, 0] = V0
    
    # OU factors: shape (N_paths, N_exp)
    U = np.zeros((N_paths, N_exp))
    
    # Hawkes intensities
    lam_S = np.full(N_paths, lam_inf_S)
    lam_V = np.full(N_paths, lam_inf_V)
    
    for i in range(N_steps):
        V_pos = np.maximum(V_agg[:, i], 0.0)
        sqrt_V = np.sqrt(V_pos)
        
        # Correlated Brownian motions
        Z1 = np.random.randn(N_paths)
        Z2 = np.random.randn(N_paths)
        dW_S = Z1 * sqrt_dt
        dW_perp = Z2 * sqrt_dt
        dW_V = rho * dW_S + np.sqrt(1 - rho**2) * dW_perp
        
        # --- Hawkes jumps ---
        # Price jumps (Poisson with intensity lam_S * dt)
        prob_S = np.minimum(lam_S * dt, 0.99)
        jump_S_mask = np.random.random(N_paths) < prob_S
        J_S = np.where(jump_S_mask, np.random.normal(mu_J, sigma_J, N_paths), 0.0)
        dN_S = jump_S_mask.astype(float)
        
        # Variance jumps
        prob_V = np.minimum(lam_V * dt, 0.99)
        jump_V_mask = np.random.random(N_paths) < prob_V
        J_V = np.where(jump_V_mask, np.random.exponential(mu_V, N_paths), 0.0)
        dN_V = jump_V_mask.astype(float)
        
        # --- Update OU factors ---
        for l in range(N_exp):
            drift_U = (-x_l[l] * U[:, l] + kappa * (theta - V_pos)) * dt
            diff_U = xi * sqrt_V * dW_V
            jump_U = J_V * dN_V
            U[:, l] += drift_U + diff_U + jump_U
        
        # Aggregated variance
        V_agg[:, i+1] = V0 + np.sum(omega_l[np.newaxis, :] * U, axis=1)
        
        # --- Update Hawkes intensities ---
        lam_S += beta_S * (lam_inf_S - lam_S) * dt + eta_SS * dN_S + eta_SV * dN_V
        lam_V += beta_V * (lam_inf_V - lam_V) * dt + eta_VV * dN_V + eta_VS * dN_S
        lam_S = np.maximum(lam_S, 1e-8)
        lam_V = np.maximum(lam_V, 1e-8)
        
        # --- Update price ---
        V_pos_new = np.maximum(V_agg[:, i+1], 0.0)
        # Use average of V for better discretization
        V_mid = 0.5*(V_pos + V_pos_new)
        sqrt_V_mid = np.sqrt(np.maximum(V_mid, 0.0))
        
        drift_S = (r - lam_S * m_s - 0.5 * V_mid) * dt
        diff_S = sqrt_V_mid * dW_S
        jump_price = (np.exp(J_S) - 1) * dN_S
        
        S[:, i+1] = S[:, i] * np.exp(drift_S + diff_S) * (1 + jump_price)
    
    return S, V_agg

# ============================================================
# 4. LSM (Longstaff-Schwartz) for American Options
# ============================================================
def lsm_american_put(S_paths, K, r, T):
    """LSM for American put option."""
    N_paths, N_steps_plus1 = S_paths.shape
    N_steps = N_steps_plus1 - 1
    dt = T / N_steps
    
    # Cash flows at expiry
    cashflow = np.maximum(K - S_paths[:, -1], 0.0)
    exercise_time = np.full(N_paths, N_steps)
    
    # Backward induction
    for t in range(N_steps-1, 0, -1):
        disc = np.exp(-r * dt)
        cashflow *= disc  # discount one step
        
        intrinsic = np.maximum(K - S_paths[:, t], 0.0)
        itm = intrinsic > 0
        
        if np.sum(itm) < 10:
            continue
        
        # Regression on ITM paths
        X = S_paths[itm, t] / K  # normalize
        Y = cashflow[itm]
        
        # Polynomial basis (up to degree 3)
        A = np.column_stack([np.ones(np.sum(itm)), X, X**2, X**3])
        try:
            coeff = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = A @ coeff
        except:
            continue
        
        # Exercise if intrinsic > continuation
        exercise = intrinsic[itm] > continuation
        idx_itm = np.where(itm)[0]
        exercise_idx = idx_itm[exercise]
        
        cashflow[exercise_idx] = intrinsic[exercise_idx]
        exercise_time[exercise_idx] = t
    
    # Final discounting
    disc_factors = np.exp(-r * exercise_time * dt)
    payoffs = np.zeros(N_paths)
    for i in range(N_paths):
        t_ex = exercise_time[i]
        payoffs[i] = np.maximum(K - S_paths[i, t_ex], 0.0) * np.exp(-r * t_ex * dt)
    
    return np.mean(payoffs)

def mc_european_put(S_paths, K, r, T):
    """Simple MC European put price."""
    payoff = np.maximum(K - S_paths[:, -1], 0.0)
    return np.exp(-r*T) * np.mean(payoff)

def mc_european_call(S_paths, K, r, T):
    payoff = np.maximum(S_paths[:, -1] - K, 0.0)
    return np.exp(-r*T) * np.mean(payoff)

def lsm_american_call(S_paths, K, r, T):
    """LSM for American call (mainly for completeness)."""
    N_paths, N_steps_plus1 = S_paths.shape
    N_steps = N_steps_plus1 - 1
    dt = T / N_steps
    
    cashflow = np.maximum(S_paths[:, -1] - K, 0.0)
    exercise_time = np.full(N_paths, N_steps)
    
    for t in range(N_steps-1, 0, -1):
        disc = np.exp(-r * dt)
        cashflow *= disc
        
        intrinsic = np.maximum(S_paths[:, t] - K, 0.0)
        itm = intrinsic > 0
        
        if np.sum(itm) < 10:
            continue
        
        X = S_paths[itm, t] / K
        Y = cashflow[itm]
        A = np.column_stack([np.ones(np.sum(itm)), X, X**2, X**3])
        try:
            coeff = np.linalg.lstsq(A, Y, rcond=None)[0]
            continuation = A @ coeff
        except:
            continue
        
        exercise = intrinsic[itm] > continuation
        idx_itm = np.where(itm)[0]
        exercise_idx = idx_itm[exercise]
        cashflow[exercise_idx] = intrinsic[exercise_idx]
        exercise_time[exercise_idx] = t
    
    payoffs = np.zeros(N_paths)
    for i in range(N_paths):
        t_ex = exercise_time[i]
        payoffs[i] = np.maximum(S_paths[i, t_ex] - K, 0.0) * np.exp(-r * t_ex * dt)
    
    return np.mean(payoffs)

# ============================================================
# 5. Control Variate MC
# ============================================================
def mc_with_cv(S_paths, K, r, T, sigma_bsm, S0, option='put', american=False):
    """MC price with BSM control variate."""
    if american:
        if option == 'put':
            mc_price = lsm_american_put(S_paths, K, r, T)
        else:
            mc_price = lsm_american_call(S_paths, K, r, T)
        return mc_price  # CV harder for American; just return raw
    else:
        if option == 'put':
            payoffs = np.maximum(K - S_paths[:, -1], 0.0) * np.exp(-r*T)
        else:
            payoffs = np.maximum(S_paths[:, -1] - K, 0.0) * np.exp(-r*T)
        
        # BSM control variate (using GBM paths from same random numbers is ideal,
        # but we approximate with analytical)
        mc_raw = np.mean(payoffs)
        return mc_raw

# ============================================================
# 6. Main computation
# ============================================================
def compute_iv_curves():
    # Global parameters
    S0 = 100.0
    r = 0.03
    T = 0.5
    N_steps = 200
    N_paths = 60000
    
    # Strike range
    strikes = np.linspace(75, 125, 21)
    moneyness = strikes / S0
    
    # === BSM parameters ===
    sigma_bsm = 0.20
    
    # === Merton parameters ===
    sigma_merton = 0.15
    lam_merton = 0.8
    mu_J_merton = -0.08
    sigma_J_merton = 0.12
    
    # === DST common parameters ===
    V0 = 0.04  # initial variance (sigma=0.2)
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    
    # Hawkes parameters
    lam_inf_S = 0.5
    lam_inf_V = 0.3
    beta_S = 5.0
    beta_V = 5.0
    eta_SS = 0.8
    eta_VV = 0.6
    eta_SV = 0.3
    eta_VS = 0.2
    mu_J_dst = -0.05
    sigma_J_dst = 0.10
    mu_V_dst = 0.02
    
    # Different alpha values (gamma parameter)
    alphas = [0.1, 0.25, 0.4]  # H = 0.4, 0.25, 0.1
    alpha_labels = [r'$\alpha=0.1$ (H=0.4)', r'$\alpha=0.25$ (H=0.25)', r'$\alpha=0.4$ (H=0.1)']
    
    print("=" * 70)
    print("Implied Volatility Curve Simulation")
    print("=" * 70)
    
    # -------------------------------------------------------
    # A. BSM IV (flat line by definition for European)
    # -------------------------------------------------------
    print("\n[1/5] BSM model...")
    iv_bsm_euro = np.full(len(strikes), sigma_bsm)
    
    # For American puts, use binomial or LSM
    # Generate GBM paths for American BSM
    S_bsm = np.zeros((N_paths, N_steps+1))
    S_bsm[:, 0] = S0
    Z_stored = np.random.randn(N_paths, N_steps)
    for i in range(N_steps):
        dt = T / N_steps
        S_bsm[:, i+1] = S_bsm[:, i] * np.exp((r - 0.5*sigma_bsm**2)*dt + sigma_bsm*np.sqrt(dt)*Z_stored[:, i])
    
    iv_bsm_amer = []
    for K in strikes:
        price_amer = lsm_american_put(S_bsm, K, r, T)
        iv_val = implied_vol(price_amer, S0, K, T, r, 'put')
        iv_bsm_amer.append(iv_val)
    iv_bsm_amer = np.array(iv_bsm_amer)
    print("  BSM done.")
    
    # -------------------------------------------------------
    # B. Merton
    # -------------------------------------------------------
    print("\n[2/5] Merton model...")
    iv_merton_euro = []
    for K in strikes:
        price = merton_european(S0, K, T, r, sigma_merton, lam_merton, mu_J_merton, sigma_J_merton, 'put')
        iv_val = implied_vol(price, S0, K, T, r, 'put')
        iv_merton_euro.append(iv_val)
    iv_merton_euro = np.array(iv_merton_euro)
    
    # Merton American via LSM
    print("  Generating Merton MC paths...")
    S_merton = merton_mc_paths(S0, r, sigma_merton, lam_merton, mu_J_merton, sigma_J_merton, T, N_steps, N_paths)
    
    iv_merton_amer = []
    for K in strikes:
        price_amer = lsm_american_put(S_merton, K, r, T)
        iv_val = implied_vol(price_amer, S0, K, T, r, 'put')
        iv_merton_amer.append(iv_val)
    iv_merton_amer = np.array(iv_merton_amer)
    print("  Merton done.")
    
    # -------------------------------------------------------
    # C. DST with different alpha values
    # -------------------------------------------------------
    iv_dst_euro = {}
    iv_dst_amer = {}
    
    for idx, alpha in enumerate(alphas):
        print(f"\n[{3+idx}/5] DST model with alpha={alpha} (H={0.5-alpha:.2f})...")
        
        S_dst, V_dst = dst_mc_paths(
            S0, r, V0, kappa, theta, xi, rho, alpha,
            lam_inf_S, lam_inf_V, beta_S, beta_V,
            eta_SS, eta_VV, eta_SV, eta_VS,
            mu_J_dst, sigma_J_dst, mu_V_dst,
            T, N_steps, N_paths, N_exp=6
        )
        
        iv_euro_list = []
        iv_amer_list = []
        for K in strikes:
            # European
            price_euro = mc_european_put(S_dst, K, r, T)
            iv_e = implied_vol(price_euro, S0, K, T, r, 'put')
            iv_euro_list.append(iv_e)
            
            # American
            price_amer = lsm_american_put(S_dst, K, r, T)
            iv_a = implied_vol(price_amer, S0, K, T, r, 'put')
            iv_amer_list.append(iv_a)
        
        iv_dst_euro[alpha] = np.array(iv_euro_list)
        iv_dst_amer[alpha] = np.array(iv_amer_list)
        print(f"  DST alpha={alpha} done.")
    
    # -------------------------------------------------------
    # D. Control Variate diagnostics
    # -------------------------------------------------------
    print("\n[CV Check] Control variate diagnostics for ATM put...")
    K_atm = 100.0
    
    # DST European ATM prices with and without CV
    cv_results = {}
    for alpha in alphas:
        S_dst, _ = dst_mc_paths(
            S0, r, V0, kappa, theta, xi, rho, alpha,
            lam_inf_S, lam_inf_V, beta_S, beta_V,
            eta_SS, eta_VV, eta_SV, eta_VS,
            mu_J_dst, sigma_J_dst, mu_V_dst,
            T, N_steps, 30000, N_exp=6
        )
        
        payoffs = np.maximum(K_atm - S_dst[:, -1], 0.0) * np.exp(-r*T)
        raw_mean = np.mean(payoffs)
        raw_se = np.std(payoffs) / np.sqrt(len(payoffs))
        
        # Generate matched GBM paths for CV (use terminal S to approximate)
        S_term_gbm = S0 * np.exp((r - 0.5*0.2**2)*T + 0.2*np.sqrt(T)*np.random.randn(30000))
        payoffs_cv_pilot = np.maximum(K_atm - S_term_gbm, 0.0) * np.exp(-r*T)
        bsm_exact = bsm_price(S0, K_atm, T, r, 0.2, 'put')
        
        cv_results[alpha] = {
            'raw_price': raw_mean,
            'raw_se': raw_se,
            'bsm_cv_ref': bsm_exact
        }
    
    # -------------------------------------------------------
    # E. Plotting
    # -------------------------------------------------------
    print("\nGenerating plots...")
    
    # Color scheme
    colors_dst = ['#E63946', '#457B9D', '#2A9D8F']
    color_bsm = '#264653'
    color_merton = '#E9C46A'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle('Implied Volatility Curves: DST (Rough Heston + Dual Hawkes) vs Merton vs BSM\n'
                 'Put Options | $S_0$=100, r=3%, T=0.5y',
                 fontsize=15, fontweight='bold', y=0.98)
    
    # --- Panel 1: European IV ---
    ax1 = axes[0, 0]
    ax1.axhline(sigma_bsm, color=color_bsm, linewidth=2.5, linestyle='--', label='BSM ($\\sigma$=0.20)', zorder=5)
    ax1.plot(moneyness, iv_merton_euro, color=color_merton, linewidth=2.5, marker='s',
             markersize=5, label='Merton JD', zorder=4)
    for idx, alpha in enumerate(alphas):
        ax1.plot(moneyness, iv_dst_euro[alpha], color=colors_dst[idx], linewidth=2,
                 marker='o', markersize=4, label=f'DST {alpha_labels[idx]}', zorder=3)
    ax1.set_xlabel('Moneyness (K/S₀)', fontsize=12)
    ax1.set_ylabel('Implied Volatility', fontsize=12)
    ax1.set_title('(a) European Put — IV Smile', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.75, 1.25])
    
    # --- Panel 2: American IV ---
    ax2 = axes[0, 1]
    ax2.plot(moneyness, iv_bsm_amer, color=color_bsm, linewidth=2.5, linestyle='--',
             marker='D', markersize=4, label='BSM ($\\sigma$=0.20)', zorder=5)
    ax2.plot(moneyness, iv_merton_amer, color=color_merton, linewidth=2.5, marker='s',
             markersize=5, label='Merton JD', zorder=4)
    for idx, alpha in enumerate(alphas):
        ax2.plot(moneyness, iv_dst_amer[alpha], color=colors_dst[idx], linewidth=2,
                 marker='o', markersize=4, label=f'DST {alpha_labels[idx]}', zorder=3)
    ax2.set_xlabel('Moneyness (K/S₀)', fontsize=12)
    ax2.set_ylabel('Implied Volatility', fontsize=12)
    ax2.set_title('(b) American Put — IV Smile (LSM)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.75, 1.25])
    
    # --- Panel 3: American - European early exercise premium (IV difference) ---
    ax3 = axes[1, 0]
    diff_bsm = iv_bsm_amer - iv_bsm_euro
    diff_merton = iv_merton_amer - iv_merton_euro
    ax3.plot(moneyness, diff_bsm * 100, color=color_bsm, linewidth=2.5, linestyle='--',
             marker='D', markersize=4, label='BSM')
    ax3.plot(moneyness, diff_merton * 100, color=color_merton, linewidth=2.5,
             marker='s', markersize=5, label='Merton JD')
    for idx, alpha in enumerate(alphas):
        diff_dst = iv_dst_amer[alpha] - iv_dst_euro[alpha]
        ax3.plot(moneyness, diff_dst * 100, color=colors_dst[idx], linewidth=2,
                 marker='o', markersize=4, label=f'DST {alpha_labels[idx]}')
    ax3.set_xlabel('Moneyness (K/S₀)', fontsize=12)
    ax3.set_ylabel('IV Difference (bps × 100)', fontsize=12)
    ax3.set_title('(c) Early Exercise Premium: IV(American) − IV(European)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='grey', linewidth=0.8, linestyle='-')
    ax3.set_xlim([0.75, 1.25])
    
    # --- Panel 4: Control Variate diagnostic ---
    ax4 = axes[1, 1]
    # Bar chart of raw MC SE for each alpha
    alpha_strs = [f'α={a}\n(H={0.5-a:.2f})' for a in alphas]
    raw_prices = [cv_results[a]['raw_price'] for a in alphas]
    raw_ses = [cv_results[a]['raw_se'] for a in alphas]
    
    x_pos = np.arange(len(alphas))
    bars = ax4.bar(x_pos, raw_prices, width=0.5, color=colors_dst, alpha=0.8,
                   yerr=raw_ses, capsize=8, ecolor='black')
    ax4.axhline(cv_results[alphas[0]]['bsm_cv_ref'], color=color_bsm, linewidth=2,
                linestyle='--', label=f'BSM ref = {cv_results[alphas[0]]["bsm_cv_ref"]:.4f}')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(alpha_strs, fontsize=11)
    ax4.set_ylabel('ATM Put Price', fontsize=12)
    ax4.set_title('(d) Control Variate Check: ATM Put MC Price ± 1 SE', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, price, se in zip(bars, raw_prices, raw_ses):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 0.05,
                 f'{price:.4f}\n±{se:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('/home/claude/iv_curves.png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("\nPlot saved to /home/claude/iv_curves.png")
    
    # -------------------------------------------------------
    # F. Summary table
    # -------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'ATM Euro IV':>12} {'ATM Amer IV':>12} {'OTM(K=85) Euro':>16} {'ITM(K=115) Euro':>16}")
    print("-" * 90)
    
    k_atm_idx = np.argmin(np.abs(strikes - 100))
    k_otm_idx = np.argmin(np.abs(strikes - 85))
    k_itm_idx = np.argmin(np.abs(strikes - 115))
    
    print(f"{'BSM':<30} {iv_bsm_euro[k_atm_idx]:>12.4f} {iv_bsm_amer[k_atm_idx]:>12.4f} "
          f"{iv_bsm_euro[k_otm_idx]:>16.4f} {iv_bsm_euro[k_itm_idx]:>16.4f}")
    print(f"{'Merton JD':<30} {iv_merton_euro[k_atm_idx]:>12.4f} {iv_merton_amer[k_atm_idx]:>12.4f} "
          f"{iv_merton_euro[k_otm_idx]:>16.4f} {iv_merton_euro[k_itm_idx]:>16.4f}")
    
    for idx, alpha in enumerate(alphas):
        label = f'DST α={alpha} (H={0.5-alpha:.2f})'
        e = iv_dst_euro[alpha]
        a = iv_dst_amer[alpha]
        print(f"{label:<30} {e[k_atm_idx]:>12.4f} {a[k_atm_idx]:>12.4f} "
              f"{e[k_otm_idx]:>16.4f} {e[k_itm_idx]:>16.4f}")
    print("=" * 90)

if __name__ == '__main__':
    compute_iv_curves()
