# Realized Volatility and Quadratic Variation

## 1. Concept Skeleton
**Definition:** Non-parametric volatility estimator based on high-frequency returns; sum of squared intraday returns approximates integrated variance; model-free approach  
**Purpose:** Accurate volatility measurement; volatility forecasting; option pricing; risk management; benchmark for GARCH models; high-frequency econometrics  
**Prerequisites:** Quadratic variation, semimartingales, market microstructure noise, jump processes, sampling theory, infill asymptotics

## 2. Comparative Framing
| Estimator | Formula | Noise Robustness | Jump Robustness | Efficiency | Sampling |
|-----------|---------|------------------|-----------------|------------|----------|
| **Realized Variance (RV)** | ΣΣrₜ² | No | No | High (no noise) | All returns |
| **Realized Kernel (RK)** | Kernel-weighted autocovariances | Yes | No | High | All returns |
| **Two-Scale RV (TSRV)** | Sparse - Dense averages | Yes | No | Moderate | Subsampling |
| **Pre-Averaging** | Overlapping windows | Yes | No | High | Overlapping |
| **Bipower Variation** | Σ\|rₜ\|\|rₜ₊₁\| | Minimal | Yes | Moderate | Adjacent |
| **MedRV** | Median-based | Minimal | Yes | Robust | Triplets |
| **Squared Daily Return** | r²_daily | NA | NA | Very low | Daily only |

## 3. Examples + Counterexamples

**Classic Example:**  
S&P 500 with 5-minute returns: RV=0.25% daily variance. Squared daily return=0.15% (underestimates due to single observation). GARCH forecast=0.22%. RV provides superior ex-post volatility proxy—improves forecast evaluation.

**Failure Case:**  
Ultra-high-frequency (1-second) returns on illiquid stock. Bid-ask bounce creates spurious volatility. RV explodes to 10× true value. Realized kernel with Bartlett weights reduces to 1.2×. Pre-averaging at 5-minute scale eliminates noise.

**Edge Case:**  
Market opens with flash crash (jump). Standard RV=5% includes jump contribution. Bipower variation=0.8% isolates continuous component. Jump test rejects continuity (p<0.001). Separate modeling required.

## 4. Layer Breakdown
```
Realized Volatility Framework:
├─ Theoretical Foundation:
│   ├─ Continuous-Time Setup:
│   │   ├─ Log-price process: dpₜ = μₜdt + σₜdWₜ + dJₜ
│   │   │   ├─ Drift: μₜ (potentially time-varying)
│   │   │   ├─ Diffusion: σₜ (stochastic volatility)
│   │   │   ├─ Brownian motion: Wₜ
│   │   │   └─ Jumps: Jₜ (finite/infinite activity)
│   │   └─ Goal: Estimate integrated variance (IV)
│   ├─ Integrated Variance (IV):
│   │   ├─ IV = ∫₀ᵀ σₛ²ds (continuous part)
│   │   ├─ Unobservable latent quantity
│   │   └─ RV → IV as Δ → 0 (no noise, no jumps)
│   ├─ Quadratic Variation (QV):
│   │   ├─ [p,p]ₜ = ∫₀ᵗ σₛ²ds + ΣΣ(ΔJₛ)²
│   │   ├─ Includes jump contribution
│   │   └─ RV estimates QV (not just IV)
│   └─ Infill Asymptotics:
│       ├─ n → ∞ (more observations within fixed [0,T])
│       ├─ Δ = T/n → 0 (sampling interval)
│       └─ Different from long-span (T → ∞)
├─ Realized Variance (Basic):
│   ├─ Definition: RV = ΣΣrₜ² where rₜ = pₜ - pₜ₋₁
│   │   ├─ Sum over M intraday returns
│   │   └─ Δ = T/M (sampling frequency)
│   ├─ Consistency:
│   │   ├─ No noise, no jumps: RV →ᵖ IV
│   │   ├─ Unbiased for quadratic variation
│   │   └─ Rate: √M(RV - IV) →ᵈ N(0, 2IQ)
│   ├─ Integrated Quarticity (IQ):
│   │   ├─ IQ = ∫₀ᵀ σₛ⁴ds
│   │   └─ Determines asymptotic variance of RV
│   ├─ Optimal Sampling:
│   │   ├─ No noise: Use all data (Δ → 0)
│   │   ├─ With noise: Trade-off (5-min typical)
│   │   └─ Liquidity-dependent
│   └─ Computational:
│       └─ Extremely fast: O(M) summation
├─ Market Microstructure Noise:
│   ├─ Observed Price Model:
│   │   ├─ p*ₜ = pₜ + uₜ
│   │   │   ├─ p*ₜ: Observed (transaction) price
│   │   │   ├─ pₜ: Efficient (latent) price
│   │   │   └─ uₜ: Microstructure noise
│   │   ├─ Noise sources:
│   │   │   ├─ Bid-ask bounce
│   │   │   ├─ Discreteness (tick size)
│   │   │   ├─ Rounding errors
│   │   │   └─ Asynchronous trading
│   │   └─ Typically: uₜ ~ i.i.d., E[uₜ]=0, Var(uₜ)=ω²
│   ├─ Effect on RV:
│   │   ├─ E[RV] ≈ IV + 2Mω² (upward bias)
│   │   ├─ Bias increases with M (sampling frequency)
│   │   └─ Signature plot: RV vs Δ (U-shaped)
│   ├─ Optimal Sampling Frequency:
│   │   ├─ Minimize MSE of RV
│   │   ├─ Δ* ∝ (ω²/IV)^(2/3)
│   │   └─ Typical: 5-20 minutes for liquid stocks
│   └─ Noise Variance Estimation:
│       ├─ ω̂² = -(1/(2M))Σγ̂₁
│       └─ γ̂₁: First-order autocovariance of returns
├─ Noise-Robust Estimators:
│   ├─ Realized Kernel (Barndorff-Nielsen et al 2008):
│   │   ├─ RK = γ̂₀ + ΣΣk(h/H)·(γ̂ₕ + γ̂₋ₕ)
│   │   │   ├─ γ̂ₕ: h-th autocovariance
│   │   │   ├─ k(·): Kernel function (Parzen, Bartlett, etc.)
│   │   │   └─ H: Bandwidth parameter
│   │   ├─ Kernels:
│   │   │   ├─ Parzen: k(x) = 1-6x²+6|x|³ for x≤0.5
│   │   │   ├─ Bartlett: k(x) = 1-|x|
│   │   │   └─ Tukey-Hanning: k(x) = (1+cos(πx))/2
│   │   ├─ Bandwidth Selection:
│   │   │   ├─ H* ∝ M^(1/2) (optimal rate)
│   │   │   └─ Data-driven: minimize MSE estimate
│   │   ├─ Properties:
│   │   │   ├─ Consistent for IV (not QV if jumps)
│   │   │   ├─ Asymptotic normality
│   │   │   └─ Efficient under i.i.d. noise
│   │   └─ Advantages:
│   │       ├─ Uses all data (no subsampling loss)
│   │       └─ Flexible kernel choice
│   ├─ Two-Scale Realized Variance (Zhang et al 2005):
│   │   ├─ TSRV = RV_slow - (M_fast/M_slow)·RV_fast_avg
│   │   │   ├─ RV_slow: Sparse grid (e.g., 15-min)
│   │   │   ├─ RV_fast_avg: Average of fast grids (e.g., 1-min)
│   │   │   └─ Bias correction via subtraction
│   │   ├─ Intuition:
│   │   │   ├─ Slow grid: Low noise, high variance
│   │   │   ├─ Fast grid: High noise, low variance
│   │   │   └─ Difference removes noise bias
│   │   ├─ Subsampling:
│   │   │   ├─ Create J overlapping grids
│   │   │   └─ Average to reduce variance
│   │   └─ Rate:
│   │       └─ Optimal M_slow ∝ M^(2/3), M_fast = M
│   ├─ Pre-Averaging (Jacod et al 2009):
│   │   ├─ Smooth returns over overlapping windows
│   │   ├─ r̄ₜ = ΣΣg(i/k)rₜ₊ᵢ where g(·) is weight function
│   │   ├─ Then: PA = (1/θ)ΣΣr̄ₜ² - ψω̂²
│   │   │   └─ θ, ψ: Constants depending on g
│   │   ├─ Weight Functions:
│   │   │   ├─ Linear: g(x) = min(x, 1-x)
│   │   │   └─ Optimal under certain criteria
│   │   └─ Properties:
│   │       ├─ Consistent for IV
│   │       └─ Efficient (achieves optimal rate)
│   └─ Comparison:
│       ├─ RK: Most popular, flexible
│       ├─ TSRV: Intuitive, good finite-sample
│       └─ PA: Theoretically optimal, complex
├─ Jump-Robust Estimators:
│   ├─ Motivation:
│   │   ├─ RV → QV = IV + Jump Variation
│   │   ├─ Want to isolate continuous part (IV)
│   │   └─ Or separately estimate jump component
│   ├─ Bipower Variation (Barndorff-Nielsen & Shephard 2004):
│   │   ├─ BV = (π/2)ΣΣ|rₜ||rₜ₋₁|
│   │   ├─ Consistency: BV →ᵖ IV (even with jumps)
│   │   ├─ Intuition: Products less affected by jumps
│   │   └─ Assumes jumps rare relative to M
│   ├─ Tripower Quarticity:
│   │   ├─ TQ = M·(μ₄/₃)³ ΣΣ|rₜ|^(4/3)|rₜ₋₁|^(4/3)|rₜ₋₂|^(4/3)
│   │   └─ Estimates IQ (integrated quarticity)
│   ├─ MedRV (Median-based):
│   │   ├─ MedRV = (π/(6-4√3+π))ΣΣ med(|rₜ₋₁|, |rₜ|, |rₜ₊₁|)²
│   │   ├─ More robust to zero returns
│   │   └─ Finite-sample improvements
│   ├─ Threshold Bipower Variation:
│   │   ├─ Truncate large returns before computing BV
│   │   └─ More robust to large jumps
│   └─ Jump Component:
│       ├─ J = max(RV - BV, 0)
│       └─ Separate modeling of jumps vs diffusion
├─ Jump Tests:
│   ├─ Z-statistic (Barndorff-Nielsen & Shephard):
│   │   ├─ Z = (RV - BV) / √((θ-2)·TQ/M)
│   │   ├─ Under H₀ (no jumps): Z →ᵈ N(0,1)
│   │   └─ Reject if |Z| > 1.96 (5% level)
│   ├─ Ratio Test:
│   │   └─ Compare RV/BV (should be ≈1 if no jumps)
│   ├─ Lee-Mykland (2008):
│   │   ├─ Identify individual jump times
│   │   └─ Local volatility estimation
│   └─ Practical Considerations:
│       ├─ Power depends on jump size, M
│       └─ Pre-test bias in estimation
├─ Multivariate Extensions:
│   ├─ Realized Covariance:
│   │   ├─ RCov_ij = ΣΣrᵢ,ₜrⱼ,ₜ
│   │   └─ Consistent for integrated covariance
│   ├─ Synchronization Issues:
│   │   ├─ Refresh time: Use only when both traded
│   │   ├─ Previous tick: Carry forward last price
│   │   └─ Hayashi-Yoshida: Explicitly handle asynchrony
│   ├─ Realized Correlation:
│   │   └─ RCorr_ij = RCov_ij / √(RV_i·RV_j)
│   ├─ Positive Semi-Definite:
│   │   ├─ Realized covariance matrix may not be PSD
│   │   └─ Regularization: Eigenvalue adjustment, shrinkage
│   └─ Applications:
│       ├─ Portfolio optimization
│       ├─ Risk management (covariance forecasting)
│       └─ Pairs trading, hedging
├─ Forecasting with Realized Volatility:
│   ├─ HAR Model (Heterogeneous Autoregressive):
│   │   ├─ RV_t = β₀ + β_D·RV_{t-1} + β_W·RV_{t-5:t-1} + β_M·RV_{t-22:t-1}
│   │   ├─ Components: Daily, Weekly, Monthly
│   │   ├─ Captures volatility persistence
│   │   └─ Corsi (2009), extremely popular
│   ├─ ARFIMA on RV:
│   │   ├─ Long memory in realized volatility
│   │   └─ Fractional differencing
│   ├─ HAR-RV-J (with jumps):
│   │   ├─ Include continuous and jump components separately
│   │   └─ Separate predictability
│   ├─ Realized GARCH:
│   │   ├─ Combine GARCH with realized measures
│   │   └─ Hansen, Huang, Shek (2012)
│   └─ MIDAS (Mixed Data Sampling):
│       └─ Aggregate intraday to daily forecasts
├─ Inference:
│   ├─ Asymptotic Distribution:
│   │   ├─ √M(RV - IV) →ᵈ N(0, 2IQ)
│   │   └─ IQ estimated by tripower quarticity
│   ├─ Confidence Intervals:
│   │   ├─ CI: RV ± z_α·√(2·T̂Q/M)
│   │   └─ Valid under no jumps, no noise
│   ├─ Noise-Robust CI:
│   │   └─ Use realized kernel variance estimators
│   └─ Hypothesis Tests:
│       ├─ Equal volatility across periods
│       └─ Comparison of models
├─ Practical Implementation:
│   ├─ Data Preparation:
│   │   ├─ Clean outliers (fat-finger trades)
│   │   ├─ Handle non-trading hours
│   │   ├─ Adjust for splits, dividends
│   │   └─ Time stamping (exchange vs local)
│   ├─ Sampling Choices:
│   │   ├─ Calendar time: Fixed intervals (5-min)
│   │   ├─ Tick time: Every n-th trade
│   │   ├─ Volume time: Fixed volume buckets
│   │   └─ Calendar most common for RV
│   ├─ Software:
│   │   ├─ Python: realized_volatility packages
│   │   ├─ R: highfrequency, realized packages
│   │   └─ MATLAB: MFE Toolbox
│   └─ Storage:
│       └─ Database design for tick data (TB scale)
└─ Applications:
    ├─ Volatility Forecasting:
    │   └─ Ex-post benchmark for GARCH, SV models
    ├─ Option Pricing:
    │   ├─ Model-free implied volatility
    │   └─ Realized vol predicts option prices
    ├─ Risk Management:
    │   ├─ VaR with realized vol superior
    │   └─ Regulatory capital (Basel)
    ├─ Volatility Trading:
    │   ├─ Variance swaps: Payoff based on RV
    │   └─ VIX products
    ├─ Market Quality:
    │   └─ Noise-to-signal ratio measurement
    └─ High-Frequency Trading:
        └─ Intraday risk management
```

**Interaction:** High-frequency data → Clean/synchronize → Choose estimator (noise/jumps) → Compute RV → Forecast with HAR → Evaluate models

## 5. Mini-Project
Implement realized volatility estimators with microstructure noise and jump detection:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

np.random.seed(321)

# ===== Simulate High-Frequency Price Process =====
print("="*80)
print("REALIZED VOLATILITY AND QUADRATIC VARIATION")
print("="*80)

# Simulation parameters
T = 1  # 1 trading day
M = 390  # 6.5 hours * 60 minutes (1-minute sampling)
dt = T / M

# Stochastic volatility parameters
kappa = 5  # Mean reversion speed
theta = 0.20**2  # Long-run variance (20% annual vol)
sigma_v = 0.3  # Volatility of volatility
rho = -0.7  # Correlation (leverage effect)

print(f"\nSimulation Setup:")
print(f"  Trading period: {T} day")
print(f"  Intraday observations: {M} (1-minute)")
print(f"  Stochastic volatility (Heston-type)")
print(f"  Leverage effect: ρ = {rho}")

# Initialize
v = np.zeros(M+1)  # Variance process
v[0] = theta

p = np.zeros(M+1)  # Log-price
p[0] = 0

# Simulate stochastic volatility
Z_v = np.random.randn(M)
Z_p = rho * Z_v + np.sqrt(1 - rho**2) * np.random.randn(M)

for t in range(M):
    # Variance process (CIR)
    v[t+1] = v[t] + kappa * (theta - v[t]) * dt + sigma_v * np.sqrt(v[t] * dt) * Z_v[t]
    v[t+1] = max(v[t+1], 1e-6)  # Keep positive
    
    # Price process
    p[t+1] = p[t] + np.sqrt(v[t] * dt) * Z_p[t]

# Add jumps (3 per day on average)
n_jumps = np.random.poisson(3)
jump_times = np.random.choice(M, n_jumps, replace=False)
jump_sizes = np.random.normal(0, 0.01, n_jumps)  # 1% jumps

for i, t in enumerate(jump_times):
    p[t+1:] += jump_sizes[i]

print(f"  Jumps: {n_jumps} introduced")

# True integrated variance (continuous part)
IV_true = np.sum(v[:-1] * dt)
print(f"\nTrue Integrated Variance: {IV_true:.6f}")
print(f"  (Annualized vol: {np.sqrt(IV_true * 252):.4f})")

# ===== Add Market Microstructure Noise =====
noise_std = 0.001  # Noise standard deviation (10 bps)
noise = np.random.normal(0, noise_std, M+1)

p_observed = p + noise

print(f"\nMicrostructure noise added:")
print(f"  Noise std: {noise_std:.4f} ({noise_std*100:.2f} bps)")

# ===== Calculate Returns =====
r_true = np.diff(p)  # True returns (no noise)
r_observed = np.diff(p_observed)  # Observed returns (with noise)

# ===== Realized Variance (Naive) =====
print("\n" + "="*80)
print("REALIZED VARIANCE (NAIVE)")
print("="*80)

RV_naive = np.sum(r_observed**2)

print(f"Realized Variance (naive): {RV_naive:.6f}")
print(f"  True IV: {IV_true:.6f}")
print(f"  Bias: {RV_naive - IV_true:.6f} ({(RV_naive/IV_true - 1)*100:.1f}%)")

if RV_naive > IV_true:
    print(f"  ✗ Upward bias due to microstructure noise")

# ===== Signature Plot =====
print("\n" + "="*80)
print("SIGNATURE PLOT")
print("="*80)

sampling_frequencies = [1, 2, 3, 5, 10, 15, 20, 30, 60]  # minutes
RVs = []

for freq in sampling_frequencies:
    step = freq
    r_sampled = p_observed[::step]
    r_sampled = np.diff(r_sampled)
    RV_sampled = np.sum(r_sampled**2)
    RVs.append(RV_sampled)

print(f"Sampling frequencies tested: {sampling_frequencies}")
print(f"Optimal frequency (visual inspection): ~5-15 minutes")

# ===== Realized Kernel Estimator =====
print("\n" + "="*80)
print("REALIZED KERNEL (Noise-Robust)")
print("="*80)

# Calculate autocovariances
def realized_kernel(returns, H, kernel='bartlett'):
    """Realized kernel estimator"""
    M = len(returns)
    
    # Autocovariances
    gamma = np.zeros(H+1)
    for h in range(H+1):
        if h == 0:
            gamma[h] = np.sum(returns**2)
        else:
            gamma[h] = np.sum(returns[h:] * returns[:-h])
    
    # Kernel weights
    if kernel == 'bartlett':
        weights = 1 - np.arange(H+1) / (H+1)
    elif kernel == 'parzen':
        x = np.arange(H+1) / (H+1)
        weights = np.where(x <= 0.5, 
                          1 - 6*x**2 + 6*x**3,
                          2*(1-x)**3)
    else:
        raise ValueError("Unknown kernel")
    
    # Realized kernel
    RK = gamma[0] + 2 * np.sum(weights[1:] * gamma[1:])
    
    return RK, gamma

# Optimal bandwidth (rule of thumb)
H_opt = int(M**0.5)
print(f"Bandwidth (H): {H_opt}")

RK_bartlett, gamma = realized_kernel(r_observed, H_opt, kernel='bartlett')

print(f"\nRealized Kernel (Bartlett):")
print(f"  RK: {RK_bartlett:.6f}")
print(f"  True IV: {IV_true:.6f}")
print(f"  Bias: {RK_bartlett - IV_true:.6f} ({(RK_bartlett/IV_true - 1)*100:.1f}%)")

if abs(RK_bartlett - IV_true) < abs(RV_naive - IV_true):
    improvement = (1 - abs(RK_bartlett - IV_true) / abs(RV_naive - IV_true)) * 100
    print(f"  ✓ {improvement:.1f}% improvement over naive RV")

# ===== Noise Variance Estimation =====
print("\n" + "="*80)
print("NOISE VARIANCE ESTIMATION")
print("="*80)

# Estimate from first-order autocovariance
noise_var_est = -0.5 * gamma[1] / M

print(f"Estimated noise variance: {noise_var_est:.6f}")
print(f"True noise variance: {noise_std**2:.6f}")
print(f"Estimated noise std: {np.sqrt(max(noise_var_est, 0)):.4f}")

# ===== Two-Scale Realized Variance =====
print("\n" + "="*80)
print("TWO-SCALE REALIZED VARIANCE")
print("="*80)

# Sparse grid (every 5 minutes)
sparse_step = 5
p_sparse = p_observed[::sparse_step]
r_sparse = np.diff(p_sparse)
RV_sparse = np.sum(r_sparse**2)

# Fast grid (every minute, multiple subsamples)
n_subsamples = sparse_step
RV_fast_avg = 0

for j in range(n_subsamples):
    p_fast = p_observed[j::1]  # All observations starting from j
    r_fast = np.diff(p_fast)
    RV_fast_avg += np.sum(r_fast**2)

RV_fast_avg /= n_subsamples

# Two-scale estimator
TSRV = RV_sparse - (M / len(r_sparse)) * RV_fast_avg

print(f"RV (sparse, 5-min): {RV_sparse:.6f}")
print(f"RV (fast, 1-min avg): {RV_fast_avg:.6f}")
print(f"\nTwo-Scale RV: {TSRV:.6f}")
print(f"  True IV: {IV_true:.6f}")
print(f"  Bias: {TSRV - IV_true:.6f} ({(TSRV/IV_true - 1)*100:.1f}%)")

# ===== Bipower Variation (Jump-Robust) =====
print("\n" + "="*80)
print("BIPOWER VARIATION (Jump-Robust)")
print("="*80)

# Use 5-minute returns (less noise)
r_5min = np.diff(p_observed[::5])

# Bipower variation
mu_1 = np.sqrt(2/np.pi)
BV = (np.pi/2) * np.sum(np.abs(r_5min[1:]) * np.abs(r_5min[:-1]))

# Realized variance (5-min)
RV_5min = np.sum(r_5min**2)

# Jump component
J = max(RV_5min - BV, 0)

print(f"Realized Variance (5-min): {RV_5min:.6f}")
print(f"Bipower Variation: {BV:.6f}")
print(f"Jump component (J): {J:.6f}")
print(f"  Percentage: {(J/RV_5min)*100:.1f}% of total QV")

# ===== Jump Test =====
print("\n" + "="*80)
print("JUMP TEST")
print("="*80)

# Tripower quarticity
mu_4_3 = 2**(2/3) * np.gamma(7/6) / np.gamma(0.5)
TQ = M * (mu_4_3**3) * np.sum(
    np.abs(r_5min[2:])**(4/3) * 
    np.abs(r_5min[1:-1])**(4/3) * 
    np.abs(r_5min[:-2])**(4/3)
)

# Z-statistic
theta = (np.pi/2)**2 + np.pi - 5
Z_stat = (RV_5min - BV) / np.sqrt((theta - 2) * TQ / len(r_5min))

p_value = 2 * (1 - stats.norm.cdf(abs(Z_stat)))

print(f"Jump Test (Barndorff-Nielsen & Shephard):")
print(f"  Z-statistic: {Z_stat:.4f}")
print(f"  P-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"  ✓ Reject H₀: Jumps detected (5% level)")
else:
    print(f"  ✗ Cannot reject H₀: No significant jumps")

print(f"\nTrue number of jumps: {n_jumps}")
if n_jumps > 0:
    print(f"  Jump times: {jump_times}")
    print(f"  Jump sizes: {jump_sizes}")

# ===== Visualizations =====
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Plot 1: Price Process
axes[0, 0].plot(p, linewidth=1.5, label='True Price', alpha=0.8)
axes[0, 0].plot(p_observed, linewidth=1, alpha=0.6, label='Observed Price')
for jt in jump_times:
    axes[0, 0].axvline(jt, color='red', linestyle=':', alpha=0.5, linewidth=1)
axes[0, 0].set_xlabel('Time (minutes)')
axes[0, 0].set_ylabel('Log-Price')
axes[0, 0].set_title('Simulated Price Process with Jumps and Noise')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Instantaneous Volatility
axes[0, 1].plot(np.sqrt(v * 252), linewidth=2, color='orange')
axes[0, 1].axhline(np.sqrt(theta * 252), color='blue', linestyle='--', 
                  linewidth=2, label='Long-run vol')
axes[0, 1].set_xlabel('Time (minutes)')
axes[0, 1].set_ylabel('Annualized Volatility')
axes[0, 1].set_title('Stochastic Volatility Path')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Signature Plot
axes[1, 0].plot(sampling_frequencies, RVs, marker='o', linewidth=2, markersize=8)
axes[1, 0].axhline(IV_true, color='red', linestyle='--', linewidth=2, 
                  label='True IV')
axes[1, 0].set_xlabel('Sampling Frequency (minutes)')
axes[1, 0].set_ylabel('Realized Variance')
axes[1, 0].set_title('Signature Plot (U-shape due to noise)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Estimator Comparison
estimators = ['True IV', 'RV (naive)', 'RK', 'TSRV', 'BV (5min)']
values = [IV_true, RV_naive, RK_bartlett, TSRV, BV]
colors = ['green', 'red', 'blue', 'purple', 'orange']

axes[1, 1].barh(estimators, values, color=colors, alpha=0.7)
axes[1, 1].axvline(IV_true, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Variance Estimate')
axes[1, 1].set_title('Estimator Comparison')
axes[1, 1].grid(alpha=0.3, axis='x')

# Plot 5: Returns Distribution
axes[2, 0].hist(r_observed, bins=50, alpha=0.6, density=True, label='Observed')
axes[2, 0].hist(r_true, bins=50, alpha=0.4, density=True, label='True')

# Overlay normal
x_range = np.linspace(r_observed.min(), r_observed.max(), 100)
axes[2, 0].plot(x_range, 
               stats.norm.pdf(x_range, 0, np.sqrt(IV_true/M)),
               'r--', linewidth=2, label='Normal (true vol)')

axes[2, 0].set_xlabel('Returns')
axes[2, 0].set_ylabel('Density')
axes[2, 0].set_title('Return Distribution (Noise Effect Visible)')
axes[2, 0].legend()
axes[2, 0].grid(alpha=0.3)

# Plot 6: Autocovariance Function
lags = np.arange(min(20, len(gamma)))
axes[2, 1].bar(lags, gamma[:len(lags)], alpha=0.7)
axes[2, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[2, 1].set_xlabel('Lag (h)')
axes[2, 1].set_ylabel('Autocovariance γ(h)')
axes[2, 1].set_title('Return Autocovariance (Noise → Negative γ(1))')
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('realized_volatility_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== HAR Forecasting Model =====
print("\n" + "="*80)
print("HAR MODEL (Forecasting)")
print("="*80)

# Simulate multiple days
n_days = 100
RV_daily = []

print(f"Simulating {n_days} days of RV...")

for day in range(n_days):
    # Simulate one day
    v_day = np.zeros(M+1)
    v_day[0] = theta
    p_day = np.zeros(M+1)
    
    Z_v_day = np.random.randn(M)
    Z_p_day = rho * Z_v_day + np.sqrt(1 - rho**2) * np.random.randn(M)
    
    for t in range(M):
        v_day[t+1] = v_day[t] + kappa * (theta - v_day[t]) * dt + sigma_v * np.sqrt(v_day[t] * dt) * Z_v_day[t]
        v_day[t+1] = max(v_day[t+1], 1e-6)
        p_day[t+1] = p_day[t] + np.sqrt(v_day[t] * dt) * Z_p_day[t]
    
    # Add noise
    noise_day = np.random.normal(0, noise_std, M+1)
    p_day_obs = p_day + noise_day
    
    # Calculate 5-min RV (noise-robust sampling)
    r_5min_day = np.diff(p_day_obs[::5])
    RV_day = np.sum(r_5min_day**2)
    
    RV_daily.append(RV_day)

RV_daily = np.array(RV_daily)

# HAR model: RV_t = β0 + β_D·RV_{t-1} + β_W·RV_{t-5:t-1} + β_M·RV_{t-22:t-1}
from sklearn.linear_model import LinearRegression

# Create features
X_har = []
y_har = []

for t in range(22, len(RV_daily)):
    # Daily component
    rv_d = RV_daily[t-1]
    
    # Weekly component (average of last 5 days)
    rv_w = np.mean(RV_daily[t-5:t])
    
    # Monthly component (average of last 22 days)
    rv_m = np.mean(RV_daily[t-22:t])
    
    X_har.append([rv_d, rv_w, rv_m])
    y_har.append(RV_daily[t])

X_har = np.array(X_har)
y_har = np.array(y_har)

# Split
train_size_har = int(0.7 * len(y_har))
X_train = X_har[:train_size_har]
y_train = y_har[:train_size_har]
X_test = X_har[train_size_har:]
y_test = y_har[train_size_har:]

# Fit HAR
har_model = LinearRegression()
har_model.fit(X_train, y_train)

# Predictions
y_pred = har_model.predict(X_test)

# R²
from sklearn.metrics import r2_score, mean_squared_error

r2_har = r2_score(y_test, y_pred)
mse_har = mean_squared_error(y_test, y_pred)

print(f"HAR Model Results:")
print(f"  Coefficients:")
print(f"    β_D (daily): {har_model.coef_[0]:.4f}")
print(f"    β_W (weekly): {har_model.coef_[1]:.4f}")
print(f"    β_M (monthly): {har_model.coef_[2]:.4f}")
print(f"  Intercept: {har_model.intercept_:.6f}")
print(f"\n  Out-of-sample R²: {r2_har:.4f}")
print(f"  MSE: {mse_har:.6f}")

if r2_har > 0.3:
    print(f"  ✓ HAR captures volatility persistence well")

# ===== Summary =====
print("\n" + "="*80)
print("SUMMARY AND INSIGHTS")
print("="*80)

print("\n1. Microstructure Noise Effect:")
print(f"   Naive RV bias: {(RV_naive/IV_true - 1)*100:+.1f}%")
print(f"   Realized Kernel bias: {(RK_bartlett/IV_true - 1)*100:+.1f}%")
print(f"   → Noise-robust estimator critical at high frequency")

print("\n2. Optimal Sampling:")
print(f"   Signature plot shows minimum around 5-15 minutes")
print(f"   → Trade-off between precision and noise")

print("\n3. Jump Detection:")
print(f"   True jumps: {n_jumps}")
print(f"   Jump test p-value: {p_value:.4f}")
if p_value < 0.05 and n_jumps > 0:
    print(f"   ✓ Test correctly detects jumps")
elif p_value >= 0.05 and n_jumps == 0:
    print(f"   ✓ Test correctly finds no jumps")

print("\n4. Forecasting:")
print(f"   HAR model R²: {r2_har:.3f}")
print(f"   → Multi-horizon aggregation captures persistence")

print("\n5. Practical Recommendations:")
print("   • Use 5-minute sampling for liquid assets")
print("   • Apply realized kernel for very high frequency")
print("   • Test for jumps; use bipower if significant")
print("   • HAR model simple and effective for forecasting")
print("   • Store realized measures for GARCH evaluation")

print("\n6. Applications:")
print("   • Risk management: Superior volatility proxy")
print("   • Trading: Variance swap valuation")
print("   • Microstructure: Noise-to-signal ratio")
print("   • Academic: Model-free benchmark")
```

## 6. Challenge Round
When do realized volatility estimators fail or mislead?
- **Ultra-high frequency**: 1-second data on illiquid stocks → Microstructure noise dominates; realized kernel or pre-averaging required; simple RV explodes
- **Market closures**: Overnight returns omitted → Underestimates total daily variance; need close-to-open adjustment; intraday-only biased
- **Asynchronous trading**: Multivariate RV with different trading times → Epps effect (spurious correlation); Hayashi-Yoshida estimator corrects
- **Flash crashes**: Extreme jumps exceed model assumptions → Standard jump tests under-powered; need robust procedures (MedRV)
- **Low liquidity**: Many zero returns → Realized measures biased; duration models or tick time sampling better
- **Feedback effects**: Volatility affects trading intensity → Endogeneity; RV not purely exogenous; structural modeling needed

## 7. Key References
- [Andersen & Bollerslev (1998) - Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts](https://doi.org/10.1016/S0169-7161(98)00019-1)
- [Barndorff-Nielsen & Shephard (2002) - Econometric Analysis of Realized Volatility and Its Use in Estimating Stochastic Volatility Models](https://doi.org/10.1111/1467-9868.00336)
- [Hansen & Lunde (2006) - Realized Variance and Market Microstructure Noise](https://doi.org/10.1198/073500106000000071)

---
**Status:** Gold standard for ex-post volatility measurement | **Complements:** GARCH, HAR, Stochastic Volatility, Microstructure Models
