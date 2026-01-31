
# Block 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# =====================================
# REAL-WORLD VS RISK-NEUTRAL SIMULATION
# =====================================
print("="*70)
print("REAL-WORLD (ℙ) vs RISK-NEUTRAL (ℚ) MEASURE COMPARISON")
print("="*70)

np.random.seed(42)

# Market parameters
S0 = 100.0
mu_physical = 0.10  # 10% expected return (real-world drift)
sigma = 0.20  # 20% volatility (same under ℙ and ℚ)
r = 0.05  # 5% risk-free rate
T = 1.0  # 1 year horizon

# Market price of risk
theta = (mu_physical - r) / sigma  # (10% - 5%) / 20% = 0.25

print("\nMarket Parameters:")
print(f"   S₀ = ${S0}")
print(f"   μ (real-world drift) = {mu_physical:.1%}")
print(f"   σ (volatility) = {sigma:.1%}")
print(f"   r (risk-free rate) = {r:.1%}")
print(f"   θ (market price of risk) = {theta:.2f} (Sharpe ratio)")

# =====================================
# MONTE CARLO SIMULATION
# =====================================
n_scenarios = 10000
n_steps = 252  # Daily steps
dt = T / n_steps
time_grid = np.linspace(0, T, n_steps + 1)

print(f"\nSimulation: {n_scenarios:,} scenarios, {n_steps} time steps")

# Simulate under ℙ (real-world measure)
S_physical = np.zeros((n_scenarios, n_steps + 1))
S_physical[:, 0] = S0

for t in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_scenarios)
    S_physical[:, t+1] = S_physical[:, t] * np.exp((mu_physical - 0.5*sigma**2)*dt + sigma*dW)

# Simulate under ℚ (risk-neutral measure)
np.random.seed(42)  # Same random numbers for comparison
S_risk_neutral = np.zeros((n_scenarios, n_steps + 1))
S_risk_neutral[:, 0] = S0

for t in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_scenarios)
    S_risk_neutral[:, t+1] = S_risk_neutral[:, t] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)

# Terminal distributions
S_T_physical = S_physical[:, -1]
S_T_risk_neutral = S_risk_neutral[:, -1]

print("\n" + "="*70)
print("TERMINAL STOCK PRICE DISTRIBUTIONS")
print("="*70)

print(f"\nℙ-Measure (Real-World):")
print(f"   Mean: ${np.mean(S_T_physical):.2f}")
print(f"   Std Dev: ${np.std(S_T_physical):.2f}")
print(f"   Theoretical Mean: ${S0 * np.exp(mu_physical * T):.2f}")
print(f"   Median: ${np.median(S_T_physical):.2f}")

print(f"\nℚ-Measure (Risk-Neutral):")
print(f"   Mean: ${np.mean(S_T_risk_neutral):.2f}")
print(f"   Std Dev: ${np.std(S_T_risk_neutral):.2f}")
print(f"   Theoretical Mean: ${S0 * np.exp(r * T):.2f}")
print(f"   Median: ${np.median(S_T_risk_neutral):.2f}")

difference = np.mean(S_T_physical) - np.mean(S_T_risk_neutral)
print(f"\nDifference (ℙ - ℚ): ${difference:.2f} ({difference/S0:.1%} of S₀)")

# =====================================
# OPTION PRICING
# =====================================
print("\n" + "="*70)
print("OPTION PRICING: BLACK-SCHOLES vs MONTE CARLO")
print("="*70)

K = 100  # At-the-money strike

# Black-Scholes formula (uses ℚ-measure)
def black_scholes_call(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

bs_price = black_scholes_call(S0, K, r, sigma, T)

# Monte Carlo under ℚ (correct pricing measure)
payoffs_q = np.maximum(S_T_risk_neutral - K, 0)
mc_price_q = np.exp(-r*T) * np.mean(payoffs_q)

# Monte Carlo under ℙ (WRONG for pricing, but shows forecasting)
payoffs_p = np.maximum(S_T_physical - K, 0)
mc_price_p_wrong = np.exp(-r*T) * np.mean(payoffs_p)  # Incorrect (uses wrong measure)
mc_price_p_correct = np.mean(payoffs_p) / (1 + mu_physical)**T  # Risk-adjusted discount

print(f"\nCall Option (K=${K}):")
print(f"   Black-Scholes: ${bs_price:.4f}")
print(f"   Monte Carlo (ℚ): ${mc_price_q:.4f} (correct, error ${abs(mc_price_q-bs_price):.4f})")
print(f"\n   Monte Carlo (ℙ, r-discounted): ${mc_price_p_wrong:.4f} (WRONG - mispriced!)")
print(f"   Monte Carlo (ℙ, μ-discounted): ${mc_price_p_correct:.4f} (risk-adjusted, but not market price)")

# =====================================
# DELTA HEDGING SIMULATION
# =====================================
print("\n" + "="*70)
print("DELTA HEDGING: P&L UNDER ℙ vs ℚ")
print("="*70)

# Delta-hedged portfolio: Short 1 call, long Δ shares
# Under ℚ: E^ℚ[P&L] = 0 (risk-neutral, fair pricing)
# Under ℙ: E^ℙ[P&L] ≠ 0 (depends on μ > r)

def delta_bs(S, K, r, sigma, T):
    """Black-Scholes delta."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# Simulate delta-hedged P&L
n_hedge_scenarios = 1000
pnl_physical = []
pnl_risk_neutral = []

for i in range(n_hedge_scenarios):
    # Initial position: short call at BS price, hedge with Δ shares
    initial_delta = delta_bs(S0, K, r, sigma, T)
    position_value_0 = -bs_price + initial_delta * S0
    
    # Terminal value (simplified: no rebalancing during path)
    # ℙ-scenario
    S_T_p = S_physical[i, -1]
    payoff_p = np.maximum(S_T_p - K, 0)
    position_value_T_p = -payoff_p + initial_delta * S_T_p
    pnl_p = position_value_T_p - position_value_0
    pnl_physical.append(pnl_p)
    
    # ℚ-scenario
    S_T_q = S_risk_neutral[i, -1]
    payoff_q = np.maximum(S_T_q - K, 0)
    position_value_T_q = -payoff_q + initial_delta * S_T_q
    pnl_q = position_value_T_q - position_value_0
    pnl_risk_neutral.append(pnl_q)

print(f"\nDelta-Hedged Portfolio (Short Call + Long {initial_delta:.3f} Shares):")
print(f"   Expected P&L under ℙ: ${np.mean(pnl_physical):.4f} (should be > 0 if μ > r)")
print(f"   Expected P&L under ℚ: ${np.mean(pnl_risk_neutral):.4f} (should be ≈ 0)")
print(f"   Std Dev (ℙ): ${np.std(pnl_physical):.4f}")
print(f"   Std Dev (ℚ): ${np.std(pnl_risk_neutral):.4f}")

# =====================================
# GIRSANOV TRANSFORMATION VISUALIZATION
# =====================================
print("\n" + "="*70)
print("GIRSANOV TRANSFORMATION")
print("="*70)

# Show how W^ℚ relates to W^ℙ
# W^ℚ_t = W^ℙ_t + θ t, where θ = (μ - r)/σ

# Sample paths
n_sample_paths = 5
np.random.seed(10)
W_P = np.zeros((n_sample_paths, n_steps + 1))
for i in range(n_sample_paths):
    for t in range(n_steps):
        W_P[i, t+1] = W_P[i, t] + np.random.normal(0, np.sqrt(dt))

# Girsanov shift
W_Q = W_P + theta * time_grid

print(f"\nGirsanov Shift: W^ℚ = W^ℙ + θ·t")
print(f"   θ = {theta:.4f}")
print(f"   Shift at T=1Y: {theta * T:.4f}")
print(f"   Interpretation: ℚ-Brownian motion drifts by {theta:.2f}σ per year")

# =====================================
# VISUALIZATION
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sample Paths
n_plot = 20
for i in range(n_plot):
    axes[0, 0].plot(time_grid, S_physical[i, :], alpha=0.3, color='blue')
    axes[0, 0].plot(time_grid, S_risk_neutral[i, :], alpha=0.3, color='red')

axes[0, 0].plot([], [], color='blue', linewidth=2, label='ℙ-Measure (μ=10%)')
axes[0, 0].plot([], [], color='red', linewidth=2, label='ℚ-Measure (μ=r=5%)')
axes[0, 0].axhline(S0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].set_xlabel('Time (years)')
axes[0, 0].set_ylabel('Stock Price ($)')
axes[0, 0].set_title(f'Sample Paths: ℙ vs ℚ ({n_plot} scenarios)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Terminal Distribution
axes[0, 1].hist(S_T_physical, bins=50, alpha=0.5, label='ℙ-Measure', density=True, color='blue', edgecolor='black')
axes[0, 1].hist(S_T_risk_neutral, bins=50, alpha=0.5, label='ℚ-Measure', density=True, color='red', edgecolor='black')
axes[0, 1].axvline(np.mean(S_T_physical), color='blue', linestyle='--', linewidth=2, label=f'ℙ Mean ${np.mean(S_T_physical):.0f}')
axes[0, 1].axvline(np.mean(S_T_risk_neutral), color='red', linestyle='--', linewidth=2, label=f'ℚ Mean ${np.mean(S_T_risk_neutral):.0f}')
axes[0, 1].set_xlabel('Terminal Stock Price S_T ($)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title(f'Terminal Distribution (T={T}Y)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Girsanov Transformation (Brownian Motions)
for i in range(n_sample_paths):
    axes[1, 0].plot(time_grid, W_P[i, :], alpha=0.6, color='blue')
    axes[1, 0].plot(time_grid, W_Q[i, :], alpha=0.6, color='red')

axes[1, 0].plot([], [], color='blue', linewidth=2, label='W^ℙ (Physical)')
axes[1, 0].plot([], [], color='red', linewidth=2, label='W^ℚ (Risk-Neutral)')
axes[1, 0].plot(time_grid, theta * time_grid, 'k--', linewidth=2, label=f'Drift θ·t (θ={theta:.2f})')
axes[1, 0].set_xlabel('Time (years)')
axes[1, 0].set_ylabel('Brownian Motion Value')
axes[1, 0].set_title(f'Girsanov Transformation: W^ℚ = W^ℙ + {theta:.2f}·t')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Delta-Hedged P&L Distributions
axes[1, 1].hist(pnl_physical, bins=30, alpha=0.6, label='ℙ-Measure', color='blue', edgecolor='black', density=True)
axes[1, 1].hist(pnl_risk_neutral, bins=30, alpha=0.6, label='ℚ-Measure', color='red', edgecolor='black', density=True)
axes[1, 1].axvline(np.mean(pnl_physical), color='blue', linestyle='--', linewidth=2, label=f'ℙ Mean ${np.mean(pnl_physical):.2f}')
axes[1, 1].axvline(np.mean(pnl_risk_neutral), color='red', linestyle='--', linewidth=2, label=f'ℚ Mean ${np.mean(pnl_risk_neutral):.2f}')
axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Delta-Hedged P&L ($)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Delta-Hedged Portfolio P&L')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"ℙ vs ℚ comparison complete:")
print(f"• ℙ-forecast (μ={mu_physical:.0%}): E[S_T] = ${np.mean(S_T_physical):.2f}")
print(f"• ℚ-forecast (r={r:.0%}): E[S_T] = ${np.mean(S_T_risk_neutral):.2f}")
print(f"• Drift difference: ${difference:.2f} over {T} year ({(mu_physical-r)*100:.0f}% per year)")
print(f"• Option pricing: Use ℚ → ${bs_price:.4f} (ℙ gives ${mc_price_p_wrong:.4f}, mispriced)")
print(f"• Market price of risk: θ = {theta:.2f} (Girsanov shift)")
print(f"• Delta-hedged P&L: ℙ ${np.mean(pnl_physical):.2f} vs ℚ ${np.mean(pnl_risk_neutral):.2f} (should be ~0)")