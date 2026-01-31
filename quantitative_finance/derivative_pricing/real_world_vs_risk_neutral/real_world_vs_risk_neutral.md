# Real-World vs Risk-Neutral Pricing

## 1. Concept Skeleton
**Definition:** Distinguish physical probability measure ℙ (real-world) from risk-neutral measure ℚ (pricing), related via Girsanov theorem and market price of risk  
**Purpose:** Price derivatives using risk-neutral probabilities (no-arbitrage), while forecasting/risk management uses real-world probabilities  
**Prerequisites:** Stochastic calculus, Brownian motion, Radon-Nikodym derivative, martingale theory, arbitrage-free pricing

## 2. Comparative Framing
| Measure | Real-World (ℙ) | Risk-Neutral (ℚ) |
|---------|---------------|------------------|
| **Purpose** | Forecasting, risk analysis, expected returns | Derivative pricing (no-arbitrage) |
| **Drift** | μ (expected return, e.g., 8-12% equity) | r (risk-free rate, e.g., 5%) |
| **Volatility** | σ_physical (calibrate to historical) | σ_implied (calibrate to option prices) |
| **Discounting** | Risk-adjusted rate (μ or WACC) | Risk-free rate r |
| **Example** | E^ℙ[S_T] = S_0 e^{μT} (forecast stock price) | E^ℚ[e^{-rT} Payoff] (option price) |

## 3. Examples + Counterexamples

**Simple Example:**  
Stock S₀=$100, μ=10%, σ=20%, r=5% → Real-world forecast E^ℙ[S₁Y]=$110, but risk-neutral E^ℚ[S₁Y]=$105.13 (used for option pricing)

**Failure Case:**  
Using real-world μ to price options → Call overpriced (drift too high) → arbitrage via delta-hedging exploits mispricing

**Edge Case:**  
Derivatives on non-traded assets (e.g., temperature, electricity): No unique ℚ (incomplete market) → multiple equivalent martingale measures, model choice matters

## 4. Layer Breakdown
```
Real-World vs Risk-Neutral Pricing:
├─ Probability Measures:
│   ├─ Physical Measure ℙ (Real-World):
│   │   ├─ Definition: True probabilities governing asset dynamics in reality
│   │   ├─ Purpose: Forecasting, VaR, stress testing, expected portfolio returns
│   │   ├─ Equity Dynamics: dS/S = μ dt + σ dW^ℙ
│   │   │   where μ = expected return (historical: 8-12% for equity)
│   │   ├─ Estimation: Historical calibration (regression, MLE from time series)
│   │   └─ Example: Forecast S₁Y ~ LogNormal(S₀ e^{μT}, σ²T)
│   ├─ Risk-Neutral Measure ℚ (Pricing Measure):
│   │   ├─ Definition: Artificial probabilities making discounted assets martingales
│   │   ├─ Purpose: Derivative pricing via no-arbitrage (Law of One Price)
│   │   ├─ Equity Dynamics: dS/S = r dt + σ dW^ℚ
│   │   │   where r = risk-free rate (replace μ with r)
│   │   ├─ Key Property: e^{-rt} S_t is a ℚ-martingale → E^ℚ[S_T | F_t] = S_t e^{r(T-t)}
│   │   └─ Pricing Formula: V₀ = E^ℚ[e^{-rT} Payoff(S_T)]
│   └─ Relationship:
│       Change of measure via Radon-Nikodym derivative (Girsanov theorem)
├─ Girsanov Theorem:
│   ├─ Statement: Change Brownian motion drift without changing volatility
│   │   dW^ℚ = dW^ℙ + θ dt (θ = market price of risk)
│   │   where θ = (μ - r) / σ (Sharpe ratio adjusted for risk-free rate)
│   ├─ Measure Change:
│   │   dℚ/dℙ = exp(-½∫₀ᵀ θ²dt - ∫₀ᵀ θ dW^ℙ) (Radon-Nikodym derivative)
│   │   ├─ Under ℚ: W^ℚ_t = W^ℙ_t + ∫₀ᵗ θ ds is ℚ-Brownian motion
│   │   └─ Novikov Condition: E[exp(½∫₀ᵀ θ²dt)] < ∞ (ensures valid measure)
│   ├─ Application to Equity:
│   │   ℙ: dS = μ S dt + σ S dW^ℙ
│   │   ℚ: dS = r S dt + σ S dW^ℚ (replace μ with r, W^ℙ with W^ℚ)
│   ├─ Market Price of Risk:
│   │   θ = (μ - r) / σ (compensation per unit of volatility)
│   │   ├─ Positive θ: Risk-averse investors demand premium over r
│   │   └─ Empirical: θ ≈ 0.2-0.4 for equity (Sharpe ratio)
│   └─ Multi-Dimensional:
│       For n risk factors: dW^ℚ = dW^ℙ + Θ dt (vector θ, correlation preserved)
├─ Fundamental Theorem of Asset Pricing:
│   ├─ First FTAP: No arbitrage ⟺ ∃ equivalent martingale measure ℚ
│   │   Under ℚ: Discounted asset prices are martingales
│   ├─ Second FTAP: Market completeness ⟺ unique ℚ
│   │   ├─ Complete: All payoffs replicable (e.g., Black-Scholes with stock + bond)
│   │   └─ Incomplete: Multiple ℚ (e.g., stochastic vol, jump models)
│   └─ Implications:
│       Derivative price = Expected payoff under ℚ, discounted at r
├─ Black-Scholes Framework:
│   ├─ ℙ-Dynamics: dS = μ S dt + σ S dW^ℙ
│   ├─ ℚ-Dynamics: dS = r S dt + σ S dW^ℚ
│   ├─ Option Pricing:
│   │   C(S,t) = E^ℚ[e^{-r(T-t)} max(S_T - K, 0) | S_t = S]
│   │   = S Φ(d₁) - K e^{-r(T-t)} Φ(d₂) (Black-Scholes formula)
│   │   where d₁ = [ln(S/K) + (r+½σ²)(T-t)] / (σ√(T-t))
│   ├─ Implied Volatility:
│   │   σ_implied calibrated to option prices under ℚ (forward-looking)
│   │   ≠ σ_historical estimated from returns under ℙ
│   └─ PDE Derivation:
│       Δ-hedging eliminates risk → risk-free return → no μ appears in PDE
│       ∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
├─ Forward vs Spot Measure:
│   ├─ Spot Measure (ℚ): Numeraire = money market account B_t = e^{rt}
│   │   Price: V₀ = E^ℚ[e^{-rT} Payoff]
│   ├─ Forward Measure (ℚ^T): Numeraire = zero-coupon bond P(t,T)
│   │   Price: V₀ = P(0,T) E^{ℚ^T}[Payoff]
│   │   ├─ Advantage: No discounting needed under ℚ^T
│   │   └─ Use: Interest rate derivatives (caps, swaptions)
│   ├─ Stock Measure (ℚ^S): Numeraire = stock S_t
│   │   Use: Quantos, change-of-numeraire techniques
│   └─ Change of Numeraire Formula:
│       dℚ^T/dℚ = [B_T / P(T,T)] / [B_0 / P(0,T)] (Radon-Nikodym derivative)
├─ Incomplete Markets:
│   ├─ Sources of Incompleteness:
│   │   ├─ Stochastic Volatility: σ_t random, not traded → can't hedge vol risk
│   │   ├─ Jumps: Infinite hedging frequency required → impractical
│   │   ├─ Non-Traded Assets: Weather, credit, energy → no replicating portfolio
│   │   └─ Transaction Costs: Continuous hedging not feasible
│   ├─ Consequence: Multiple ℚ satisfy no-arbitrage → derivative price range
│   │   V_min = inf_{ℚ∈M} E^ℚ[e^{-rT} Payoff]
│   │   V_max = sup_{ℚ∈M} E^ℚ[e^{-rT} Payoff]
│   │   where M = set of equivalent martingale measures
│   ├─ Approaches:
│   │   ├─ Utility Indifference: Price such that agent indifferent to trading
│   │   ├─ Good-Deal Bounds: Restrict Sharpe ratio to reasonable range
│   │   └─ Minimal Entropy: Choose ℚ closest to ℙ (minimize KL divergence)
│   └─ Example (Stochastic Vol):
│       Heston model has non-unique ℚ → specify market price of vol risk λ_v
│       dv = κ(θ - v)dt + σ_v √v (dW^ℙ_v + λ_v dt) (ℙ-dynamics)
│       dv = κ(θ - v - λ_v σ_v √v)dt + σ_v √v dW^ℚ_v (ℚ-dynamics)
├─ Practical Differences:
│   ├─ Option Pricing:
│   │   ℚ: Use implied vol σ_implied, drift = r
│   │   Example: σ_implied = 25% (from market), σ_historical = 20% (from data)
│   │   → Price with 25% vol, not 20%
│   ├─ VaR / Risk Management:
│   │   ℙ: Use historical μ and σ, simulate realistic scenarios
│   │   Example: 1-day 99% VaR uses ℙ to forecast portfolio loss distribution
│   ├─ Monte Carlo:
│   │   ℙ: dS = μ S dt + σ S dW^ℙ (forecast paths)
│   │   ℚ: dS = r S dt + σ S dW^ℚ (pricing paths)
│   │   → Same volatility σ, different drift
│   └─ Hedging:
│       Delta Δ = ∂V/∂S same under ℙ and ℚ (local property)
│       But expected P&L differs: E^ℙ[ΔP&L] ≠ 0, E^ℚ[ΔP&L] = 0 (ℚ is pricing measure)
├─ Quanto Adjustments:
│   ├─ Problem: Derivative payoff in foreign currency, settle in domestic
│   │   Example: European call on Nikkei (¥ payoff) settled in USD
│   ├─ Standard Pricing: E^{ℚ_¥}[e^{-r_¥ T} Payoff_¥] converted at spot FX
│   ├─ Quanto Pricing: Fix FX rate at inception (no FX risk)
│   │   Quanto_Call = E^{ℚ_$}[e^{-r_$ T} max(S_T^{¥/$} - K, 0)]
│   │   Requires change of measure ℚ_¥ → ℚ_$
│   ├─ Quanto Adjustment:
│   │   Drift adjusted by: μ_quanto = μ - ρ σ_S σ_FX
│   │   where ρ = correlation(Stock, FX), σ_FX = FX volatility
│   └─ Application: Quanto CDS, quanto equity options common in structured products
└─ Calibration:
    ├─ ℙ-Calibration (Historical):
    │   ├─ Data: Time series of returns r_t = ln(S_{t+1}/S_t)
    │   ├─ μ estimate: Sample mean μ̂ = (1/n)Σ r_t
    │   ├─ σ estimate: Sample std dev σ̂ = √Var(r_t)
    │   └─ Adjustment: Annualize (μ_annual = μ_daily × 252, σ_annual = σ_daily × √252)
    ├─ ℚ-Calibration (Market):
    │   ├─ Input: Option prices across strikes and maturities
    │   ├─ Implied Vol: Invert Black-Scholes to extract σ_implied(K,T)
    │   ├─ Model Parameters: Calibrate Heston, SABR to match option surface
    │   └─ Risk-Free Rate: Use OIS or Treasury curve (r from market)
    └─ Hybrid:
        Use ℙ for μ, ℚ for σ → convert between measures via Girsanov
```

**Interaction:** Calibrate ℙ from historical data (μ, σ) → Define market price of risk θ = (μ-r)/σ → Apply Girsanov to obtain ℚ-dynamics (replace μ with r) → Price derivatives under ℚ using Monte Carlo or PDE

## 5. Mini-Project
Compare real-world forecasting vs risk-neutral option pricing with Monte Carlo:
```python
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
```

## 6. Challenge Round
Why can't we use real-world probabilities ℙ to price options?
- **Arbitrage:** Delta-hedging creates risk-free portfolio → must earn r, not μ → pricing PDE has no μ term (eliminated by hedging argument)
- **Risk preferences:** Different investors have different μ expectations (bulls vs bears) → no unique price without no-arbitrage framework
- **Replication:** Option payoff replicated by dynamic trading (stock + bond) → value = cost of replication independent of μ
- **Martingale property:** Under ℚ, discounted prices are martingales (fair game) → E^ℚ[e^{-rT} S_T] = S_0, simplifies math
- **Market completeness:** Black-Scholes assumes complete market → unique ℚ, any derivative replicable
- **Empirical failure:** Using ℙ with historical μ=10% drastically overprices calls (arbitrageurs would sell), market converges to ℚ-price

Reality: ℚ is artificial construct ensuring no-arbitrage, ℙ retained for risk management (VaR, stress tests, expected returns forecasting), but pricing must use ℚ.

## 7. Key References
- [Shreve (2004) Stochastic Calculus for Finance II, Ch. 5](https://link.springer.com/book/10.1007/978-1-4939-2867-8) - Girsanov theorem, change of measure, FTAP proofs
- [Björk (2009) Arbitrage Theory in Continuous Time, Ch. 10-11](https://oxford.universitypressscholarship.com/view/10.1093/oso/9780198851615.001.0001/oso-9780198851615) - Change of numeraire, forward measures
- [Musiela & Rutkowski (2005) Martingale Methods in Financial Modelling](https://link.springer.com/book/10.1007/b137866) - Advanced measure theory, incomplete markets
- [Duffie (2001) Dynamic Asset Pricing Theory, Ch. 6](https://press.princeton.edu/books/hardcover/9780691139852/dynamic-asset-pricing-theory) - Risk-neutral valuation foundations

---
**Status:** Foundational pricing theory | **Complements:** Black-Scholes, Monte Carlo, Stochastic calculus
