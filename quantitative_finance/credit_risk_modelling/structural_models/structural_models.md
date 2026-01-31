# Structural Models (Merton Model)

## 1. Concept Skeleton
**Definition:** Default occurs when firm asset value falls below debt level; PD derived from asset value dynamics and capital structure  
**Purpose:** Theory-driven approach linking default probability to firm economics; market-implied PD from equity volatility  
**Prerequisites:** Option pricing theory, stochastic processes, balance sheet structure, equity volatility

## 2. Comparative Framing
| Model | Input Data | PD Driver | Calibration | Interpretation |
|-------|-----------|-----------|------------|-----------------|
| **Merton/Structural** | Equity vol, leverage | Asset > Debt threshold | Equity price → asset vol | Economic intuition |
| **Scorecard** | Credit attributes | Default odds from variables | Historical default rates | Statistical pattern |
| **CDS-Implied** | Market spreads | Market consensus | Option-adjusted spread | Forward-looking |
| **Transition Matrix** | Credit ratings | Historical migration | Historical rating changes | Empirical regularity |

## 3. Examples + Counterexamples

**Simple Example:**  
Firm: Assets $100M, Debt $60M, Asset volatility 20%, risk-free rate 3%. Black-Scholes: 1-year PD ≈ 2%

**Failure Case:**  
Merton underestimates distress PD; model assumes continuous asset value (no jumps). 2008 Lehman collapse: gap risk missed

**Edge Case:**  
Negative equity (liabilities > assets); Merton formula breaks. Financial crisis: Multiple firms with negative book equity

## 4. Layer Breakdown
```
Merton Structural Model Framework:
├─ Core Concept:
│   ├─ Firm modeled as call option on assets
│   ├─ Equity = max(Assets - Debt, 0)
│   ├─ Default when Assets < Debt at maturity
│   └─ Firm value follows geometric Brownian motion
├─ Mathematical Framework:
│   ├─ dV/V = μdt + σdW  (asset dynamics)
│   ├─ D = default barrier (debt level)
│   ├─ T = time to maturity
│   ├─ Distance to Default (DD) = (ln(V/D) + (μ - σ²/2)T) / (σ√T)
│   └─ PD = N(-DD)  (N = standard normal CDF)
├─ Implementation Steps:
│   ├─ 1. Estimate current asset value V₀
│   ├─ 2. Estimate asset volatility σ
│   ├─ 3. Define default barrier D (debt)
│   ├─ 4. Calculate distance to default
│   ├─ 5. Convert DD to PD using normal distribution
│   └─ 6. Validate against market CDS spread
├─ Key Parameters:
│   ├─ Asset value (V): From balance sheet or equity → assets (inverse problem)
│   ├─ Asset volatility (σ): From equity vol via Itô lemma
│   ├─ Debt (D): Book value or market value
│   ├─ Time horizon (T): Usually 1 year for comparison
│   └─ Risk-free rate (r): Government bond yield
├─ Two-Way Linkages:
│   ├─ Equity = call option: Equity_value = Assets × N(d₁) - Debt × e^(-rT) × N(d₂)
│   ├─ Leverage effect: Higher debt → Higher PD
│   └─ Volatility feedback: Higher asset vol → Higher PD
└─ Extensions:
    ├─ Multi-period: Simulate paths, calculate default probability
    ├─ Stochastic rates: Interest rates vary over time
    ├─ Jump risk: Asset value can jump down (gap risk)
    └─ Barrier models: Early warning when DD crosses threshold
```

## 5. Mini-Project
Build and calibrate Merton model:
```python
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(42)

# Example: Firm balance sheet data
print("=== Merton Structural Model ===")

# Firm data
equity_price = 50  # Market cap: $50 per share
shares_outstanding = 10  # 10M shares
equity_market_value = equity_price * shares_outstanding

book_debt = 150  # $150M debt (book value)
market_debt = 140  # $140M (estimated market value, may differ)

# Market data
equity_volatility = 0.35  # Stock returns volatility: 35%
risk_free_rate = 0.03  # 3% risk-free rate
time_horizon = 1.0  # 1-year horizon

print(f"Firm Equity Market Value: ${equity_market_value}M")
print(f"Firm Debt (Book): ${book_debt}M")
print(f"Equity Volatility: {equity_volatility:.1%}")

# Inverse problem: Estimate asset value and volatility from equity
# Equity = call option on assets
# Need to solve: E = A × N(d₁) - D × e^(-rT) × N(d₂)
# Also: σ_E × E = σ_A × A × N(d₁)

def merton_equations(params, E, D, sigma_E, r, T):
    """
    Solve for asset value A and volatility sigma_A
    Equations:
    1. E = A*N(d1) - D*exp(-rT)*N(d2)
    2. sigma_E * E = sigma_A * A * N(d1)
    """
    A, sigma_A = params
    
    if A <= D or sigma_A <= 0:
        return [1e10, 1e10]
    
    d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    call_value = A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    nd1_value = norm.cdf(d1) * A
    
    eq1 = call_value - E
    eq2 = sigma_E * E - sigma_A * nd1_value
    
    return eq1**2 + eq2**2

# Initial guess
initial_guess = [equity_market_value + book_debt * 0.5, equity_volatility / 0.8]

# Solve for asset value and volatility
result = minimize(lambda params: merton_equations(params, equity_market_value, book_debt, 
                                                   equity_volatility, risk_free_rate, time_horizon),
                 initial_guess, method='Nelder-Mead')

A_est, sigma_A_est = result.x

print(f"\n=== Calibration Results ===")
print(f"Implied Asset Value: ${A_est:.1f}M")
print(f"Implied Asset Volatility: {sigma_A_est:.1%}")
print(f"Assets / Debt ratio: {A_est / book_debt:.2f}x")

# Calculate Merton distance to default
dd_merton = (np.log(A_est / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * time_horizon) / (sigma_A_est * np.sqrt(time_horizon))
pd_merton = norm.cdf(-dd_merton)

print(f"\nDistance to Default (DD): {dd_merton:.2f}")
print(f"1-Year PD (Merton): {pd_merton:.2%}")

# Risk-neutral adjustment (market risk premium)
market_risk_premium = 0.06  # 6% excess return on market
mu_risk_neutral = risk_free_rate  # Under risk-neutral measure

dd_rn = (np.log(A_est / book_debt) + (mu_risk_neutral - 0.5 * sigma_A_est**2) * time_horizon) / (sigma_A_est * np.sqrt(time_horizon))
pd_rn = norm.cdf(-dd_rn)

print(f"PD (Risk-Neutral): {pd_rn:.2%}")

# Relationship to CDS spread (approx: spread ≈ PD × LGD / T)
lgd_assumption = 0.40
implied_cds_spread = (pd_rn * lgd_assumption / time_horizon) * 10000  # in basis points

print(f"Implied CDS Spread: {implied_cds_spread:.0f} bps")

# Scenario analysis: Asset value changes
print(f"\n=== Sensitivity Analysis ===")
print("Asset Value (% of current) | Distance to Default | PD")
print("-" * 55)

asset_scenarios = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
for scenario in asset_scenarios:
    A_scenario = A_est * scenario
    dd_scenario = (np.log(A_scenario / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * time_horizon) / (sigma_A_est * np.sqrt(time_horizon))
    pd_scenario = norm.cdf(-dd_scenario)
    print(f"{scenario*100:6.0f}% ({A_scenario:6.1f}M)      | {dd_scenario:18.2f} | {pd_scenario:6.2%}")

# Monte Carlo simulation: Asset paths and default probability
print(f"\n=== Monte Carlo Default Simulation ===")
n_paths = 10000
dt = 1/252  # Daily steps
n_steps = int(time_horizon / dt)

asset_paths = np.zeros((n_paths, n_steps + 1))
asset_paths[:, 0] = A_est

np.random.seed(42)
for step in range(n_steps):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)
    asset_paths[:, step+1] = asset_paths[:, step] * np.exp((risk_free_rate - 0.5 * sigma_A_est**2) * dt + sigma_A_est * dW)

# Check for defaults (crossing below debt barrier)
min_asset_value = asset_paths.min(axis=1)
defaults_mc = (min_asset_value < book_debt).astype(int)
pd_mc = defaults_mc.mean()

print(f"Monte Carlo PD (n_paths={n_paths}): {pd_mc:.2%}")
print(f"Merton PD: {pd_merton:.2%}")
print(f"Difference: {(pd_mc - pd_merton)*100:.2f} percentage points")

# Multi-period PD
print(f"\n=== Multi-Year PD ===")
for T_year in [1, 3, 5]:
    dd_year = (np.log(A_est / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * T_year) / (sigma_A_est * np.sqrt(T_year))
    pd_year = norm.cdf(-dd_year)
    print(f"{T_year}-Year PD: {pd_year:.2%}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Asset paths
ax1 = axes[0, 0]
time_axis = np.arange(n_steps + 1) * dt
sample_paths = asset_paths[:100, :]
for path in sample_paths:
    ax1.plot(time_axis, path, alpha=0.2, color='blue')
ax1.axhline(book_debt, color='r', linestyle='--', linewidth=2, label=f'Debt Level = ${book_debt}M')
ax1.fill_between(time_axis, 0, book_debt, alpha=0.1, color='red', label='Default Zone')
ax1.set_xlabel('Time (Years)')
ax1.set_ylabel('Asset Value ($M)')
ax1.set_title('Monte Carlo Asset Paths\n(Sample of 100 paths)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Terminal asset distribution
ax2 = axes[0, 1]
ax2.hist(asset_paths[:, -1], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(book_debt, color='r', linestyle='--', linewidth=2, label=f'Debt = ${book_debt}M')
ax2.axvline(A_est, color='g', linestyle='--', linewidth=2, label=f'Current = ${A_est:.0f}M')
ax2.set_xlabel('Asset Value at Maturity ($M)')
ax2.set_ylabel('Frequency')
ax2.set_title('Terminal Asset Value Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Payoff diagram
ax3 = axes[0, 2]
asset_range = np.linspace(50, 350, 100)
equity_payoff = np.maximum(asset_range - book_debt, 0)
debt_payoff = np.minimum(asset_range, book_debt)

ax3.plot(asset_range, equity_payoff, linewidth=2, label='Equity (Call on Assets)')
ax3.plot(asset_range, debt_payoff, linewidth=2, label='Debt (Risky Bond)')
ax3.axvline(A_est, color='g', linestyle='--', alpha=0.5, label='Current Asset Value')
ax3.set_xlabel('Asset Value at Maturity ($M)')
ax3.set_ylabel('Payoff ($M)')
ax3.set_title('Merton Payoff Structure')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Distance to Default over time
ax4 = axes[1, 0]
time_range = np.linspace(0.01, 5, 50)
dd_range = [(np.log(A_est / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * T) / (sigma_A_est * np.sqrt(T)) 
            for T in time_range]
pd_range = [norm.cdf(-dd) for dd in dd_range]

ax4.plot(time_range, dd_range, linewidth=2)
ax4.axhline(0, color='r', linestyle='--', alpha=0.5, label='DD = 0 (50% PD)')
ax4.set_xlabel('Time to Maturity (Years)')
ax4.set_ylabel('Distance to Default')
ax4.set_title('DD Over Time\n(Lower DD = Higher default risk)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: PD sensitivity to volatility
ax5 = axes[1, 1]
sigma_range = np.linspace(0.1, 0.5, 30)
pd_sigma = []
for sigma in sigma_range:
    dd_sigma = (np.log(A_est / book_debt) + (risk_free_rate - 0.5 * sigma**2) * time_horizon) / (sigma * np.sqrt(time_horizon))
    pd_sigma.append(norm.cdf(-dd_sigma))

ax5.plot(sigma_range, np.array(pd_sigma)*100, linewidth=2)
ax5.axvline(sigma_A_est, color='g', linestyle='--', label=f'Current σ = {sigma_A_est:.1%}')
ax5.set_xlabel('Asset Volatility')
ax5.set_ylabel('1-Year PD (%)')
ax5.set_title('PD Sensitivity to Volatility\n(Volatility increase → PD increase)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Scenario analysis
ax6 = axes[1, 2]
scenarios_name = ['Recovery', 'Base', 'Stress', 'Severe']
scenarios_value = [1.2, 1.0, 0.9, 0.8]
scenarios_pd = []

for val in scenarios_value:
    A_scen = A_est * val
    dd_scen = (np.log(A_scen / book_debt) + (risk_free_rate - 0.5 * sigma_A_est**2) * time_horizon) / (sigma_A_est * np.sqrt(time_horizon))
    pd_scen = norm.cdf(-dd_scen)
    scenarios_pd.append(pd_scen * 100)

colors_scen = ['green', 'blue', 'orange', 'red']
ax6.bar(scenarios_name, scenarios_pd, color=colors_scen, alpha=0.7, edgecolor='black')
ax6.set_ylabel('1-Year PD (%)')
ax6.set_title('Scenario Analysis:\nAsset Value Impact on PD')
for i, pd in enumerate(scenarios_pd):
    ax6.text(i, pd + 0.5, f'{pd:.1f}%', ha='center', va='bottom')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== Merton Model Summary ===")
print(f"Approach: Structural model linking firm value to default risk")
print(f"Key insight: Default is rational economic decision when assets < debt")
print(f"Limitation: Assumes continuous asset paths (ignores jump risk)")
```

## 6. Challenge Round
When is Merton problematic?
- **Gap risk**: Asset value can jump below debt suddenly; continuous assumption fails (2008 Lehman collapse)
- **Equity vol instability**: Equity volatility changes rapidly; implied asset vol unstable
- **Capital structure complexity**: Multiple debt classes, covenants, priority; simple debt level insufficient
- **Endogenous default**: Firm can strategically default (Chapter 11 option); not purely economic threshold
- **Balance sheet quality**: Book debt may not reflect true obligations (pensions, contingencies, leases)

## 7. Key References
- [Merton Model](https://en.wikipedia.org/wiki/Merton_model) - Original 1974 paper framework
- [Black-Scholes Option Pricing](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) - Underlying valuation methodology
- [Credit Spread Determinants](https://www.bis.org/publ/work291.pdf) - BIS research on structural model empirical performance

---
**Status:** Theory-driven PD approach with economic intuition | **Complements:** Scorecard models, market-implied PD, validation
