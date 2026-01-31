# Risk Parity

## 1. Concept Skeleton
**Definition:** Portfolio construction approach allocating capital inversely to asset risk; aims for equal risk contribution (not capital weight) from each position  
**Purpose:** Diversify portfolio by risk, not capital; reduce concentration in low-volatility assets; improve risk-adjusted returns across market regimes; capture all risk premia  
**Prerequisites:** Portfolio construction, volatility concepts, correlation matrices, covariance structures, leverage, risk budgeting

## 2. Comparative Framing
| Aspect | Market Cap Weight | Risk Parity | Equal Weight | Min Variance | Optimal (MV) |
|--------|-------------------|-------------|--------------|--------------|--------------|
| **Weight Basis** | Market cap | Risk contribution | Equal dollars | Min portfolio vol | Sharpe ratio |
| **Leverage Used** | None/rare | Often required | No | Sometimes | Rarely |
| **Volatility Concentration** | High (largest assets) | Equal | Moderate | Minimal | Balanced |
| **Bond/Equity Mix** | Market-determined | RP dictates | 50/50 | Bonds preferred | Balanced |
| **Rebalancing** | Passive (drift) | Active (maintain vol) | Regular | Occasional | Strategic |
| **Downside Protection** | Moderate | Good | Moderate | Excellent | Depends |
| **Cost** | Lowest | Higher (rebalance) | Low | Moderate | Moderate |

## 3. Examples + Counterexamples

**Simple Case:**  
Stock portfolio: Volatility 15%, Bonds: 5%. Market-cap weight might be 80/20. Risk parity: If equal risk wanted, weight Bonds higher. Calculate: w_stocks × 15% = w_bonds × 5% with w_stocks + w_bonds = 1. Solve: w_bonds = 0.60 (60% bonds!), w_stocks = 0.40. Bonds absorb less capital but contribute same risk.

**Crisis Year (2008):**  
Market cap weight: Overweight equities (largest market cap historically). When stocks crash -40%, portfolio heavily hit. Risk parity: Equal risk pre-crisis means already loaded in bonds. When crisis: Bonds up +10%, stocks down -40%. RP portfolio blends better, less tail damage.

**Bull Market Problem:**  
Equities outperform bonds for 10 years. Market cap weight: Naturally skews to equities (they're now bigger). Risk parity: Force rebalancing (sell winners, buy losers). Costs drag performance in bull market. Tradeoff: Good downside, mediocre upside.

**Leverage Requirement:**  
US Equities: 15% vol. Bonds: 5% vol. Commodities: 12% vol. Equal risk: w_eq = x, w_bnd = 3x, w_com = 1.25x. If w_eq + w_bnd + w_com = 1: x ≈ 0.2, bond weight = 0.6. To get 100% invested fully, use 1.5x leverage on bond allocation. Leverage = major feature of RP.

**Skewed Returns (Hedge Funds):**  
Hedge fund: Low vol, high skew (mediocre average, rare crashes). Stock: High vol, low skew (normal distribution). Market cap: If hedge fund larger, overweight it despite risk. Risk parity: Equal vol → more stock, less hedge fund. Better tail profile.

**Regime Switch:**  
Normal times: Risk parity ≈ 60/40 stocks/bonds. Crisis: Correlations → 1, both down. But bonds fall less. RP rebalances into bonds (contrarian). After recovery: Rebalances into stocks. Anti-momentum, anti-herding behavior.

## 4. Layer Breakdown
```
Risk Parity Framework:

├─ Core Concept:
│  ├─ Risk Budget:
│  │   Total portfolio risk σp = target risk
│  │   Allocate equally: RC_i = σp / n
│  │   Each asset contributes σp / n to portfolio vol
│  ├─ Risk Contribution (RC):
│  │   RC_i = w_i × MRC_i
│  │   MRC_i = marginal risk contribution = ∂σp / ∂w_i
│  │   MRC_i = (Cov_matrix @ w)_i / σp
│  ├─ Equal Risk Constraint:
│  │   RC_1 = RC_2 = ... = RC_n
│  │   ⟹ w_1 × MRC_1 = w_2 × MRC_2 = ...
│  ├─ Mathematical Formulation:
│  │   min_w ||RC_i - (σp/n)||² subject to Σ w_i = 1
│  │   Or: find w such that Σ w_i × MRC_i = equal
│  ├─ Connection to Leverage:
│  │   If σ_i vary widely, need leverage to enforce equal risk
│  │   Leverage multiplier: Σ w_i > 1 possible (within limits)
│  └─ Portfolio Volatility:
│      σp = √(w^T @ Cov @ w)
│      Cov = covariance matrix of assets
├─ Implementation Steps:
│  ├─ Step 1: Estimate Covariance Matrix
│  │   Historical window (1-3 years typical)
│  │   Daily returns → correlation, standard deviations
│  │   ρ_ij = correlation between assets i, j
│  │   Σ_ij = ρ_ij × σ_i × σ_j
│  ├─ Step 2: Calculate Marginal Risk Contributions
│  │   MRC_i = (Σ @ w)_i / σp
│  │   Vector operation: multiply covariance matrix by weights
│  │   Divide by portfolio vol to normalize
│  ├─ Step 3: Solve for RP Weights
│  │   Set RC_i = RC_j for all pairs
│  │   Solve: w_i / w_j = MRC_j / MRC_i
│  │   Constraint: Σ w_i = 1 (or > 1 if leveraged)
│  ├─ Step 4: Apply Leverage (if needed)
│  │   Scale weights to target portfolio vol
│  │   Scale factor: λ = Target_σ / Current_σ
│  │   New weights: w' = λ × w
│  ├─ Step 5: Enforce Bounds
│  │   Long-only constraint: w_i ≥ 0
│  │   Leverage limit: Σ |w_i| ≤ leverage_max
│  ├─ Step 6: Rebalance Periodically
│  │   As volatilities drift, RC diverges from equal
│  │   Monthly/quarterly rebalancing typical
│  │   Trade-off: Transaction costs vs risk drift
│  └─ Implementation Issues:
│      Covariance estimation error (especially correlations)
│      Non-linearities in optimization
│      Leverage constraints, margin requirements
├─ Asset Class Perspective:
│  ├─ Traditional RP (Stocks + Bonds):
│  │   Stocks: ~15% vol, Bonds: ~5% vol
│  │   RP weights: ~25% stocks, ~75% bonds (overleveraged bonds)
│  │   Leverage: 1.3x total (leverage bonds 0.75 × 1.3 = 0.975)
│  ├─ Multi-Asset RP (Stocks + Bonds + Commodities):
│  │   Stocks: 15%, Bonds: 5%, Commodities: 12%
│  │   RP weights: Stocks < Commodities << Bonds
│  │   More complex leverage, higher rebalancing
│  ├─ Sector RP:
│  │   Within equities, allocate by sector vol
│  │   High-vol sectors (Tech): Lower weight
│  │   Low-vol sectors (Utilities): Higher weight
│  ├─ Currency RP:
│  │   Global portfolio allocate to currency risk
│  │   High-vol currencies: Hedge or underweight
│  ├─ Factor RP:
│  │   Allocate to value, momentum, quality, etc.
│  │   Equal risk contribution from each factor
│  │   Beta-neutral (market-neutral) RP possible
│  └─ Geographic RP:
│      Allocate across regions by vol
│      Diversifies geographic/economic shocks
├─ Advantages of Risk Parity:
│  ├─ Diversification:
│  │   True risk diversification (not just capital)
│  │   Low-vol assets get more capital (more shares)
│  │   Reduces portfolio concentration
│  ├─ Performance Across Regimes:
│  │   Bull: Stocks high vol → underweight → drag
│  │   Bear: Bonds low vol → overweight → protection
│  │   Balanced performance across cycles
│  ├─ Crisis Resilience:
│  │   Built-in bond hedge
│  │   Automatic rebalancing buys dips (contrarian)
│  │   Historical Sharpe ratio > 60/40 in many periods
│  ├─ Simpler than Optimization:
│  │   No need to estimate expected returns (only vol)
│  │   Objective purely diversification
│  │   Less model risk than mean-variance optimization
│  ├─ Lower Tail Risk:
│  │   Less equity concentration
│  │   Max drawdown often lower than MV portfolios
│  │   CVaR/VaR improvements documented
│  └─ Institutional Appeal:
│      Rules-based (algorithmic)
│      Transparency (equal risk concept clear)
│      Systematic rebalancing
├─ Disadvantages & Challenges:
│  ├─ Leverage Required:
│  │   Not all investors can leverage
│  │   Margin costs, counterparty risk
│  │   Regulatory constraints
│  ├─ Bull Market Underperformance:
│  │   If one asset booms, RP sells winners
│  │   Drag during prolonged equity rallies
│  │   2010s: Equities outperformed, RP lagged
│  ├─ Covariance Estimation Error:
│  │   RP highly sensitive to correlation estimates
│  │   Tail correlation ≠ average correlation
│  │   Model risk significant
│  ├─ Rebalancing Costs:
│  │   Active rebalancing incurs transaction costs
│  │   Market impact, bid-ask spreads
│  │   Drags performance in volatile periods
│  ├─ Leverage Constraints:
│  │   Leverage limit (2x, 3x typical) may not achieve RP
│  │   Constraints force compromise allocation
│  ├─ Correlation Spikes:
│  │   Crisis: Stock-bond correlation → +1
│  │   RP breaks down (all assets move together)
│  │   No diversification benefit in true crisis
│  ├─ Volatility Regime Shifts:
│  │   If vol structure changes persistently, RP stale
│  │   Frequent rebalancing needed
│  └─ Opportunity Cost:
│      Ignores expected returns (only uses vol)
│      May underweight high-return assets
│      Foregoes alpha opportunities
├─ Variations & Extensions:
│  ├─ Targeted RP:
│  │   Weight toward higher-performing asset classes
│  │   RP + alpha overlay
│  │   More pragmatic than pure RP
│  ├─ Conditional RP:
│  │   Adjust risk targets based on market regime
│  │   Higher equity RP in low-vol environment
│  │   Lower equity RP in high-vol environment
│  ├─ Fuzzy RP:
│  │   RC_i ≈ RC_j (approximately equal, not exact)
│  │   Reduces rebalancing frequency
│  │   Lowers transaction costs
│  ├─ Volatility-Targeting RP:
│  │   RP with dynamic leverage
│  │   Scale portfolio vol to target (e.g., 12% annual)
│  │   High vol periods → reduce leverage, vice versa
│  ├─ Downside RP:
│  │   Use downside vol (semi-variance) instead of vol
│  │   Asymmetric risk allocation
│  │   Focus on losses, not gains
│  └─ Factor RP:
│      Allocate to market, value, momentum, quality factors
│      Equal risk from each factor
│      Beta-neutral or long-only variants
├─ Comparison to Alternatives:
│  ├─ vs Market Cap Weight:
│  │   MV: Passive, low cost, drift in weights
│  │   RP: Active, rebalancing, controlled vol
│  ├─ vs Equal Weight:
│  │   EW: Equal capital, no rebalancing
│  │   RP: Equal risk, requires rebalancing
│  ├─ vs Min Variance:
│  │   MV: Minimize all vol (may concentrate in few)
│  │   RP: Balance risk across assets (more diversified)
│  ├─ vs Efficient Frontier (MV optimized):
│  │   MV Optimal: Best Sharpe (requires return estimates)
│  │   RP: Rule-based, avoids return estimation error
│  └─ vs 60/40 Benchmark:
│      Traditional: Fixed allocation, drifts
│      RP: Dynamic, maintains risk parity, rebalances
└─ Practical Implementation:
   ├─ Hedge Funds:
   │   Risk parity hedge funds: AQR, Bridgewater (Pure Alpha)
   │   Target: Better Sharpe than traditional allocation
   │   Reality: Mixed, depends on period
   ├─ Smart Beta ETFs:
   │   RP-based ETFs: RPAR, RPAGG (global RP)
   │   Available to retail investors
   │   Lower cost than hedge fund version
   ├─ Internal Implementation:
   │   Systematic rebalancing algorithm
   │   Software: Python (cvxpy), MATLAB
   │   Risk systems: Bloomberg, FactSet, Axioma
   ├─ Data Requirements:
   │   Daily returns (1-3 year window)
   │   Covariance matrix estimation
   │   Correlation forecasting (EVT or CCC-GARCH)
   └─ Regulatory Considerations:
       Leverage classification
       Counterparty risk (collateral)
       Leverage limits (SEC, FINRA)
```

**Interaction:** Estimate covariance → Calculate marginal risk contributions → Solve for equal risk weights → Apply leverage → Rebalance periodically → Monitor risk drift.

## 5. Mini-Project
Construct and analyze risk parity portfolio:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Historical annual volatility and correlation estimates
assets = ['US Equities', 'US Bonds', 'Commodities']
vols = np.array([0.15, 0.05, 0.12])  # std dev
corr_matrix = np.array([
    [1.00, -0.20, 0.50],
    [-0.20, 1.00, -0.10],
    [0.50, -0.10, 1.00]
])

# Build covariance matrix
cov_matrix = np.outer(vols, vols) * corr_matrix

# Risk Parity Optimization
def risk_parity_weights(cov_matrix, target_leverage=1.0):
    """Calculate risk parity weights"""
    n = cov_matrix.shape[0]
    vols = np.sqrt(np.diag(cov_matrix))
    
    # Initial inverse volatility weights
    w_init = 1 / vols
    w_init = w_init / w_init.sum()
    
    # Refine: iterate to achieve equal risk
    def risk_contribution(w):
        """Return risk contributions"""
        port_vol = np.sqrt(w @ cov_matrix @ w)
        mrc = (cov_matrix @ w) / port_vol
        rc = w * mrc
        return rc
    
    def objective(w):
        """Minimize variance in risk contributions"""
        rc = risk_contribution(w)
        target_rc = 1.0 / n  # target: equal risk
        return np.sum((rc - target_rc)**2)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, None) for _ in range(n)]
    
    result = minimize(objective, w_init, method='SLSQP', 
                     constraints=constraints, bounds=bounds)
    
    w_rp = result.x
    
    # Apply leverage if needed
    port_vol_rp = np.sqrt(w_rp @ cov_matrix @ w_rp)
    w_rp_leveraged = w_rp * target_leverage / port_vol_rp
    
    return w_rp, w_rp_leveraged

# Comparison portfolios
w_market_cap = np.array([0.70, 0.25, 0.05])  # typical market cap weight
w_equal = np.array([1/3, 1/3, 1/3])
w_inverse_vol = 1 / vols
w_inverse_vol = w_inverse_vol / w_inverse_vol.sum()

w_rp, w_rp_lev = risk_parity_weights(cov_matrix, target_leverage=1.3)

portfolios = {
    'Market Cap': w_market_cap,
    'Equal Weight': w_equal,
    'Inverse Vol': w_inverse_vol,
    'Risk Parity': w_rp,
    'RP Leveraged': w_rp_lev
}

# Calculate portfolio metrics
results = {}
for pname, w in portfolios.items():
    port_vol = np.sqrt(w @ cov_matrix @ w)
    
    # Risk contributions
    mrc = (cov_matrix @ w) / port_vol
    rc = w * mrc
    
    # Gross exposure
    gross_exposure = np.sum(np.abs(w))
    
    results[pname] = {
        'weights': w,
        'port_vol': port_vol,
        'risk_contrib': rc,
        'gross_exposure': gross_exposure,
    }

# Print results
print("="*90)
print("RISK PARITY PORTFOLIO ANALYSIS")
print("="*90)
print(f"\nAsset Volatilities: {dict(zip(assets, vols))}")
print(f"\nCorrelation Matrix:\n{corr_matrix}")

print(f"\n{'Portfolio':<20} ", end='')
for asset in assets:
    print(f"{asset:<15} ", end='')
print(f"{'Gross Exp':<10} {'Vol':<8}")
print("-"*90)

for pname, metrics in results.items():
    w = metrics['weights']
    vol = metrics['port_vol']
    gross = metrics['gross_exposure']
    print(f"{pname:<20} ", end='')
    for i in range(len(assets)):
        print(f"{w[i]:<15.2%} ", end='')
    print(f"{gross:<10.2f}x {vol:<8.2%}")

print(f"\n{'Portfolio':<20} ", end='')
for asset in assets:
    print(f"RC: {asset:<12} ", end='')
print()
print("-"*90)

for pname, metrics in results.items():
    rc = metrics['risk_contrib']
    print(f"{pname:<20} ", end='')
    for i in range(len(assets)):
        print(f"{rc[i]:<15.2%} ", end='')
    print()

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Weight comparison
ax = axes[0, 0]
x = np.arange(len(assets))
width = 0.15
for i, (pname, metrics) in enumerate(results.items()):
    ax.bar(x + i*width, metrics['weights'], width, label=pname, alpha=0.8)
ax.set_ylabel('Weight')
ax.set_title('Portfolio Weights Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(assets)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 2: Risk contributions
ax = axes[0, 1]
for i, (pname, metrics) in enumerate(results.items()):
    ax.bar(x + i*width, metrics['risk_contrib'], width, label=pname, alpha=0.8)
ax.axhline(y=1/3, color='red', linestyle='--', linewidth=2, label='Equal Risk Target')
ax.set_ylabel('Risk Contribution')
ax.set_title('Risk Contributions (RP Target: 33.3% each)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(assets)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 3: Portfolio volatility comparison
ax = axes[0, 2]
vols_port = [results[pname]['port_vol'] for pname in results]
colors = ['red' if results[pname]['gross_exposure'] > 1 else 'blue' for pname in results]
bars = ax.bar(results.keys(), vols_port, color=colors, alpha=0.7)
ax.set_ylabel('Portfolio Volatility')
ax.set_title('Portfolio Volatility (Red=Leveraged)')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Risk decomposition (stacked)
ax = axes[1, 0]
rc_matrix = np.array([results[pname]['risk_contrib'] for pname in results]).T
rc_matrix_pct = rc_matrix * 100
bottom = np.zeros(len(results))
colors_stack = ['steelblue', 'orange', 'green']
for i, asset in enumerate(assets):
    ax.bar(results.keys(), rc_matrix_pct[i], bottom=bottom, label=asset, 
          color=colors_stack[i], alpha=0.8)
    bottom += rc_matrix_pct[i]
ax.set_ylabel('Risk Contribution (%)')
ax.set_title('Risk Decomposition by Asset')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 5: Correlation matrix heatmap
ax = axes[1, 1]
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(assets)))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(assets, rotation=45, ha='right')
ax.set_yticklabels(assets)
for i in range(len(assets)):
    for j in range(len(assets)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10)
ax.set_title('Asset Correlation Matrix')
plt.colorbar(im, ax=ax)

# Plot 6: Gross exposure summary
ax = axes[1, 2]
gross_exps = [results[pname]['gross_exposure'] for pname in results]
colors_exp = ['red' if g > 1 else 'green' for g in gross_exps]
ax.bar(results.keys(), gross_exps, color=colors_exp, alpha=0.7)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No Leverage')
ax.set_ylabel('Gross Exposure (Leverage Multiple)')
ax.set_title('Gross Exposure Summary')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*90)
print("KEY INSIGHTS:")
print("="*90)
print(f"Risk Parity trades off leverage for diversification:")
print(f"  - Market Cap weight: High equity (70%), low leverage")
print(f"  - Risk Parity: Lower equity (careful RP calcs), may need leverage")
print(f"  - Benefit: Equal risk contribution across assets")
print(f"  - Tradeoff: Requires rebalancing, transaction costs")
```

## 6. Challenge Round
- Derive risk parity weights from covariance matrix analytically
- Implement risk parity with no-shorting constraints
- Design dynamic risk parity with volatility targeting
- Compare risk parity performance: 2008 crisis vs 2017 bull market
- Explain correlation spillovers and risk parity breakdown in extreme events

## 7. Key References
- [Bridgewater, "Engineering Targeted Returns and Risk" (2012)](https://www.bridgewater.com/) — Risk parity pioneer
- [Asness et al, "The Value of Risk Parity" (2012)](https://www.aqr.com/insights/research/white-papers/) — Academic validation
- [Maillard et al, "The Properties of Equally Weighted Risk Contribution Portfolios" (2010)](https://www.jstor.org/) — Mathematical foundations
- [Dalio, "Principles for a New World Order" (2014)](https://www.principles.com/)

---
**Status:** Diversification by risk, not capital | **Complements:** Leverage, Portfolio Construction, Volatility Targeting, Rebalancing
