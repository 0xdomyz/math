# Portfolio Construction

## Concept Skeleton

Portfolio construction allocates capital across assets to optimize risk-return tradeoffs, balancing diversification, expected returns, constraints (sector limits, position sizes), and transaction costs. Modern approaches combine mean-variance optimization (Markowitz), factor models (exposures to value, momentum, quality), and risk budgeting (allocating volatility across bets) to build efficient portfolios aligned with investment objectives and risk tolerance.

**Core Components:**
- **Asset universe**: Stocks, bonds, alternatives; filtration by liquidity, market cap, investability
- **Weight optimization**: Solving for portfolio weights maximizing Sharpe ratio or minimizing risk subject to constraints
- **Constraints**: Long-only vs. long-short, sector neutrality, position limits (e.g., no single stock >5%), turnover caps
- **Risk models**: Covariance matrix estimation (sample, shrinkage, factor-based), forecasting volatilities and correlations
- **Expected return forecasts**: Historical means, factor models, alpha signals, analyst estimates

**Why it matters:** Disciplined portfolio construction separates systematic edge from idiosyncratic noise, manages unintended risks (concentration, factor tilts), and scales strategies from research to production.

---

## Comparative Framing

| Dimension | **Mean-Variance Optimization (MVO)** | **Risk Parity** | **Equal Weight** |
|-----------|--------------------------------------|-----------------|------------------|
| **Objective** | Maximize Sharpe ratio (return/risk) | Equalize risk contribution across assets | Equal capital allocation |
| **Return forecasts** | Required (drives allocation) | Not required (risk-focused) | Not required |
| **Risk model** | Full covariance matrix | Volatilities and correlations | Ignores correlations |
| **Concentration risk** | Can be extreme (few assets dominate) | Diversified by design | Diversified (equal weights) |
| **Sensitivity to inputs** | High (small forecast errors → large weight changes) | Moderate (depends on vol estimates) | None (fixed 1/N) |
| **Turnover** | High (frequent rebalancing) | Moderate (risk evolves slowly) | Low (only rebalance on entry/exit) |

**Key insight:** MVO is optimal *if* forecasts are accurate but unstable with estimation error; risk parity diversifies risk but ignores return forecasts; equal weight is naive but surprisingly robust baseline.

---

## Examples & Counterexamples

### Examples of Portfolio Construction

1. **Long-Only Equity Portfolio with Sector Constraints**  
   - Universe: S&P 500 stocks  
   - Objective: Maximize Sharpe ratio  
   - Constraints: Long-only (weights ≥ 0), sector weights ±5% of benchmark, no stock >3%  
   - Inputs: Expected returns from momentum + value signals, covariance from factor model  
   - Output: 120 positions, sector-neutral, tilted toward high-momentum/low-valuation stocks  

2. **Risk Parity Multi-Asset Portfolio**  
   - Assets: 60% equities, 30% bonds, 10% commodities (naive allocation)  
   - Risk parity allocation: Bonds get higher weight (lower volatility) to equalize risk contribution  
   - Example: 40% equities (15% vol), 50% bonds (5% vol), 10% commodities (20% vol)  
   - Each contributes ~33% of portfolio risk  

3. **Factor-Neutral Long-Short Portfolio**  
   - Alpha signal: Earnings surprise  
   - Constraints: Market-neutral (beta=0), sector-neutral, size-neutral  
   - Construction: Rank stocks by signal, long top quintile, short bottom quintile, weight to neutralize factors  
   - Result: Pure alpha exposure, isolated from market/sector/size risks

### Non-Examples (or Edge Cases)

- **Holding only top-ranked stock**: Concentration risk; no diversification; portfolio construction requires spreading capital.
- **Using 1-month sample covariance for 100 stocks**: Estimation error dominates (100×100 matrix = 5,050 parameters from ~20 observations); risk model collapses.
- **Optimizing without transaction costs**: Produces unrealistic high-turnover portfolios; real-world construction includes cost-aware optimization.

---

## Layer Breakdown

**Layer 1: Asset Universe Definition**  
Filter investable universe: liquidity (average daily volume > threshold), price (no penny stocks), data quality (exclude corporate actions errors). For factor strategies, may exclude utilities, financials (different accounting). Result: ~500–2,000 stocks for liquid equity strategies.

**Layer 2: Risk Model Specification**  
Covariance matrix \(\Sigma\) drives portfolio risk: \(\sigma_p^2 = w' \Sigma w\).  
**Approaches:**  
- **Sample covariance**: Historical returns; unstable for large N (curse of dimensionality)  
- **Factor models**: \(\Sigma = B F B' + D\) where \(B\) = factor loadings, \(F\) = factor covariance, \(D\) = idiosyncratic variances (reduces parameters from \(N^2\) to \(N \times K\))  
- **Shrinkage**: Blend sample covariance with structured target (e.g., identity matrix, single-index model) via Ledoit-Wolf estimator

**Layer 3: Mean-Variance Optimization**  
Solve quadratic program:  
\[
\max_w \quad w' \mu - \frac{\lambda}{2} w' \Sigma w
\]  
subject to:  
\[
\sum w_i = 1, \quad w_i \geq 0 \text{ (long-only)}, \quad \text{sector/factor constraints}
\]  
where \(\mu\) = expected returns, \(\lambda\) = risk aversion, \(\Sigma\) = covariance matrix.  
**Output**: Optimal weights \(w^*\).

**Layer 4: Post-Processing and Trade Generation**  
Current holdings → target weights → trades (buy/sell quantities). Apply turnover constraints:  
\[
\sum |w_i^{\text{target}} - w_i^{\text{current}}| \leq \text{Turnover Limit}
\]  
Tax-loss harvesting: Defer gains, realize losses. Round lots for small positions (avoid odd-lot execution costs).

---

## Mini-Project: Mean-Variance Portfolio Optimization

**Goal:** Construct optimal portfolio with long-only and position-size constraints.

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Simulate 10 assets: expected returns and covariance
np.random.seed(42)
n_assets = 10
asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

# Expected annual returns (random 5%-15%)
expected_returns = np.random.uniform(0.05, 0.15, n_assets)

# Covariance matrix (realistic correlation structure)
correlation = np.random.uniform(0.1, 0.6, (n_assets, n_assets))
np.fill_diagonal(correlation, 1.0)
correlation = (correlation + correlation.T) / 2  # Symmetrize
volatilities = np.random.uniform(0.10, 0.30, n_assets)  # 10%-30% annual vol
cov_matrix = np.outer(volatilities, volatilities) * correlation

# Portfolio optimization: maximize Sharpe ratio
# Sharpe = (w'mu - rf) / sqrt(w'Sigma w)
# Equivalent: minimize -Sharpe or minimize variance for target return

def portfolio_stats(weights, returns, cov):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

def negative_sharpe(weights, returns, cov):
    _, _, sharpe = portfolio_stats(weights, returns, cov)
    return -sharpe

# Constraints: weights sum to 1, long-only, max 20% per asset
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
]
bounds = [(0, 0.20) for _ in range(n_assets)]  # Long-only, max 20% per asset

# Initial guess: equal weight
initial_weights = np.ones(n_assets) / n_assets

# Optimize
result = minimize(
    negative_sharpe,
    initial_weights,
    args=(expected_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
opt_return, opt_vol, opt_sharpe = portfolio_stats(optimal_weights, expected_returns, cov_matrix)

# Compare with equal-weight portfolio
equal_weights = np.ones(n_assets) / n_assets
eq_return, eq_vol, eq_sharpe = portfolio_stats(equal_weights, expected_returns, cov_matrix)

# Display results
print("=" * 60)
print("PORTFOLIO CONSTRUCTION RESULTS")
print("=" * 60)
print("\nOptimal Portfolio Weights:")
for asset, weight in zip(asset_names, optimal_weights):
    if weight > 0.01:  # Show only meaningful positions
        print(f"  {asset:12s}: {weight:>6.2%}")

print(f"\nOptimal Portfolio:")
print(f"  Expected Return: {opt_return:>6.2%}")
print(f"  Volatility:      {opt_vol:>6.2%}")
print(f"  Sharpe Ratio:    {opt_sharpe:>6.2f}")

print(f"\nEqual-Weight Portfolio:")
print(f"  Expected Return: {eq_return:>6.2%}")
print(f"  Volatility:      {eq_vol:>6.2%}")
print(f"  Sharpe Ratio:    {eq_sharpe:>6.2f}")

print(f"\nImprovement:")
print(f"  Sharpe Gain:     {opt_sharpe - eq_sharpe:>+6.2f} ({(opt_sharpe/eq_sharpe - 1)*100:>+.1f}%)")

# Risk decomposition (marginal contribution to risk)
marginal_risk = np.dot(cov_matrix, optimal_weights) / opt_vol
risk_contribution = optimal_weights * marginal_risk
print(f"\nTop 3 Risk Contributors:")
top_contributors = np.argsort(risk_contribution)[-3:][::-1]
for idx in top_contributors:
    print(f"  {asset_names[idx]:12s}: {risk_contribution[idx]/opt_vol:>6.2%} of portfolio risk")
```

**Expected Output (illustrative):**
```
============================================================
PORTFOLIO CONSTRUCTION RESULTS
============================================================

Optimal Portfolio Weights:
  Asset_1     :  20.00%
  Asset_3     :  18.45%
  Asset_5     :  20.00%
  Asset_7     :  15.32%
  Asset_9     :  20.00%
  Asset_10    :   6.23%

Optimal Portfolio:
  Expected Return:  11.34%
  Volatility:       14.25%
  Sharpe Ratio:      0.80

Equal-Weight Portfolio:
  Expected Return:   9.87%
  Volatility:       16.53%
  Sharpe Ratio:      0.60

Improvement:
  Sharpe Gain:      +0.20 (+33.3%)

Top 3 Risk Contributors:
  Asset_5     :  24.12% of portfolio risk
  Asset_1     :  22.87% of portfolio risk
  Asset_9     :  19.34% of portfolio risk
```

**Interpretation:**  
- Optimal portfolio concentrates in high-return, low-volatility assets (hits 20% max constraint).  
- Sharpe improvement (+33%) demonstrates value of optimization vs. naive equal-weight.  
- Risk decomposition identifies dominant risk sources; rebalance if concentration becomes excessive.

---

## Challenge Round

1. **Estimation Error Amplification**  
   MVO with sample covariance produces extreme weights (80% in one asset). Why, and how to fix?

   <details><summary>Hint</summary>Sample covariance is noisy; small forecast errors magnified by optimizer (allocates heavily to assets with upward-biased returns or downward-biased volatility). **Solutions:** (1) Shrink covariance toward structured target (Ledoit-Wolf), (2) Add regularization (penalize sum of squared weights), (3) Use robust optimization (consider range of scenarios), (4) Apply position limits (e.g., max 10% per asset).</details>

2. **Corner Solutions and Constraints**  
   Unconstrained MVO allocates negative weight to low-return assets. Long-only constraint forces those weights to zero. What is the economic interpretation?

   <details><summary>Solution</summary>Negative weights = short positions in mean-variance optimum. Long-only constraint eliminates shorting, forcing portfolio to hold zero (instead of short) in unattractive assets. This increases portfolio volatility (less diversification) but simplifies implementation (no borrowing costs, margin requirements).</details>

3. **Risk Parity vs. MVO**  
   Portfolio has 2 assets: Stocks (12% return, 18% vol), Bonds (4% return, 6% vol), correlation 0.3. Calculate MVO weights (max Sharpe, no constraints) and risk parity weights.

   <details><summary>Solution</summary>
   **MVO (max Sharpe):** Requires solving for tangency portfolio. Shortcut: Higher Sharpe ratio asset gets higher weight. Stocks Sharpe = 12/18 = 0.67; Bonds Sharpe = 4/6 = 0.67 (same!). Equal Sharpe → weights depend on correlation; approximately 50/50 if correlation is moderate.  
   **Risk Parity:** Equalize risk contribution. Stock weight × stock marginal risk = Bond weight × bond marginal risk. Since bonds have 1/3 the volatility, they get 3× the weight. Risk parity ≈ 25% stocks, 75% bonds (exact calculation requires correlation).  
   **Trade-off:** MVO tilts toward return forecasts; risk parity ignores returns, focuses on diversification.
   </details>

4. **Turnover-Constrained Rebalancing**  
   Current portfolio: 60% stocks, 40% bonds. Target (MVO optimal): 80% stocks, 20% bonds. Turnover limit: 10% (one-way). What trades to execute?

   <details><summary>Solution</summary>
   Desired change: +20% stocks, -20% bonds (total 40% two-way turnover).  
   Turnover limit: 10% one-way = 20% two-way.  
   Achievable: +10% stocks, -10% bonds.  
   **Executed portfolio:** 70% stocks, 30% bonds (halfway to target).  
   Next rebalancing period: Continue moving toward 80/20 if market conditions persist.
   </details>

---

## Key References

- **Markowitz (1952)**: "Portfolio Selection" ([Journal of Finance](https://www.jstor.org/stable/2975974))
- **Ledoit & Wolf (2004)**: "Honey, I Shrunk the Sample Covariance Matrix" ([Journal of Portfolio Management](https://www.jstor.org/))
- **Black & Litterman (1992)**: "Global Portfolio Optimization" (combines views with market equilibrium) ([Financial Analysts Journal](https://www.jstor.org/))
- **Meucci (2005)**: *Risk and Asset Allocation* (comprehensive portfolio construction) ([Springer](https://www.springer.com/))

**Further Reading:**  
- Robust portfolio optimization (worst-case scenarios, uncertainty sets)  
- Hierarchical risk parity (clustering-based diversification)  
- Factor risk budgeting (allocate risk to systematic factors, not assets)
