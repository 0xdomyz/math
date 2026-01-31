# Indifference Curves & Optimal Portfolio Selection

## 1. Concept Skeleton
**Definition:** Indifference curves in mean-return/volatility space connect all portfolio combinations yielding identical utility; investor indifferent among points on same curve but strictly prefers curves farther from origin (higher utility)  
**Purpose:** Visualize investor preferences geometrically; determine optimal portfolio as tangency with Capital Allocation Line (CAL); explain why rational investors choose market portfolio (two-fund separation)  
**Prerequisites:** Mean-variance utility, risk aversion coefficient λ, efficient frontier, CAL

---

## 2. Comparative Framing

| Aspect | Risk-Averse Investor (λ=5) | Moderate Investor (λ=2) | Risk-Seeking Investor (λ=0.5) |
|--------|---------------------------|--------------------------|-------------------------------|
| **Indifference Curve Shape** | Very steep (steep MRS) | Moderate slope | Flat (shallow MRS) |
| **Tangency Point** | Early on CAL (low σp) | Middle of CAL | Far right on CAL (high σp) |
| **Optimal Portfolio Return** | ~4-5% (close to Rf) | ~7-8% (market level) | ~10%+ (leveraged) |
| **Optimal Portfolio Vol** | ~3-5% | ~12-15% | ~20%+ |
| **Leverage Decision** | Borrow to buy risk-free? No | No (already at market) | Yes (lever risky portfolio) |
| **Equity Allocation** | 20-30% stocks | 60-70% stocks | 100%+ stocks (leveraged) |
| **Bond Allocation** | 70-80% bonds | 30-40% bonds | 0-10% bonds |
| **MRS Interpretation** | Give up 3-4% return to cut vol by 1% | Give up 1% return to cut vol by 1% | Give up 0.3-0.5% return to cut vol by 1% |
| **Willingness to Trade Shift** | Prefers bonds → stocks more as stocks outperform | Maintains 60-70 split despite performance | Seeks leverage despite rising rates |
| **Behavioral Signal** | Rebalances into stocks after crash (buy dips) | Passive rebalancing annually | Adds leverage when rates rise (doesn't sell) |

**Key Insight:** Indifference curve steepness (MRS = marginal rate of substitution) captures how much return investor willing to sacrifice for unit volatility reduction. Steeper curve → more risk-averse → willing to give up more return.

---

## 3. Examples + Counterexamples

**Example 1: Finding Optimal Allocation Graphically**

Investor faces:
- Risk-free asset: rf = 2%
- Market portfolio: μm = 10%, σm = 15%
- Risk aversion: λ = 4

Analysis:
1. **CAL equation:** E[Rp] = 2% + [(10%-2%)/15%] × σp = 2% + 0.533 × σp
   - Interpretation: Each 1% vol accepted → 0.533% return gain

2. **Indifference curve:** U = E[Rp] - (4/2) × σp² = E[Rp] - 2σp²
   - At indifference level U₀: E[Rp] = U₀ + 2σp²

3. **Tangency condition:** MRS (indifference curve slope) = CAL slope
   - MRS = dE[Rp]/dσp along indifference curve
   - Differentiating U = const: dE[Rp] = 4σp dσp
   - MRS = 4σp = CAL slope (0.533)
   - Solution: σp* = 0.533/4 = 0.133 (13.3%)

4. **Optimal allocation:**
   - Portfolio vol: σp* = 13.3%
   - Expected return: E[Rp*] = 2% + 0.533 × 13.3% = 9.09%
   - Weights: y* = σp*/σm = 13.3%/15% = 0.887 in risky; 1-0.887 = 11.3% in risk-free
   - Final: 88.7% market portfolio + 11.3% T-bills

5. **Utility at optimum:** U* = 9.09% - 2 × (13.3%)² = 9.09% - 3.54% = 5.55%

**Key Insight:** Optimal portfolio found where investor's preference (indifference curve slope) exactly matches available trade-off (CAL slope). This is the sweet spot: can't do better.

**Example 2: Why Everyone Holds Market Portfolio (Two-Fund Separation)**

Setup: Multiple investors (λ₁, λ₂, λ₃) see same CAL

- All have access to: rf = 2%, market portfolio (μm = 10%, σm = 15%)
- Investor A (λ₁ = 2, aggressive): Solves MRS = 2σp = 0.533 → σp = 0.267 (26.7%, leveraged)
- Investor B (λ₂ = 4, moderate): Solves MRS = 4σp = 0.533 → σp = 0.133 (13.3%, cash buffer)
- Investor C (λ₃ = 8, conservative): Solves MRS = 8σp = 0.533 → σp = 0.067 (6.7%, mostly risk-free)

Key observation: All hold SAME risky portfolio (market), but in different proportions:
- A holds 178% market + borrows 78% at rf
- B holds 89% market + 11% T-bills
- C holds 44% market + 56% T-bills

Implication: Under mean-variance optimization with same beliefs, all investors agree on ONE risky portfolio composition (market portfolio). They disagree only on how much risk to take.

**Example 3: Borrowing Constraint Impact**

Without leverage (wm ≤ 100%):
- Investor A cannot reach λ₁ = 2 optimal (would need 178% market)
- Constrained optimal: Allocate 100% to market portfolio
- Utility: U_constrained = 10% - 2 × (15%)² = 10% - 4.5% = 5.5% < unconstrained 5.55%
- Welfare loss: 0.05% (small, but measurable)

With leverage allowed (borrow at rf):
- Investor A achieves unconstrained optimum: U = 5.55%
- Welfare gain: 0.05%

Implication: Borrowing constraints bind differently for different λ values; aggressive investors hurt most by leverage restrictions.

**Example 4: Indifference Curve Crossing (Preference Shift)**

After market crash (-30%), investor observes:
- Market portfolio now has expected return = 7% (down from 10%)
- Market volatility = 20% (up from 15%)
- New CAL slope = (7% - 2%) / 20% = 0.25 (less steep than 0.533)

Impact on optimal allocation:
- Old optimum: MRS = 4σp = 0.533 → σp = 13.3%
- New optimum: MRS = 4σp = 0.25 → σp = 6.25% (much lower!)
- Portfolio shift: From 88.7% → 41.7% in market (moves right on indifference curve toward risk-free)

Behavioral observation: Investor faces lower Sharpe ratio after crash; indifference curve hasn't shifted, but CAL has become flatter. Optimal allocation moves toward safety (bond-heavy).

Interpretation: NOT change in risk aversion (λ unchanged), but change in risk-return trade-off available. CAL rotation.

**Counterexample: Preference Reversals (Violation of Indifference Curves)**

Experiment (Tversky-Kahneman):
- Portfolio A: 90% win $1M, 10% win $0 (E = $900k, std high)
- Portfolio B: 100% win $900k (E = $900k, std = 0)

Under mean-variance utility: Both have same expected value; indifference irrelevant (differ only in variance). But:

Expected choice: Prefer B (certain $900k)
BUT if reframed as:
- Portfolio A': 90% lose $0, 10% lose $1M (same as A, reframed)
- Portfolio B': 100% lose $100k (loss version of B)

Actual choice: PREFER A' (gamble to avoid loss)

Implication: Same portfolio, different preference depending on frame → indifference curves not well-defined; preference is reference-dependent (prospect theory violation).

Consequence: Mean-variance indifference curves oversimplify; behavioral factors matter.

---

## 4. Layer Breakdown

```
Indifference Curves & Optimal Portfolio Selection Architecture:

├─ Indifference Curve Geometry:
│   ├─ Definition: Set of (σp, E[Rp]) pairs yielding same utility U₀
│   │   │ For mean-variance utility: U₀ = E[Rp] - (λ/2)σp²
│   │   │ Rearranging: E[Rp] = U₀ + (λ/2)σp²  ← PARABOLA, not straight line
│   │   │
│   │   ├─ Vertex: (σp=0, E[Rp]=U₀) – point on vertical axis (only risk-free return achievable)
│   │   ├─ Curvature: d²E[Rp]/dσp² = λ > 0 (concave up; opens rightward)
│   │   ├─ Slope (MRS): dE[Rp]/dσp = λ × σp (increases with volatility!)
│   │   │   └─ At σp=5%: MRS = λ × 0.05
│   │   │   └─ At σp=10%: MRS = λ × 0.10 (steeper)
│   │   │   └─ Implication: Indifference curves fan out; wider spacing for high σp
│   │   │
│   │   └─ Shape Interpretation:
│   │       ├─ Small λ (risk-seeking): Flatter curves; little return needed for more vol
│   │       ├─ Large λ (risk-averse): Steeper curves; demands high return premium for vol
│   │       └─ Comparison: λ₁ curves lie ABOVE λ₂ curves (higher utility available)
│   │
│   ├─ Indifference Curve Map (Family of Curves):
│   │   ├─ U₀ < U₁ < U₂: Higher utility curves farther from origin
│   │   ├─ All curves have same shape (parabolic) but shifted vertically
│   │   ├─ No curves intersect (non-crossing property; rational preferences)
│   │   └─ Denser curves at high σp (flattening becomes steeper relative to vol)
│   │
│   ├─ Marginal Rate of Substitution (MRS):
│   │   ├─ Definition: How much return investor willing to sacrifice per unit vol reduction
│   │   ├─ Formula: MRS = λ × σp
│   │   ├─ Interpretation:
│   │   │   ├─ MRS = 0.5 means: "I'll accept 0.5% lower return to cut vol by 1%"
│   │   │   ├─ MRS = 2.0 means: "I'll accept only 2% lower return to cut vol by 1%"
│   │   │   └─ Higher MRS = more willing to trade return for vol reduction
│   │   │
│   │   └─ Relationship to λ:
│   │       ├─ MRS = λσp → steeper slope for higher λ (at any σp)
│   │       ├─ MRS increases with σp → curves fan out (increasing slope)
│   │       └─ Practical: At σp = 15%, risk-averse investor (λ=8) has MRS = 1.2
│   │           (willing to give up 1.2% return for 1% vol reduction)
│   │
│   └─ Effective Marginal Rate:
│       ├─ True MRS from indifference curve: λσp
│       ├─ Perceived MRS from investor interview: Often inconsistent/noisy
│       ├─ Revealed MRS from portfolio choice: (Rp - Rf) / σp × (some adjustment)
│       └─ Practical: Use revealed > stated > questioned
│
├─ Capital Allocation Line (CAL):
│   ├─ Definition: Feasible set of (σp, E[Rp]) from combining risk-free + risky assets
│   │   │ E[Rp] = rf + (μm - rf) / σm × σp = rf + SR_m × σp
│   │   │ where SR_m = Sharpe ratio of market portfolio
│   │   │
│   │   ├─ Linear (straight line, not curved!)
│   │   ├─ Y-intercept: E[Rp=0] = rf (risk-free rate)
│   │   ├─ Slope: (μm - rf) / σm = Sharpe ratio = risk premium per unit vol
│   │   └─ Extends rightward from Rf: Increase σp → proportional increase in E[Rp]
│   │
│   ├─ Properties:
│   │   ├─ Accessible to all investors (assuming rf borrowing rate = rf lending)
│   │   ├─ ALL risky portfolios lie BELOW CAL (lower Sharpe ratio than market)
│   │   ├─ Market portfolio (μm, σm) lies exactly ON CAL
│   │   ├─ Leverage extends CAL rightward (borrow at rf, buy market)
│   │   └─ Lending portion CAL extends leftward (invest in rf + market)
│   │
│   ├─ Sharpe Ratio (Slope):
│   │   ├─ SR = (μm - rf) / σm = risk premium / risk
│   │   ├─ Typical: SR ≈ 0.4-0.6 (e.g., 8% risk premium / 15% vol = 0.53)
│   │   ├─ Interpretation: 0.53 means 0.53% return per 1% of volatility
│   │   ├─ Comparison: Steeper CAL = better risk-return available
│   │   └─ Implication: All investors prefer steeper CAL (higher market Sharpe)
│   │
│   └─ Economic Significance:
│       ├─ Market prices risk → CAL slope reflects consensus risk premium
│       ├─ Arbitrage prevents profitable deviations (all efficient portfolios on CAL)
│       ├─ CAL shift: Economic news → market expected return or vol changes
│       └─ Investors adjust along CAL (change allocation), not their preferences
│
├─ Optimal Portfolio Selection (Geometric Solution):
│   ├─ Optimization Problem:
│   │   │ Max U(w) = E[Rp(w)] - (λ/2)σp²(w)
│   │   │ Subject to: Σ wi = 1
│   │   │            w ≥ 0 (or allow leverage)
│   │   │
│   │   ├─ Objective: Maximize utility (find highest indifference curve achievable)
│   │   ├─ Constraint: Feasible allocations lie on CAL
│   │   └─ Solution: Tangency point (where indifference curve touches CAL)
│   │
│   ├─ Tangency Condition (First-Order Condition):
│   │   │ At optimum: MRS (indifference) = CAL slope (market trade-off)
│   │   │ λσp* = (μm - rf) / σm
│   │   │ Solving: σp* = (μm - rf) / (λ × σm)
│   │   │         E[Rp*] = rf + [(μm - rf) / σm] × σp*
│   │   │
│   │   ├─ Intuition: If MRS < CAL slope → CAL offers better return/vol trade → move right
│   │   ├─ If MRS > CAL slope → CAL unfavorable → move left
│   │   ├─ Equilibrium: MRS = slope → can't improve further
│   │   └─ Second-order: Indifference curve curvature > CAL curvature (0) → guarantees max
│   │
│   ├─ Closed-Form Solution (for unconstrained portfolio):
│   │   │ Optimal weight in risky (market): w_risky = (μm - rf) / (λ × σm²)
│   │   │ Optimal weight in risk-free: w_rf = 1 - w_risky
│   │   │
│   │   ├─ Properties:
│   │   │   ├─ Higher λ → lower w_risky (more risk-averse → less risky)
│   │   │   ├─ Higher (μm - rf) → higher w_risky (more attractive risk premium)
│   │   │   ├─ Higher σm → lower w_risky (more expensive risk)
│   │   │   ├─ Linear in λ⁻¹: Doubling λ halves risky allocation
│   │   │   └─ Can be >1 (leverage): If w_risky > 1 → borrow at rf, buy risky
│   │   │
│   │   └─ Example Calculation (rf=2%, μm=10%, σm=15%, λ=2):
│   │       ├─ w_risky = (10%-2%) / (2 × 15%²) = 8% / (2 × 2.25%) = 8% / 4.5% = 1.78 (leverage!)
│   │       ├─ w_rf = 1 - 1.78 = -0.78 (borrow 78% to leverage up)
│   │       ├─ Optimal portfolio: 178% market + borrow 78% at 2%
│   │       └─ Expected return: 10% × 1.78 + 2% × (-0.78) = 17.8% - 1.56% = 16.24%
│   │           Volatility: 15% × 1.78 = 26.7%
│   │
│   └─ Graphical Representation:
│       ├─ Plot 1: CAL and indifference curves for different λ
│       ├─ Plot 2: Tangency point shifts right (lower λ) or left (higher λ)
│       ├─ Plot 3: Optimal allocation moves along CAL
│       └─ Plot 4: Utility at optimum increases as λ decreases
│
├─ Two-Fund Separation Theorem:
│   ├─ Statement: Every rational investor holds a combination of TWO funds:
│   │   │ 1) Risk-free bond fund (rf)
│   │   │ 2) Market portfolio (μm, σm)
│   │   │
│   │   ├─ Implication: No need to hold 1000s of stocks; just rf + market portfolio
│   │   ├─ Proof: All optimal portfolios lie on CAL; CAL connects rf to market
│   │   └─ Practical: Justifies index funds; don't beat market through picking
│   │
│   ├─ Conditions for Two-Fund Separation:
│   │   ├─ All investors have mean-variance utility (same form)
│   │   ├─ All investors have same beliefs (agree on μ, Σ)
│   │   ├─ Risk-free rate same for borrowing/lending
│   │   ├─ No taxes or transactions costs
│   │   └─ No short-sale constraints (or same for all)
│   │
│   ├─ Why Markets Price Assets This Way:
│   │   ├─ If someone beats market → arbitrageurs copy → market adjusts
│   │   ├─ Equilibrium: No arbitrage → CAL tangent to efficient frontier
│   │   ├─ Market portfolio emerges as consensus tangency portfolio
│   │   └─ Individual stocks priced so efficient frontier touches CAL at market
│   │
│   └─ Breakdown Conditions (Reality Adjustments):
│       ├─ Different beliefs → different optimal portfolios (but still two funds each)
│       ├─ Taxes differ by person → optimal mix of rf + market varies
│       ├─ Short-sale constraints → inefficient frontier; need three or more funds
│       ├─ Leverage constraints → poor investors stuck at 100% market; can't lever
│       └─ Multiple risk-free rates (borrow > lend) → CAL has kink; multiple segments
│
├─ Optimal Allocation Mechanics:
│   ├─ Adjustment Process:
│   │   ├─ Step 1: Estimate market expected return μm and volatility σm
│   │   ├─ Step 2: Assess risk-free rate rf
│   │   ├─ Step 3: Determine your risk aversion coefficient λ (questionnaire or revealed)
│   │   ├─ Step 4: Solve tangency condition: σp* = (μm - rf) / (λ × σm)
│   │   ├─ Step 5: Allocate: w_risky = σp*/σm; w_rf = 1 - w_risky
│   │   └─ Step 6: Rebalance periodically (as beliefs/λ change)
│   │
│   ├─ Sensitivity to Parameters:
│   │   ├─ 1% increase in μm → higher risky allocation, more return
│   │   ├─ 1% increase in σm → lower risky allocation, less leverage
│   │   ├─ 1% increase in rf → lower risky allocation (rf more attractive)
│   │   ├─ 1 unit increase in λ → halve risky allocation (rough approximation)
│   │   └─ Elasticities: w_risky ∝ (μm - rf), w_risky ∝ σm⁻², w_risky ∝ λ⁻¹
│   │
│   └─ Monitoring & Rebalancing:
│       ├─ Quarterly: Check if market performance caused allocation drift
│       ├─ Semi-annually: Reassess beliefs (μm, σm changed?)
│       ├─ Annually: Recalibrate λ (life changes: promotion, kids, inheritance?)
│       ├─ Threshold-based: Rebalance if allocation drifts >5% from target
│       └─ Mechanical: Execute rebalancing regardless of market timing views
│
└─ Beyond Two Funds (Real-World Complications):
    ├─ Transaction costs: Optimal allocation shifts; high turnover costly
    ├─ Taxes: Different effective risk-free rates (after-tax); may need three funds (taxable bonds)
    ├─ Regulatory constraints: Pension funds can't borrow; insurance firms limited leverage
    ├─ Information asymmetry: Some investors have better beliefs than others
    ├─ Behavioral factors: Investor may not follow optimal allocation (home bias, familiarity)
    └─ Model risk: Uncertainty in μ, Σ, λ → robust optimization with constraints
```

**Mathematical Formulas:**

Indifference curve (parametric):
$$E[R_p] = U_0 + \frac{\lambda}{2}\sigma_p^2$$

Marginal Rate of Substitution:
$$MRS = \frac{dE[R_p]}{d\sigma_p} = \lambda \sigma_p$$

Capital Allocation Line:
$$E[R_p] = r_f + \frac{\mu_m - r_f}{\sigma_m} \sigma_p = r_f + SR_m \cdot \sigma_p$$

Optimal volatility (tangency condition):
$$\sigma_p^* = \frac{\mu_m - r_f}{\lambda \sigma_m}$$

Optimal allocation (two-fund separation):
$$w_{risky}^* = \frac{\sigma_p^*}{\sigma_m} = \frac{\mu_m - r_f}{\lambda \sigma_m^2}$$

---

## 5. Mini-Project: Visualizing Indifference Curves & Optimal Selection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import yfinance as yf

# Compute and visualize indifference curves; find optimal allocation graphically

def get_efficient_frontier_data(start_date, end_date):
    """
    Fetch market data to estimate frontier and CAL.
    """
    tickers = ['SPY', 'BND']  # Stocks and Bonds
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns


def estimate_cov_matrix(returns):
    """
    Estimate covariance matrix and mean returns.
    """
    annual_returns = returns.mean() * 252
    annual_cov = returns.cov() * 252
    
    return annual_returns, annual_cov


def compute_efficient_frontier_points(annual_returns, annual_cov, n_points=50):
    """
    Compute efficient frontier from 0% to 100% in risky asset.
    """
    results = []
    
    for w_risky in np.linspace(0, 1.5, n_points):  # Allow leverage up to 150%
        w = np.array([w_risky, 1 - w_risky])
        
        portfolio_return = np.sum(w * annual_returns)
        portfolio_var = w @ annual_cov @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        results.append({
            'w_risky': w_risky,
            'return': portfolio_return,
            'volatility': portfolio_vol
        })
    
    return pd.DataFrame(results)


def compute_indifference_curves(lambda_coeff, utility_levels, vol_range):
    """
    For each utility level, compute E[Rp] = U + (λ/2) σp²
    """
    curves = {}
    
    for u_level in utility_levels:
        returns = u_level + (lambda_coeff / 2) * vol_range**2
        curves[f'U={u_level:.3f}'] = returns
    
    return curves


def find_tangency_portfolio(annual_returns, annual_cov, rf_rate):
    """
    Find portfolio with highest Sharpe ratio (tangent to CAL).
    """
    n_assets = len(annual_returns)
    
    def neg_sharpe(w):
        p_return = np.sum(w * annual_returns)
        p_var = w @ annual_cov @ w
        p_vol = np.sqrt(p_var)
        sharpe = (p_return - rf_rate) / p_vol
        return -sharpe
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([0.6, 0.4])
    
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    w_tangent = result.x
    p_return = np.sum(w_tangent * annual_returns)
    p_vol = np.sqrt(w_tangent @ annual_cov @ w_tangent)
    sharpe = (p_return - rf_rate) / p_vol
    
    return w_tangent, p_return, p_vol, sharpe


def compute_cal(rf_rate, tangent_return, tangent_vol, vol_range):
    """
    Capital Allocation Line: E[Rp] = rf + (E[Rm] - rf)/σm × σp
    """
    sharpe = (tangent_return - rf_rate) / tangent_vol
    cal_returns = rf_rate + sharpe * vol_range
    return cal_returns


def optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate):
    """
    Find optimal allocation on CAL for given λ.
    """
    sharpe = (tangent_return - rf_rate) / tangent_vol
    
    # Tangency condition: λ σp = sharpe
    opt_vol = sharpe / lambda_coeff
    opt_return = rf_rate + sharpe * opt_vol
    
    return opt_vol, opt_return


# Main Analysis
print("=" * 100)
print("INDIFFERENCE CURVES & OPTIMAL PORTFOLIO SELECTION")
print("=" * 100)

# 1. Data
print("\n1. MARKET DATA & PARAMETERS")
print("-" * 100)

returns = get_efficient_frontier_data('2015-01-01', '2024-01-01')
annual_returns, annual_cov = estimate_cov_matrix(returns)
rf_rate = 0.025

print(f"Risk-free rate: {rf_rate:.2%}")
print(f"Stocks (SPY): {annual_returns['SPY']:.2%} return, {np.sqrt(annual_cov.iloc[0, 0]):.2%} vol")
print(f"Bonds (BND):  {annual_returns['BND']:.2%} return, {np.sqrt(annual_cov.iloc[1, 1]):.2%} vol")
print(f"Correlation: {annual_cov.iloc[0, 1] / (np.sqrt(annual_cov.iloc[0, 0]) * np.sqrt(annual_cov.iloc[1, 1])):.2f}")

# 2. Tangency portfolio
print("\n2. TANGENCY PORTFOLIO (Highest Sharpe Ratio)")
print("-" * 100)

w_tangent, tangent_return, tangent_vol, tangent_sharpe = find_tangency_portfolio(
    annual_returns, annual_cov, rf_rate
)

print(f"Expected return: {tangent_return:.2%}")
print(f"Volatility: {tangent_vol:.2%}")
print(f"Sharpe ratio: {tangent_sharpe:.3f}")
print(f"Allocation: {w_tangent[0]:.1%} stocks, {w_tangent[1]:.1%} bonds")

# 3. Optimal allocations for different λ values
print("\n3. OPTIMAL ALLOCATION BY RISK AVERSION COEFFICIENT")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
optimal_allocations = {}

for lambda_coeff in lambda_values:
    opt_vol, opt_return = optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate)
    
    w_risky = opt_vol / tangent_vol
    w_rf = 1 - w_risky
    
    optimal_allocations[lambda_coeff] = {
        'vol': opt_vol,
        'return': opt_return,
        'w_risky': w_risky,
        'w_rf': w_rf
    }
    
    print(f"\nλ = {lambda_coeff}:")
    print(f"  Optimal vol: {opt_vol:.2%}")
    print(f"  Optimal return: {opt_return:.2%}")
    print(f"  Allocation: {w_risky:5.1%} risky portfolio + {w_rf:5.1%} risk-free")
    print(f"  Utility: {opt_return - (lambda_coeff/2) * opt_vol**2:.4f}")

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Indifference curves and CAL
ax = axes[0, 0]

vol_range = np.linspace(0, 0.40, 200)

# CAL
cal_returns = compute_cal(rf_rate, tangent_return, tangent_vol, vol_range)
ax.plot(vol_range * 100, cal_returns * 100, 'k-', linewidth=3, label='Capital Allocation Line (CAL)')

# Indifference curves for different λ
colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
for i, lambda_coeff in enumerate(lambda_values):
    opt_vol, opt_return = optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate)
    utility = opt_return - (lambda_coeff / 2) * opt_vol**2
    
    indiff_returns = utility + (lambda_coeff / 2) * vol_range**2
    ax.plot(vol_range * 100, indiff_returns * 100, '--', color=colors[i], linewidth=2, 
            label=f'Indifference (λ={lambda_coeff}), U={utility:.4f}')
    
    # Mark optimal point
    ax.scatter(opt_vol * 100, opt_return * 100, s=200, color=colors[i], marker='*', zorder=5)

# Risk-free asset
ax.scatter([0], [rf_rate * 100], s=300, marker='o', color='green', label='Risk-free asset', zorder=5)

# Tangency portfolio
ax.scatter([tangent_vol * 100], [tangent_return * 100], s=300, marker='s', color='red', 
          label='Tangency portfolio', zorder=5)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Indifference Curves & Capital Allocation Line', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim([0, 40])
ax.set_ylim([0, 15])

# Plot 2: Optimal allocation vs λ
ax = axes[0, 1]

lambdas = list(optimal_allocations.keys())
risky_weights = [optimal_allocations[l]['w_risky'] for l in lambdas]
rf_weights = [optimal_allocations[l]['w_rf'] for l in lambdas]

x = np.arange(len(lambdas))
width = 0.35

ax.bar(x - width/2, [w * 100 for w in risky_weights], width, label='Risky portfolio', color='#3498db', alpha=0.8)
ax.bar(x + width/2, [w * 100 for w in rf_weights], width, label='Risk-free', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Allocation (%)', fontsize=12)
ax.set_title('Optimal Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'λ={l}' for l in lambdas])
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.axhline(y=0, color='k', linewidth=0.8)

# Plot 3: Utility by λ
ax = axes[1, 0]

utilities = [optimal_allocations[l]['return'] - (l / 2) * optimal_allocations[l]['vol']**2 
            for l in lambdas]

ax.plot(lambdas, utilities, 'o-', linewidth=2.5, markersize=8, color='#2ecc71')
ax.fill_between(lambdas, utilities, alpha=0.3, color='#2ecc71')

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Utility', fontsize=12)
ax.set_title('Optimal Utility by Risk Aversion', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Plot 4: Tangency condition illustration
ax = axes[1, 1]

vol_range = np.linspace(0, 0.40, 200)
lambda_coeff = 4.0

# Indifference curve for λ=4
opt_vol, opt_return = optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate)
utility = opt_return - (lambda_coeff / 2) * opt_vol**2
indiff_returns = utility + (lambda_coeff / 2) * vol_range**2

# MRS along indifference curve
mrs = lambda_coeff * vol_range

# CAL slope (constant)
cal_slope = (tangent_return - rf_rate) / tangent_vol

ax.plot(vol_range * 100, mrs * 100, 'b-', linewidth=2.5, label=f'MRS (λ={lambda_coeff})')
ax.axhline(y=cal_slope * 100, color='r', linestyle='--', linewidth=2.5, label=f'CAL slope = {cal_slope:.3f}')

# Tangency point
ax.scatter([opt_vol * 100], [lambda_coeff * opt_vol * 100], s=300, marker='*', 
          color='green', label='Optimal (MRS = CAL slope)', zorder=5)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Marginal Rate of Substitution (dE[R]/dσ)', fontsize=12)
ax.set_title('Tangency Condition: MRS = CAL Slope', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 40])

plt.tight_layout()
plt.savefig('indifference_curves_optimal_selection.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: indifference_curves_optimal_selection.png")
plt.show()

# 5. Key insights
print("\n4. KEY INSIGHTS & INTERPRETATION")
print("-" * 100)
print(f"""
INDIFFERENCE CURVE GEOMETRY:
├─ Parabolic shape: E[Rp] = U₀ + (λ/2)σp²
├─ Curvature increases with λ (risk-averse investors have more curved preferences)
├─ Higher indifference curves represent higher utility (farther from origin)
└─ Curves never intersect (transitive preferences)

MARGINAL RATE OF SUBSTITUTION (MRS):
├─ Formula: MRS = λ × σp (slope of indifference curve)
├─ Interpretation: How much return willing to sacrifice per unit volatility reduction
├─ Increases with σp: Steeper slopes at higher volatility (flattens demand for more vol)
├─ Higher λ: Steeper curve at all σp (more risk-averse; demands higher return per vol)
└─ Example: At σp=10% with λ=4, MRS=0.4 (willing to sacrifice 0.4% return per 1% vol cut)

CAPITAL ALLOCATION LINE (CAL):
├─ Linear opportunity set: E[Rp] = {rf_rate:.2%} + {cal_slope:.3f} × σp
├─ Tangent portfolio composition: {w_tangent[0]:.1%} stocks, {w_tangent[1]:.1%} bonds
├─ Slope (Sharpe ratio): {cal_slope:.3f} (risk premium per unit risk)
├─ All investors lie on CAL; choice determined by λ (how much risk to take)
└─ Leverage extends CAL rightward; leverage is controlled by borrowing

OPTIMAL PORTFOLIO SELECTION:
├─ Tangency condition: MRS = CAL slope at optimum
├─ Optimization: Max utility subject to feasibility on CAL
├─ Solution: Higher λ → move left on CAL (lower vol, lower return)
├─ Solution: Lower λ → move right on CAL (higher vol, higher return, may lever)
└─ Example: λ=2 investor holds {optimal_allocations[2.0]['w_risky']:.1%} risky + {optimal_allocations[2.0]['w_rf']:.1%} risk-free

TWO-FUND SEPARATION:
├─ Implication: All investors hold combination of risk-free + one risky portfolio
├─ Market portfolio = optimal risky portfolio (for investor with market's λ)
├─ Institutional practice: Index funds (market proxy) + bonds/bills
├─ Theoretical: Explains why beating market hard (everyone agrees on asset pricing)
└─ Violation: Home bias, behavioral preferences → people don't follow pure two-fund

PRACTICAL RECOMMENDATIONS:
├─ Estimate λ from questionnaire or revealed preference (historical allocation)
├─ Find tangency portfolio (market portfolio often serves this)
├─ Locate optimal point on CAL based on your λ
├─ Allocate: w_risky = {cal_slope:.3f} / (λ × {tangent_vol:.2%}) = {cal_slope:.3f} / ({tangent_vol:.2%}λ)
├─ Rebalance periodically (quarterly) to maintain allocation
└─ Adjust λ if life changes: retirement, inheritance, income shock, time horizon shift
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Leverage vs Constraint:** Conservative investor (λ=8) on unconstrained CAL achieves w_risky = 22%. If leverage prohibited (max 100% risky), constrained optimum is w_risky = 100%. What is utility loss from constraint? How does utility gap scale with λ?

2. **Borrowing Rate Asymmetry:** What if borrowing rate (5%) > lending rate (2%)? How does this kink the CAL? Can aggressive investor still lever profitably? At what λ does kink become binding?

3. **Belief Disagreement:** Two investors see same efficient frontier but have different beliefs about rf (one thinks 2%, other thinks 5%). How do their indifference curve tangencies differ? Who prefers more risk?

4. **Time Horizon & λ:** Young investor (30 years) effective λ = 0.5; retiree (10 years) effective λ = 8. Why does λ increase with age? Is this rational (human capital horizon) or behavioral (loss aversion)?

5. **Indifference Curve Estimation:** From portfolio choice data (investor holds 60% stocks, 40% bonds), can you reverse-engineer λ? What assumptions needed? How sensitive is inference to belief assumptions?

---

## 7. Key References

- **Markowitz, H. (1959).** "Portfolio Selection: Efficient Diversification of Investments" – Geometric foundation of indifference curves; optimality conditions for portfolio selection.

- **Sharpe, W.F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium" – Introduced Capital Allocation Line (CAL); Security Market Line (SML); two-fund separation theorem.

- **Tobin, J. (1958).** "Liquidity Preference as Behavior Towards Risk" – Separated portfolio choice into allocation to risky portfolio vs risk-free; CAL concept.

- **Lintner, J. (1965).** "The Valuation of Risk Assets and the Selection of Risky Investments" – Generalized CAPM; indifference curve tangency with market equilibrium.

- **Merton, R.C. (1972).** "An Analytic Derivation of the Efficient Portfolio Frontier" – Mathematical treatment of efficient frontier geometry; optimization foundations.

- **CFA Level I Curriculum:** Portfolio Management chapter on utility functions and optimal selection – Practical implementation of indifference curves.

- **Investopedia: Capital Allocation Line** – https://www.investopedia.com/terms/c/cal.asp – Accessible explanation of CAL and optimal portfolio selection.

