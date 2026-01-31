# Capital Allocation Line & Capital Market Line (CAL/CML)

## 1. Concept Skeleton
**Definition:** Capital Allocation Line (CAL): feasible return combinations from mixing risk-free asset (rf) with single risky portfolio; E[Rp] = rf + (μm - rf)/σm × σp. Capital Market Line (CML): special case using market portfolio as risky component; equilibrium line on which all efficient portfolios lie.  
**Purpose:** Visualize risk-return trade-off; determine optimal portfolio allocation (where indifference curve tangent to CAL); establish market equilibrium; explain why all investors hold market + risk-free combination  
**Prerequisites:** Risk-free rate, efficient frontier, Sharpe ratio, mean-variance optimization

---

## 2. Comparative Framing

| Aspect | Capital Allocation Line (CAL) | Capital Market Line (CML) | Efficient Frontier | Indifference Curve |
|--------|------------------------------|--------------------------|-----------------|-------------------|
| **Definition** | Return combinations from rf + any risky portfolio | Return combinations from rf + market portfolio | All efficient portfolios (Pareto-optimal risk-return) | Investor preferences (constant utility) |
| **Equation** | E[Rp] = rf + (μr - rf)/σr × σp | E[Rp] = rf + (μm - rf)/σm × σp | Parabolic frontier (min var to tangency) | E[Rp] = U + (λ/2)σp² (parabolic upward) |
| **Shape** | Linear (straight line) | Linear (special case of CAL) | Curved (parabolic upward) | Parabolic (upward-opening) |
| **Y-intercept** | rf (when σp = 0) | rf (when σp = 0) | rf (at min var portfolio) | U (utility level at σ=0) |
| **Slope** | (μr - rf)/σr = Sharpe ratio of risky | (μm - rf)/σm = market Sharpe ratio | N/A (curved) | λσp (increases with volatility) |
| **Risky Component** | Any portfolio (user-specified) | Market portfolio (all assets weighted by cap) | Mix of all assets | N/A (pure preference) |
| **Uniqueness** | Different CAL for each risky portfolio | One CML (equilibrium line) | One efficient frontier | Different curve for each investor (λ) |
| **Practical Use** | Portfolio construction (choose risky, then allocate) | Asset pricing (equilibrium returns) | Benchmark (compare to efficient set) | Investor-specific optimization |
| **Tangency** | Tangent to CAL at investor's optimal | All efficient portfolios lie ON CML | CML tangent to frontier at market portfolio | Tangent to CAL at optimal choice |
| **Example** | rf=2%, risky portfolio μ=8%, σ=15% → CAL slope=0.4 | rf=2%, market μ=10%, σ=15% → CML slope=0.533 | All combinations of assets shown | U=5, λ=2 → indifference curve drawn |

**Key Insight:** CAL is specific to chosen risky portfolio; CML uses market portfolio (equilibrium). Optimal allocation found at tangency of indifference curve with CAL (which is CML in equilibrium).

---

## 3. Examples + Counterexamples

**Example 1: Constructing CAL for Single Risky Portfolio**

Investor considers mixing:
- Risk-free: T-bills at 2.5%
- Risky portfolio: Tech stocks with E[R] = 12%, σ = 20%

CAL equation:
$$E[R_p] = 0.025 + \frac{0.12 - 0.025}{0.20} \times \sigma_p = 0.025 + 0.475 \times \sigma_p$$

Points on CAL:
- At σp = 0%: E[Rp] = 2.5% (100% risk-free)
- At σp = 10%: E[Rp] = 2.5% + 0.475 × 10% = 7.25% (50% tech, 50% T-bills)
- At σp = 20%: E[Rp] = 2.5% + 0.475 × 20% = 12.0% (100% tech)
- At σp = 30%: E[Rp] = 2.5% + 0.475 × 30% = 16.75% (150% tech, -50% T-bills = leveraged)

CAL slope = 0.475 (Sharpe ratio of tech portfolio)

**Interpretation:** Every 1% of volatility accepted → 0.475% return gained (favorable trade-off).

**Example 2: Comparing Two CALs (Which Risky Portfolio Better?)**

Portfolio A (Tech): E[R] = 12%, σ = 20%, rf = 2.5%
- CAL_A slope = (12% - 2.5%) / 20% = 0.475

Portfolio B (Bonds + Stocks): E[R] = 8%, σ = 12%, rf = 2.5%
- CAL_B slope = (8% - 2.5%) / 12% = 0.458

Comparison:
- CAL_A slope (0.475) > CAL_B slope (0.458)
- **Implication:** Portfolio A offers better risk-adjusted return (higher Sharpe ratio)
- **Decision:** Rational investors prefer to combine rf with Portfolio A over Portfolio B
- **Why:** More return per unit risk (0.475 > 0.458)

**Graphically:** CAL_A lies above CAL_B for all risk levels > 0; dominates CAL_B.

**Example 3: Capital Market Line in Equilibrium**

Market equilibrium:
- Market portfolio: E[Rm] = 10%, σm = 15%, β = 1.0 (definition)
- Risk-free rate: rf = 2.5%
- CML slope: (10% - 2.5%) / 15% = 0.5

CML equation:
$$E[R_p] = 0.025 + 0.5 \times \sigma_p$$

Two-fund separation: All investors hold:
- Combination 1 (Conservative): 20% market + 80% risk-free
  - E[Rp] = 0.025 + 0.5 × (0.2 × 15%) = 0.025 + 1.5% = 3.75%
  - σp = 0.2 × 15% = 3%
  
- Combination 2 (Moderate): 60% market + 40% risk-free
  - E[Rp] = 0.025 + 0.5 × (0.6 × 15%) = 0.025 + 4.5% = 6.5%
  - σp = 0.6 × 15% = 9%
  
- Combination 3 (Aggressive): 100% market + 0% risk-free
  - E[Rp] = 10%, σp = 15%
  
- Combination 4 (Leveraged): 150% market - 50% risk-free
  - E[Rp] = 0.025 + 0.5 × (1.5 × 15%) = 0.025 + 11.25% = 13.25%
  - σp = 1.5 × 15% = 22.5%

**Key insight:** All combinations are efficient (lie on CML); difference determined by risk aversion (λ).

**Example 4: Deviations from CML (Arbitrage Opportunity)**

Suppose individual stock has:
- Expected return: 8%
- Beta: 0.8
- **CAPM prediction:** E[Ri] = 2.5% + 0.8 × (10% - 2.5%) = 8.5%
- **Actual return:** 8%
- **Difference:** α = 8% - 8.5% = -0.5% (negative alpha; underpriced risk?)

Wait, -0.5% alpha means stock offers 0.5% below CAPM prediction (overpriced):
- Return too low for risk level
- Investor can short stock, buy CML portfolio with same beta
- Capture 0.5% riskless return (arbitrage)

**Arbitrage strategy (zero-cost portfolio):**
- Short 1 unit of stock (E[R] = 8%, β = 0.8)
- Long 0.8 units of market (E[R] = 10%, β = 0.8)
- Long 0.2 units of risk-free (E[R] = 2.5%, β = 0)
- **Portfolio return:** -8% + 0.8 × 10% + 0.2 × 2.5% = -8% + 8% + 0.5% = 0.5% (riskless!)
- **Portfolio beta:** -0.8 + 0.8 × 1.0 + 0.2 × 0 = 0 (no market risk)

If repeated by many arbitrageurs, stock price falls → expected return rises to 8.5% (CML).

**Implication:** Arbitrage keeps all stocks on or near CML; deviations temporary.

**Example 5: Leverage Constraint (Borrowing Limit)**

Investor wants allocation:
- 120% market (leveraged) + -20% risk-free (borrowed)
- E[Rp] = 0.025 + 0.5 × (1.2 × 15%) = 12.5%
- σp = 1.2 × 15% = 18%

But broker limits leverage to maximum 50% (can borrow up to 50% of portfolio):
- Constrained optimum: 150% market - 50% risk-free (maximum leverage allowed)
- E[Rp] = 0.025 + 0.5 × (1.5 × 15%) = 13.25%
- σp = 22.5%

**Implication:** Constraint binds for aggressive investor; constrained optimum worse than unconstrained; welfare loss from leverage limit.

---

## 4. Layer Breakdown

```
Capital Allocation Line & Capital Market Line Architecture:

├─ Capital Allocation Line (CAL) Definition:
│   ├─ Feasible set of portfolios combining risk-free + risky asset
│   │   │ Allocation: w_rf + w_risky = 1 (fully invested or leveraged)
│   │   │ Return: Rp = w_rf × rf + w_risky × R_risky
│   │   │ Expected return: E[Rp] = w_rf × rf + w_risky × E[R_risky]
│   │   │
│   │   │ For varying w_risky ∈ [0, ∞):
│   │   │ E[Rp] = (1 - w_risky) × rf + w_risky × E[R_risky]
│   │   │       = rf + w_risky × (E[R_risky] - rf)
│   │   │
│   │   │ Volatility: σp = w_risky × σ_risky (risk-free uncorrelated with risky)
│   │   │ Substituting w_risky = σp / σ_risky:
│   │   │ E[Rp] = rf + (E[R_risky] - rf) / σ_risky × σp  ← CAL equation
│   │   │
│   │   └─ Slope: (E[R_risky] - rf) / σ_risky = Sharpe ratio (return per unit risk)
│   │
│   ├─ Properties of CAL:
│   │   ├─ Linear (straight line, not curved)
│   │   ├─ Y-intercept: rf (when σp = 0)
│   │   ├─ Slope: Sharpe ratio of risky portfolio (risk premium / volatility)
│   │   ├─ Extends from (0, rf) rightward infinitely (via leverage)
│   │   ├─ All points on CAL feasible via allocation decision
│   │   ├─ Points above CAL: infeasible (cannot achieve without accepting more risk)
│   │   └─ Points below CAL: inefficient (suboptimal allocation)
│   │
│   ├─ Multiple CALs:
│   │   ├─ Different risky portfolio → Different CAL (different slope)
│   │   ├─ Steeper CAL: Higher Sharpe ratio (better risk-return trade-off)
│   │   ├─ Flatter CAL: Lower Sharpe ratio (worse risk-return trade-off)
│   │   ├─ Rational investor choice: Select risky portfolio with highest Sharpe
│   │   ├─ Implication: One portfolio dominates others (tangency portfolio)
│   │   └─ In equilibrium: Market portfolio = tangency portfolio
│   │
│   └─ CAL Parametrization:
│       ├─ By weight w_risky ∈ [0, ∞):
│       │   ├─ w_risky = 0 → 100% rf (σp=0, E[Rp]=rf)
│       │   ├─ w_risky = 1 → 100% risky (σp=σ_risky, E[Rp]=E[R_risky])
│       │   ├─ w_risky > 1 → Leverage (borrow at rf, invest in risky)
│       │   └─ w_risky < 0 → Short risky (sell risky, lend at rf, unfeasible usually)
│       │
│       ├─ By volatility target σp:
│       │   └─ Directly solve for allocation via w_risky = σp / σ_risky
│       │
│       └─ By return target E[Rp]:
│           └─ Directly solve for volatility via σp = (E[Rp] - rf) × σ_risky / (E[R_risky] - rf)
│
├─ Capital Market Line (CML) as Special Case:
│   ├─ Definition: CAL using market portfolio as risky component
│   │   │ CML = CAL with R_risky = market portfolio (M)
│   │   │ E[Rp] = rf + (E[Rm] - rf) / σm × σp
│   │   │
│   │   ├─ Where:
│   │   │   ├─ rf = risk-free rate (T-bill or Treasury)
│   │   │   ├─ E[Rm] = expected market return
│   │   │   ├─ σm = market volatility (standard deviation)
│   │   │   ├─ σp = portfolio volatility (on CML)
│   │   │   └─ (E[Rm] - rf) = equity risk premium (market premium)
│   │   │
│   │   └─ CML slope = Market Sharpe ratio (most efficient in equilibrium)
│   │
│   ├─ Why CML = Equilibrium Line:
│   │   ├─ Efficient frontier theorem: In equilibrium, all efficient portfolios lie on CML
│   │   │   └─ Proof: If portfolio P on efficient frontier but below CML, arbitrage exists
│   │   ├─ Market portfolio uniqueness: Only ONE portfolio that satisfies tangency
│   │   │   ├─ Tangency: Highest Sharpe ratio
│   │   │   ├─ Market composition: All assets by market capitalization
│   │   │   └─ All investors agree on risky portfolio composition (two-fund separation)
│   │   │
│   │   ├─ Why everyone holds market:
│   │   │   ├─ If someone holds non-market risky portfolio, arbitrage improves it
│   │   │   ├─ Demand shifts toward higher-Sharpe-ratio assets
│   │   │   ├─ Prices adjust until market portfolio = highest Sharpe
│   │   │   └─ Equilibrium: All investors hold market + rf (proportions differ by λ)
│   │   │
│   │   └─ Implication: No one can beat market through stock picking
│   │       └─ (Consistent with efficient markets hypothesis)
│   │
│   ├─ Distinction from CAL:
│   │   ├─ CAL: For any arbitrary risky portfolio (user-chosen)
│   │   ├─ CML: Specifically for market portfolio (equilibrium component)
│   │   ├─ All CALs different → One CML in equilibrium
│   │   ├─ CAL practical: Portfolio construction (choose risky, then allocate)
│   │   └─ CML theoretical: Asset pricing (equilibrium returns on CML)
│   │
│   └─ CML vs CAPM (Related but Distinct):
│       ├─ CML: Defines efficient frontier in return-risk space
│       │   └─ ONLY applies to efficient portfolios (on frontier)
│       ├─ CAPM: Defines expected return for ANY asset (efficient or not)
│       │   └─ E[Ri] = rf + βi(E[Rm] - rf)
│       │   └─ Uses beta (systematic risk), not total volatility
│       ├─ Connection: CML special case of CAPM for efficient portfolios
│       │   └─ If portfolio on CML, can derive β, verify E[Rp] via CAPM
│       └─ Practical: CML for efficient portfolios; CAPM for individual assets
│
├─ Optimal Allocation (Tangency with Indifference Curve):
│   ├─ Optimization Problem:
│   │   │ Max U(Rp) = E[Rp] - (λ/2)σp²
│   │   │ Subject to: E[Rp] = rf + (E[Rm] - rf)/σm × σp  ← CAL constraint
│   │   │
│   │   ├─ Interpretation: Maximize utility subject to feasibility on CAL
│   │   ├─ Solution: Tangency point (indifference curve touches CAL)
│   │   └─ Uniqueness: Only ONE point on CAL maximizes U (for given λ)
│   │
│   ├─ Tangency Condition (First-Order):
│   │   │ MRS (indifference) = CAL slope (market trade-off)
│   │   │ λσp* = (E[Rm] - rf) / σm
│   │   │ Solving: σp* = (E[Rm] - rf) / (λ × σm)
│   │   │         E[Rp*] = rf + (E[Rm] - rf) / σm × σp*
│   │   │
│   │   ├─ Interpretation:
│   │   │   ├─ Higher λ (more risk-averse) → lower σp* (less risk taken)
│   │   │   ├─ Higher market premium → higher σp* (more reward for risk)
│   │   │   ├─ Higher σm (more volatile market) → lower σp* (lower efficient frontier)
│   │   │   └─ At optimum: Cannot improve by moving along CAL
│   │   │
│   │   └─ Second-order condition: d²U/dσp² < 0 (concavity ensures maximum)
│   │
│   ├─ Graphical Solution:
│   │   ├─ Plot 1: CAL (linear) and indifference curves (parabolic, fanning out)
│   │   ├─ Plot 2: Find point where indifference curve is tangent to CAL
│   │   ├─ Plot 3: At tangency, slope of indifference = slope of CAL
│   │   └─ Plot 4: Higher utility curves (shifted outward) not reachable on CAL
│   │
│   ├─ Allocation Decision:
│   │   ├─ Once σp* found, determine portfolio weights:
│   │   │   ├─ w_market = σp* / σm (fraction in market portfolio)
│   │   │   ├─ w_rf = 1 - w_market (fraction in risk-free)
│   │   │   └─ If w_market > 1: Borrow (leverage)
│   │   │
│   │   ├─ Example: σp* = 10%, σm = 15%
│   │   │   ├─ w_market = 10% / 15% = 66.7%
│   │   │   ├─ w_rf = 1 - 0.667 = 33.3% (safe cushion)
│   │   │   └─ Portfolio: 66.7% market + 33.3% T-bills
│   │   │
│   │   └─ Sensitivity:
│   │       ├─ If λ doubled: σp* halved → move toward risk-free
│   │       ├─ If market premium doubled: σp* doubled → take more risk
│   │       └─ Small changes in λ, premium → large allocation shifts
│   │
│   └─ Welfare Implications:
│       ├─ Utility at optimum: U* = E[Rp*] - (λ/2)(σp*)²
│       ├─ If constrained (e.g., max 50% stock): Utility loss from sub-optimality
│       ├─ Larger λ (conservative) → Utility gain from constraint (more predictable)
│       └─ Smaller λ (aggressive) → Utility loss from constraint (cannot lever enough)
│
├─ Borrowing & Leverage Considerations:
│   ├─ Unlimited Borrowing (Theoretical):
│   │   ├─ CAL extends infinitely right (σp can be arbitrarily large)
│   │   ├─ Aggressive investors can borrow at rf, invest all in market
│   │   ├─ Optimal allocation: w_market = (E[Rm] - rf) / (λ σm²) can exceed 1
│   │   └─ Example: w_market = 1.5 means borrow 50% of wealth at rf
│   │
│   ├─ Borrowing Constraints (Reality):
│   │   ├─ Investors cannot borrow at rf (borrow rate > rf by credit spread)
│   │   ├─ Leverage often limited to max 50% (broker/regulator rule)
│   │   ├─ If constraint binds: w_market = limit, allocate remainder to rf
│   │   ├─ Constrained optimal: Allocate 100% to market (maximum w_market)
│   │   └─ Welfare loss: Cannot reach unconstrained tangency point
│   │
│   ├─ Short-Selling Constraints:
│   │   ├─ If cannot short: w_rf ≥ 0 (cannot borrow, only lend)
│   │   ├─ CAL cut off at w_market = 1 (100% market, 0% risk-free)
│   │   ├─ Aggressive investor stuck at market point (cannot lever)
│   │   └─ Conservative investor can move left on CAL (toward rf)
│   │
│   └─ Implication:
│       ├─ Borrowing constraints push allocation toward 100% market
│       ├─ Conservative investors prioritize risk-free over risky
│       ├─ Aggregate: May cause two-CAL regime (different borrowing rates)
│       └─ Practical: Real portfolios deviate from pure two-fund model
│
├─ Time Horizon & Dynamic CAL:
│   ├─ Single-Period CAL (Static):
│   │   ├─ Portfolio held to end of 1 period (e.g., 1 year)
│   │   ├─ Return = (end wealth - initial) / initial
│   │   ├─ CAL valid for this horizon only
│   │   └─ Multi-period planning: Assumptions weaken (reinvestment risk)
│   │
│   ├─ Multi-Period CAL (Dynamic):
│   │   ├─ Investor rebalances periodically (quarterly, annually)
│   │   ├─ CAL shifts over time (market expected return, volatility change)
│   │   ├─ Optimal allocation may change with new expectations
│   │   └─ Example: After market crash, expected return ↑, CAL slopes steeper
│   │
│   └─ Liability-Driven Investing (LDI):
│       ├─ When investor has fixed liability (e.g., pension payment $1M in 5 years)
│       ├─ Risk-free = bond matching liability duration
│       ├─ CAL modified: Constraint is achieving liability, not arbitrary return
│       ├─ Allocation: Hedge liability (immunization) + excess into risky
│       └─ Practical: Institutions modify CAL for liability structure
│
└─ Empirical & Practical Considerations:
    ├─ CML vs Empirical Efficient Frontier:
    │   ├─ CML prediction: All efficient portfolios lie on straight line
    │   ├─ Empirical observation: Frontier curved (especially low-risk end)
    │   ├─ Reason: CML assumes mean-variance utility + normality (not exact)
    │   ├─ Deviations: Higher moments matter (skewness, kurtosis)
    │   └─ Implication: CML useful baseline, but not perfect model
    │
    ├─ Market Portfolio Identification:
    │   ├─ Theoretical: Market = all assets, market-cap weighted
    │   ├─ Practical: Use broad index (S&P 500, total market fund)
    │   ├─ Concern: S&P 500 ≈ 80% U.S. GDP, missing alternatives (RE, commodities)
    │   ├─ Adjustment: Include alternative assets in risky portfolio
    │   └─ Data: Market portfolio not directly observable; use proxies
    │
    ├─ Rebalancing Mechanics:
    │   ├─ Once allocation determined (e.g., 70% market, 30% rf)
    │   ├─ As time passes, weights drift (market outperforms rf)
    │   ├─ Rebalancing: Sell market winners, buy rf losers (contrarian)
    │   ├─ Frequency: Quarterly typical (balance with tax, transaction costs)
    │   └─ Benefit: Locks in gains, maintains target risk
    │
    └─ Alternative to Two-Fund:
        ├─ Multi-fund approach: rf + market + alternatives (commodities, RE)
        ├─ Three-fund portfolio: Total stock + total bond + total intl
        ├─ Motivation: Better diversification, tail risk hedging
        ├─ Trade-off: Complexity vs benefits (marginal gain for most)
        └─ Practical: Two-fund sufficient for most investors
```

**Mathematical Formulas:**

Capital Allocation Line:
$$E[R_p] = r_f + \frac{E[R_{risky}] - r_f}{\sigma_{risky}} \sigma_p$$

Capital Market Line (market portfolio):
$$E[R_p] = r_f + \frac{E[R_m] - r_f}{\sigma_m} \sigma_p$$

Optimal volatility (tangency condition):
$$\sigma_p^* = \frac{E[R_m] - r_f}{\lambda \sigma_m}$$

Allocation weights:
$$w_{market} = \frac{\sigma_p^*}{\sigma_m}, \quad w_{rf} = 1 - w_{market}$$

Sharpe ratio:
$$SR = \frac{E[R] - r_f}{\sigma}$$

---

## 5. Mini-Project: Visualizing CAL, CML & Optimal Allocation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Visualize Capital Allocation Line, Capital Market Line, optimal allocation

def get_market_data(start_date, end_date):
    """
    Fetch market and risk-free rate data.
    """
    # Market proxy (S&P 500)
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
    market_returns = spy_data.pct_change().dropna()
    
    # Risk-free rate approximation (T-bills, use constant for simplicity)
    rf = 0.04  # 4% current approximate rate
    
    return market_returns, rf


def compute_market_parameters(market_returns):
    """
    Estimate market expected return and volatility.
    """
    annual_return = market_returns.mean() * 252
    annual_vol = market_returns.std() * np.sqrt(252)
    
    return annual_return, annual_vol


def compute_cal(rf, risky_return, risky_vol, vol_range):
    """
    Compute Capital Allocation Line: E[Rp] = rf + (E[R_risky] - rf) / σ_risky × σp
    """
    sharpe_ratio = (risky_return - rf) / risky_vol
    cal_returns = rf + sharpe_ratio * vol_range
    return cal_returns, sharpe_ratio


def compute_indifference_curves(lambda_values, utility_levels, vol_range):
    """
    Compute indifference curves for different risk aversions.
    E[Rp] = U + (λ/2) σp²
    """
    curves = {}
    for lambda_coeff in lambda_values:
        curve_dict = {}
        for u_level in utility_levels:
            returns = u_level + (lambda_coeff / 2) * vol_range ** 2
            curve_dict[f'U={u_level:.3f}'] = returns
        curves[f'λ={lambda_coeff}'] = curve_dict
    return curves


def optimal_allocation(rf, market_return, market_vol, lambda_coeff):
    """
    Compute optimal allocation on CAL for given λ.
    Tangency: λσp* = (E[Rm] - rf) / σm
    """
    market_premium = market_return - rf
    optimal_vol = market_premium / (lambda_coeff * market_vol)
    optimal_return = rf + market_premium / market_vol * optimal_vol
    
    # Weights
    w_market = optimal_vol / market_vol
    w_rf = 1 - w_market
    
    return optimal_vol, optimal_return, w_market, w_rf


def compute_efficient_frontier_simple(rf, market_return, market_vol, num_points=50):
    """
    Efficient frontier approximated by CML.
    """
    vol_range = np.linspace(0, 0.4, num_points)
    cal_returns, _ = compute_cal(rf, market_return, market_vol, vol_range)
    return vol_range, cal_returns


# Main Analysis
print("=" * 100)
print("CAPITAL ALLOCATION LINE & CAPITAL MARKET LINE")
print("=" * 100)

# 1. Data
print("\n1. MARKET DATA & PARAMETERS")
print("-" * 100)

market_returns, rf = get_market_data('2015-01-01', '2024-01-01')
market_return, market_vol = compute_market_parameters(market_returns)

print(f"Risk-free rate (assumed): {rf*100:.2f}%")
print(f"Market expected return: {market_return*100:.2f}%")
print(f"Market volatility: {market_vol*100:.2f}%")
print(f"Market Sharpe ratio: {(market_return - rf) / market_vol:.3f}")

# 2. CAL slope (Sharpe ratio)
print("\n2. CAPITAL ALLOCATION LINE (CAL)")
print("-" * 100)

_, sharpe_ratio = compute_cal(rf, market_return, market_vol, np.array([0.15]))
print(f"\nCML equation: E[Rp] = {rf*100:.2f}% + {sharpe_ratio:.3f} × σp")
print(f"\nInterpretation: Every 1% volatility accepted → {sharpe_ratio*100:.2f}% return gained")

# Points on CAL
print(f"\nPoints on CML:")
print(f"{'Volatility':<15} {'Expected Return':<20} {'Allocation':<30}")
print("-" * 65)

for w_market in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    vol = w_market * market_vol
    ret = rf + (market_return - rf) * w_market
    w_rf = 1 - w_market
    allocation = f"{w_market*100:.0f}% market, {w_rf*100:.0f}% rf"
    if w_rf < 0:
        allocation = f"{w_market*100:.0f}% market, {abs(w_rf)*100:.0f}% borrowed"
    
    print(f"{vol*100:<15.2f} {ret*100:<20.2f} {allocation:<30}")

# 3. Optimal allocation for different λ values
print("\n3. OPTIMAL ALLOCATION BY RISK AVERSION")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
optimal_results = {}

for lambda_coeff in lambda_values:
    opt_vol, opt_return, w_market, w_rf = optimal_allocation(rf, market_return, market_vol, lambda_coeff)
    utility = opt_return - (lambda_coeff / 2) * opt_vol ** 2
    optimal_results[lambda_coeff] = {
        'vol': opt_vol,
        'return': opt_return,
        'w_market': w_market,
        'w_rf': w_rf,
        'utility': utility
    }
    
    print(f"\nλ = {lambda_coeff} (Risk aversion):")
    print(f"  Optimal volatility: {opt_vol*100:.2f}%")
    print(f"  Optimal return: {opt_return*100:.2f}%")
    print(f"  Utility: {utility:.4f}")
    print(f"  Allocation: {w_market*100:.1f}% market, {w_rf*100:.1f}% risk-free")
    if w_rf < 0:
        print(f"             (Leverage: borrow {abs(w_rf)*100:.1f}% to buy market)")

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: CAL and CML with indifference curves
ax = axes[0, 0]

vol_range = np.linspace(0, 0.35, 200)
cal_returns, _ = compute_cal(rf, market_return, market_vol, vol_range)

# CML
ax.plot(vol_range * 100, cal_returns * 100, 'k-', linewidth=3.5, label='Capital Market Line (CML)', zorder=3)

# Indifference curves
colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
for i, lambda_coeff in enumerate(lambda_values):
    opt_vol, opt_return, _, _ = optimal_allocation(rf, market_return, market_vol, lambda_coeff)
    utility = opt_return - (lambda_coeff / 2) * opt_vol ** 2
    
    indiff_returns = utility + (lambda_coeff / 2) * vol_range ** 2
    ax.plot(vol_range * 100, indiff_returns * 100, '--', color=colors[i], linewidth=2,
            label=f'Indifference (λ={lambda_coeff})', alpha=0.7)
    
    # Optimal point
    ax.scatter(opt_vol * 100, opt_return * 100, s=250, color=colors[i], marker='*', zorder=5)

# Risk-free and market
ax.scatter([0], [rf * 100], s=300, marker='o', color='green', label='Risk-free', zorder=5)
ax.scatter([market_vol * 100], [market_return * 100], s=300, marker='s', color='red',
          label='Market Portfolio', zorder=5)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Capital Market Line & Indifference Curves', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim([0, 35])
ax.set_ylim([2, 20])

# Plot 2: Allocation by λ
ax = axes[0, 1]

lambdas = list(optimal_results.keys())
market_weights = [optimal_results[l]['w_market'] for l in lambdas]
rf_weights = [optimal_results[l]['w_rf'] for l in lambdas]

x = np.arange(len(lambdas))
width = 0.35

ax.bar(x - width/2, [w * 100 for w in market_weights], width, label='Market Portfolio', 
      color='#3498db', alpha=0.8)
ax.bar(x + width/2, [w * 100 for w in rf_weights], width, label='Risk-Free', 
      color='#e74c3c', alpha=0.8)

ax.set_ylabel('Allocation (%)', fontsize=12)
ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_title('Optimal Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'λ={l}' for l in lambdas])
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.axhline(y=0, color='k', linewidth=0.8)

# Plot 3: Risk-return by λ
ax = axes[1, 0]

vols = [optimal_results[l]['vol'] for l in lambdas]
rets = [optimal_results[l]['return'] for l in lambdas]

ax.plot(lambdas, [v * 100 for v in vols], 'o-', linewidth=2.5, markersize=8, 
       label='Volatility', color='#e74c3c')
ax_twin = ax.twinx()
ax_twin.plot(lambdas, [r * 100 for r in rets], 's-', linewidth=2.5, markersize=8,
            label='Expected Return', color='#27ae60')

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Volatility (%)', fontsize=12, color='#e74c3c')
ax_twin.set_ylabel('Expected Return (%)', fontsize=12, color='#27ae60')
ax.set_title('Optimal Portfolio Risk-Return by λ', fontweight='bold', fontsize=13)
ax.tick_params(axis='y', labelcolor='#e74c3c')
ax_twin.tick_params(axis='y', labelcolor='#27ae60')
ax.grid(alpha=0.3)

# Plot 4: Utility by λ
ax = axes[1, 1]

utilities = [optimal_results[l]['utility'] for l in lambdas]

ax.plot(lambdas, utilities, 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
ax.fill_between(lambdas, utilities, alpha=0.3, color='#2ecc71')

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Utility', fontsize=12)
ax.set_title('Optimal Utility by Risk Aversion', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cal_cml_optimal_allocation.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: cal_cml_optimal_allocation.png")
plt.show()

# 5. Comparison with constrained optimization
print("\n4. IMPACT OF LEVERAGE CONSTRAINT")
print("-" * 100)

for lambda_coeff in [1.0, 2.0]:
    opt_vol, opt_return, w_market, w_rf = optimal_allocation(rf, market_return, market_vol, lambda_coeff)
    
    if w_market > 1.0:
        # Constraint binds
        w_market_constrained = 1.0
        w_rf_constrained = 0.0
        vol_constrained = w_market_constrained * market_vol
        ret_constrained = rf + (market_return - rf) * w_market_constrained
        utility_unconstrained = opt_return - (lambda_coeff / 2) * opt_vol ** 2
        utility_constrained = ret_constrained - (lambda_coeff / 2) * vol_constrained ** 2
        
        print(f"\nλ = {lambda_coeff}:")
        print(f"  Unconstrained optimal:")
        print(f"    w_market = {w_market*100:.1f}% (leverage), w_rf = {w_rf*100:.1f}%")
        print(f"    Utility = {utility_unconstrained:.4f}")
        print(f"  Constrained optimal (max leverage = 0%):")
        print(f"    w_market = {w_market_constrained*100:.1f}%, w_rf = {w_rf_constrained*100:.1f}%")
        print(f"    Utility = {utility_constrained:.4f}")
        print(f"  Welfare loss from constraint: {(utility_unconstrained - utility_constrained)*100:.2f} bp")

print("\n" + "=" * 100)
```

---

## 6. Challenge Round

1. **CAL vs CML:** If individual stock has Sharpe ratio < market Sharpe ratio, which CAL dominates? Can investor improve by holding stock instead of market portfolio? When would an investor hold non-market stocks?

2. **Leverage Impact:** Aggressive investor (λ=1) wants σp=25% but market σm=15%. How much must they leverage? If borrow rate = 5% (vs rf=3%), how does CAL slope change? Does constraint matter?

3. **Horizon Mismatch:** Portfolio optimized for 1-year horizon (using 1Y Treasury as rf) but investor realizes actual horizon is 5 years. Should allocation change? Does rf matter for long-term investors?

4. **Dynamic CAL:** Market crash occurs; E[Rm] drops 3%, σm rises 5%. How does CML slope change (steeper or flatter)? Does optimal allocation shift? Which investor (conservative or aggressive) affected more?

5. **Two-Fund Separation Breakdown:** If all investors have different beliefs about market return (not same µm), does two-fund separation still hold? If yes, what changes? If no, what alternative model?

---

## 7. Key References

- **Sharpe, W.F. (1964).** "Capital Asset Prices: A Theory of Market Equilibrium" – CAPM and CML derivation; Nobel Prize 1990.

- **Tobin, J. (1958).** "Liquidity Preference as Behavior Towards Risk" – Introduced Capital Allocation Line; separation of risky/risk-free decision.

- **Lintner, J. (1965).** "The Valuation of Risk Assets and the Selection of Risky Investments" – CML equilibrium and portfolio selection.

- **Merton, R.C. (1972).** "An Analytic Derivation of the Efficient Portfolio Frontier" – Mathematical foundations of CAL/CML; continuous optimization.

- **Markowitz, H.M. (1959).** "Portfolio Selection: Efficient Diversification of Investments" – Foundational efficient frontier; geometric foundations.

- **CFA Institute:** "Portfolio Management" – Professional curriculum on CAL/CML application; practical implementation.

- **Investopedia:** Capital Market Line – https://www.investopedia.com/terms/c/cml.asp – Practical explanation with examples.

