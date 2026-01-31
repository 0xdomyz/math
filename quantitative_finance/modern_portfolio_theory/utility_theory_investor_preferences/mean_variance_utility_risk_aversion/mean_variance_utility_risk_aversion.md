# Mean-Variance Utility Function & Risk Aversion

## 1. Concept Skeleton
**Definition:** Mathematical representation of investor preferences: utility U = E[Rp] - (λ/2)σp², where expected return positively valued, variance (risk) negatively valued with strength determined by risk aversion coefficient λ  
**Purpose:** Quantify investor risk tolerance, determine optimal asset allocation, explain portfolio construction decisions across different investor types  
**Prerequisites:** Probability, expected value, variance, utility functions, optimization

---

## 2. Comparative Framing

| Aspect | Risk-Averse Investor (λ = 10) | Moderate Investor (λ = 2) | Risk-Seeking Investor (λ = 0.2) |
|--------|-------------------------------|--------------------------|--------------------------------|
| **Utility Function** | U = E[R] - 5σ² (steep penalty for risk) | U = E[R] - σ² (standard) | U = E[R] - 0.1σ² (minimal penalty) |
| **Portfolio Choice** | Conservative (40% stocks, 60% bonds) | Moderate (60% stocks, 40% bonds) | Aggressive (90% stocks, 10% bonds) |
| **Expected Return Preference** | Accepts low return (5%) for safety | Target moderate return (8%) | Seeks high return (12%) |
| **Volatility Tolerance** | Avoids volatility (5% target vol) | Accepts standard volatility (12%) | Embraces volatility (25%+) |
| **Example Trade-off** | Willing to give up 1% return for 10% vol reduction | Willing to give up 1% return for 1% vol reduction | Willing to give up 1% return for 0.1% vol reduction |
| **Reaction to Loss** | Deeply distressed (-$10k loss devastating) | Moderately concerned (-$10k loss manageable) | Relatively indifferent (-$10k loss acceptable) |
| **Optimal Allocation** | Capital Allocation Line (CAL) near risk-free | CAL at market tangent | CAL extended beyond market (leverage) |
| **Indifference Curve Slope** | Very steep (willing to trade little return for risk reduction) | Moderate slope | Flat (little return sacrifice for more risk) |

**Key Insight:** Mean-variance utility captures fundamental principle: rational investors trade off expected return for lower risk, but strength of trade-off varies dramatically with λ (personal risk tolerance).

---

## 3. Examples + Counterexamples

**Example 1: Risk Aversion in Retirement Planning**
- Investor A (60-year-old, 10 years to retirement): λ = 20 (very risk-averse)
  - Portfolio: 30% stocks, 70% bonds (conservative)
  - Expected return: 4.5% p.a., volatility: 5%
  - Utility: 4.5% - (20/2) × 0.05² = 4.5% - 0.025% ≈ 4.475
  
- Investor B (30-year-old, 35 years to retirement): λ = 0.5 (risk-seeking)
  - Portfolio: 90% stocks, 10% bonds (aggressive)
  - Expected return: 10% p.a., volatility: 16%
  - Utility: 10% - (0.5/2) × 0.16² = 10% - 0.0064% ≈ 9.994
  
- Comparison: Investor B's utility (9.994) >> Investor A's (4.475) due to longer horizon and higher λ tolerance
- Lesson: Risk aversion coefficient λ captures time horizon, financial obligations, psychological comfort

**Example 2: Lottery Ticket Paradox (Challenges Mean-Variance Theory)**
- Rational prediction: Lottery tickets have negative expected return (-50% typical), high variance
  - Utility: -50% - (λ/2) × 100² < 0 for any reasonable λ
  - Prediction: No one should buy lottery tickets
- Actual behavior: Millions buy tickets (negative expected return!!)
- Explanation: Mean-variance utility incomplete; fails to capture appeal of small probability of huge gain
- Implication: Investors may have non-mean-variance utility (enjoy skewness, not just variance)
- Consequence: Mean-variance theory explains most portfolios but not speculation

**Example 3: Insurance Purchase (Negative Expected Value)**
- Homeowner: House worth $500k, fire probability 0.1% annually
- Expected loss: $500k × 0.001 = $500
- Insurance quote: $2,000 annually (4× expected loss)
- Expected value: -$1,500 (should not buy)
- Mean-variance analysis: Buying insurance = accepting negative return with volatility reduction
  - U_uninsured = 0% return - (λ/2) × large σ² (highly variable: catastrophic loss if fire)
  - U_insured = -0.4% return - (λ/2) × small σ² (smooth, predictable loss)
- With high λ, utility improvement from variance reduction > cost of negative return
- Lesson: Risk aversion λ explains insurance purchases; willing to pay premium for certainty

**Example 4: Coefficient of Risk Aversion Estimation (Personal)**
- Investor asked: "Coin flip: 50% gain $1M, 50% lose $500k. Accept?"
- Analysis: E[return] = 0.5 × $1M - 0.5 × $500k = $250k (positive; should accept if risk-neutral)
- Actual response: "No, too risky"
- Implied λ: If reject, loss aversion > gain pleasure
  - Must satisfy: 0.5 × 1M - 0.5 × 500k < (λ/2) × variance penalty
  - Solving: λ > 0.8 (moderate-to-high risk aversion)
- Implication: Rejecting positive EV gambles implies λ ≥ 1

**Counterexample: Behavioral Utility (Non-Mean-Variance)**
- Investor faces two portfolios:
  - Portfolio A: 50% gain, 50% loss of equal size (symmetric, mean-variance neutral)
  - Portfolio B: 50% gain, 50% loss but loss larger (skewed, mean-variance worse)
- Mean-variance prediction: Prefer A (lower variance)
- Actual choice: Many prefer B (value the small probability of big gain)
- Explanation: Prospect theory utility ≠ mean-variance utility; overweight probability of gains
- Implication: Mean-variance utility model incomplete for explaining actual investor behavior

---

## 4. Layer Breakdown

```
Mean-Variance Utility & Risk Aversion Architecture:

├─ Mathematical Foundation:
│   ├─ Quadratic Utility Function:
│   │   U(W) = W - (λ/2)W² where W = wealth
│   │   Derivatives: U'(W) > 0 (more wealth better)
│   │               U''(W) < 0 (diminishing marginal utility; risk averse)
│   │   Limitation: Exhibits increasing risk aversion (unrealistic; people don't become more risk-averse as wealth increases)
│   │
│   ├─ Mean-Variance Utility (Portfolio Context):
│   │   U(Rp) = E[Rp] - (λ/2)σp²
│   │   ├─ Linear in expected return (more return always better)
│   │   ├─ Quadratic in variance (diminishing penalty; 4% vol twice as bad as 2% vol, but not 4×)
│   │   ├─ Parameter λ > 0: Risk aversion coefficient
│   │   │   └─ Higher λ = stronger penalty for risk (more risk-averse investor)
│   │   │   └─ Typical range: λ ∈ [0.1, 20] depending on investor
│   │   ├─ Practical appeal: Simple, tractable; optimal portfolio closed-form solution
│   │   └─ Limitation: Ignores skewness, kurtosis, higher moments (non-normal returns matter)
│   │
│   └─ Expected Utility Theory (Axiomatic Foundation):
│       ├─ von Neumann-Morgenstern axioms:
│       │   ├─ Completeness: Investor can rank any two portfolios
│       │   ├─ Transitivity: If A > B and B > C, then A > C
│       │   ├─ Independence: Preference for A vs B unaffected by adding C
│       │   └─ Continuity: For uncertain outcomes, sure thing equivalence exists
│       ├─ Implication: Any rational investor's preferences representable by utility function
│       └─ Form: Expected utility across states U(Rp) = Σ p(s) u(R(s))
│
├─ Risk Aversion Coefficient λ (Absolute vs Relative):
│   ├─ Absolute Risk Aversion (ARA):
│   │   A(W) = -U''(W) / U'(W) (utility curvature normalized by marginal utility)
│   │   ├─ Quadratic utility: Increasing ARA = risk aversion increases with wealth (problematic)
│   │   ├─ Log utility: Constant relative RA; relative constant; matches observed behavior better
│   │   └─ Interpretation: How much investor willing to sacrifice return for volatility reduction
│   │
│   ├─ Relative Risk Aversion (RRA):
│   │   R(W) = W × A(W) (ARA adjusted for wealth level)
│   │   ├─ Log utility: RRA = 1 (risk aversion scales with wealth; reasonable)
│   │   ├─ Power utility: RRA = γ (exogenous risk aversion parameter)
│   │   └─ Interpretation: Percentage of wealth willing to allocate to risky assets
│   │
│   ├─ Calibration to Behavior:
│   │   ├─ Stock market participation: ~50% of investors; suggests ARA ≈ 2-5
│   │   ├─ Equity premium puzzle: Requires λ ≈ 8-15 to rationalize 6% risk premium
│   │   ├─ Home bias: λ ≈ 2-3 suggests psychological risk aversion ≠ utility λ
│   │   └─ Optimal retirement asset allocation: λ = 1-2 for typical life-cycle
│   │
│   └─ Variation by Investor Type:
│       ├─ Retirees (need income security): λ = 8-15
│       ├─ Working professionals (long horizon): λ = 1-3
│       ├─ Institutions (large pools): λ = 0.5-1
│       ├─ Young investors (can recover from loss): λ = 0.3-0.7
│       └─ Institutions with leverage constraints: λ = 1-2 (cannot short)
│
├─ Portfolio Optimization with Mean-Variance Utility:
│   ├─ Objective: Max U(Rp) = E[Rp] - (λ/2)σp²
│   │           = Σ wi E[Ri] - (λ/2) Σ wi² σi² - (λ) Σ Σ wi wj Cov(Ri, Rj)
│   │           Subject to: Σ wi = 1 (fully invested or allow leverage)
│   │                      wi ≥ 0 (no short selling, optional constraint)
│   │
│   ├─ Lagrangian Solution:
│   │   L = E[Rp] - (λ/2)σp² - ν(Σ wi - 1) - Σ μi max(0, -wi)
│   │   First-order conditions:
│   │   ∂L/∂wi = E[Ri] - λ Σ Cov(Ri, Rj)wj - ν - μi = 0
│   │   
│   ├─ Optimal Portfolio (Two-Fund Separation):
│   │   w* = (1/λ) × Σ^-1 × (E[R] - rf × 1)
│   │   └─ All investors hold market portfolio (scaled by λ)
│   │   └─ Higher λ → lighter holding of risky assets; more in risk-free
│   │
│   ├─ Efficient Frontier:
│   │   ├─ Lower risk (low λ): More concentrated in risk-free; E[Rp] ≈ rf, σp ≈ 0
│   │   ├─ Moderate risk (λ = 2): Balanced allocation; typical institutional investor
│   │   ├─ High risk (high λ^-1): Concentrated in risky portfolio; leverage possible
│   │   └─ All efficient portfolios on Capital Allocation Line (CAL)
│   │
│   └─ Sensitivity to λ:
│       ├─ Small change in λ → large change in optimal weights (instability)
│       ├─ Reason: Inverse relationship; λ in denominator of weight formula
│       ├─ Practical: Even small uncertainty in λ → suboptimal allocation
│       └─ Mitigation: Robust optimization; constraint portfolio concentration
│
├─ Implications for Portfolio Construction:
│   ├─ Risk Tolerance Questionnaires:
│   │   ├─ Purpose: Estimate λ from investor responses
│   │   ├─ Typical questions: 1) Age, 2) Horizon, 3) Reaction to loss, 4) Gambling preferences
│   │   ├─ Limitations: Stated preferences ≠ revealed preferences; biased, inconsistent
│   │   └─ Better: Use historical allocation choices to back out implied λ
│   │
│   ├─ Life-Cycle Asset Allocation:
│   │   ├─ Young (30y, λ = 0.5-1): 90% risky (high human capital, long horizon)
│   │   ├─ Middle-age (50y, λ = 1-2): 70% risky (reduced human capital, moderate horizon)
│   │   ├─ Pre-retirement (60y, λ = 3-5): 50% risky (need capital preservation)
│   │   ├─ Retirement (70y+, λ = 5-10): 30% risky (need liquidity, income)
│   │   └─ Pattern: λ increases with age as human capital (ability to work and earn) decreases
│   │
│   ├─ Allocation Constraints:
│   │   ├─ Min/Max sector weights: Prevent extreme concentration from high λ uncertainty
│   │   ├─ Diversification: "1/N rule" (equal weight) often beats mean-variance for small λ variations
│   │   ├─ Leverage limits: Restrict leverage to avoid amplifying estimation error
│   │   └─ Minimum/maximum risk: Set portfolio vol bounds [V_min, V_max]
│   │
│   └─ Rebalancing Frequency:
│       ├─ Optimal rebalancing depends on λ: High λ (conservative) → less rebalancing
│       ├─ Tax efficiency: Annual rebalancing typical (balance with tax drag)
│       ├─ Threshold-based: Rebalance when weights drift >5% (avoid timing luck)
│       └─ Mechanical discipline: Quarterly/annual rules work better than discretionary
│
└─ Beyond Mean-Variance Utility:
    ├─ Prospect Theory Utility (S-shaped, reference dependence):
    │   └─ Does not follow quadratic form; may overweight tail probabilities
    │
    ├─ Consumption-Based Utility (Time-Separable CRRA):
    │   └─ U = E[Σ δ^t u(Ct)]; captures intertemporal substitution
    │
    ├─ Disappointment Aversion:
    │   └─ Overweight probability of regret; impacts willingness to take risk
    │
    └─ Ambiguity Aversion:
        └─ Uncertainty about distributions increases risk aversion; Knightian risk
```

**Mathematical Formulas:**

Mean-Variance Utility:
$$U(R_p) = E[R_p] - \frac{\lambda}{2}\sigma_p^2$$

Optimal portfolio weights (unconstrained):
$$w^* = \frac{1}{\lambda}\Sigma^{-1}(E[R] - r_f \mathbf{1})$$

Certainty equivalent return:
$$R_{ce} = E[R_p] - \frac{\lambda}{2}\sigma_p^2 = U(R_p)$$

Risk aversion coefficient from indifference (pricing risk):
$$\lambda = \frac{E[R_m] - r_f}{\sigma_m^2/2} \approx \frac{0.06}{0.16^2/2} \approx 4.7$$

(Equity premium puzzle: observed λ ≈ 4-8, but implied should be ~1 under CAPM)

---

## 5. Mini-Project: Estimating Risk Aversion & Optimal Allocation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Estimate investor risk aversion from preferences and compute optimal allocation

def estimate_lambda_from_questionnaire(responses):
    """
    Estimate risk aversion coefficient λ from investor questionnaire.
    Responses: dict with keys like 'age', 'loss_comfort', 'portfolio_risk', 'gambling'
    """
    lambda_estimate = 1.0  # Start with neutral
    
    # Age-based adjustment
    age = responses.get('age', 50)
    lambda_estimate *= (age / 40)  # Older → higher λ
    
    # Loss comfort ("How would you feel losing 20% in one year?")
    # 1=devastating, 5=manageable
    loss_comfort = responses.get('loss_comfort', 3)
    lambda_estimate *= (6 - loss_comfort) / 3  # Lower comfort → higher λ
    
    # Portfolio risk tolerance ("What's ideal stock allocation?")
    # Measured as % stocks (higher = lower λ)
    stock_pct = responses.get('portfolio_risk', 60) / 100
    lambda_estimate *= (1 - stock_pct)  # Inverse relationship
    
    # Gambling preference ("Would you buy a 50-50 coin flip gamble?")
    # 1=never, 5=often
    gambling = responses.get('gambling', 2)
    lambda_estimate *= (3 - gambling) / 2  # More gambling → lower λ
    
    return lambda_estimate


def get_market_data(start_date, end_date):
    """
    Fetch asset returns for optimization.
    """
    tickers = {
        'Stocks': 'SPY',
        'Bonds': 'AGG',
        'Real Estate': 'VNQ',
        'Commodities': 'GSG',
    }
    
    data = yf.download(list(tickers.values()), start=start_date, end=end_date, 
                      progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns, tickers


def optimize_portfolio_mean_variance(returns, lambda_coeff, rf_rate=0.02):
    """
    Solve for optimal portfolio weights given risk aversion coefficient.
    Maximize: E[Rp] - (λ/2) σp²
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Objective function: -Utility (minimize negative)
    def neg_utility(w):
        portfolio_return = np.sum(mean_returns * w)
        portfolio_var = w @ cov_matrix @ w
        utility = portfolio_return - (lambda_coeff / 2) * portfolio_var
        return -utility
    
    # Constraints: weights sum to 1, no short selling
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weight
    x0 = np.array([1 / n_assets] * n_assets)
    
    result = minimize(neg_utility, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, -result.fun


def compute_efficient_frontier(returns, lambda_range, rf_rate=0.02):
    """
    Compute efficient frontier for different risk aversion coefficients.
    """
    frontier = []
    
    for lambda_coeff in lambda_range:
        weights, utility = optimize_portfolio_mean_variance(returns, lambda_coeff, rf_rate)
        
        portfolio_return = returns.mean() * 252
        portfolio_var = ((weights * returns).std() * np.sqrt(252)) ** 2
        
        expected_return = np.sum(weights * returns.mean() * 252)
        portfolio_vol = np.sqrt(weights @ (returns.cov() * 252) @ weights)
        
        frontier.append({
            'lambda': lambda_coeff,
            'expected_return': expected_return,
            'volatility': portfolio_vol,
            'utility': utility,
            'weights': weights
        })
    
    return pd.DataFrame(frontier)


def compute_capital_allocation_line(returns, rf_rate=0.02):
    """
    Compute CAL using tangency portfolio.
    CAL: E[Rp] = rf + (E[Rm] - rf) / σm × σp
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Objective: maximize Sharpe ratio
    def neg_sharpe(w):
        portfolio_return = np.sum(mean_returns * w)
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)
        sharpe = (portfolio_return - rf_rate) / portfolio_vol
        return -sharpe
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([1 / n_assets] * n_assets)
    
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    tangency_weights = result.x
    tangency_return = np.sum(mean_returns * tangency_weights)
    tangency_vol = np.sqrt(tangency_weights @ cov_matrix @ tangency_weights)
    tangency_sharpe = (tangency_return - rf_rate) / tangency_vol
    
    return tangency_weights, tangency_return, tangency_vol, tangency_sharpe


# Main Analysis
print("=" * 100)
print("MEAN-VARIANCE UTILITY & RISK AVERSION COEFFICIENT ESTIMATION")
print("=" * 100)

# 1. Risk aversion questionnaire
print("\n1. INVESTOR QUESTIONNAIRE & LAMBDA ESTIMATION")
print("-" * 100)

investor_profiles = {
    'Conservative (Retiree)': {
        'age': 72,
        'loss_comfort': 1,
        'portfolio_risk': 30,
        'gambling': 1
    },
    'Moderate (Mid-Career)': {
        'age': 45,
        'loss_comfort': 3,
        'portfolio_risk': 60,
        'gambling': 2
    },
    'Aggressive (Young Professional)': {
        'age': 32,
        'loss_comfort': 4,
        'portfolio_risk': 85,
        'gambling': 4
    },
}

lambda_estimates = {}
for profile_name, responses in investor_profiles.items():
    lambda_val = estimate_lambda_from_questionnaire(responses)
    lambda_estimates[profile_name] = lambda_val
    print(f"\n{profile_name}:")
    print(f"  Age: {responses['age']}, Loss comfort: {responses['loss_comfort']}/5, Portfolio risk: {responses['portfolio_risk']}%")
    print(f"  Estimated λ: {lambda_val:.2f}")

# 2. Get market data
print("\n2. ASSET CLASS DATA")
print("-" * 100)

returns, tickers = get_market_data('2015-01-01', '2024-01-01')

print(f"\nAsset Classes:")
for asset, ticker in tickers.items():
    mean_ret = returns[ticker].mean() * 252
    volatility = returns[ticker].std() * np.sqrt(252)
    print(f"  {asset}: Expected return {mean_ret:.2%}, Volatility {volatility:.2%}")

# 3. Optimal allocation for each investor
print("\n3. OPTIMAL PORTFOLIO ALLOCATION BY INVESTOR TYPE")
print("-" * 100)

allocations = {}
for profile_name, lambda_val in lambda_estimates.items():
    weights, utility = optimize_portfolio_mean_variance(returns, lambda_val)
    allocations[profile_name] = weights
    
    portfolio_return = np.sum(weights * (returns.mean() * 252))
    portfolio_vol = np.sqrt(weights @ (returns.cov() * 252) @ weights)
    
    print(f"\n{profile_name} (λ = {lambda_val:.2f}):")
    print(f"  Expected return: {portfolio_return:.2%}")
    print(f"  Volatility: {portfolio_vol:.2%}")
    print(f"  Utility: {utility:.4f}")
    print(f"  Allocation:")
    for i, (asset, ticker) in enumerate(tickers.items()):
        print(f"    {asset}: {weights[i]:5.1%}")

# 4. Capital Allocation Line
print("\n4. TANGENCY PORTFOLIO & CAPITAL ALLOCATION LINE (CAL)")
print("-" * 100)

rf_rate = 0.02
tangency_weights, tangency_return, tangency_vol, tangency_sharpe = compute_capital_allocation_line(returns, rf_rate)

print(f"Risk-free rate: {rf_rate:.2%}")
print(f"\nTangency Portfolio (Highest Sharpe Ratio):")
print(f"  Expected return: {tangency_return:.2%}")
print(f"  Volatility: {tangency_vol:.2%}")
print(f"  Sharpe ratio: {tangency_sharpe:.3f}")
print(f"  Allocation:")
for i, (asset, ticker) in enumerate(tickers.items()):
    print(f"    {asset}: {tangency_weights[i]:5.1%}")

print(f"\nCAL Equation: E[Rp] = {rf_rate:.2%} + {(tangency_return - rf_rate)/tangency_vol:.3f} × σp")

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Utility function illustration
ax = axes[0, 0]
sigma_range = np.linspace(0, 0.25, 100)
for profile_name, lambda_val in lambda_estimates.items():
    fixed_return = 0.08
    utilities = fixed_return - (lambda_val / 2) * sigma_range**2
    ax.plot(sigma_range * 100, utilities, label=f'{profile_name} (λ={lambda_val:.1f})', linewidth=2)

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Utility (for fixed 8% return)')
ax.set_title('Mean-Variance Utility by Risk Aversion Coefficient', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Optimal allocations by investor
ax = axes[0, 1]
profile_names = list(allocations.keys())
asset_names = list(tickers.keys())
allocation_matrix = np.array([allocations[p] for p in profile_names]).T

x = np.arange(len(profile_names))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, asset in enumerate(asset_names):
    offset = (i - 1.5) * width
    ax.bar(x + offset, allocation_matrix[i], width, label=asset, color=colors[i], alpha=0.8)

ax.set_ylabel('Allocation (%)')
ax.set_title('Optimal Portfolio Allocation by Investor Type', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(profile_names, rotation=15, ha='right', fontsize=9)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Efficient frontier
ax = axes[1, 0]
lambda_range = np.logspace(-1, 1, 50)  # λ from 0.1 to 10
frontier = compute_efficient_frontier(returns, lambda_range, rf_rate)

ax.plot(frontier['volatility'] * 100, frontier['expected_return'] * 100, 
        linewidth=2.5, color='darkblue', label='Efficient Frontier')
ax.scatter([tangency_vol * 100], [tangency_return * 100], 
          s=300, color='red', marker='*', label='Tangency Portfolio', zorder=5)
ax.scatter([rf_rate * 100], [rf_rate * 100], 
          s=200, color='green', marker='o', label='Risk-Free Asset', zorder=5)

# Add CAL line
cal_vol_range = np.linspace(0, 0.25, 100)
cal_return = rf_rate + (tangency_return - rf_rate) / tangency_vol * cal_vol_range
ax.plot(cal_vol_range * 100, cal_return * 100, 'r--', linewidth=2, label='CAL', alpha=0.7)

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Expected Return (%)')
ax.set_title('Efficient Frontier & Capital Allocation Line', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Risk-return for different λ values
ax = axes[1, 1]
for profile_name, lambda_val in lambda_estimates.items():
    profile_data = frontier[frontier['lambda'] == lambda_val].iloc[0]
    ax.scatter(profile_data['volatility'] * 100, profile_data['expected_return'] * 100,
              s=200, alpha=0.7, label=f'{profile_name} (λ={lambda_val:.1f})')

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Expected Return (%)')
ax.set_title('Optimal Portfolios on Efficient Frontier', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mean_variance_utility_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: mean_variance_utility_analysis.png")
plt.show()

# 6. Key insights
print("\n5. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 100)
print(f"""
RISK AVERSION COEFFICIENT (λ) INTERPRETATION:
├─ Low λ (0.5-1.0): Risk-seeking; willing to accept 2-4% volatility for 1% return
├─ Moderate λ (1.5-3.0): Standard investor; willing to accept 1-2% vol for 1% return
├─ High λ (5-10): Risk-averse; willing to accept <0.5% vol for 1% return
└─ Very high λ (>10): Conservative/retiree; prioritize capital preservation

ESTIMATED INVESTOR PROFILES:
├─ Conservative (λ ≈ {lambda_estimates['Conservative (Retiree)']:.1f}): Retirement focused; capital preservation paramount
├─ Moderate (λ ≈ {lambda_estimates['Moderate (Mid-Career)']:.1f}): Long-term growth; can accept volatility
└─ Aggressive (λ ≈ {lambda_estimates['Aggressive (Young Professional)']:.1f}): Growth-oriented; high risk tolerance

IMPLICATIONS:
├─ Allocation changes dramatically with λ (even 1-unit difference → 10-15% weight shift)
├─ Estimation uncertainty: ±20% in λ → suboptimal allocation costs ~0.5-1% annually
├─ Rebalancing frequency: Conservative investors benefit from quarterly rebalancing (lock in diversification benefit)
├─ Leverage: Aggressive investors can improve utility through leverage (borrow at rf, invest in risky)
└─ Time horizon: λ increases with age (shorter horizon → higher risk aversion)

PRACTICAL RECOMMENDATIONS:
├─ Use questionnaires for initial λ estimate, then refine based on behavior
├─ Consider revealed preferences: What allocation would this investor actually choose?
├─ Build in constraints: Min 20% bonds (retirees), max 50% volatility (safety)
├─ Annual review: Recalibrate λ as life circumstances change (retirement, inheritance, etc.)
└─ Use robust optimization: Assume λ range rather than point estimate; hedge uncertainty
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Lambda Estimation Accuracy:** If you estimate λ = 3 but true value is λ = 2, what is the suboptimality of your allocation? How much lower is expected utility? What if you're off by 5 points?

2. **Revealed vs Stated Preferences:** Investor states λ = 10 (conservative) via questionnaire, but holds 80% stocks (suggests λ ≈ 0.5). Which reveals true preference? How would you reconcile the discrepancy?

3. **Dynamic Risk Aversion:** After market crash (portfolio down 30%), does investor's λ change? Increase (loss aversion activated) or decrease (buying opportunity)? How should rebalancing adapt?

4. **Leverage Optimization:** Young aggressive investor (λ = 0.5) can borrow at 2% to invest in 8% expected portfolio. Should they lever 2× (100% portfolio + 100% leverage)? When does leverage become suboptimal?

5. **Ambiguity Aversion:** During high-uncertainty periods, investor seems to act with λ = 5 (conservative) even though normal λ = 2. How would you model uncertainty → risk aversion? Should allocations be dynamic?

---

## 7. Key References

- **Markowitz, H. (1952).** "Portfolio Selection" – Foundational mean-variance framework; Nobel Prize 1990; established λ as key parameter in portfolio choice.

- **Tobin, J. (1958).** "Liquidity Preference as Behavior Towards Risk" – Extended Markowitz to include risk-free asset; introduced Capital Allocation Line (CAL) concept.

- **Pratt, J.W. (1964).** "Risk Aversion in the Small and in the Large" – Rigorous treatment of absolute and relative risk aversion; mathematical foundations.

- **Arrow, K.J. (1965).** "Aspects of the Theory of Risk-Bearing" – Theoretical analysis of how risk aversion depends on wealth; Pratt-Arrow framework.

- **von Neumann, J. & Morgenstern, O. (1944).** "Theory of Games and Economic Behavior" – Axiomatic foundation of expected utility theory; underlies all modern portfolio theory.

- **Kahneman, D. & Tversky, A. (1979).** "Prospect Theory: An Analysis of Decision under Risk" – Behavioral alternative to mean-variance utility; explains observed deviations.

- **Investopedia: Risk Aversion** – https://www.investopedia.com/terms/r/riskaverse.asp – Practical overview with examples.

- **CFA Institute: Utility Theory** – https://www.cfainstitute.org – Professional curriculum on utility functions and optimal choice.

