# Certainty Equivalent & Expected Utility Theory

## 1. Concept Skeleton
**Definition:** Certainty equivalent (CE): risk-free return that yields same utility as uncertain portfolio; CE = E[Rp] - (λ/2)σp². Expected Utility: axiomatic framework (von Neumann-Morgenstern) justifying utility-based decision-making under uncertainty.  
**Purpose:** Quantify risk premium (CE < E[Rp]); understand rational choice foundations; explain when investors trade certain outcome for uncertain higher-return gamble; measure welfare cost of risk  
**Prerequisites:** Probability, utility functions, mean-variance utility, risk aversion

---

## 2. Comparative Framing

| Aspect | Expected Value (EV) Framework | Expected Utility (EU) Framework | Certainty Equivalent (CE) |
|--------|------------------------------|--------------------------------|--------------------------|
| **Decision Criterion** | Choose option with highest EV | Choose option with highest EU | Indifferent between CE (certain) and portfolio (uncertain) |
| **Example Choice** | 50% win $1M vs 100% win $600k | Prefer certain $600k (λ > 0) | CE of lottery = $600k; willing to sell for this |
| **Risk Premium** | Not applicable (ignores variance) | RP = E[R] - CE; captures risk cost | RP = (λ/2)σ² for mean-variance utility |
| **Rational Prediction** | Would always gamble (higher EV) | Explains risk aversion; may refuse negative-EV gambles | Reveals λ (higher RP → higher λ) |
| **Behavioral Match** | Poor; contradicts observation | Good; explains insurance, caution, diversification | Excellent; practical welfare measure |
| **Calculation** | Simple: Σ p(s) × return(s) | Complex: Σ p(s) × u(return(s)) | CE = U⁻¹[EU]; inverse utility applied |
| **Lottery Preference** | Depends on probabilities only | Depends on probabilities + utility shape | Depends on utility concavity (risk aversion) |
| **Portfolio Decision** | Ignore variance; chase return | Balance return + risk via λ | Accept portfolio if CE > alternative (e.g., bonds) |
| **Equity Premium Puzzle** | Stock premium 6% rational if μ_stock = E[R] + 6% | PUZZLE: Why is λ ≈ 8 empirically but ≈1 theoretically? | CE_stocks much lower than E[R] due to high variance |
| **Implication** | People irrational or information-asymmetric | People rational but risk-averse; preferences matter | CE reveals true welfare; higher CE = better off |

**Key Insight:** Expected Value ignores risk (variance); Expected Utility incorporates it via concave utility function; Certainty Equivalent converts uncertain portfolio into equivalent sure amount, revealing risk penalty paid.

---

## 3. Examples + Counterexamples

**Example 1: Insurance Purchase (Negative Expected Value)**

Homeowner facing fire risk:
- House: $500,000
- Annual fire probability: 0.1%
- Expected loss: $500 × 0.001 = $500
- Insurance quote: $2,000/year (quotes $1,500 premium above expected loss)
- **EV Analysis:** Should NOT buy; negative EV of $1,500
- **CE Analysis:** 
  - Without insurance: EU = U(500,000 - 0.001 × 500,000) = U(499,500) with variance
    - More precisely: 99.9% keep $500k, 0.1% lose $500k → wealth ranges from 0 to $500k
  - With insurance: CE = $500k - $2,000 = $498,000 (certain)
  - Utility without: U_uninsured = U[0.999 × $500k + 0.001 × $0] - (λ/2) × [variance of loss]
    - Catastrophic variance → huge utility penalty
  - Utility with: U_insured = U($498k) (no variance)
  - **Decision:** If U($498k) > U_uninsured, buy insurance
  - **Why:** Risk aversion λ makes variance penalty huge; CE loss of $2k acceptable for variance elimination

**Example 2: Equity Premium Puzzle (The Paradox)**

Historical data:
- Treasury bills: 2% return, σ = 0%
- Stocks: 10% return, σ = 15%
- Observed equity risk premium: 8%

Theoretical prediction (mean-variance utility):
- Risk premium = (λ/2) × σ² ≈ 8%
- Solving: λ = 2 × 8% / (15%)² ≈ 0.71

But actual behavior (from allocation surveys + questionnaires):
- Implied λ from allocation choices ≈ 2-3
- Should result in RP ≈ 2-3% (not 8%)

**The Paradox:** 
- If λ = 0.71 (low risk aversion), why do people hold bonds at all?
- If λ = 2-3 (observed from allocations), why is equity premium only 8% (would need RP ≈ 3%, not 8%)?
- **CE explanation:** 
  - CE of stocks = 10% - (3/2) × (15%)² ≈ 10% - 3.4% ≈ 6.6%
  - CE of bonds = 2% (certain)
  - Gain from stocks: 6.6% - 2% = 4.6% (not 8%)
  - Puzzle: Why does market price in 8% premium if investors only value 4.6% gain?
  - Potential resolutions: 
    * Higher λ historically (λ ≈ 5-8) → larger RP justified
    * Disaster risk / rare crashes not captured by variance
    * Behavioral: people overlook stocks despite good CE
    * Limited market participation: only sophisticated investors hold equities

**Example 3: Certainty Equivalent vs Expected Value**

Three portfolios:
- **Portfolio A:** 50% gain $1M, 50% gain $0
  - E[A] = $500k
  - σ_A = $500k
  - U_A = E[R] - (λ/2)σ² = 500k - (2/2) × (500k)² = 500k - 500B (very negative!)
  - CE_A = U_A = essentially 0 or negative (lottery has huge variance penalty)
  - Revelation: Despite $500k expected value, investor very averse

- **Portfolio B:** 100% gain $300k
  - E[B] = $300k
  - σ_B = $0 (certain)
  - U_B = 300k (no variance penalty)
  - CE_B = $300k

- **Choice:** Most people prefer B (certain $300k) over A (50-50 $1M or $0)
  - Expected value prefers A ($500k > $300k)
  - Expected utility prefers B (U_B > U_A due to variance penalty)
  - CE shows B is equivalent to $300k certain, A equivalent to much less

**Example 4: Risk Premium from CE Calculation**

Investor evaluates portfolio allocation:
- Current: 100% bonds yielding 3% (CE = 3%)
- Proposed: 70% stocks (μ=10%, σ=15%) + 30% bonds (μ=3%, σ=0%), correlation = 0.2
- Portfolio: E[Rp] = 0.7 × 10% + 0.3 × 3% = 7.9%
- Volatility: σp = √(0.7² × 15² + 0.3² × 0² + 2 × 0.7 × 0.3 × 0 × 0.2) = 10.5%
- CE = 7.9% - (2/2) × (10.5%)² = 7.9% - 1.1% = 6.8%

Comparison:
- Keep bonds: CE = 3%
- Switch to portfolio: CE = 6.8%
- **Gain:** 6.8% - 3% = 3.8% certainty equivalent improvement
- **Risk premium paid:** 7.9% - 6.8% = 1.1% (variance penalty for uncertainty)
- **Interpretation:** Investor willing to accept 1.1% return penalty for diversification; net CE gain of 3.8%

**Example 5: Allais Paradox (Violation of Expected Utility)**

Classic experiment revealing preference reversal:

Choice 1:
- A: 100% win $1M (E[A] = $1M, U_A = U($1M))
- B: 89% win $1M, 10% win $5M, 1% win $0 (E[B] = 0.89M + 0.1 × 5M = 1.39M)
- Actual choice: Most people prefer A (certain outcome)

Choice 2:
- C: 11% win $1M, 89% win $0 (E[C] = 0.11M)
- D: 10% win $5M, 90% win $0 (E[D] = 0.5M)
- Actual choice: Most people prefer D (higher expected value)

Paradox: If indifferent to A vs B scale, should be indifferent to C vs D (same scale factors), but people's preferences reverse!

Implication:
- **Expected Utility Violation:** Choices depend on reference point, not just objective probabilities
- **CE Interpretation:** For A, CE ≈ $1M (certain). For B, CE < $1.39M due to diminishing utility of extra $5M. For C, D: CE of D > CE of C despite lower prob.
- **Behavioral:** People overweight probability of zero (avoid ruin) in Choice 1; overweight upside gain in Choice 2
- **Conclusion:** Mean-variance utility or expected utility alone insufficient; reference dependence (prospect theory) needed

---

## 4. Layer Breakdown

```
Certainty Equivalent & Expected Utility Theory Architecture:

├─ Certainty Equivalent (CE) Definition & Interpretation:
│   ├─ Formal Definition:
│   │   │ CE is the risk-free return yielding same utility as uncertain portfolio
│   │   │ U(CE) = E[U(Rp)]
│   │   │ CE = U⁻¹[E[U(Rp)]]  ← Inverse utility function applied to expected utility
│   │   │
│   │   ├─ For mean-variance utility U(Rp) = E[Rp] - (λ/2)σp²:
│   │   │   └─ CE = E[Rp] - (λ/2)σp²  (explicit formula!)
│   │   │
│   │   ├─ For log utility U(W) = ln(W):
│   │   │   └─ CE = exp[E[ln(Rp)]]  (geometric mean return)
│   │   │
│   │   ├─ For power utility U(W) = W^(1-γ) / (1-γ), γ = relative risk aversion:
│   │   │   └─ CE = [E[Rp^(1-γ)]]^[1/(1-γ)]
│   │   │
│   │   └─ General Principle: CE always ≤ E[Rp] (risk has negative value)
│   │       └─ CE = E[Rp] only when variance = 0 (no risk)
│   │
│   ├─ Interpretation of CE:
│   │   ├─ Risk Premium (RP) = E[Rp] - CE = amount of return sacrificed for uncertainty
│   │   ├─ For mean-variance: RP = (λ/2)σp²
│   │   ├─ Practical example: Portfolio has E[R]=8%, σ=12%
│   │   │   ├─ If λ=2: CE = 8% - (2/2)(12%)² = 8% - 1.44% = 6.56%
│   │   │   ├─ If λ=4: CE = 8% - (4/2)(12%)² = 8% - 2.88% = 5.12%
│   │   │   ├─ If λ=0 (risk-neutral): CE = E[R] = 8% (no risk discount)
│   │   │   └─ Implication: Higher λ → lower CE (more averse to risk)
│   │   │
│   │   ├─ Welfare Interpretation:
│   │   │   ├─ Investor indifferent between:
│   │   │   │   ├─ Uncertain portfolio with (E[Rp], σp)
│   │   │   │   └─ Certain return of CE
│   │   │   ├─ If offered CE or better → accept (better off)
│   │   │   ├─ If offered less than CE → reject (worse off)
│   │   │   └─ CE is break-even point
│   │   │
│   │   └─ Preference Revelation:
│   │       ├─ From observed choice of portfolio, can infer λ
│   │       ├─ If investor chooses risky over risk-free at CE, reveals λ preference
│   │       ├─ Combining multiple choices → triangulate λ estimate
│   │       └─ Revealed preference > stated preference (more credible)
│   │
│   └─ CE vs Expected Value vs Expected Utility:
│       ├─ EV (ignores risk): E[R] = Σ p(s) R(s)
│       ├─ EU (incorporates risk): E[U(R)] = Σ p(s) u(R(s))
│       ├─ CE (converts to equivalent sure outcome): CE = U⁻¹[E[U(R)]]
│       └─ Relationship: CE ≤ E[R] by Jensen's inequality (u concave)
│
├─ Expected Utility Theory (von Neumann-Morgenstern Axioms):
│   ├─ Axiomatic Foundations (why rational decision-makers use EU):
│   │   │
│   │   ├─ Axiom 1: Completeness (Comparability)
│   │   │   │ For any two lotteries A, B, investor can rank: A ≻ B, B ≻ A, or A ~ B
│   │   │   ├─ Implication: No indecision; preferences are complete
│   │   ├─ Axiom 2: Transitivity (Consistency)
│   │   │   │ If A ≻ B and B ≻ C, then A ≻ C
│   │   │   ├─ Implication: Preferences don't cycle; rational choice possible
│   │   ├─ Axiom 3: Continuity (No Jumps)
│   │   │   │ If A ≻ B ≻ C, then exists p ∈ [0,1] such that pA + (1-p)C ~ B
│   │   │   ├─ Implication: Preferences can be represented numerically
│   │   └─ Axiom 4: Independence (Irrelevance of Common Outcomes)
│   │       │ If A ~ B, then pA + (1-p)C ~ pB + (1-p)C for any p, C
│   │       ├─ Implication: Preference for A vs B unaffected by mixing with C
│   │       ├─ Violation: Allais paradox (Choice 1 & 2 reversal)
│   │       └─ Consequence: People behave non-EU in some settings
│   │
│   ├─ Theorem (vNM): If investor satisfies axioms 1-4, preferences representable by
│   │   │ utility function u(·), and decision rule is: choose option maximizing E[u(R)]
│   │   ├─ Proof sketch: Assign utilities u(best)=1, u(worst)=0; interpolate via continuity
│   │   ├─ Uniqueness: u(·) unique up to affine transformation u'(x) = a + b × u(x)
│   │   └─ Implication: EU formulation fundamental to rational choice
│   │
│   ├─ Utility Function Shape (Determines Risk Aversion):
│   │   ├─ Risk-averse (λ > 0, concave u): u''(x) < 0
│   │   │   ├─ Marginal utility decreases (diminishing marginal utility)
│   │   │   ├─ Example: u(x) = ln(x), u(x) = √x, u(x) = -e^(-λx)
│   │   │   ├─ Implies: E[u] < u(E[x]) (Jensen's inequality)
│   │   │   └─ Consequence: Prefer certain to uncertain with same expected value
│   │   │
│   │   ├─ Risk-neutral (λ = 0, linear u): u''(x) = 0
│   │   │   ├─ Utility proportional to payoff: u(x) = ax + b
│   │   │   ├─ Decision rule: Max E[R] (ignore variance)
│   │   │   └─ Implication: Would accept any positive-EV gamble
│   │   │
│   │   └─ Risk-seeking (λ < 0, convex u): u''(x) > 0
│   │       ├─ Marginal utility increases (rare; pathological)
│   │       ├─ Example: u(x) = x², u(x) = e^(λx) for λ > 0
│   │       ├─ Implies: E[u] > u(E[x])
│   │       └─ Consequence: Prefer uncertain to certain with same expected value (gamble)
│   │
│   └─ Evidence For & Against EU:
│       ├─ Supporting Evidence:
│       │   ├─ Market prices consistent with EU (asset pricing)
│       │   ├─ Portfolio choices generally follow mean-variance (EU special case)
│       │   ├─ Insurance purchases explained by risk-averse utility
│       │   └─ Negotiated prices in experiments cluster at EU-predicted levels
│       │
│       └─ Violations (Behavioral):
│           ├─ Allais Paradox: Preference reversals violate independence axiom
│           ├─ Framing Effects: Same choice, different presentation → different choices
│           ├─ Reference Dependence: Utility depends on starting wealth, not absolute
│           ├─ Loss Aversion: Losing $X worse than gaining $X is good (λ ≠ symmetric)
│           └─ Implication: EU useful baseline, but behavioral extensions needed
│
├─ Risk Aversion Measures:
│   ├─ Absolute Risk Aversion (ARA):
│   │   ├─ Definition: A(W) = -u''(W) / u'(W)  ← Utility curvature per marginal utility
│   │   ├─ Interpretation: How much investor willing to pay (in $) to eliminate risk
│   │   ├─ Formula: Premium paid ≈ (1/2) × A(W) × variance
│   │   ├─ Quadratic utility: ARA = λ (constant)
│   │   ├─ Log utility: ARA = 1/W (decreases with wealth; realistic)
│   │   ├─ Exponential utility u(W) = 1 - e^(-aW): ARA = a (constant)
│   │   └─ Practical: Higher ARA → higher risk premium demanded
│   │
│   ├─ Relative Risk Aversion (RRA):
│   │   ├─ Definition: R(W) = W × A(W) = -W × u''(W) / u'(W)  ← ARA scaled by wealth
│   │   ├─ Interpretation: Percentage of wealth willing to allocate to risky asset
│   │   ├─ Logarithmic utility: RRA = 1 (proportion constant regardless of wealth)
│   │   ├─ Power utility u(W) = W^(1-γ)/(1-γ): RRA = γ (exogenous constant)
│   │   ├─ Quadratic utility: RRA = λW (increases with wealth; unrealistic)
│   │   └─ Empirical: RRA ≈ 1-4 (people allocate 1-4× risky assets as proportion of wealth)
│   │
│   ├─ Pratt-Arrow Measures (Rigorous Treatment):
│   │   ├─ Theorem: u is more risk-averse than v (in certainty-equivalent sense) iff
│   │   │   A_u(W) ≥ A_v(W) for all W
│   │   ├─ Application: Quadratic vs exponential utility; which truly more risk-averse?
│   │   └─ Implication: ARA ordering complete and transitive
│   │
│   └─ Empirical Calibration:
│       ├─ Stock market data: λ ≈ 1-2 (implied from Sharpe ratio)
│       ├─ Life-cycle surveys: λ ≈ 2-5 (people describe themselves as moderate)
│       ├─ Lottery experiments: λ ≈ 0.5-1 (lower under lab conditions)
│       ├─ Insurance purchases: λ ≈ 5-10 (very high for risk-averse acts)
│       └─ Reconciliation: λ varies by context (domain-specific risk aversion)
│
├─ Risk Premium Decomposition:
│   ├─ Total Risk Premium:
│   │   │ RP_total = E[Rp] - Rf = return above risk-free
│   │   ├─ Equity risk premium: 6-8% historically
│   │   ├─ Bond credit risk premium: 2-5% (investment grade)
│   │   ├─ Liquidity premium: 0.5-2% (hard to trade)
│   │   └─ Inflation premium: 1-3% (protect purchasing power)
│   │
│   ├─ Variance Risk Premium:
│   │   │ RP_var = (λ/2) × σp²  ← Premium paid for variance alone
│   │   ├─ Example: 15% vol stock with λ=2
│   │   │   └─ Variance premium = (2/2) × (15%)² = 2.25%
│   │   ├─ Doubling volatility quadruples premium (convex in vol)
│   │   └─ Implication: Tail risk matters disproportionately
│   │
│   ├─ Skewness & Tail Risk Premium:
│   │   ├─ Beyond variance: E[u(R)] depends on full distribution, not just mean/var
│   │   ├─ Negative skewness: Right tail cut off → higher risk premium
│   │   ├─ Fat tails / kurtosis: Crashes more likely → premium higher
│   │   ├─ Example: Crash risk 1x every 20 years (-50% loss) → CE dramatically lower
│   │   └─ Implication: Mean-variance utility underprices tail risk; true premium > (λ/2)σ²
│   │
│   └─ Market Premium Puzzle:
│       ├─ Observed: Stocks 8% premium over bonds
│       ├─ Variance justified: (λ/2) × (15%)² → λ ≈ 0.71 implied
│       ├─ But revealed λ ≈ 2-3; conflicts with 8% premium
│       ├─ Resolutions:
│       │   ├─ Rare disasters: Crash probability > variance suggests → higher λ
│       │   ├─ Peso problem: Markets price in low-prob events (currency crisis, war)
│       │   ├─ Behavioral: Myopic loss aversion (evaluate annually, not lifetime)
│       │   ├─ Information: Some investors believe stocks outperform; disagreement
│       │   └─ Frictions: Taxes, costs, constraints reduce investor allocations
│       └─ Current thinking: Combination of all factors; no single resolution
│
└─ Application: Computing CE for Portfolio Decisions:
    ├─ Step 1: Collect returns data for candidate portfolios
    ├─ Step 2: Estimate mean μp and variance σp² for each
    ├─ Step 3: Determine investor's risk aversion λ (survey, revealed, or domain-specific)
    ├─ Step 4: Calculate CE for each portfolio: CE = μp - (λ/2)σp²
    ├─ Step 5: Choose portfolio with highest CE (best risk-adjusted welfare)
    ├─ Step 6: Compare CE to alternatives (bonds, prior allocation) to assess gain
    └─ Step 7: Monitor realized returns vs CE forecast; recalibrate if systematic error
```

**Mathematical Formulas:**

Certainty Equivalent (general):
$$U(CE) = E[U(R_p)]$$

Certainty Equivalent (mean-variance):
$$CE = E[R_p] - \frac{\lambda}{2}\sigma_p^2$$

Risk Premium:
$$RP = E[R_p] - CE = \frac{\lambda}{2}\sigma_p^2$$

Expected Utility (general lottery):
$$E[U(R)] = \sum_{s=1}^{S} p_s \cdot u(R_s)$$

Absolute Risk Aversion:
$$A(W) = -\frac{u''(W)}{u'(W)}$$

Relative Risk Aversion:
$$R(W) = W \cdot A(W) = -\frac{W \cdot u''(W)}{u'(W)}$$

---

## 5. Mini-Project: Computing & Comparing Certainty Equivalents

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import yfinance as yf

# Compute certainty equivalents for portfolios; compare to alternatives

def get_portfolio_returns(start_date, end_date):
    """
    Fetch portfolio returns for analysis.
    """
    tickers = {
        'Aggressive (80% SPY, 20% BND)': ['SPY', 'BND'],
        'Moderate (60% SPY, 40% BND)': ['SPY', 'BND'],
        'Conservative (40% SPY, 60% BND)': ['SPY', 'BND'],
        'Bonds Only (100% BND)': ['BND'],
        'Stocks Only (100% SPY)': ['SPY'],
    }
    
    # Download data
    all_tickers = list(set([t for ts in tickers.values() for t in ts]))
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    portfolio_returns = {}
    for portfolio_name, weights_dict in [
        ('Aggressive (80% SPY, 20% BND)', {'SPY': 0.8, 'BND': 0.2}),
        ('Moderate (60% SPY, 40% BND)', {'SPY': 0.6, 'BND': 0.4}),
        ('Conservative (40% SPY, 60% BND)', {'SPY': 0.4, 'BND': 0.6}),
        ('Bonds Only (100% BND)', {'BND': 1.0}),
        ('Stocks Only (100% SPY)', {'SPY': 1.0}),
    ]:
        port_ret = sum(weights_dict.get(t, 0) * returns[t] for t in all_tickers if t in returns.columns)
        portfolio_returns[portfolio_name] = port_ret
    
    return pd.DataFrame(portfolio_returns), returns[['SPY', 'BND']]


def compute_certainty_equivalents(returns_dict, lambda_values):
    """
    Compute CE for each portfolio and λ value.
    """
    results = []
    
    for portfolio_name, port_returns in returns_dict.items():
        annual_mean = port_returns.mean() * 252
        annual_std = port_returns.std() * np.sqrt(252)
        annual_var = annual_std ** 2
        
        for lambda_coeff in lambda_values:
            ce = annual_mean - (lambda_coeff / 2) * annual_var
            risk_premium = (lambda_coeff / 2) * annual_var
            
            results.append({
                'Portfolio': portfolio_name,
                'Lambda': lambda_coeff,
                'Expected Return': annual_mean,
                'Volatility': annual_std,
                'Variance': annual_var,
                'CE': ce,
                'Risk Premium': risk_premium,
                'Sharpe Ratio': (annual_mean - 0.02) / annual_std if annual_std > 0 else 0
            })
    
    return pd.DataFrame(results)


def simulate_wealth_paths(returns, initial_wealth, n_paths, n_steps):
    """
    Monte Carlo simulation of portfolio wealth paths.
    """
    mean_return = returns.mean() * 252
    std_return = returns.std() * np.sqrt(252)
    
    # Geometric random walk
    random_returns = np.random.normal(mean_return, std_return, (n_paths, n_steps))
    
    wealth_paths = np.zeros((n_paths, n_steps + 1))
    wealth_paths[:, 0] = initial_wealth
    
    for t in range(n_steps):
        wealth_paths[:, t + 1] = wealth_paths[:, t] * (1 + random_returns[:, t])
    
    return wealth_paths


def interpret_ce_as_insurance_value(ce, expected_return, risk_aversion):
    """
    Interpret CE as implied insurance value.
    """
    return expected_return - ce  # Amount investor willing to pay to eliminate uncertainty


# Main Analysis
print("=" * 100)
print("CERTAINTY EQUIVALENT & EXPECTED UTILITY THEORY")
print("=" * 100)

# 1. Data
print("\n1. PORTFOLIO DATA & RETURNS")
print("-" * 100)

portfolio_returns, asset_returns = get_portfolio_returns('2015-01-01', '2024-01-01')

print("\nHistorical Returns (2015-2024):")
print(portfolio_returns.describe().loc[['mean', 'std']].T)

# Annualize
annual_summary = pd.DataFrame({
    'Annual Return': portfolio_returns.mean() * 252,
    'Annual Volatility': portfolio_returns.std() * np.sqrt(252),
    'Sharpe Ratio': (portfolio_returns.mean() * 252 - 0.02) / (portfolio_returns.std() * np.sqrt(252))
})
print("\n" + annual_summary.to_string())

# 2. Certainty equivalents for different λ
print("\n2. CERTAINTY EQUIVALENTS BY RISK AVERSION COEFFICIENT")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
ce_results = compute_certainty_equivalents(portfolio_returns, lambda_values)

# Pivot for readability
ce_pivot = ce_results.pivot_table(values='CE', index='Portfolio', columns='Lambda')
print("\nCertainty Equivalent (%):  [Expected Return - Risk Premium]")
print((ce_pivot * 100).to_string())

rp_pivot = ce_results.pivot_table(values='Risk Premium', index='Portfolio', columns='Lambda')
print("\n\nRisk Premium (%) paid for variance:")
print((rp_pivot * 100).to_string())

# 3. Insurance value interpretation
print("\n3. IMPLIED INSURANCE VALUE (What investor would pay to eliminate uncertainty)")
print("-" * 100)

for lambda_coeff in [2.0, 4.0]:
    subset = ce_results[ce_results['Lambda'] == lambda_coeff]
    print(f"\nFor Risk Aversion λ = {lambda_coeff}:")
    for _, row in subset.iterrows():
        insurance_value = row['Expected Return'] - row['CE']
        print(f"  {row['Portfolio']:30s}: Would pay {insurance_value*100:5.2f}% of return to eliminate risk")

# 4. Comparison: Which portfolio best for different investors?
print("\n4. OPTIMAL PORTFOLIO CHOICE BY RISK AVERSION")
print("-" * 100)

for lambda_coeff in [1.0, 2.0, 4.0, 8.0]:
    subset = ce_results[ce_results['Lambda'] == lambda_coeff].sort_values('CE', ascending=False)
    best = subset.iloc[0]
    print(f"\nλ = {lambda_coeff} (Risk aversion):")
    print(f"  Best choice: {best['Portfolio']}")
    print(f"    Expected return: {best['Expected Return']*100:.2f}%")
    print(f"    Volatility: {best['Volatility']*100:.2f}%")
    print(f"    Certainty equivalent: {best['CE']*100:.2f}%")
    print(f"    Risk premium paid: {best['Risk Premium']*100:.2f}%")

# 5. Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: CE by portfolio and λ
ax = axes[0, 0]
for portfolio in portfolio_returns.columns:
    subset = ce_results[ce_results['Portfolio'] == portfolio]
    ax.plot(subset['Lambda'], subset['CE'] * 100, 'o-', label=portfolio, linewidth=2, markersize=6)

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=11)
ax.set_ylabel('Certainty Equivalent (%)', fontsize=11)
ax.set_title('Certainty Equivalent by Portfolio & λ', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

# Plot 2: Risk premium by portfolio and λ
ax = axes[0, 1]
for portfolio in portfolio_returns.columns:
    subset = ce_results[ce_results['Portfolio'] == portfolio]
    ax.plot(subset['Lambda'], subset['Risk Premium'] * 100, 's--', label=portfolio, linewidth=2, markersize=6)

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=11)
ax.set_ylabel('Risk Premium Paid (%)', fontsize=11)
ax.set_title('Risk Premium by Portfolio & λ', fontweight='bold', fontsize=12)
ax.legend(fontsize=9, loc='best')
ax.grid(alpha=0.3)

# Plot 3: E[R] vs CE scatter
ax = axes[0, 2]
for portfolio in portfolio_returns.columns:
    subset = ce_results[ce_results['Portfolio'] == portfolio].drop_duplicates('Portfolio', keep='last')
    ax.scatter(subset['Expected Return'] * 100, subset['CE'] * 100, s=150, label=portfolio, alpha=0.7)
    
    # Add 45-degree line (E[R] = CE, risk-neutral)
    ax.plot([0, 15], [0, 15], 'k--', alpha=0.3, linewidth=1)

ax.set_xlabel('Expected Return (%)', fontsize=11)
ax.set_ylabel('Certainty Equivalent (%)', fontsize=11)
ax.set_title('Expected Return vs Certainty Equivalent', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 4: Utility comparison for moderate investor (λ=2)
ax = axes[1, 0]
lambda_coeff = 2.0
subset = ce_results[ce_results['Lambda'] == lambda_coeff].sort_values('Expected Return')

ax.barh(subset['Portfolio'], subset['CE'] * 100, label='Certainty Equivalent', alpha=0.7, color='#2ecc71')
ax.barh(subset['Portfolio'], (subset['Expected Return'] - subset['CE']) * 100, 
        left=subset['CE'] * 100, label='Risk Premium Paid', alpha=0.7, color='#e74c3c')

ax.set_xlabel('Return (%)', fontsize=11)
ax.set_title(f'Return Composition (λ={lambda_coeff})', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='x')

# Plot 5: Wealth distribution (MC simulation)
ax = axes[1, 1]

moderate_returns = portfolio_returns['Moderate (60% SPY, 40% BND)']
wealth_paths = simulate_wealth_paths(moderate_returns, 100000, 1000, 50)

percentiles = [10, 25, 50, 75, 90]
colors_dist = ['#e74c3c', '#f39c12', '#2ecc71', '#f39c12', '#e74c3c']

for percentile, color in zip(percentiles, colors_dist):
    ax.plot(np.arange(51), np.percentile(wealth_paths, percentile, axis=0), 
           label=f'{percentile}th percentile', linewidth=2, color=color)

final_wealth = wealth_paths[:, -1]
final_mean = final_wealth.mean()
final_ce = final_mean - (lambda_coeff / 2) * final_wealth.std()**2  # Approximate CE

ax.axhline(y=final_mean, color='blue', linestyle='--', linewidth=2, label='Expected wealth')
ax.axhline(y=final_ce, color='green', linestyle='--', linewidth=2, label=f'CE (λ={lambda_coeff})')

ax.set_xlabel('Time Steps (5-year horizon)', fontsize=11)
ax.set_ylabel('Wealth ($)', fontsize=11)
ax.set_title('Wealth Paths & Certainty Equivalent', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 6: Allais Paradox illustration
ax = axes[1, 2]

# Outcomes (in $000s)
outcomes = ['Gamble 1', 'Gamble 2', 'Gamble 3', 'Gamble 4']
ev = [500, 510, 110, 100]  # Expected values
ce_illustrative = [400, 410, 50, 90]  # Certainty equivalents (with λ=2, λ=4 for last two)

x = np.arange(len(outcomes))
width = 0.35

ax.bar(x - width/2, ev, width, label='Expected Value', alpha=0.7, color='#3498db')
ax.bar(x + width/2, ce_illustrative, width, label='Certainty Equivalent (λ=2-4)', alpha=0.7, color='#e74c3c')

ax.set_ylabel('Value ($000s)', fontsize=11)
ax.set_title('Allais Paradox: EU vs EV Preference Reversals', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(outcomes, fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('certainty_equivalent_expected_utility.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: certainty_equivalent_expected_utility.png")
plt.show()

# 6. Key insights
print("\n5. KEY INSIGHTS & INTERPRETATION")
print("-" * 100)
print(f"""
CERTAINTY EQUIVALENT DEFINITION:
├─ CE is risk-free return investor indifferent between having & uncertain portfolio
├─ Formula (mean-variance): CE = E[Rp] - (λ/2)σp²
├─ Interpretation: E[Rp] - CE = risk premium (return sacrificed for uncertainty)
└─ Key insight: Higher λ → lower CE (more averse; demands return for risk)

EXPECTED UTILITY THEORY (von Neumann-Morgenstern):
├─ Axiom 1 (Completeness): Can rank all options
├─ Axiom 2 (Transitivity): Preferences consistent (A>B, B>C → A>C)
├─ Axiom 3 (Continuity): Numerical representation exists
├─ Axiom 4 (Independence): C irrelevant if choosing between A vs B
├─ Theorem: If satisfy all 4, rational to maximize E[U(R)]
└─ Practical: Explains insurance, diversification, risk management

RISK AVERSION COEFFICIENTS:
├─ Absolute (ARA = -u''/u'): How much $ pay to eliminate risk
├─ Relative (RRA = W × ARA): % of wealth to allocate to risky
├─ Empirical: λ ≈ 1-2 (stocks), λ ≈ 5-10 (insurance), λ ≈ 0.5-1 (lab)
└─ Calibration: Match observed allocation choices to infer λ

EQUITY PREMIUM PUZZLE:
├─ Observed: Stocks 8% premium, variance implies λ ≈ 0.7
├─ But people act with λ ≈ 2-3, which implies 2-3% premium (not 8%)
├─ Resolutions: Rare disasters, myopic loss aversion, information disagreement, frictions
└─ Current thinking: Combination; no single explanation

PRACTICAL DECISION RULE:
├─ Step 1: Estimate portfolio E[R] and σ
├─ Step 2: Determine your λ (questionnaire or revealed)
├─ Step 3: Compute CE = E[R] - (λ/2)σ²
├─ Step 4: Compare CE across options; choose highest
├─ Step 5: Sensitivity analysis (if λ ±1, does choice change?)
└─ Step 6: Rebalance if belief or λ changes (life events, market shocks)

VIOLATIONS & EXTENSIONS:
├─ Allais Paradox: Preference reversal violates Independence axiom
├─ Framing Effects: Same choice, different wording → different decision
├─ Reference Dependence: CE depends on starting wealth, not absolute level
├─ Loss Aversion: Losing $X worse than gaining $X is good (λ asymmetric)
└─ Modern: Prospect theory, behavioral portfolio theory address these

YOUR ANALYSIS:
├─ Best portfolio for λ=2 (moderate): {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Portfolio']}
├─  Expected return: {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Expected Return']*100:.2f}%
├─ Certainty equivalent: {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['CE']*100:.2f}%
├─ Risk premium paid: {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Risk Premium']*100:.2f}%
└─ Interpretation: Willing to give up {ce_results[ce_results['Lambda']==2.0].sort_values('CE', ascending=False).iloc[0]['Risk Premium']*100:.2f}% return for volatility reduction
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Risk Premium Scaling:** If you double volatility σ → 2σ, how does risk premium scale (linear, quadratic, other)? If λ = 2 and σ goes from 10% → 20%, does RP go from 1% → 2% or 1% → 4%? Why matters for tail risk.

2. **CE Estimation Accuracy:** If you estimate λ = 3 but true value λ = 2, by how much is inferred CE biased? Which direction? Over/underestimate true welfare gain?

3. **Allais Paradox Resolution:** Why do people prefer certain $1M over 89% chance of $1M but prefer uncertain $5M over certain $1M? Is this rational? Can it be explained within EU framework with appropriate u(·)?

4. **Reference Dependence:** If portfolio has same distribution but investor's reference changes (e.g., inheritance), does CE change? How does reference point interact with λ? Is this a model failure?

5. **Myopic Loss Aversion:** If investor evaluates portfolio annually vs lifetime, does optimal λ change? How does evaluation frequency affect risk tolerance? Why might 401k investors have lower λ than pension funds?

---

## 7. Key References

- **von Neumann, J. & Morgenstern, O. (1944).** "Theory of Games and Economic Behavior" – Axiomatic foundations of expected utility; foundational for all modern decision theory.

- **Pratt, J.W. (1964).** "Risk Aversion in the Small and in the Large" – Rigorous treatment of absolute and relative risk aversion; Pratt-Arrow framework.

- **Arrow, K.J. (1965).** "Aspects of the Theory of Risk-Bearing" – Theoretical analysis linking risk aversion to wealth; foundational for insurance & portfolio demand.

- **Kahneman, D. & Tversky, A. (1979).** "Prospect Theory: An Analysis of Decision under Risk" – Behavioral alternative to EU; explains reference dependence, loss aversion, probability weighting.

- **Tversky, A. & Kahneman, D. (1981).** "The Framing of Decisions and the Psychology of Choice" – Demonstrates framing effects, violating EU independence axiom; seminal behavioral work.

- **Shiller, R.J. (1981).** "Do Stock Prices Move Too Much to Be Justified by Subsequent Changes in Dividends?" – Early equity premium puzzle evidence; questions standard asset pricing.

- **Mehra, R. & Prescott, E.C. (1985).** "The Equity Premium Puzzle" – Formalized the puzzle; asks why historical premium (8%) >> theoretical prediction (~2%).

- **CFA Institute:** "Expected Utility & Decision Theory" – https://www.cfainstitute.org – Professional curriculum; applied EU in portfolio management.

