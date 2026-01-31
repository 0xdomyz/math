# Modern Portfolio Theory: Foundations and Assumptions

## 1. Concept Skeleton
**Definition:** Core postulates, mathematical framework, and underlying assumptions enabling Markowitz mean-variance optimization; investor rationality; market efficiency; divisibility and liquidity assumptions; homogeneous expectations; risk-return trade-off axiomatization  
**Purpose:** Establish theoretical foundation for portfolio construction; understand when MPT applies/breaks; identify critical assumptions vulnerable to violation; enable model robustness analysis; bridge theory to practice  
**Prerequisites:** Probability theory, statistics (mean/variance/correlation), optimization (Lagrange multipliers), utility theory, financial markets basics

## 2. Comparative Framing
| Assumption | Standard MPT | Alternative Models | Implications for Portfolio |
|-----------|-----------|-------------------|---------------------------|
| **Rationality** | Investors maximize expected utility | Behavioral: Loss aversion, herding, overconfidence | May not select mean-var efficient portfolios; momentum, anomalies |
| **Normality** | Returns ~ N(μ, σ²) | Fat tails, skewness, kurtosis (t-dist, Laplace) | Risk underestimated; VaR/CVaR needed; tail hedges valuable |
| **Linearity** | Utility quadratic (mean-var) | Non-quadratic: Higher moments matter | Alternative risk measures; derivatives valuable |
| **Divisibility** | Arbitrary position sizes | Integer/lot constraints, min trade size | Discrete optimization; tracking error; rebalancing costs |
| **Liquidity** | Markets liquid, bid-ask 0 | Bid-ask spread, market depth, price impact | Transaction costs critical; passive > active for many |
| **Correlation** | Constant over time | Time-varying (GARCH, regime-switching) | Dynamic hedging; correlations spike in crashes; diversification fails |
| **Homogeneous Expectations** | All investors have same forecasts | Heterogeneous beliefs; different opinions | Active management possible; market can be inefficient |
| **No Arbitrage** | Markets frictionless | Arbitrage constraints, costs of carry | Mispricing opportunities; statistical arbitrage possible |
| **Information** | Common knowledge | Asymmetric information; slow dissemination | Insiders trade profitably; passive underperforms timing |
| **Short-Selling** | Unlimited short sales allowed | Short sale constraints; margin requirements | Cannot achieve tangency portfolio; kinks in frontier |

| Model Variant | Key Feature | When Appropriate | Limitations |
|---------------|-----------|------------------|------------|
| **Markowitz** | Mean-variance optimization, quadratic utility | Small portfolios, liquid assets | Ignores higher moments, constraints, costs |
| **CAPM** | Single-factor (market), linear relationship | Aggregate market portfolio, large investor base | Beta inadequate; multiple factors needed |
| **Black-Litterman** | Incorporate views on expected returns | Active management, strategic bets | Complex; view specification subjective |
| **Multi-factor Models** | Multiple factors (size, value, momentum) | Explain cross-sectional variation | Factor risk premia unstable; not traded |
| **Robust Optimization** | Hedge against parameter uncertainty | Small samples; robust weights | Overly conservative; higher tracking error |

## 3. Examples + Counterexamples

**Simple Example:**  
Two assets: Stock (μ=10%, σ=20%, correlation with bond=0) + Bond (μ=4%, σ=5%). Mean-variance frontier is hyperbola. Investor with risk aversion A=2 selects weight w_s ≈ 40%, w_b ≈ 60%, expected return 6.4%, volatility 7%. Quadratic utility governs choice.

**Perfect Fit:**  
Diversified portfolio of 50 liquid large-cap stocks, daily rebalancing, no short sales, normal returns. MPT framework applies well: efficient frontier computable, optimization stable, constraints handled, transaction costs negligible.

**Non-Normal Returns:**  
Hedge fund returns: Positive skew (good), but negative tail (May 2020 short-squeeze losses). VaR at 95% underestimates true tail risk. 1% worst day loss > 3-sigma prediction. CVaR (expected shortfall) more appropriate.

**Correlation Breakdown:**  
2008 financial crisis: Stock-bond correlation rose from -0.3 to +0.5 (during diversification failure). Portfolios constructed assuming correlation=-0.3 experienced massive drawdowns. Dynamic correlation models needed.

**Short Sale Constraints:**  
Optimal (unconstrained) portfolio: 150% Stocks, -50% Bonds. With no short sales: 100% Stocks. Constrained frontier dominates unconstrained only at boundaries. Tracking error vs unrestricted portfolio.

**Over-Optimization:**  
Fit mean-variance model to 3 years daily returns of 1000 stocks. Select weights based on estimated means and covariance. Forward test: massive underperformance. Estimation error dominates optimization solution (Jaggernaut effect).

**Illiquid Assets:**  
Private equity position: High expected return but 2% bid-ask spread, $50M minimum. Correlation to public equities estimated unreliably (few transactions). Price impact when liquidating exists. Standard MPT portfolio rules underestimate true cost.

## 4. Layer Breakdown
```
Modern Portfolio Theory: Foundations Framework:

├─ Core Mathematical Framework:
│  ├─ Portfolio Return:
│  │   R_p = Σ w_i * R_i  (weighted average return)
│  │   w_i: Weight of asset i
│  │   Σ w_i = 1 (fully invested)
│  │   Expected return: E[R_p] = Σ w_i * μ_i
│  ├─ Portfolio Variance:
│  │   σ_p² = Σ w_i² σ_i² + Σ Σ w_i w_j ρ_ij σ_i σ_j  (i ≠ j)
│  │   Σ w_i² σ_i²: Variance term (own risk)
│  │   Σ Σ w_i w_j ρ_ij σ_i σ_j: Covariance term (diversification)
│  │   Covariance matrix: Σ = [σ_ij]
│  │   Matrix form: σ_p² = w^T Σ w
│  ├─ Correlation Structure:
│  │   ρ_ij = Cov(R_i, R_j) / (σ_i * σ_j)
│  │   -1 ≤ ρ_ij ≤ 1
│  │   Perfect positive (ρ=1): No diversification benefit
│  │   Perfect negative (ρ=-1): Perfect hedge
│  │   Zero correlation (ρ=0): Partial diversification
│  ├─ Efficient Frontier:
│  │   Set of portfolios minimizing variance for given return
│  │   min σ_p² = w^T Σ w
│  │   s.t. E[R_p] = Σ w_i μ_i = μ_target
│  │        Σ w_i = 1
│  │   Solution: Hyperbola in (σ, μ) space
│  │   Lower branch (minimum variance): Inefficient
│  │   Upper branch: Efficient (rational investors choose here)
│  └─ Capital Allocation Line (CAL):
│      Combines risk-free asset with risky portfolio
│      E[R_p] = r_f + θ * σ_p
│      θ = (E[R_m] - r_f) / σ_m (slope = Sharpe ratio of risky portfolio)
│      CAL is tangent to efficient frontier at optimal risky portfolio
├─ Key Assumptions (Foundations):
│  ├─ Rationality:
│  │   ├─ Definition:
│  │   │   Investors choose portfolios to maximize expected utility
│  │   │   Preference for higher returns, lower risk
│  │   │   Consistent preferences (transitivity)
│  │   │   No arbitrage exploitation errors
│  │   ├─ Utility Function:
│  │   │   U(R) = E[R] - (A/2) * Var(R)  [Quadratic]
│  │   │   A: Risk aversion coefficient (A > 0)
│  │   │   Higher A: More risk-averse (prefer lower volatility)
│  │   │   Indifference curves: Upward sloping in (σ, μ) space
│  │   ├─ Optimal Portfolio Selection:
│  │   │   Investor chooses portfolio on CAL maximizing U(R)
│  │   │   y* = (E[R_m] - r_f) / (A * σ_m²)  [Proportion in risky portfolio]
│  │   │   More risk-averse (high A): Lower y*, closer to r_f
│  │   │   Less risk-averse (low A): Higher y*, leverage with r_f
│  │   └─ Violation: Behavioral finance shows investors are NOT rational
│  │       Loss aversion (more pain from loss than joy from gain)
│  │       Overconfidence (overestimate ability)
│  │       Herding (follow crowds)
│  ├─ Normality of Returns:
│  │   ├─ Assumption:
│  │   │   R_i ~ N(μ_i, σ_i²)  [Jointly normal for portfolio]
│  │   │   Daily/weekly returns approximately normal
│  │   ├─ Justification:
│  │   │   Central Limit Theorem: Returns = sum of many trades
│  │   │   Aggregation: Many random events → normal
│  │   ├─ Properties under normality:
│  │   │   Mean and variance fully characterize distribution
│  │   │   Linear combinations of normals → normal
│  │   │   Optimizing E[R] and Var(R) → optimal for any utility
│  │   ├─ Violations in Reality:
│  │   │   Fat tails (more extreme events): Kurtosis > 3
│  │   │   Skewness (asymmetry): Negative for equities, positive for call options
│  │   │   Example: Oct 1987 crash (19.7% down) ≈ 22 sigma event
│  │   ├─ Consequences:
│  │   │   Mean-variance optimization underestimates risk in tails
│  │   │   VaR/CVaR more appropriate than volatility
│  │   │   Tail hedges valuable (out-of-money puts)
│  │   └─ Better Assumptions:
│  │       Student t-distribution (heavier tails)
│  │       Mixture models (normal times + rare crisis)
│  │       Copulas (non-normal dependence structure)
│  ├─ Quadratic Utility:
│  │   ├─ Assumption:
│  │   │   U(R) = aR - bR² (quadratic, concave, a > 0, b > 0)
│  │   │   Equivalent to E[R] - (A/2)*Var(R)
│  │   │   Only mean and variance matter
│  │   ├─ Justification:
│  │   │   Simplifies optimization (only two moments)
│  │   │   Locally reasonable approximation near R = a/(2b)
│  │   ├─ Consequences:
│  │   │   Ignores skewness (prefer positive skew)
│  │   │   Ignores kurtosis (vulnerable to tail risk)
│  │   │   Portfolio with high skew, low variance preferred equally
│  │   ├─ Violations:
│  │   │   Real investors care about skewness (prefer positive)
│  │   │   Options valued on skewness expectations (volatility smile)
│  │   │   Hedge funds valued partly on positive skew profile
│  │   └─ Multi-Moment Models:
│  │       Incorporate skewness: U = E[R] - (A/2)*Var - (B/6)*Skew
│  │       But optimization becomes non-convex (harder)
│  ├─ Divisibility and Continuous Proportions:
│  │   ├─ Assumption:
│  │   │   Can invest any amount w_i ∈ [0,1] (or [-∞, +∞] with short sales)
│  │   │   Infinitesimal positions allowed
│  │   │   No discrete lot constraints
│  │   ├─ Market Realities:
│  │   │   Minimum order sizes (e.g., 100 shares)
│  │   │   Odd lots have worse execution
│  │   │   Small positions < commission cost
│  │   ├─ Consequences:
│  │   │   Optimal continuous solution may not be achievable
│  │   │   Integer/mixed-integer programming needed
│  │   │   Tracking error when rounding to integers
│  │   └─ Solutions:
│  │       Discrete optimization (computationally harder)
│  │       Penalize small positions (min position size threshold)
│  │       Accept approximate solution (tracking error)
│  ├─ Perfect Liquidity and Zero Transaction Costs:
│  │   ├─ Assumption:
│  │   │   Can trade any size at mid-quote, no spread
│  │   │   No slippage, market impact, commissions
│  │   │   Buy/sell at same price (symmetric)
│  │   ├─ Market Realities:
│  │   │   Bid-ask spread (typically 1-5 bps for liquid assets)
│  │   │   Market impact (large orders move price)
│  │   │   Commissions (0.5-1 bp on equities)
│  │   │   Borrowing costs for short sales (10-100 bps annually)
│  │   ├─ Impact by Asset Class:
│  │   │   Stocks: Low transaction costs (0.1-0.5%), liquid
│  │   │   Bonds: Higher spreads (5-50 bps), lower volume
│  │   │   Alternatives: Very high (2-5%), illiquid, long lock-ups
│  │   ├─ Consequences:
│  │   │   Frequent rebalancing becomes uneconomical
│  │   │   Tracking error widens with turnover
│  │   │   Active management must overcome transaction costs
│  │   └─ Solutions:
│  │       Incorporate transaction costs in objective:
│  │       min [w^T Σ w + λ * (cost of trade)]
│  │       Reduce rebalancing frequency (quarterly vs daily)
│  │       Hold passive indices (minimize turnover)
│  ├─ Constant Correlation:
│  │   ├─ Assumption:
│  │   │   ρ_ij constant over time
│  │   │   Covariance matrix stable
│  │   ├─ Reality:
│  │   │   Correlations vary with market regime
│  │   │   During crises: Correlations spike toward 1 (portfolio diversification fails)
│  │   │   During boom: Correlations lower
│  │   │   Example: 2008 financial crisis
│  │   │   Stock-bond correlation: -0.3 (normal) → +0.5 (crisis)
│  │   ├─ Time-Varying Models:
│  │   │   GARCH (conditional variance varies)
│  │   │   Regime-switching (two or more correlation regimes)
│  │   │   Dynamic correlation (exponential smoothing)
│  │   ├─ Consequences:
│  │   │   Portfolio constructed assuming fixed correlation fails in stress
│  │   │   VaR estimates underestimate tail joint risk
│  │   │   Hedging strategies require correlation updates
│  │   └─ Practical Solutions:
│  │       Use stressed correlations (crisis values)
│  │       Monte Carlo with correlation changes
│  │       Include tail hedges
│  ├─ Homogeneous Expectations:
│  │   ├─ Assumption:
│  │   │   All investors have identical beliefs about means, variances, correlations
│  │   │   Market equilibrium: Everyone holds market portfolio (adjusted for risk aversion)
│  │   ├─ Reality:
│  │   │   Investors have heterogeneous views (different forecasts)
│  │   │   Information dissemination is not instant
│  │   │   Some investors more informed than others
│  │   ├─ Implications:
│  │   │   Active management possible (exploit view differences)
│  │   │   Smart Beta: Systematic factors outperform cap-weighted
│  │   │   Mispricing opportunities exist
│  │   ├─ If True (Homogeneous):
│  │   │   Market portfolio is efficient (CAPM)
│  │   │   No active manager can beat market (after fees)
│  │   │   Value investing, momentum are anomalies
│  │   └─ Black-Litterman Model:
│  │       Blends market equilibrium (homogeneous) with investor views
│  │       Starts with cap-weighted implied returns
│  │       Adjusts toward investor views (with uncertainty)
│  │       Result: More stable, less extreme positions
│  ├─ No Arbitrage and Market Efficiency:
│  │   ├─ Assumption:
│  │   │   Markets are informationally efficient (EMH)
│  │   │   No arbitrage opportunities (prices correct, no free lunches)
│  │   │   Mispricing only due to execution friction
│  │   ├─ Weak-Form Efficiency:
│  │   │   Past prices/volume contain no predictive power
│  │   │   Technical analysis futile
│  │   │   Returns are i.i.d. (white noise)
│  │   ├─ Semi-Strong Efficiency:
│  │   │   Public information instantly reflected in prices
│  │   │   Fundamental analysis futile
│  │   │   Beat market only with inside information
│  │   ├─ Strong Efficiency:
│  │   │   All information (public + private) reflected in prices
│  │   │   Insiders cannot profit
│  │   │   Extreme claim; few believe true
│  │   ├─ Violations (Anomalies):
│  │   │   Momentum (prices drift for months)
│  │   │   Mean reversion (prices overreact then revert)
│  │   │   Size effect (small caps outperform)
│  │   │   Value effect (cheap stocks outperform)
│  │   ├─ Explanations:
│  │   │   Data mining/luck (especially over long periods)
│  │   │   Risk-based (factor loaded; higher expected return)
│  │   │   Behavioral (irrational investors; over/under-react)
│  │   ├─ Practical Consequence:
│  │   │   If markets fully efficient: Passive indexing optimal
│  │   │   If not: Active management can add value
│  │   │   Consensus: Semi-strong efficiency mostly true, but factors exist
│  │   └─ Statistical Arbitrage:
│  │       Exploit small mispricings (too small to profit individually)
│  │       Scale up across many positions (law of large numbers)
│  │       Pairs trading, factor convergence trades
│  ├─ Short Sale Constraints:
│  │   ├─ Standard Assumption:
│  │   │   Can short-sell any amount (w_i → -∞ allowed)
│  │   │   Sell borrowed securities, pocket proceeds
│  │   ├─ Reality:
│  │   │   Short sales banned for some securities
│  │   │   Borrow fees (especially for low-float stocks): 10-100% annually
│  │   │   Uptick rule or locate requirement
│  │   │   High margin requirements (typically 150% of short value)
│  │   ├─ Consequences:
│  │   │   Tangency portfolio may be not achievable
│  │   │   Efficient frontier constrained at upper boundary (100% long)
│  │   │   Hedging strategies restricted
│  │   ├─ Impact on Optimization:
│  │   │   If tangency portfolio w_i ≥ 0 ∀i: No constraint binding
│  │   │   If some w_i < 0: Set w_i = 0, re-optimize
│  │   │   Constrained frontier may be significantly different
│  │   └─ Practical Solutions:
│  │       Optimize with w_i ≥ 0 constraint
│  │       Use hedging derivatives (put options) instead of shorts
│  │       Identify low-cost short candidates
│  ├─ Single Period, Static Optimization:
│  │   ├─ Assumption:
│  │   │   Optimize portfolio once (at t=0)
│  │   │   Hold for fixed horizon (T)
│  │   │   No rebalancing during period
│  │   ├─ Reality:
│  │   │   Markets change; rebalancing beneficial
│  │   │   Multi-period (stochastic control formulation)
│  │   │   Utility depends on final wealth, not intermediate
│  │   ├─ Consequences:
│  │   │   Static allocation: w maintained fixed (e.g., 60/40)
│  │   │   Drifts with market: Equities may be 70% after bull market
│  │   │   Rebalancing forces selling winners, buying losers (contrarian)
│  │   ├─ Rebalancing Benefits:
│  │   │   Risk control (keep volatility target)
│  │   │   Forces disciplined buying low, selling high
│  │   │   Costs vs benefits vary by market regime
│  │   └─ Multi-Period Framework:
│  │       Stochastic control: Optimal policy depends on current state
│  │       Dynamic programming: Recursive solution
│  │       Path-dependent utility (not just final wealth)
│  └─ Parameter Estimation:
│      ├─ Estimation Risk:
│      │   Optimal weights depend on estimated μ, σ, ρ
│      │   Small sample sizes introduce error
│      │   Example: N=100 assets, T=250 days (1 year)
│      │   Estimate 100 means, 100 variances, ~5000 correlations
│      │   But only 250 observations: Severely under-identified
│      ├─ Consequences:
│      │   Estimated frontier overstates achievable returns
│      │   Optimal weights are extreme (100% in highest μ asset)
│      │   Small changes in estimates → large weight changes
│      │   Forward performance much worse than back-test
│      ├─ Estimation Errors:
│      │   μ hardest to estimate (high noise-to-signal)
│      │   σ easier (volatility clustering captures trend)
│      │   ρ easier than μ (relative to signals)
│      ├─ Solutions:
│      │   Shrinkage estimators (pull toward market average)
│      │   Robust optimization (worst-case parameter uncertainty)
│      │   Empirical Bayes (prior beliefs + data)
│      │   Factor models (estimate fewer parameters)
│      └─ Optimal Shrinkage:
│          Ledoit-Wolf: Ŝ = αS + (1-α)F * I (mix sample cov with identity)
│          α chosen to minimize MSE
│          Typically α = 0.2-0.4 for 250 observations
│          Significant out-of-sample improvement
├─ Behavioral Violations of Rationality:
│  ├─ Loss Aversion:
│  │   Investors feel loss more than equivalent gain
│  │   Weight loss at 2x gain (approximately)
│  │   V(x) = x for x ≥ 0, V(x) = 2.25x for x < 0
│  │   Leads to: Reluctance to sell losers (disposition effect)
│  ├─ Overconfidence:
│  │   Investors overestimate ability to predict returns
│  │   Generate expectations above market average
│  │   Result: Higher portfolio turnover, lower performance
│  ├─ Anchoring:
│  │   Investors anchor on past prices (mental accounting)
│  │   Reluctant to sell below "cost basis"
│  │   Irrelevant to forward-looking decisions (sunk cost)
│  ├─ Recency Bias:
│  │   Over-weight recent performance
│  │   Buy hot funds after bull markets (bad timing)
│  │   Sell beaten-down sectors before rebounds
│  ├─ Home Bias:
│  │   Allocate disproportionately to home country
│  │   U.S. investors: ~80% U.S. stocks (should be ~50% globally)
│  │   Reasons: Familiarity, information asymmetry, tax
│  └─ Herding:
│      Follow crowds (other investors)
│      Buy when others buy (bubbles)
│      Sell when others sell (crashes)
│      Collective action → amplified volatility
├─ Comparison: Theory vs Practice:
│  ├─ Market Portfolio Identification:
│  │   Theory: Market portfolio = all risky assets in value-weight
│  │   Practice: Use S&P 500, Russell 2000, global index
│  │   Problem: Many assets (real estate, private equity, art) excluded
│  ├─ Risk-Free Rate:
│  │   Theory: Single r_f, constant over time
│  │   Practice: T-bills (short-term), T-bonds (long), varies by maturity
│  │   Problem: Long-term projects need long-term r_f; term structure matters
│  ├─ Risk Aversion:
│  │   Theory: Estimate from investor preferences or historical equity premium
│  │   Practice: Varies by investor, changes with wealth, age, goals
│  │   Problem: Personalization critical; one A doesn't fit all
│  ├─ Expected Returns:
│  │   Theory: Based on factor models (CAPM, Fama-French)
│  │   Practice: Analyst forecasts, survey expectations, historical averages
│  │   Problem: Highly uncertain; active management profits depend on accuracy
│  └─ Rebalancing Discipline:
│      Theory: Rebalance to maintain target weights
│      Practice: Trade-off vs transaction costs
│      Problem: Optimal frequency depends on costs, volatility, investor goals
└─ Extensions and Modern Developments:
   ├─ Black-Litterman Model:
   │   Blend market equilibrium with investor views
   │   Solves problem: Unconstrained optimization yields extreme weights
   │   Process:
   │   1. Start with implied returns from market prices (equilibrium)
   │   2. Specify views on expected returns (with confidence levels)
   │   3. Combine: Posterior = Prior + Views (Bayesian blend)
   │   Result: Stabilized, reasonable weights
   ├─ Robust Optimization:
   │   Account for parameter uncertainty
   │   Optimize worst-case (parameter value causes lowest return)
   │   Minimize: min_w max_μ (subject to μ in uncertainty set) -μ^T w
   │   More stable weights; higher tracking error but risk mitigation
   ├─ Multi-Period Stochastic Optimization:
   │   Optimal portfolio depends on current state (wealth, market conditions)
   │   Dynamic policy: w_t* = f(W_t, μ_t, σ_t)
   │   Target-date funds: Glide path based on time horizon
   │   Life-cycle portfolio: Age-dependent allocation
   ├─ Tail Risk Management:
   │   Use CVaR (expected shortfall) instead of variance
   │   Include tail hedges (options, bonds) to protect downside
   │   Stress testing: How does portfolio perform in worst scenarios?
   ├─ Factor Investing:
   │   Replace individual asset returns with factor exposures
   │   Factors: Market, size, value, momentum, quality, volatility
   │   Risk decomposition: Which factors drive portfolio risk?
   │   Efficient frontier in factor space (fewer dimensions)
   ├─ ESG (Environmental, Social, Governance):
   │   Add constraints: Exclude carbon-heavy stocks, governance poor
   │   May reduce returns (opportunity cost)
   │   Argument: Lower tail risk from regulatory/social pressure
   └─ Machine Learning Extensions:
      Estimate means/covariance with ML (neural nets, random forests)
      Potentially better than traditional statistics (feature engineering)
      Risk: Overfitting; data leakage in backtests
      Practical: Blend traditional + ML approaches
```

**Interaction:** Identify investor objectives and constraints → Choose framework (Markowitz vs Black-Litterman vs Robust) → Estimate parameters (μ, Σ) or elicit views → Optimize portfolio weights → Set rebalancing rules → Monitor performance vs benchmarks → Update estimates → Re-optimize.

## 5. Mini-Project
Comprehensive analysis of MPT foundations and assumption violations:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import norm, t
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("MODERN PORTFOLIO THEORY: FOUNDATIONS AND ASSUMPTIONS")
print("="*80)

class PortfolioOptimizer:
    """Core portfolio optimization tools"""
    
    def __init__(self, returns_df):
        """
        Initialize with returns DataFrame
        rows: dates, columns: assets
        """
        self.returns = returns_df
        self.n_assets = returns_df.shape[1]
        self.asset_names = returns_df.columns.tolist()
        
        # Compute statistics
        self.mu = returns_df.mean()
        self.sigma = returns_df.std()
        self.cov_matrix = returns_df.cov()
        self.corr_matrix = returns_df.corr()
        self.returns_annual = self.mu * 252
        self.sigma_annual = self.sigma * np.sqrt(252)
    
    def min_variance_portfolio(self, constrain_short_sales=True):
        """Minimum variance portfolio"""
        def objective(w):
            return np.sqrt(w @ self.cov_matrix @ w)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        if constrain_short_sales:
            bounds = Bounds(0, 1)
        else:
            bounds = Bounds(-10, 10)
        
        result = minimize(
            objective,
            x0=np.ones(self.n_assets)/self.n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def max_sharpe_portfolio(self, r_f=0.02, constrain_short_sales=True):
        """Maximum Sharpe ratio portfolio"""
        def neg_sharpe(w):
            ret = w @ self.mu * 252
            vol = np.sqrt(w @ self.cov_matrix @ w * 252)
            return -(ret - r_f) / vol
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        if constrain_short_sales:
            bounds = Bounds(0, 1)
        else:
            bounds = Bounds(-10, 10)
        
        result = minimize(
            neg_sharpe,
            x0=np.ones(self.n_assets)/self.n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def efficient_frontier(self, target_returns=None, constrain_short_sales=True, r_f=0.02):
        """Compute efficient frontier"""
        if target_returns is None:
            target_returns = np.linspace(self.returns_annual.min(), self.returns_annual.max(), 50)
        
        frontier_vols = []
        frontier_rets = []
        frontier_weights = []
        
        for target_ret in target_returns:
            def objective(w):
                return np.sqrt(w @ self.cov_matrix @ w)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ self.mu * 252 - target_ret}
            ]
            
            if constrain_short_sales:
                bounds = Bounds(0, 1)
            else:
                bounds = Bounds(-10, 10)
            
            try:
                result = minimize(
                    objective,
                    x0=np.ones(self.n_assets)/self.n_assets,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-9}
                )
                
                if result.success:
                    vol = np.sqrt(result.x @ self.cov_matrix @ result.x) * np.sqrt(252)
                    frontier_vols.append(vol)
                    frontier_rets.append(target_ret)
                    frontier_weights.append(result.x)
            except:
                continue
        
        return {
            'returns': frontier_rets,
            'volatilities': frontier_vols,
            'weights': frontier_weights
        }
    
    def portfolio_stats(self, weights):
        """Compute portfolio statistics"""
        ret = weights @ self.mu * 252
        vol = np.sqrt(weights @ self.cov_matrix @ weights * 252)
        sharpe = (ret - 0.02) / vol if vol > 0 else 0
        
        return {
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe
        }

class AssumptionAnalyzer:
    """Analyze violation of MPT assumptions"""
    
    def __init__(self, returns_df):
        self.returns = returns_df
        self.n_assets = returns_df.shape[1]
        self.T = returns_df.shape[0]
    
    def test_normality(self):
        """Test normality assumption (Jarque-Bera equivalent)"""
        results = []
        
        for col in self.returns.columns:
            r = self.returns[col]
            
            # Skewness and Kurtosis
            skew = ((r - r.mean())**3).mean() / (r.std()**3)
            kurt = ((r - r.mean())**4).mean() / (r.std()**4)
            
            # Excess kurtosis (normal = 0)
            excess_kurt = kurt - 3
            
            results.append({
                'asset': col,
                'skewness': skew,
                'excess_kurtosis': excess_kurt,
                'normal_test': 'Fail' if abs(skew) > 0.5 or excess_kurt > 1 else 'Pass'
            })
        
        return pd.DataFrame(results)
    
    def test_constant_correlation(self, window=60):
        """Test for constant correlation (rolling window)"""
        rolling_corrs = []
        
        for i in range(len(self.returns) - window):
            window_returns = self.returns.iloc[i:i+window]
            corr = window_returns.corr().values[0, 1]  # Correlation of first two assets
            rolling_corrs.append(corr)
        
        return {
            'mean_correlation': np.mean(rolling_corrs),
            'std_correlation': np.std(rolling_corrs),
            'min_correlation': np.min(rolling_corrs),
            'max_correlation': np.max(rolling_corrs),
            'rolling_corrs': rolling_corrs
        }
    
    def test_parameter_stability(self):
        """Test stability of mean and variance estimates"""
        # Split sample in half
        mid = len(self.returns) // 2
        first_half = self.returns.iloc[:mid]
        second_half = self.returns.iloc[mid:]
        
        mu_diff = (first_half.mean() - second_half.mean()).abs().mean()
        sigma_diff = (first_half.std() - second_half.std()).abs().mean()
        
        return {
            'mean_stability': mu_diff,
            'volatility_stability': sigma_diff,
            'stable': 'Yes' if mu_diff < 0.003 and sigma_diff < 0.01 else 'No'
        }
    
    def tail_risk_analysis(self):
        """Analyze tail risk (Value at Risk, Expected Shortfall)"""
        results = {}
        
        for col in self.returns.columns:
            r = self.returns[col]
            
            # Normal VaR (95%)
            var_95_normal = r.mean() - 1.645 * r.std()
            
            # Empirical VaR
            var_95_empirical = r.quantile(0.05)
            
            # CVaR (Expected Shortfall)
            cvar_95 = r[r <= var_95_empirical].mean()
            
            results[col] = {
                'VaR_95_Normal': var_95_normal,
                'VaR_95_Empirical': var_95_empirical,
                'CVaR_95': cvar_95,
                'Diff': var_95_normal - var_95_empirical
            }
        
        return pd.DataFrame(results).T

# Generate synthetic returns
np.random.seed(42)
n_days = 500
assets = ['Stock A', 'Stock B', 'Bond', 'Commodity']
n_assets = len(assets)

# Generate multivariate normal returns
mean_returns = np.array([0.0005, 0.0004, 0.0002, 0.0003])  # Daily
cov_matrix_true = np.array([
    [0.0001, 0.00005, -0.00002, 0.000001],
    [0.00005, 0.00012, -0.00003, 0.000002],
    [-0.00002, -0.00003, 0.00003, -0.000001],
    [0.000001, 0.000002, -0.000001, 0.00006]
])

# Generate normal returns
returns_normal = np.random.multivariate_normal(mean_returns, cov_matrix_true, n_days)

# Add some fat tails and skewness to first asset
tail_indices = np.random.choice(n_days, size=int(0.02*n_days), replace=False)
returns_normal[tail_indices, 0] *= 3  # Amplify extremes

returns_df = pd.DataFrame(returns_normal, columns=assets)

print(f"\nData: {n_days} daily returns for {n_assets} assets")
print(f"Sample period: {n_days} days (~{n_days/252:.1f} years)")

# 1. Test Assumptions
print("\n" + "="*80)
print("1. TESTING MPT ASSUMPTIONS")
print("="*80)

analyzer = AssumptionAnalyzer(returns_df)

# Normality test
print("\nNormality Test (Jarque-Bera style):")
normality = analyzer.test_normality()
print(normality.to_string(index=False))

# Correlation stability
print("\nCorrelation Stability Test:")
corr_stability = analyzer.test_constant_correlation(window=60)
print(f"  Mean correlation: {corr_stability['mean_correlation']:.4f}")
print(f"  Std dev: {corr_stability['std_correlation']:.4f}")
print(f"  Range: [{corr_stability['min_correlation']:.4f}, {corr_stability['max_correlation']:.4f}]")
print(f"  Assessment: {'Stable' if corr_stability['std_correlation'] < 0.05 else 'Varying significantly'}")

# Parameter stability
print("\nParameter Stability Test (First half vs Second half):")
param_stability = analyzer.test_parameter_stability()
print(f"  Mean estimate difference: {param_stability['mean_stability']:.6f}")
print(f"  Volatility estimate difference: {param_stability['volatility_stability']:.6f}")
print(f"  Overall: {param_stability['stable']}")

# Tail risk
print("\nTail Risk Analysis (95% confidence level):")
tail_risk = analyzer.tail_risk_analysis()
print(tail_risk)

# 2. Optimization: No constraints vs Short-sale constraints
print("\n" + "="*80)
print("2. OPTIMAL PORTFOLIO COMPARISON")
print("="*80)

optimizer = PortfolioOptimizer(returns_df)

# Minimum variance portfolios
w_minvar_unconstrained = optimizer.min_variance_portfolio(constrain_short_sales=False)
w_minvar_constrained = optimizer.min_variance_portfolio(constrain_short_sales=True)

print("\nMinimum Variance Portfolio (Unconstrained):")
for i, (asset, weight) in enumerate(zip(assets, w_minvar_unconstrained)):
    print(f"  {asset}: {weight:7.2%}")

print("\nMinimum Variance Portfolio (No Short Sales):")
for i, (asset, weight) in enumerate(zip(assets, w_minvar_constrained)):
    print(f"  {asset}: {weight:7.2%}")

# Maximum Sharpe portfolios
w_sharpe_unconstrained = optimizer.max_sharpe_portfolio(constrain_short_sales=False)
w_sharpe_constrained = optimizer.max_sharpe_portfolio(constrain_short_sales=True)

print("\nMaximum Sharpe Ratio Portfolio (Unconstrained):")
for i, (asset, weight) in enumerate(zip(assets, w_sharpe_unconstrained)):
    print(f"  {asset}: {weight:7.2%}")

print("\nMaximum Sharpe Ratio Portfolio (No Short Sales):")
for i, (asset, weight) in enumerate(zip(assets, w_sharpe_constrained)):
    print(f"  {asset}: {weight:7.2%}")

# 3. Efficient Frontier
print("\n" + "="*80)
print("3. EFFICIENT FRONTIER COMPUTATION")
print("="*80)

frontier_unconstrained = optimizer.efficient_frontier(constrain_short_sales=False)
frontier_constrained = optimizer.efficient_frontier(constrain_short_sales=True)

print(f"\nUnconstrained frontier: {len(frontier_unconstrained['returns'])} portfolios")
print(f"  Return range: [{min(frontier_unconstrained['returns'])*100:.2f}%, {max(frontier_unconstrained['returns'])*100:.2f}%]")
print(f"  Volatility range: [{min(frontier_unconstrained['volatilities'])*100:.2f}%, {max(frontier_unconstrained['volatilities'])*100:.2f}%]")

print(f"\nConstrained frontier: {len(frontier_constrained['returns'])} portfolios")
print(f"  Return range: [{min(frontier_constrained['returns'])*100:.2f}%, {max(frontier_constrained['returns'])*100:.2f}%]")
print(f"  Volatility range: [{min(frontier_constrained['volatilities'])*100:.2f}%, {max(frontier_constrained['volatilities'])*100:.2f}%]")

# 4. Asset characteristics
print("\n" + "="*80)
print("4. INDIVIDUAL ASSET CHARACTERISTICS")
print("="*80)

print(f"\n{'Asset':<15} {'Annual Return':<15} {'Annual Vol':<15} {'Sharpe Ratio':<15}")
print("-" * 60)
for i, asset in enumerate(assets):
    w = np.zeros(n_assets)
    w[i] = 1.0
    stats = optimizer.portfolio_stats(w)
    print(f"{asset:<15} {stats['return']*100:>13.2f}% {stats['volatility']*100:>13.2f}% {stats['sharpe']:>13.2f}")

# 5. Estimation error analysis
print("\n" + "="*80)
print("5. ESTIMATION ERROR IMPACT")
print("="*80)

# Split into estimation and test periods
split = 250
est_period = returns_df.iloc[:split]
test_period = returns_df.iloc[split:]

optimizer_est = PortfolioOptimizer(est_period)
w_optimal_est = optimizer_est.max_sharpe_portfolio(constrain_short_sales=True)

# Compute out-of-sample performance
optimizer_test = PortfolioOptimizer(test_period)
test_stats = optimizer_test.portfolio_stats(w_optimal_est)

print(f"\nPortfolio optimized on first {split} days")
print(f"Weights: {dict(zip(assets, [f'{w:.2%}' for w in w_optimal_est]))}")
print(f"\nIn-sample (estimation period) performance:")
in_sample_stats = optimizer_est.portfolio_stats(w_optimal_est)
print(f"  Return: {in_sample_stats['return']*100:.2f}%")
print(f"  Volatility: {in_sample_stats['volatility']*100:.2f}%")
print(f"  Sharpe: {in_sample_stats['sharpe']:.4f}")

print(f"\nOut-of-sample (test period) performance:")
print(f"  Return: {test_stats['return']*100:.2f}%")
print(f"  Volatility: {test_stats['volatility']*100:.2f}%")
print(f"  Sharpe: {test_stats['sharpe']:.4f}")

print(f"\nEstimation error impact:")
print(f"  Return degradation: {(in_sample_stats['return'] - test_stats['return'])*100:.2f}%")
print(f"  Sharpe degradation: {(in_sample_stats['sharpe'] - test_stats['sharpe']):.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Efficient Frontier
ax = axes[0, 0]
if frontier_unconstrained['returns']:
    ax.plot(np.array(frontier_unconstrained['volatilities'])*100, 
            np.array(frontier_unconstrained['returns'])*100, 
            'b-', linewidth=2, label='Unconstrained')
if frontier_constrained['returns']:
    ax.plot(np.array(frontier_constrained['volatilities'])*100,
            np.array(frontier_constrained['returns'])*100,
            'r-', linewidth=2, label='No Short Sales')

# Individual assets
for i, asset in enumerate(assets):
    w = np.zeros(n_assets)
    w[i] = 1.0
    stats = optimizer.portfolio_stats(w)
    ax.scatter(stats['volatility']*100, stats['return']*100, s=100, label=asset)

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Expected Return (%)')
ax.set_title('Efficient Frontier: Constrained vs Unconstrained')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Optimal weights comparison
ax = axes[0, 1]
x = np.arange(n_assets)
width = 0.35
ax.bar(x - width/2, w_sharpe_unconstrained, width, label='Unconstrained', alpha=0.8)
ax.bar(x + width/2, w_sharpe_constrained, width, label='Constrained', alpha=0.8)
ax.set_ylabel('Weight')
ax.set_title('Maximum Sharpe Ratio Portfolio: Weights Comparison')
ax.set_xticks(x)
ax.set_xticklabels(assets, rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)

# Plot 3: Normality test (Q-Q plot)
ax = axes[0, 2]
returns_stock_a = returns_df.iloc[:, 0]
sorted_returns = np.sort(returns_stock_a)
theoretical = norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
ax.scatter(theoretical, sorted_returns, alpha=0.6)
ax.plot(theoretical, sorted_returns, 'r-', linewidth=1)
ax.set_xlabel('Theoretical Quantiles (Normal)')
ax.set_ylabel('Sample Quantiles')
ax.set_title('Q-Q Plot: Testing Normality Assumption')
ax.grid(alpha=0.3)

# Plot 4: Rolling correlation
ax = axes[1, 0]
ax.plot(corr_stability['rolling_corrs'], linewidth=1.5)
ax.axhline(corr_stability['mean_correlation'], color='r', linestyle='--', label='Mean')
ax.fill_between(range(len(corr_stability['rolling_corrs'])),
                corr_stability['mean_correlation'] - corr_stability['std_correlation'],
                corr_stability['mean_correlation'] + corr_stability['std_correlation'],
                alpha=0.2)
ax.set_xlabel('Time Window')
ax.set_ylabel('Correlation')
ax.set_title('Rolling Correlation: Testing Constant Correlation')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Skewness and Excess Kurtosis
ax = axes[1, 1]
normality_results = analyzer.test_normality()
x = np.arange(len(assets))
width = 0.35
ax.bar(x - width/2, normality_results['skewness'], width, label='Skewness', alpha=0.8)
ax.bar(x + width/2, normality_results['excess_kurtosis'], width, label='Excess Kurtosis', alpha=0.8)
ax.set_ylabel('Value')
ax.set_title('Normality Tests: Skewness and Kurtosis')
ax.set_xticks(x)
ax.set_xticklabels(assets, rotation=45)
ax.legend()
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.grid(alpha=0.3, axis='y')

# Plot 6: In-sample vs Out-of-sample
ax = axes[1, 2]
metrics = ['Return', 'Volatility', 'Sharpe']
in_sample = [in_sample_stats['return']*100, in_sample_stats['volatility']*100, in_sample_stats['sharpe']]
out_sample = [test_stats['return']*100, test_stats['volatility']*100, test_stats['sharpe']]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, in_sample, width, label='In-Sample', alpha=0.8)
ax.bar(x + width/2, out_sample, width, label='Out-of-Sample', alpha=0.8)
ax.set_ylabel('Value')
ax.set_title('Estimation Error: In-Sample vs Out-of-Sample')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("KEY INSIGHTS FROM ASSUMPTION TESTING")
print("="*80)
print(f"1. Normality: Fat tails detected ({normality.loc[normality['normal_test']=='Fail'].shape[0]} assets fail)")
print(f"2. Correlation: {'Stable' if corr_stability['std_correlation'] < 0.05 else 'Time-varying'}")
print(f"3. Parameter Stability: {param_stability['stable']}")
print(f"4. Optimization Impact: Constraints reduce variance by {((w_minvar_constrained @ optimizer.cov_matrix @ w_minvar_constrained)**0.5 - (w_minvar_unconstrained @ optimizer.cov_matrix @ w_minvar_unconstrained)**0.5)*np.sqrt(252)*100:.2f}%")
print(f"5. Estimation Error: Out-of-sample Sharpe {(test_stats['sharpe'] - in_sample_stats['sharpe']):.4f} points lower")
```

## 6. Challenge Round
1. **Shrinkage Estimators:** Implement Ledoit-Wolf shrinkage for covariance matrix. Compare efficient frontiers: sample cov vs shrunk. Which has better out-of-sample performance? Estimate optimal shrinkage intensity.

2. **Black-Litterman:** Start with market cap weights (equilibrium). Specify bullish view on Stock A (expect +5% excess return). Blend with market equilibrium. Compare to unconstrained optimization (extreme weights).

3. **Regime Switching:** Model two correlation regimes (normal, crisis). Estimate transition probabilities. Optimize portfolio conditional on current regime. Does portfolio shift between regimes?

4. **Tail Risk Optimization:** Use CVaR (expected shortfall) instead of variance as risk metric. Optimize portfolio minimizing CVaR at 95% level. Compare weights to traditional mean-variance.

5. **Parameter Uncertainty:** Simulate parameter estimation error (sample covariance has noise). Generate 100 portfolio samples with perturbed parameters. Plot distribution of realized Sharpe ratios. How much does estimation error matter?

## 7. Key References
- [Markowitz, "Portfolio Selection" (1952)](https://www.jstor.org/stable/2975974) - foundational mean-variance framework
- [Black & Litterman, "Global Portfolio Optimization" (1992)](https://www.ssrnpapers.com/sol3/papers.cfm?abstract_id=2382500) - incorporating views and market equilibrium
- [Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix" (2004)](https://www.jstor.org/stable/1392074) - shrinkage estimator reducing estimation error

---
**Status:** Theoretical foundation for portfolio management | **Complements:** Mean-Variance Optimization, CAPM, Risk Measures, Behavioral Finance, Multi-Factor Models
