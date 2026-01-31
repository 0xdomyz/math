# Portfolio Optimization

## 1. Concept Skeleton
**Definition:** Selecting asset weights to maximize expected return for given risk or minimize risk for target return (Markowitz mean-variance framework)  
**Purpose:** Construct efficient portfolios, balance diversification vs concentration, manage estimation error impact on weights  
**Prerequisites:** Covariance matrix, expected returns estimation, quadratic optimization, estimation uncertainty

## 2. Comparative Framing
| Approach | Objective | Constraints | Inputs Required | Sensitivity to Error |
|----------|-----------|-------------|-----------------|----------------------|
| **Mean-Variance** | Maximize return per unit risk | Long-only, budget | μ̂, Σ̂ (n + n²/2 parameters) | Extreme (corner solutions) |
| **Min-Variance** | Minimize variance | Long-only, budget | Σ̂ only (n²/2 parameters) | Moderate (stable) |
| **Risk Parity** | Equal risk contribution | Long-only, budget | Σ̂ only | Low (robust) |
| **1/N (Naive)** | Equal weight | Long-only | None (0 parameters) | None (benchmark) |
| **Black-Litterman** | Bayesian blend | Market equilibrium prior | Views + confidence | Moderate (shrinkage effect) |

## 3. Examples + Counterexamples

**Simple Example:**  
2 assets: μ₁=10%, σ₁=15%; μ₂=8%, σ₂=10%; ρ=0.3 → Optimal w*=[0.60, 0.40] (12% target return); σ_p=10.5% (diversification reduces risk below either asset)

**Failure Case:**  
Estimate μ̂₁=12% (true 10%); μ̂₂=7% (true 8%) → Optimal ŵ=[0.95, 0.05] (extreme overweight asset 1); realized return 10.1% with σ_p=14.5% (worse than naive 1/N)

**Edge Case:**  
10 assets; 5 years data (60 months) → Covariance matrix 55 parameters estimated; rank deficient (60<55+10); optimization unstable (negative weights, turnover >200%)

## 4. Layer Breakdown
```
Portfolio Optimization Structure:
├─ Mean-Variance Framework (Markowitz 1952):
│   ├─ Problem formulation:
│   │   ├─ Objective: min w'Σw subject to w'μ = μ_target, w'1 = 1
│   │   │   ├─ w: Portfolio weights (n×1 vector)
│   │   │   ├─ Σ: Covariance matrix (n×n; symmetric positive definite)
│   │   │   ├─ μ: Expected return vector (n×1)
│   │   │   └─ μ_target: Target portfolio return
│   │   ├─ Lagrangian: L = w'Σw - λ₁(w'μ - μ_target) - λ₂(w'1 - 1)
│   │   ├─ First-order condition: 2Σw - λ₁μ - λ₂1 = 0
│   │   └─ Solution: w* = Σ⁻¹(λ₁μ + λ₂1) where λ₁, λ₂ from constraints
│   ├─ Efficient Frontier:
│   │   ├─ Definition: Set of portfolios with min variance for each return level
│   │   │   ├─ Parametric form: σ²_p(μ_p) = A - 2Bμ_p + Cμ²_p
│   │   │   ├─ A = d·1'Σ⁻¹1, B = d·1'Σ⁻¹μ, C = d·μ'Σ⁻¹μ
│   │   │   ├─ d = 1/Δ where Δ = (1'Σ⁻¹1)(μ'Σ⁻¹μ) - (1'Σ⁻¹μ)²
│   │   │   └─ Shape: Hyperbola in (σ, μ) space; upper branch is efficient
│   │   ├─ Global minimum variance portfolio:
│   │   │   ├─ w_gmv = Σ⁻¹1 / (1'Σ⁻¹1)
│   │   │   ├─ Return: μ_gmv = 1'Σ⁻¹μ / (1'Σ⁻¹1)
│   │   │   ├─ Variance: σ²_gmv = 1 / (1'Σ⁻¹1)
│   │   │   └─ Example: 10 assets → μ_gmv=7.5%, σ_gmv=8.2%
│   │   └─ Tangency portfolio (with risk-free asset):
│   │       ├─ w_tan ∝ Σ⁻¹(μ - r_f·1) (weights proportional to excess return per risk)
│   │       ├─ Sharpe ratio: SR_tan = √[(μ-r_f·1)'Σ⁻¹(μ-r_f·1)]
│   │       ├─ Capital Market Line: μ_p = r_f + SR_tan·σ_p
│   │       └─ All efficient portfolios: Combinations of r_f and tangency
│   ├─ Two-Fund Separation Theorem:
│   │   ├─ Any efficient portfolio = combination of 2 frontier portfolios
│   │   ├─ Corollary: With risk-free → all investors hold tangency + r_f
│   │   └─ Implication: CAPM (market portfolio = tangency)
│   └─ Constraints:
│       ├─ Long-only: w_i ≥ 0 for all i (inequality constraints)
│       │   ├─ Solution: Quadratic programming (QP) with bounds
│       │   ├─ Effect: Reduces extreme weights; frontier shifts right (higher risk for same return)
│       │   └─ Example: Unconstrained w*=[-0.3, 0.7, 0.6]; constrained w=[0, 0.55, 0.45]
│       ├─ Turnover constraint: ||w_new - w_old||₁ ≤ τ
│       │   ├─ Limits transaction costs
│       │   ├─ Typical τ: 20-50% quarterly rebalance
│       │   └─ Trade-off: Lower turnover vs suboptimal weights
│       ├─ Sector constraints: Σ(w_i for i in sector S) ≤ c_S
│       │   ├─ Risk management: Limit concentration
│       │   ├─ Example: Tech ≤30%, Finance ≤25%
│       │   └─ Implementation: Linear inequality Aw ≤ b
│       └─ Tracking error constraint: (w-w_benchmark)'Σ(w-w_benchmark) ≤ TE²
│           ├─ Active management with benchmark tracking
│           ├─ Typical TE: 2-5% annually for active funds
│           └─ Trade-off: Active bets vs index hugging
├─ Estimation Error Impact:
│   ├─ Parameter uncertainty:
│   │   ├─ Inputs required: n expected returns, n(n+1)/2 covariances
│   │   │   ├─ For n=100: 100 + 5,050 = 5,150 parameters
│   │   │   ├─ Typical data: 60 months (5 years) × 100 assets = 6,000 obs
│   │   │   └─ Problem: Near-singular covariance; unstable inversion
│   │   ├─ Estimation error model:
│   │   │   ├─ True: μ, Σ; Estimated: μ̂ ~ N(μ, Σ/T), Σ̂ ~ Wishart(Σ, T)
│   │   │   ├─ Optimal weights: w* = f(μ, Σ); Estimated: ŵ = f(μ̂, Σ̂)
│   │   │   ├─ Error: ||ŵ - w*|| = O(1/√T) (converges slowly)
│   │   │   └─ Magnification: Mean estimation error dominates (σ(μ̂) >> σ(Σ̂) relative impact)
│   │   └─ Simulation evidence (Michaud 1989):
│   │       ├─ True Sharpe 0.5 → Realized Sharpe 0.2 (out-of-sample)
│   │       ├─ Extreme weights: w_i ∈ [-2, 3] (short 200%, long 300%)
│   │       └─ Naive 1/N often outperforms (DeMiguel et al. 2009)
│   ├─ Sources of error:
│   │   ├─ Mean estimation:
│   │   │   ├─ Standard error: SE(μ̂_i) = σ_i/√T
│   │   │   ├─ Example: σ=20%, T=60 months → SE=2.6% (huge relative to μ=10%)
│   │   │   ├─ Consequence: Overweights assets with upward-biased μ̂
│   │   │   └─ Reversal: Out-of-sample underperformance
│   │   ├─ Covariance estimation:
│   │   │   ├─ Sample covariance: Σ̂ = (1/T)Σr_t r'_t
│   │   │   ├─ Condition number: κ(Σ̂) = λ_max/λ_min (high when T≈n)
│   │   │   ├─ Inversion instability: Small eigenvalues → large errors in Σ̂⁻¹
│   │   │   └─ Example: n=100, T=60 → Σ̂ nearly singular; weights unstable
│   │   └─ Non-stationarity:
│   │       ├─ Parameters change over time (regime shifts)
│   │       ├─ Historical estimates: Use past 5 years; may not reflect future
│   │       └─ Example: Pre-COVID Σ̂ → Useless March 2020 (correlations spiked to 0.9+)
│   ├─ Consequences:
│   │   ├─ Extreme weights: ŵ_i >> 1 or << 0 (optimizer exploits noise)
│   │   ├─ Concentration: Top 3 assets get 80%+ weight
│   │   ├─ High turnover: Small estimate changes → large reallocation
│   │   └─ Out-of-sample underperformance: Realized Sharpe 50% lower
│   └─ Mitigation strategies:
│       ├─ Shrinkage estimators:
│       │   ├─ James-Stein: μ̂_shrink = (1-λ)μ̂ + λμ̄·1 (shrink toward grand mean)
│       │   ├─ Ledoit-Wolf: Σ̂_shrink = (1-δ)Σ̂ + δ·I·trace(Σ̂)/n
│       │   ├─ Optimal shrinkage: δ* minimizes MSE (data-driven)
│       │   └─ Effect: Reduces extreme estimates; improves out-of-sample
│       ├─ Robust optimization:
│       │   ├─ Worst-case: max_w min_{μ∈U} w'μ subject to w'Σw≤σ² (robust to uncertainty set U)
│       │   ├─ Bayesian: Incorporate prior distribution on μ
│       │   └─ Resampling: Bootstrap confidence intervals; average weights
│       ├─ Regularization:
│       │   ├─ Ridge: min w'Σw + γ||w-w_prior||² (penalize deviation from prior)
│       │   ├─ LASSO: min w'Σw + γ||w||₁ (sparse portfolio; many w_i=0)
│       │   └─ Parameter: γ controls simplicity vs fit (cross-validation)
│       └─ Dimension reduction:
│           ├─ Factor models: r_t = Bf_t + ε_t (k<n factors)
│           ├─ PCA: Retain top 5-10 principal components
│           └─ Effect: n(n+1)/2 → nk + k(k+1)/2 parameters (massive reduction)
├─ Minimum Variance Portfolio:
│   ├─ Formulation: min w'Σw subject to w'1=1 (no return target)
│   │   ├─ Solution: w_mv = Σ⁻¹1 / (1'Σ⁻¹1)
│   │   ├─ Only uses Σ (not μ); less sensitive to estimation error
│   │   └─ Example: n=50 → Σ has 1,275 parameters vs 1,325 for mean-variance
│   ├─ Performance:
│   │   ├─ Empirical: Sharpe ratio 0.6-0.8 (competitive with mean-variance)
│   │   ├─ Stability: Low turnover (10-20% annually)
│   │   └─ Out-of-sample: Often beats mean-variance (robustness)
│   └─ Interpretation:
│       ├─ Weights: High on low-volatility, low-correlation assets
│       ├─ Low-volatility anomaly: Min-vol outperforms market (Haugen & Baker 1991)
│       └─ Explanation: Leverage constraints → investors overbid high-beta
├─ Risk Parity:
│   ├─ Principle: Equalize risk contribution across assets
│   │   ├─ Marginal risk: MR_i = (Σw)_i = ∂σ_p/∂w_i
│   │   ├─ Risk contribution: RC_i = w_i·MR_i (amount of total risk from asset i)
│   │   ├─ Objective: RC_1 = RC_2 = ... = RC_n = σ_p/n
│   │   └─ Solution: Nonlinear system; iterative (Newton's method)
│   ├─ Implementation:
│   │   ├─ Simplified (inverse volatility): w_i ∝ 1/σ_i
│   │   ├─ General: Solve (Σw)_i = c/w_i for all i (c = constant)
│   │   └─ Example: 2 assets σ₁=15%, σ₂=10%, ρ=0.3 → w=[0.40, 0.60]
│   ├─ Properties:
│   │   ├─ Tilts toward low-volatility assets
│   │   ├─ Diversification: No single asset dominates risk
│   │   └─ Leverage: Often requires leverage to reach target return
│   └─ Performance:
│       ├─ Sharpe ratio: 0.5-0.7 (stable across regimes)
│       ├─ Drawdowns: -20% vs -40% market (defensive)
│       └─ Critique: Ignores expected returns (may underweight high-return assets)
├─ Black-Litterman Model (1992):
│   ├─ Motivation: Incorporate investor views + market equilibrium
│   │   ├─ Problem: Historical μ̂ unreliable; pure market weights too passive
│   │   ├─ Solution: Bayesian framework blending equilibrium + views
│   │   └─ Output: Posterior expected returns E[μ|views]
│   ├─ Equilibrium (prior):
│   │   ├─ Reverse optimization: μ_eq = λΣw_mkt
│   │   │   ├─ w_mkt: Market cap weights
│   │   │   ├─ λ: Risk aversion (typical 2.5-3.5)
│   │   │   └─ Interpretation: Expected returns consistent with market holdings
│   │   ├─ Example: Asset 1 has w_mkt=0.60, σ₁=15% → μ_eq,1 = 2.5×0.6×0.15²≈0.034 (3.4%)
│   │   └─ Advantage: Avoids extreme estimates (stable baseline)
│   ├─ Views (updates):
│   │   ├─ Absolute: E[r_A] = 12% with 2% uncertainty
│   │   ├─ Relative: E[r_A - r_B] = 3% with 1.5% uncertainty
│   │   ├─ Matrix form: P·μ = q + ε where ε ~ N(0, Ω)
│   │   │   ├─ P: View matrix (k×n; each row is a view)
│   │   │   ├─ q: View values (k×1)
│   │   │   └─ Ω: Uncertainty (k×k diagonal; confidence in each view)
│   │   └─ Example: View 1: Tech outperforms 5% (P=[1,0,-1], q=0.05, Ω₁₁=0.02²)
│   ├─ Bayesian update:
│   │   ├─ Posterior mean: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹μ_eq + P'Ω⁻¹q]
│   │   ├─ Posterior covariance: Σ_BL = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
│   │   ├─ τ: Scaling (typically 0.01-0.05; uncertainty in equilibrium)
│   │   └─ Interpretation: Weighted average of prior and views (high Ω → views matter less)
│   ├─ Optimal weights:
│   │   ├─ w_BL = Σ⁻¹μ_BL / (1'Σ⁻¹μ_BL) (mean-variance with posterior)
│   │   └─ Advantage: Moderate weights (between market and extreme MV)
│   └─ Performance:
│       ├─ Sharpe ratio: 0.7-0.9 (incorporates views without instability)
│       ├─ Turnover: 30-50% annually (moderate; stable tilt toward views)
│       └─ Adoption: Popular among institutional investors (AQR, Bridgewater)
├─ 1/N Naive Diversification:
│   ├─ Strategy: Equal weight all assets (w_i = 1/n)
│   │   ├─ No estimation (0 parameters)
│   │   ├─ Rebalancing: Annually or quarterly to maintain equal weights
│   │   └─ Example: S&P 500 equal-weight vs cap-weight
│   ├─ Theoretical justification:
│   │   ├─ DeMiguel et al. (2009): 1/N beats 14 optimization models out-of-sample
│   │   ├─ Short sample: T < n² → Estimation error dominates optimization gains
│   │   └─ Diversification: As n→∞, σ²_p → E[σ_ij] (avg covariance; idiosyncratic risk eliminated)
│   ├─ Performance:
│   │   ├─ Sharpe ratio: 0.6-0.8 (surprisingly competitive)
│   │   ├─ Turnover: 10-20% (low; only rebalancing drift)
│   │   └─ Robustness: No parameter risk; stable across regimes
│   └─ When it fails:
│       ├─ Extreme differences: Mix stocks + bonds → suboptimal (ignore σ differences)
│       ├─ Large n: n=500 equally weights micro-caps (liquidity issues)
│       └─ Strong views: If genuinely have superior μ forecast → use it
└─ Transaction Costs & Turnover:
    ├─ Cost model: TC = Σ|w_new,i - w_old,i|·(spread_i/2 + commission_i)
    │   ├─ Bid-ask spread: 0.01% (large cap) to 1% (small cap)
    │   ├─ Commissions: 0-0.5 bps (institutional) to 1 bps (retail)
    │   ├─ Market impact: √(Trade size/ADV) × volatility × 0.1 (temporary)
    │   └─ Example: $100M portfolio, 50% turnover, 10 bps total cost → $50K annually
    ├─ Turnover-constrained optimization:
    │   ├─ Objective: max w'μ - λw'Σw - κ·Σ|w_i - w_old,i|
    │   ├─ κ: Cost penalty (proportional to cost rate)
    │   └─ Solution: Sparse updates (only adjust if benefit > cost threshold)
    ├─ Rebalancing frequency:
    │   ├─ High-frequency (daily): High costs, small drift correction
    │   ├─ Low-frequency (annual): Low costs, large drift (suboptimal weights)
    │   └─ Optimal: Quarterly to semi-annually (balance trade-off)
    └─ Tax considerations:
        ├─ Capital gains: Short-term (ordinary rate) vs long-term (lower rate)
        ├─ Tax-loss harvesting: Sell losers to offset gains
        └─ Turnover penalty: Realized gains → tax drag 1-2% annually
```

**Key Insight:** Estimation error dominates optimization gains; simple robust methods (min-variance, risk parity, 1/N) often outperform unconstrained mean-variance out-of-sample

## 5. Mini-Project
Mean-variance optimization with estimation error analysis:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_assets = 10
T_estimation = 60  # 5 years of monthly data
T_test = 120  # 10 years out-of-sample
n_simulations = 100

# True parameters (unknown to optimizer)
mu_true = np.random.uniform(0.06, 0.14, n_assets) / 12  # Monthly returns 6-14% annual
Sigma_true = np.random.randn(n_assets, n_assets) * 0.03
Sigma_true = Sigma_true @ Sigma_true.T  # Ensure positive definite
Sigma_true += np.diag(np.random.uniform(0.001, 0.004, n_assets))  # Add idiosyncratic risk

# Helper functions
def generate_returns(mu, Sigma, T):
    """Generate multivariate normal returns"""
    return np.random.multivariate_normal(mu, Sigma, T)

def mean_variance_weights(mu, Sigma, target_return=None):
    """Compute mean-variance optimal weights"""
    n = len(mu)
    
    if target_return is None:
        # Tangency portfolio (max Sharpe; assume r_f=0)
        Sigma_inv = np.linalg.inv(Sigma)
        w = Sigma_inv @ mu
        w /= np.sum(w)
    else:
        # Target return portfolio
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(n)
        
        A = ones.T @ Sigma_inv @ ones
        B = ones.T @ Sigma_inv @ mu
        C = mu.T @ Sigma_inv @ mu
        
        lambda1 = (C - target_return * B) / (A * C - B**2)
        lambda2 = (target_return * A - B) / (A * C - B**2)
        
        w = Sigma_inv @ (lambda1 * ones + lambda2 * mu)
    
    return w

def min_variance_weights(Sigma):
    """Compute minimum variance weights"""
    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(len(Sigma))
    w = Sigma_inv @ ones
    return w / np.sum(w)

def naive_weights(n):
    """1/N equal weights"""
    return np.ones(n) / n

def portfolio_performance(w, mu, Sigma):
    """Calculate portfolio return and volatility"""
    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

# Simulation: Compare strategies with estimation error
results = {
    'Mean-Variance': {'returns': [], 'vols': [], 'sharpes': []},
    'Min-Variance': {'returns': [], 'vols': [], 'sharpes': []},
    '1/N Naive': {'returns': [], 'vols': [], 'sharpes': []}
}

print("="*70)
print("Portfolio Optimization: Estimation Error Impact")
print("="*70)
print(f"Assets: {n_assets}")
print(f"Estimation period: {T_estimation} months")
print(f"Out-of-sample test: {T_test} months")
print(f"Simulations: {n_simulations}")
print("")

for sim in range(n_simulations):
    # Generate estimation data
    returns_est = generate_returns(mu_true, Sigma_true, T_estimation)
    
    # Estimate parameters (with error)
    mu_hat = returns_est.mean(axis=0)
    Sigma_hat = np.cov(returns_est.T)
    
    # Compute optimal weights using estimated parameters
    try:
        w_mv = mean_variance_weights(mu_hat, Sigma_hat)
        w_minvar = min_variance_weights(Sigma_hat)
    except np.linalg.LinAlgError:
        # Singular matrix (happens with estimation error)
        continue
    
    w_naive = naive_weights(n_assets)
    
    # Generate out-of-sample test data
    returns_test = generate_returns(mu_true, Sigma_true, T_test)
    
    # Calculate realized performance
    for name, w in [('Mean-Variance', w_mv), ('Min-Variance', w_minvar), ('1/N Naive', w_naive)]:
        # Realized returns using true distribution
        portfolio_returns = returns_test @ w
        realized_mean = portfolio_returns.mean()
        realized_vol = portfolio_returns.std()
        realized_sharpe = realized_mean / realized_vol if realized_vol > 0 else 0
        
        results[name]['returns'].append(realized_mean * 12)  # Annualize
        results[name]['vols'].append(realized_vol * np.sqrt(12))  # Annualize
        results[name]['sharpes'].append(realized_sharpe * np.sqrt(12))  # Annualize

# Compute statistics across simulations
print("Out-of-Sample Performance (Annualized):")
print("="*70)
print(f"{'Strategy':<20} {'Return':<12} {'Volatility':<12} {'Sharpe Ratio':<12}")
print("-"*70)

for name in results:
    mean_ret = np.mean(results[name]['returns'])
    mean_vol = np.mean(results[name]['vols'])
    mean_sharpe = np.mean(results[name]['sharpes'])
    
    print(f"{name:<20} {mean_ret:>8.2%}    {mean_vol:>8.2%}    {mean_sharpe:>8.3f}")

# Statistical tests
mv_sharpes = np.array(results['Mean-Variance']['sharpes'])
naive_sharpes = np.array(results['1/N Naive']['sharpes'])
diff = mv_sharpes - naive_sharpes

from scipy import stats as sp_stats
t_stat, p_value = sp_stats.ttest_rel(mv_sharpes, naive_sharpes)

print("")
print("Statistical Test: Mean-Variance vs 1/N")
print("-"*70)
print(f"Mean Sharpe difference: {diff.mean():>8.3f}")
print(f"t-statistic: {t_stat:>8.3f}")
print(f"p-value: {p_value:>8.4f}")
print(f"Result: {'1/N significantly better' if p_value < 0.05 and diff.mean() < 0 else 'No significant difference'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of Sharpe ratios
axes[0, 0].hist(results['Mean-Variance']['sharpes'], bins=20, alpha=0.6, label='Mean-Variance', color='blue')
axes[0, 0].hist(results['Min-Variance']['sharpes'], bins=20, alpha=0.6, label='Min-Variance', color='green')
axes[0, 0].hist(results['1/N Naive']['sharpes'], bins=20, alpha=0.6, label='1/N Naive', color='orange')
axes[0, 0].axvline(np.mean(results['Mean-Variance']['sharpes']), color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(np.mean(results['1/N Naive']['sharpes']), color='orange', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Sharpe Ratio')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Out-of-Sample Sharpe Ratio Distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Risk-return scatter
for name, color in [('Mean-Variance', 'blue'), ('Min-Variance', 'green'), ('1/N Naive', 'orange')]:
    axes[0, 1].scatter(results[name]['vols'], results[name]['returns'], 
                      alpha=0.5, s=30, label=name, color=color)
    
    # Plot mean
    mean_vol = np.mean(results[name]['vols'])
    mean_ret = np.mean(results[name]['returns'])
    axes[0, 1].scatter(mean_vol, mean_ret, s=200, marker='*', 
                      edgecolor='black', linewidth=2, color=color)

axes[0, 1].set_xlabel('Volatility (annualized)')
axes[0, 1].set_ylabel('Return (annualized)')
axes[0, 1].set_title('Out-of-Sample Risk-Return')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Weight distribution for one simulation
returns_est_sample = generate_returns(mu_true, Sigma_true, T_estimation)
mu_hat_sample = returns_est_sample.mean(axis=0)
Sigma_hat_sample = np.cov(returns_est_sample.T)

w_mv_sample = mean_variance_weights(mu_hat_sample, Sigma_hat_sample)
w_minvar_sample = min_variance_weights(Sigma_hat_sample)
w_naive_sample = naive_weights(n_assets)

x = np.arange(n_assets)
width = 0.25
axes[1, 0].bar(x - width, w_mv_sample, width, label='Mean-Variance', color='blue', alpha=0.7)
axes[1, 0].bar(x, w_minvar_sample, width, label='Min-Variance', color='green', alpha=0.7)
axes[1, 0].bar(x + width, w_naive_sample, width, label='1/N Naive', color='orange', alpha=0.7)
axes[1, 0].axhline(0, color='black', linewidth=0.8)
axes[1, 0].set_xlabel('Asset')
axes[1, 0].set_ylabel('Weight')
axes[1, 0].set_title('Portfolio Weights (Sample Estimation)')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f'A{i+1}' for i in range(n_assets)])
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Efficient frontier (in-sample vs out-of-sample)
# In-sample frontier (estimated parameters)
target_returns = np.linspace(mu_hat_sample.min(), mu_hat_sample.max(), 50)
frontier_vols_insample = []
for target in target_returns:
    w = mean_variance_weights(mu_hat_sample, Sigma_hat_sample, target)
    _, vol, _ = portfolio_performance(w, mu_hat_sample, Sigma_hat_sample)
    frontier_vols_insample.append(vol * np.sqrt(12))

# Out-of-sample frontier (true parameters; unknown)
frontier_vols_true = []
for target in target_returns:
    w = mean_variance_weights(mu_hat_sample, Sigma_hat_sample, target)  # Use estimated weights
    _, vol, _ = portfolio_performance(w, mu_true, Sigma_true)  # But evaluate with true params
    frontier_vols_true.append(vol * np.sqrt(12))

axes[1, 1].plot(frontier_vols_insample, target_returns * 12, 
               linewidth=2, label='In-Sample Frontier', color='blue', linestyle='--')
axes[1, 1].plot(frontier_vols_true, target_returns * 12, 
               linewidth=2, label='Out-of-Sample (Realized)', color='red')

# Plot naive portfolio
ret_naive, vol_naive, _ = portfolio_performance(w_naive_sample, mu_true, Sigma_true)
axes[1, 1].scatter(vol_naive * np.sqrt(12), ret_naive * 12, 
                  s=200, marker='*', color='orange', edgecolor='black', 
                  linewidth=2, label='1/N Naive', zorder=5)

axes[1, 1].set_xlabel('Volatility (annualized)')
axes[1, 1].set_ylabel('Return (annualized)')
axes[1, 1].set_title('Efficient Frontier: Estimation Error Impact')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

print("")
print("="*70)
print("Key Insights:")
print("="*70)
print("1. Mean-Variance optimization sensitive to estimation error")
print("   → Extreme weights from noisy μ̂ estimates")
print("")
print("2. Simple strategies (1/N, Min-Variance) often competitive")
print("   → Robustness to parameter uncertainty")
print("")
print("3. In-sample frontier ≠ Out-of-sample performance")
print("   → Overfitting to historical data")
```

## 6. Challenge Round
When optimization fails or misleads:
- **Short estimation window**: T=60, n=50 → 1,275 parameters, near-singular Σ̂ → extreme weights (w_i>2 or <-1); use shrinkage or min-variance
- **Non-stationarity**: 2007 Σ̂ useless in 2008 crisis (correlations 0.3→0.9); rolling windows or regime-switching models needed
- **Mean estimation dominance**: SE(μ̂_i)=2% vs μ_i=10% (20% relative error) → huge impact on weights; use equilibrium (Black-Litterman) or ignore means (min-variance)
- **Transaction costs**: Optimal weights change 50%; costs 0.5%; net gain -0.3% (loss after costs); constrain turnover or rebalance less frequently
- **Short constraints**: Unconstrained w_i=-0.5 (short 50%); but borrowing costs 2-3%; long-only constraint or adjust for costs
- **Leverage**: Tangency requires 150% leverage; but margin costs 3-5%; max Sharpe ≠ max utility after costs; constrain leverage or adjust r_f for cost

## 7. Key References
- [Markowitz: Portfolio Selection (1952)](https://www.jstor.org/stable/2975974) - Mean-variance optimization foundation
- [DeMiguel, Garlappi & Uppal: 1/N vs Optimization (2009)](https://academic.oup.com/rfs/article/22/5/1915/1592668) - Estimation error dominates; 1/N competitive
- [Black & Litterman: Global Portfolio Optimization (1992)](https://www.jstor.org/stable/4479577) - Bayesian framework with equilibrium + views

---
**Status:** Core portfolio management | **Complements:** Asset Return Properties, Risk Management Models, Factor Models, Performance Attribution
