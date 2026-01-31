# Estimation Error and Input Sensitivity

## 1. Concept Skeleton
**Definition:** Small changes in expected returns/covariance estimates cause large portfolio weight changes; mean-variance optimization amplifies estimation noise  
**Purpose:** Understand practical limitations of MPT; quantify impact of parameter uncertainty on portfolio construction  
**Prerequisites:** Mean-variance optimization, covariance matrices, statistical estimation, portfolio weights

## 2. Comparative Framing
| Aspect | Mean-Variance Optimization | Robust Methods | Data-Driven Approaches |
|--------|---------------------------|----------------|----------------------|
| **Input Sensitivity** | Very high (unstable weights) | Moderate (constraints reduce) | Low (shrinkage/regularization) |
| **Estimation Error** | Large (directly in objective) | Built-in robustness | Penalizes extreme weights |
| **Practical Use** | Limited without safeguards | Better for constraints | Often preferred in practice |
| **Rebalancing** | Frequent (if inputs change) | Less frequent | Stable allocations |
| **Turnover** | High (tracking error costs) | Medium | Lower |

## 3. Examples + Counterexamples

**Estimation Error Problem:**  
Asset A: E[R] = 10% ± 2% (true range), sample estimate = 12%  
Change weight from 5% to 40% based on estimation error  
Realization: Actual return = 8% → underperformance vs true optimum

**Covariance Instability:**  
100 assets, 5 years of data, need ~5,000 covariance estimates  
Only 250 returns observations available  
Correlation estimates very noisy, optimization becomes unstable

**Extreme Weights:**  
Optimal portfolio: Long $100M in small-cap, Short $80M  
Implausible leverage; transaction costs prohibitive; execution risk high

**Reality Check:**  
Black-Litterman model vs straight mean-variance  
BL dampens extreme positions using market equilibrium + views  
Produces more reasonable, tradable portfolios

## 4. Layer Breakdown
```
Estimation Error Framework:
├─ Sources of Estimation Error:
│   ├─ Expected Returns (μ):
│   │   ├─ Historical mean: √(n) sampling error, non-stationary
│   │   ├─ Bias: Mean reversion, survivorship, time-period dependent
│   │   ├─ Parameter uncertainty: Longer history needed than volatility
│   │   ├─ Forecast error: Future ≠ past (structural breaks, regimes)
│   │   └─ Estimation quality: Most important, least precise
│   ├─ Covariance Matrix (Σ):
│   │   ├─ p×p matrix for p assets (e.g., 100×100 for 100 assets = 5,050 elements)
│   │   ├─ Symmetric, positive definite (mathematical constraints)
│   │   ├─ Sample covariance: Σ_sample = (1/n) Σ(r_t - r̄)(r_t - r̄)'
│   │   ├─ Bias when p large relative to n (denominator too small)
│   │   ├─ Singular or near-singular when p > n (non-invertible)
│   │   └─ Estimation error increases with dimensionality
│   ├─ Correlation Coefficients:
│   │   ├─ ρ_ij sample correlation from n observations
│   │   ├─ Variance: ~(1-ρ²)²/n (narrow for ρ near ±1)
│   │   ├─ Fat tails: Realized correlations spike in crises
│   │   └─ Structural changes: Pre-crisis ≠ crisis ≠ recovery correlations
│   └─ Risk-Free Rate:
│       ├─ Usually less uncertain (T-bill/swap rates observable)
│       ├─ But economically more uncertain (time-varying term premium)
│       └─ Relatively minor source vs return/covariance error
├─ Impact on Portfolio Weights:
│   ├─ Sensitivity Formula:
│   │   ├─ w ≈ Σ⁻¹(μ - rf·1) / normalization
│   │   ├─ Small change in μ: Δμ → Δw = (Σ⁻¹ · Δμ) / normalization
│   │   ├─ Amplification factor: Depends on condition number of Σ⁻¹
│   │   ├─ Low eigenvalues: High amplification (near-singular)
│   │   └─ Estimation error directly magnified by (Σ⁻¹)
│   ├─ Empirical Studies:
│   │   ├─ Chopra-Ziemba (1993): 1% error in returns → 15-30% weight change
│   │   ├─ Kozers-Murrin: 20% error in expected returns common
│   │   ├─ Weight changes → turnover, transaction costs
│   │   └─ Practical alpha wiped out by rebalancing costs
│   └─ Extreme Positions:
│       ├─ Tiny differences in expected returns
│       ├─ Can drive allocations to 100%+ one asset
│       ├─ Short positions for diversification (leveraged)
│       ├─ Violates practical constraints
│       └─ Signals: If weights extreme, likely estimation error
├─ Manifestations of Estimation Error:
│   ├─ Weight Instability:
│   │   ├─ Portfolio A (Jan data): w_1 = 30%, w_2 = 20%
│   │   ├─ Portfolio B (Feb data): w_1 = 5%, w_2 = 55%
│   │   ├─ Change driven by one month data, not fundamental change
│   │   ├─ Rebalancing costs erase value
│   │   └─ Optimal weights vs feasible allocations
│   ├─ Out-of-Sample Performance Degradation:
│   │   ├─ Historical optimization: High Sharpe on training data
│   │   ├─ Forward testing: Lower returns in-sample
│   │   ├─ Difference: Parameter uncertainty + overfitting
│   │   ├─ In-sample R² high, out-of-sample low
│   │   └─ Optimal rule: Simpler models generalize better
│   ├─ Correlation Breakdowns:
│   │   ├─ Diversification assumes stable correlations
│   │   ├─ Crisis: Most assets correlate to 1 (diversification fails)
│   │   ├─ Historical correlation ≠ crisis correlation
│   │   ├─ Model based on normal times breaks in extremes
│   │   └─ Stressed correlations exceed normal estimates
│   └─ Missing Data:
│       ├─ Incomplete history (new IPOs)
│       ├─ Survivorship bias (delisted firms excluded)
│       ├─ Penny stocks: No trade, stale data
│       └─ Estimation: Biased or impossible
├─ Solutions & Mitigation Strategies:
│   ├─ Longer Historical Data:
│   │   ├─ More observations → lower sampling error
│   │   ├─ Trade-off: More history, but parameter non-stationarity
│   │   ├─ Optimal: 5-10 years (balances precision and regime shifts)
│   │   └─ Too long: Structural breaks render data obsolete
│   ├─ Shrinkage Estimators:
│   │   ├─ Combine sample estimates with prior (target)
│   │   ├─ Formula: Σ_shrink = λ·Σ_target + (1-λ)·Σ_sample
│   │   ├─ Ledoit-Wolf shrinkage: Optimal λ to minimize MSE
│   │   ├─ Reduces condition number → more stable weights
│   │   └─ Improves out-of-sample performance substantially
│   ├─ Constraints on Weights:
│   │   ├─ No short selling: w_i ≥ 0
│   │   ├─ Concentration limits: w_i ≤ 0.05 (5% max per asset)
│   │   ├─ Sector limits: Prevent all tech, all financial, etc.
│   │   ├─ Leverage limits: Σ |w_i| ≤ 2 (can't be >200% invested)
│   │   └─ Constraints automatically stabilize weights
│   ├─ Factor Models (Reduce Dimensionality):
│   │   ├─ Instead of 100 assets: Use 5 factors
│   │   ├─ Estimate fewer parameters (10 factor loadings vs 5,050 covariances)
│   │   ├─ Factor covariance matrix much smaller, more stable
│   │   ├─ Reduces estimation error dramatically
│   │   └─ Trade-off: Model risk (wrong factors) vs estimation risk
│   ├─ Equal-Weight Baseline:
│   │   ├─ w_i = 1/n for all i (naive diversification)
│   │   ├─ No estimation error (no parameters to estimate)
│   │   ├─ Out-of-sample: Often beats mean-variance on Sharpe!
│   │   ├─ Empirical puzzle: Simple > complex
│   │   └─ Suggests optimization error exceeds gain from optimization
│   ├─ Bayes Methods & Prior Elicitation:
│   │   ├─ Incorporate expert views, market prices
│   │   ├─ Black-Litterman: Start with market portfolio (prior)
│   │   ├─ Adjust for views, produce stable allocations
│   │   └─ Combines information efficiently
│   └─ Resampling (Michaud):
│       ├─ Bootstrap historical returns 1000+ times
│       ├─ Optimize on each sample
│       ├─ Average weights across resamples
│       ├─ Produces "expected" portfolio (considers estimation uncertainty)
│       └─ Improves out-of-sample Sharpe vs traditional optimization
├─ Practical Implementation Guidelines:
│   ├─ Data Preparation:
│   │   ├─ Use minimum 3 years (monthly), 5 years better (balances noise/stationarity)
│   │   ├─ Adjust for stock splits, mergers, special dividends
│   │   ├─ Handle missing data: Forward-fill or drop
│   │   ├─ Check for outliers: Delisting days, trading halts
│   │   └─ Align rebalancing dates (avoid stale price bias)
│   ├─ Input Validation:
│   │   ├─ Expected returns: Sanity check vs industry forecasts
│   │   ├─ Correlations: Should be [-1, 1], positive definite
│   │   ├─ Volatilities: Reasonable (stocks 15-30%, bonds 5-10%)
│   │   ├─ Covariance matrix: Eigenvalue check (no negative, max/min ratio)
│   │   └─ If inputs fail validation: Use regularized estimates
│   ├─ Portfolio Monitoring:
│   │   ├─ Track weight drift (actual vs target)
│   │   ├─ Monitor input changes (re-estimate return expectations)
│   │   ├─ Compare realized vs predicted performance
│   │   ├─ Rebalance when drift exceeds threshold (5-10%)
│   │   └─ Document parameter changes (audit trail)
│   └─ Stress Testing:
│       ├─ Sensitivity analysis: Vary μ, Σ by ±10-20%
│       ├─ Check weight stability to input changes
│       ├─ Scenario analysis: Historical crises (2008, 2020)
│       ├─ Portfolio performance: Would allocation protect in stress?
│       └─ If portfolio breaks under stress: Use more constraints
└─ When to Worry Most:
    ├─ Many assets, short history (high estimation error)
    ├─ Expected returns very different across assets
    ├─ Correlations near 1 (numerical issues, singular matrix)
    ├─ Constraints tight (binding constraints dominate optimization)
    ├─ Frequent rebalancing (transaction costs exceed benefit)
    └─ Use simpler models in uncertain environments
```

**Interaction:** Estimation error in μ and Σ directly amplified in optimal weights; constraints naturally regularize

## 5. Mini-Project
Analyze estimation error impact on portfolio optimization:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# Download asset data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'XOM', 'PG', 'JNJ', 'MA', 'V']
end_date = datetime.now()
start_date = datetime(2019, 1, 1)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

# Split into estimation and validation periods
split_date = len(returns) // 2
estimation_returns = returns.iloc[:split_date]
validation_returns = returns.iloc[split_date:]

print(f"\nData periods:")
print(f"  Estimation: {estimation_returns.index[0].date()} to {estimation_returns.index[-1].date()}")
print(f"  Validation: {validation_returns.index[0].date()} to {validation_returns.index[-1].date()}")

# Calculate historical statistics
mu_est = estimation_returns.mean() * 252  # Annualized
sigma_est = estimation_returns.cov() * 252
rf = 0.02  # Risk-free rate

def portfolio_stats(weights, mu, sigma, rf=0):
    """Calculate portfolio return, volatility, Sharpe"""
    p_return = weights @ mu
    p_vol = np.sqrt(weights @ sigma @ weights)
    sharpe = (p_return - rf) / p_vol if p_vol > 0 else 0
    return p_return, p_vol, sharpe

def optimize_portfolio(mu, sigma, rf=0, constraint_type='unconstrained'):
    """
    Optimize portfolio weights
    
    constraint_type: 'unconstrained', 'no_short', 'concentration'
    """
    n = len(mu)
    
    if constraint_type == 'unconstrained':
        # Unconstrained (allow short selling)
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            # Singular matrix: use Moore-Penrose pseudoinverse
            sigma_inv = np.linalg.pinv(sigma)
        
        w = sigma_inv @ (mu - rf)
        w = w / w.sum()  # Normalize
        
    elif constraint_type == 'no_short':
        # No short selling constraint
        def objective(w):
            ret, vol, _ = portfolio_stats(w, mu, sigma, rf)
            return -ret / vol if vol > 0 else 0  # Negative for minimization
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = tuple((0, 1) for _ in range(n))
        
        result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        w = result.x if result.success else np.ones(n) / n
        
    elif constraint_type == 'concentration':
        # Concentration limit: max 10% per asset
        def objective(w):
            ret, vol, _ = portfolio_stats(w, mu, sigma, rf)
            return -ret / vol if vol > 0 else 0
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = tuple((0, 0.1) for _ in range(n))  # Max 10% per asset
        
        result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        w = result.x if result.success else np.ones(n) / n
    
    return w / w.sum()  # Ensure normalized

# Optimization scenarios
print("\n" + "="*100)
print("OPTIMIZATION RESULTS (Estimation Period Data)")
print("="*100)

w_unconstrained = optimize_portfolio(mu_est, sigma_est, rf, 'unconstrained')
w_no_short = optimize_portfolio(mu_est, sigma_est, rf, 'no_short')
w_concentration = optimize_portfolio(mu_est, sigma_est, rf, 'concentration')
w_equal = np.ones(len(tickers)) / len(tickers)

weights_df = pd.DataFrame({
    'Equal-Weight': w_equal,
    'Unconstrained': w_unconstrained,
    'No Short': w_no_short,
    'Concentration': w_concentration
}, index=tickers)

print("\nPortfolio Weights:")
print(weights_df.round(4))

# In-sample performance (estimation period)
print("\n" + "="*100)
print("IN-SAMPLE PERFORMANCE (Estimation Period)")
print("="*100)

portfolios = {
    'Equal-Weight': w_equal,
    'Unconstrained': w_unconstrained,
    'No Short': w_no_short,
    'Concentration': w_concentration
}

insample_stats = {}
for name, w in portfolios.items():
    ret, vol, sharpe = portfolio_stats(w, mu_est, sigma_est, rf)
    insample_stats[name] = {'Return': ret, 'Volatility': vol, 'Sharpe': sharpe}

insample_df = pd.DataFrame(insample_stats).T
print(insample_df.round(4))

# Out-of-sample performance (validation period)
mu_val = validation_returns.mean() * 252
sigma_val = validation_returns.cov() * 252

print("\n" + "="*100)
print("OUT-OF-SAMPLE PERFORMANCE (Validation Period)")
print("="*100)

oosample_stats = {}
for name, w in portfolios.items():
    ret, vol, sharpe = portfolio_stats(w, mu_val, sigma_val, rf)
    oosample_stats[name] = {'Return': ret, 'Volatility': vol, 'Sharpe': sharpe}

oosample_df = pd.DataFrame(oosample_stats).T
print(oosample_df.round(4))

# Performance degradation
print("\n" + "="*100)
print("IN-SAMPLE vs OUT-OF-SAMPLE DEGRADATION")
print("="*100)

degradation = pd.DataFrame({
    'In-Sample Sharpe': insample_df['Sharpe'],
    'Out-of-Sample Sharpe': oosample_df['Sharpe'],
    'Degradation': insample_df['Sharpe'] - oosample_df['Sharpe'],
    'Degradation %': ((insample_df['Sharpe'] - oosample_df['Sharpe']) / insample_df['Sharpe'] * 100)
})

print(degradation.round(4))
print("\nNote: Large degradation indicates overfitting / estimation error")

# Weight stability analysis (sensitivity to estimation period)
print("\n" + "="*100)
print("WEIGHT STABILITY ANALYSIS")
print("="*100)

# Reestimate with slightly different periods
windows = [20, 40, 60, 80]  # Different training window sizes
sensitivity_results = {ticker: [] for ticker in tickers}

for window in windows:
    ret_window = estimation_returns.iloc[-window:]  # Most recent window months
    mu_window = ret_window.mean() * 252
    sigma_window = ret_window.cov() * 252
    
    w_window = optimize_portfolio(mu_window, sigma_window, rf, 'no_short')
    
    for i, ticker in enumerate(tickers):
        sensitivity_results[ticker].append(w_window[i])

sensitivity_df = pd.DataFrame(sensitivity_results, index=[f'{w}-month' for w in windows])

print("\nPortfolio Weight Sensitivity (No-Short Portfolio, Different Windows):")
print(sensitivity_df.round(4))

print("\nWeight Range by Asset (Min - Max):")
for ticker in tickers:
    w_range = sensitivity_df[ticker].max() - sensitivity_df[ticker].min()
    print(f"  {ticker}: {w_range:.4f} ({sensitivity_df[ticker].min():.4f} to {sensitivity_df[ticker].max():.4f})")

# Estimation error empirical study
print("\n" + "="*100)
print("ESTIMATION ERROR SIMULATION")
print("="*100)

# Bootstrap: Resample returns, reoptimize
n_simulations = 100
bootstrap_weights = {name: np.zeros((n_simulations, len(tickers))) 
                     for name in portfolios.keys()}
bootstrap_returns = []

np.random.seed(42)
for i in range(n_simulations):
    # Resample returns with replacement
    indices = np.random.choice(len(estimation_returns), len(estimation_returns), replace=True)
    ret_boot = estimation_returns.iloc[indices]
    
    mu_boot = ret_boot.mean() * 252
    sigma_boot = ret_boot.cov() * 252
    
    # Reoptimize
    for name, w in portfolios.items():
        w_boot = optimize_portfolio(mu_boot, sigma_boot, rf, 'no_short' if 'no' in name or 'Conc' in name else 'unconstrained')
        bootstrap_weights[name][i] = w_boot

# Analyze weight stability across bootstrap samples
print("\nWeight Standard Deviation Across Bootstrap Samples:")
weight_stability = {}
for name in portfolios.keys():
    weight_std = bootstrap_weights[name].std(axis=0)
    weight_stability[name] = weight_std.mean()  # Average across assets
    print(f"  {name:>20}: {weight_std.mean():.4f}")

print("\nInterpretation: Higher std dev = more unstable weights = estimation error problem")

# Correlation matrix analysis (condition number)
print("\n" + "="*100)
print("COVARIANCE MATRIX CONDITIONING")
print("="*100)

eigenvalues = np.linalg.eigvals(sigma_est)
eigenvalues_sorted = np.sort(eigenvalues)[::-1]

condition_number = eigenvalues_sorted[0] / eigenvalues_sorted[-1]

print(f"\nEigenvalue Range: {eigenvalues_sorted[-1]:.4f} to {eigenvalues_sorted[0]:.4f}")
print(f"Condition Number: {condition_number:.4f}")
print(f"Ratio Max/Min: {condition_number:.2f}x")

if condition_number > 100:
    print(f"\n⚠️  HIGH CONDITION NUMBER: Covariance matrix ill-conditioned")
    print(f"    Weights are very sensitive to estimation error")
    print(f"    Recommendation: Use shrinkage, constraints, or factor models")
else:
    print(f"\n✓ Moderate condition number: Matrix well-conditioned")

# Shrinkage Estimator (Ledoit-Wolf)
def ledoit_wolf_shrinkage(returns_data):
    """
    Ledoit-Wolf shrinkage estimator
    Combines sample covariance with shrinkage target (scaled identity matrix)
    """
    X = returns_data.values
    n, p = X.shape
    
    # Sample covariance
    S = np.cov(X.T)
    
    # Shrinkage target: scaled identity
    F = np.eye(p) * np.trace(S) / p
    
    # Optimal shrinkage intensity
    X_centered = X - X.mean(axis=0)
    
    # Empirical shrinkage coefficient (Ledoit-Wolf formula)
    d2 = np.sum((X_centered ** 2) @ (S.T ** 2))
    b2 = d2 / (n * np.trace(S @ S))
    b_hat = np.min([b2, 1.0])
    
    # Shrunk covariance
    S_shrink = b_hat * F + (1 - b_hat) * S
    
    return S_shrink, b_hat

sigma_shrink, shrink_coeff = ledoit_wolf_shrinkage(estimation_returns)

print(f"\n" + "="*100)
print("SHRINKAGE ESTIMATOR (Ledoit-Wolf)")
print("="*100)
print(f"Shrinkage Coefficient: {shrink_coeff:.4f}")
print(f"Interpretation: {shrink_coeff*100:.1f}% toward identity matrix, {(1-shrink_coeff)*100:.1f}% sample covariance")

# Optimize with shrunk covariance
w_shrink = optimize_portfolio(mu_est, sigma_shrink, rf, 'no_short')

print(f"\nPortfolio Weights (Shrunk Covariance):")
w_comparison = pd.DataFrame({
    'Sample Cov': optimize_portfolio(mu_est, sigma_est, rf, 'no_short'),
    'Shrunk Cov': w_shrink
}, index=tickers)
print(w_comparison.round(4))

# Out-of-sample Sharpe comparison
ret_shrink, vol_shrink, sharpe_shrink = portfolio_stats(w_shrink, mu_val, sigma_val, rf)

print(f"\nOut-of-Sample Performance (Validation Period):")
print(f"  Sample Covariance: Sharpe = {oosample_df.loc['No Short', 'Sharpe']:.4f}")
print(f"  Shrunk Covariance: Sharpe = {sharpe_shrink:.4f}")
print(f"  Improvement: {(sharpe_shrink - oosample_df.loc['No Short', 'Sharpe']):.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: In-sample vs Out-of-sample Sharpe
strategies = list(insample_stats.keys())
insample_sharpes = [insample_stats[s]['Sharpe'] for s in strategies]
oosample_sharpes = [oosample_stats[s]['Sharpe'] for s in strategies]

x = np.arange(len(strategies))
width = 0.35

bars1 = axes[0, 0].bar(x - width/2, insample_sharpes, width, label='In-Sample', alpha=0.8)
bars2 = axes[0, 0].bar(x + width/2, oosample_sharpes, width, label='Out-of-Sample', alpha=0.8)

axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')
axes[0, 0].set_ylabel('Sharpe Ratio')
axes[0, 0].set_title('Estimation Error: In-Sample vs Out-of-Sample Performance')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Eigenvalue spectrum
axes[0, 1].bar(range(len(eigenvalues_sorted)), eigenvalues_sorted, alpha=0.7)
axes[0, 1].set_xlabel('Eigenvalue Index')
axes[0, 1].set_ylabel('Eigenvalue')
axes[0, 1].set_title('Eigenvalue Spectrum (Covariance Matrix Conditioning)')
axes[0, 1].grid(alpha=0.3)

# Add horizontal line for ratio info
axes[0, 1].text(len(eigenvalues_sorted)-2, max(eigenvalues_sorted)*0.7,
               f'Max/Min = {condition_number:.1f}x', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Plot 3: Weight stability (bootstrap)
stability_values = [weight_stability[name] for name in portfolios.keys()]
bars = axes[1, 0].bar(portfolios.keys(), stability_values, alpha=0.7, color='orange')
axes[1, 0].set_ylabel('Avg Weight Std Dev (Bootstrap)')
axes[1, 0].set_title('Weight Stability Across Bootstrap Samples')
axes[1, 0].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, stability_values):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Unconstrained weights distribution (showing concentration)
unconstrained_positive = w_unconstrained[w_unconstrained > 0].sum()
unconstrained_negative = w_unconstrained[w_unconstrained < 0].sum()

positions = np.arange(len(tickers))
colors = ['green' if w > 0 else 'red' for w in w_unconstrained]

bars = axes[1, 1].bar(positions, w_unconstrained, color=colors, alpha=0.7)
axes[1, 1].set_xticks(positions)
axes[1, 1].set_xticklabels(tickers, rotation=45, ha='right')
axes[1, 1].set_ylabel('Portfolio Weight')
axes[1, 1].set_title('Unconstrained Optimization: Extreme Positions')
axes[1, 1].axhline(0, color='black', linewidth=0.5)
axes[1, 1].grid(axis='y', alpha=0.3)

# Add statistics
total_long = w_unconstrained[w_unconstrained > 0].sum()
total_short = abs(w_unconstrained[w_unconstrained < 0].sum())
gross_exposure = total_long + total_short

axes[1, 1].text(len(tickers)-2, max(w_unconstrained)*0.7,
               f'Long: {total_long:.2f}\nShort: {total_short:.2f}\nGross: {gross_exposure:.2f}',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS: ESTIMATION ERROR AND INPUT SENSITIVITY")
print("="*100)
print("1. Unconstrained optimization highly sensitive to input estimates")
print("2. Out-of-sample performance substantially worse than in-sample (overfitting)")
print("3. Equal-weight often competitive with optimized portfolio (beats average)")
print("4. Constraints (no short, concentration limits) naturally stabilize allocations")
print("5. High condition number → ill-conditioned matrix → extreme weight swings")
print("6. Shrinkage estimators improve out-of-sample performance significantly")
print("7. Weight instability indicates estimation error problem, not portfolio quality")
print("8. Simpler models often generalize better than optimization-based approaches")
print("9. Use multiple estimation windows to assess weight stability")
print("10. Always implement constraints matching practical implementation")
```

## 6. Challenge Round
Why is estimation error such a big problem for mean-variance optimization?
- Direct amplification: Portfolio weights w ∝ Σ⁻¹(μ - rf); small μ error amplified by Σ⁻¹
- Ill-conditioning: When correlations high, eigenvalues spread → large condition number → magnification
- Parameter uncertainty: Returns hardest to estimate; covariance easier; risk-free easiest
- Curse of dimensionality: p assets need p(p+1)/2 covariance estimates; limited data
- Non-stationarity: Market regime changes; past parameters don't predict future

Why do simple portfolios (equal-weight) often beat optimization?
- Estimation error dominates: Optimization error > diversification benefit
- Overfitting: Mean-variance fits past data perfectly, not predictive
- No parameter estimation: Equal-weight = zero estimation risk
- Robustness: Simple allocation works in multiple regimes
- Empirical puzzle: Suggests practitioners overestimate their forecasting ability

Solutions and trade-offs:
- Constraints: No short, concentration limits, leverage caps naturally regularize
- Shrinkage: Combine sample estimates with prior (target), reduce noise
- Factor models: Reduce dimensionality; estimate fewer parameters, more stable
- Black-Litterman: Use market equilibrium prior + views; produces stable allocations
- Resampling: Bootstrap historical data; average weights over samples
- Regular rebalancing: Monitor inputs, reoptimize only when drift significant

## 7. Key References
- [Chopra, V.K. & Ziemba, W.T. (1993) "The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice"](https://www.jstor.org/stable/2328922)
- [Ledoit, O. & Wolf, M. (2004) "Honey, I Shrunk the Sample Covariance Matrix"](https://www.jstor.org/stable/3598851)
- [Michaud, R.O. (1998) "Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization"](https://www.wiley.com/en-us/Efficient+Asset+Management%3A+A+Practical+Guide+to+Stock+Portfolio+Optimization%2C+Revised+Edition-p-9781883823597)
- [Investopedia - Estimation Risk](https://www.investopedia.com/terms/e/estimation-risk.asp)

---
**Status:** Critical practical limitation of MPT | **Complements:** Black-Litterman, Shrinkage Estimators, Robust Optimization
