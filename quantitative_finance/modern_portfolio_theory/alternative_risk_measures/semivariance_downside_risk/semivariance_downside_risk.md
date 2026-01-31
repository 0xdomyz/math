# Semi-Variance and Downside Risk

## 1. Concept Skeleton
**Definition:** Variance of returns below target/mean; measures downside volatility only; asymmetric risk measure  
**Purpose:** Focus on harmful volatility (losses), ignore upside variation; downside deviation calculation  
**Prerequisites:** Variance, expected value, probability distributions, risk aversion

## 2. Comparative Framing
| Risk Measure | Variance | Semi-Variance | Lower Partial Moment | VaR | CVaR | Max Drawdown |
|--------------|----------|---------------|---------------------|-----|------|--------------|
| **Focus** | Total variation | Below-target only | Below-target powers | Threshold | Tail average | Peak decline |
| **Symmetry** | Symmetric | Asymmetric | Asymmetric | Asymmetric | Asymmetric | Asymmetric |
| **Upside** | Penalizes | Ignores | Ignores | Ignores | Ignores | Ignores |
| **Target** | Mean | Mean/custom | Custom | Quantile | Beyond quantile | Historical path |
| **Coherent** | Yes | Yes | Depends on order | No | Yes | No |
| **Computation** | Simple | Simple | Simple | Moderate | Moderate | Path-dependent |

## 3. Examples + Counterexamples

**Key Distinction:**  
Portfolio A: Returns = [-5%, -3%, 0%, +3%, +5%, +10%, +15%]  
Portfolio B: Returns = [-5%, -3%, 0%, +3%, +5%, +6%, +7%]  
Both have similar means, but different variances due to upside  
Semi-variance focuses on [-5%, -3%, 0%] for both → more similar

**Why Semi-Variance Matters:**  
Investor preference: Upside volatility is good (large gains), downside is bad (losses)  
Variance treats ±10% equally; semi-variance only penalizes -10%

**Sortino Ratio Example:**  
Fund A: Return 12%, Std Dev 15%, Downside Dev 8% → Sortino = (12%-2%)/8% = 1.25  
Fund B: Return 10%, Std Dev 10%, Downside Dev 7% → Sortino = (10%-2%)/7% = 1.14  
Sharpe would pick B, Sortino picks A (better downside-adjusted return)

## 4. Layer Breakdown
```
Semi-Variance and Downside Risk Framework:
├─ Mathematical Definitions:
│   ├─ Semi-Variance (below mean):
│   │   ├─ SV = (1/n) Σ [min(r_i - μ, 0)]²
│   │   ├─ Or: SV = E[(min(R - μ, 0))²]
│   │   ├─ Downside Deviation: √SV
│   │   └─ Only negative deviations contribute
│   ├─ Target Semi-Variance (below target τ):
│   │   ├─ SV_τ = (1/n) Σ [min(r_i - τ, 0)]²
│   │   ├─ Common targets: τ = 0 (losses), τ = rf (underperformance)
│   │   └─ Minimum Acceptable Return (MAR)
│   ├─ Lower Partial Moment (LPM):
│   │   ├─ LPM_n(τ) = E[(max(τ - R, 0))^n]
│   │   ├─ LPM_0: Shortfall probability
│   │   ├─ LPM_1: Expected shortfall magnitude
│   │   ├─ LPM_2: Semi-variance (τ = mean)
│   │   └─ Higher orders: Penalize larger deviations more
│   └─ Relationship to Variance:
│       ├─ Var(R) = SV⁺ + SV⁻ (upside + downside)
│       ├─ For symmetric distributions: SV = Var/2
│       └─ For skewed: SV ≠ Var/2 (captures asymmetry)
├─ Sortino Ratio:
│   ├─ Formula: S = (R_p - MAR) / DD
│   ├─ R_p: Portfolio return
│   ├─ MAR: Minimum Acceptable Return (often risk-free rate)
│   ├─ DD: Downside Deviation = √(SV)
│   ├─ Interpretation: Return per unit of downside risk
│   ├─ vs Sharpe: Sortino only penalizes bad volatility
│   └─ Higher values better (more return per downside risk)
├─ Calculation Methods:
│   ├─ Historical Semi-Variance:
│   │   ├─ Target = mean: SV = mean([min(r - μ, 0)]²)
│   │   ├─ Target = 0%: SV = mean([min(r, 0)]²)
│   │   ├─ Denominator: Use n or n-1 (sample correction)
│   │   └─ Only observations below target contribute
│   ├─ Parametric (Normal):
│   │   ├─ For symmetric normal: SV = σ²/2
│   │   ├─ Not useful (ignores asymmetry advantage)
│   │   └─ Better for skewed distributions (t, GED)
│   └─ Forward-Looking:
│       ├─ Scenario analysis with probabilities
│       ├─ Monte Carlo with return model
│       └─ Options-implied downside risk
├─ Mean-Semivariance Optimization:
│   ├─ Objective: min SV_τ(w'R) subject to E[w'R] ≥ target
│   ├─ Equivalent to: min Σ_i p_i [min(r_i'w - τ, 0)]²
│   ├─ Quadratic programming (like mean-variance)
│   ├─ Advantages:
│   │   ├─ Focuses on downside risk only
│   │   ├─ Better for asymmetric returns
│   │   ├─ Aligns with investor preferences
│   │   └─ Convex optimization (tractable)
│   ├─ Challenges:
│   │   ├─ Requires longer history (fewer observations)
│   │   ├─ Estimation error in tail
│   │   └─ Choice of target τ subjective
│   └─ Mean-Semivariance Frontier:
│       ├─ Similar shape to mean-variance
│       ├─ Often more efficient for skewed assets
│       └─ Tangency: max (R - rf) / DD
├─ Downside Beta:
│   ├─ Definition: β⁻ = Cov(R_i, R_m | R_m < μ_m) / Var(R_m | R_m < μ_m)
│   ├─ Systematic risk in down markets only
│   ├─ Conditional on market decline
│   ├─ Captures "crash risk" exposure
│   ├─ vs Regular Beta: Can be higher (more downside sensitive)
│   └─ Applications: D-CAPM (Downside CAPM)
├─ Applications:
│   ├─ Performance Evaluation:
│   │   ├─ Sortino Ratio for fund ranking
│   │   ├─ Omega Ratio: Gains/Losses around threshold
│   │   └─ Upside/Downside Capture Ratios
│   ├─ Portfolio Construction:
│   │   ├─ Mean-semivariance optimization
│   │   ├─ Downside risk parity
│   │   └─ Minimum downside deviation portfolios
│   ├─ Asset Pricing:
│   │   ├─ Downside CAPM (Estrada, Bawa)
│   │   ├─ Only downside beta priced
│   │   └─ Better explanation for emerging markets
│   └─ Risk Budgeting:
│       ├─ Allocate based on downside contribution
│       ├─ Component downside deviation
│       └─ Marginal downside risk
├─ Advantages:
│   ├─ Investor-friendly: Matches risk perception (loss aversion)
│   ├─ Asymmetry: Captures skewness in returns
│   ├─ Convex: Optimization is tractable
│   ├─ Coherent: Semi-deviation satisfies axioms
│   └─ Interpretable: Downside volatility easy to explain
└─ Limitations:
    ├─ Data Requirements: Fewer observations (only downside)
    ├─ Estimation Error: Tail estimation less precise
    ├─ Target Choice: Subjective (0%, mean, rf, MAR?)
    ├─ Diversification: Less benefit measured if uncorrelated in tails
    └─ Normal Distribution: No advantage if returns symmetric
```

**Interaction:** Semi-variance isolates downside volatility, aligning risk measurement with investor loss aversion

## 5. Mini-Project
Implement semi-variance analysis and compare with traditional variance:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats

# Download data
tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'VNQ']
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

def semivariance(returns, target=None, below=True):
    """
    Calculate semi-variance (variance of returns below/above target)
    
    Parameters:
    - returns: Series or array of returns
    - target: Threshold (default: mean)
    - below: If True, variance below target; if False, above target
    """
    if target is None:
        target = returns.mean()
    
    if below:
        # Below target (downside)
        deviations = returns[returns < target] - target
    else:
        # Above target (upside)
        deviations = returns[returns > target] - target
    
    if len(deviations) > 0:
        sv = (deviations ** 2).mean()
    else:
        sv = 0
    
    return sv

def downside_deviation(returns, target=None):
    """
    Calculate downside deviation (square root of semi-variance)
    """
    sv = semivariance(returns, target, below=True)
    return np.sqrt(sv)

def sortino_ratio(returns, target=0, periods=252):
    """
    Calculate Sortino Ratio
    
    Parameters:
    - returns: Daily returns
    - target: Minimum Acceptable Return (MAR)
    - periods: Annualization factor
    """
    excess_returns = returns - target/periods
    mean_excess = excess_returns.mean() * periods
    dd = downside_deviation(returns, target/periods) * np.sqrt(periods)
    
    if dd > 0:
        sortino = mean_excess / dd
    else:
        sortino = np.nan
    
    return sortino

def lower_partial_moment(returns, target=0, order=2):
    """
    Calculate Lower Partial Moment of given order
    
    LPM_n(τ) = E[(max(τ - R, 0))^n]
    """
    shortfalls = np.maximum(target - returns, 0)
    lpm = (shortfalls ** order).mean()
    return lpm

def upside_downside_capture(returns, benchmark_returns, periods=252):
    """
    Calculate upside and downside capture ratios
    """
    # Identify up and down periods for benchmark
    up_periods = benchmark_returns > 0
    down_periods = benchmark_returns < 0
    
    # Average returns in those periods
    asset_up = returns[up_periods].mean() * periods
    asset_down = returns[down_periods].mean() * periods
    bench_up = benchmark_returns[up_periods].mean() * periods
    bench_down = benchmark_returns[down_periods].mean() * periods
    
    # Capture ratios
    upside_capture = (asset_up / bench_up) * 100 if bench_up != 0 else np.nan
    downside_capture = (asset_down / bench_down) * 100 if bench_down != 0 else np.nan
    
    return upside_capture, downside_capture

def downside_beta(returns, market_returns):
    """
    Calculate downside beta (beta conditional on market declines)
    """
    # Only periods when market is down
    market_mean = market_returns.mean()
    down_periods = market_returns < market_mean
    
    if down_periods.sum() > 10:  # Need sufficient observations
        returns_down = returns[down_periods]
        market_down = market_returns[down_periods]
        
        # Covariance and variance in down periods
        cov_down = np.cov(returns_down, market_down)[0, 1]
        var_down = market_down.var()
        
        beta_down = cov_down / var_down if var_down > 0 else np.nan
    else:
        beta_down = np.nan
    
    return beta_down

# Calculate metrics for all assets
metrics = {}

for ticker in tickers:
    ret = returns[ticker]
    
    # Traditional metrics
    annual_return = ret.mean() * 252
    volatility = ret.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / volatility if volatility > 0 else np.nan
    
    # Downside metrics
    sv_mean = semivariance(ret, target=ret.mean())
    sv_zero = semivariance(ret, target=0)
    dd_mean = downside_deviation(ret, target=ret.mean()) * np.sqrt(252)
    dd_zero = downside_deviation(ret, target=0) * np.sqrt(252)
    
    sortino = sortino_ratio(ret, target=0.02, periods=252)
    
    # LPMs
    lpm0 = lower_partial_moment(ret, target=0, order=0)  # Shortfall probability
    lpm1 = lower_partial_moment(ret, target=0, order=1) * 252  # Expected shortfall
    lpm2 = lower_partial_moment(ret, target=0, order=2) * 252  # Semi-variance
    
    # Upside/Downside capture vs SPY
    if ticker != 'SPY':
        up_cap, down_cap = upside_downside_capture(ret, returns['SPY'])
    else:
        up_cap, down_cap = 100, 100
    
    # Downside beta
    if ticker != 'SPY':
        dbeta = downside_beta(ret, returns['SPY'])
    else:
        dbeta = 1.0
    
    metrics[ticker] = {
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Downside Dev (mean)': dd_mean,
        'Downside Dev (0%)': dd_zero,
        'Sortino Ratio': sortino,
        'Shortfall Prob': lpm0,
        'Expected Shortfall': lpm1,
        'Upside Capture': up_cap,
        'Downside Capture': down_cap,
        'Downside Beta': dbeta
    }

metrics_df = pd.DataFrame(metrics).T

print("\n" + "="*110)
print("DOWNSIDE RISK METRICS COMPARISON")
print("="*110)
print(metrics_df.round(4))

# Mean-Semivariance Optimization
def portfolio_downside_deviation(weights, returns, target=0):
    """
    Calculate portfolio downside deviation
    """
    portfolio_returns = returns @ weights
    dd = downside_deviation(portfolio_returns, target=target)
    return dd

def optimize_mean_semivariance(returns, target_return=None, target_threshold=0):
    """
    Optimize portfolio to minimize downside deviation
    """
    n_assets = returns.shape[1]
    
    # Initial guess
    w0 = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
    ]
    
    if target_return is not None:
        mean_returns = returns.mean()
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ mean_returns - target_return
        })
    
    # Bounds (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        lambda w: portfolio_downside_deviation(w, returns, target_threshold),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

def optimize_mean_variance(returns, target_return=None):
    """
    Traditional mean-variance optimization for comparison
    """
    n_assets = returns.shape[1]
    cov_matrix = returns.cov()
    
    w0 = np.ones(n_assets) / n_assets
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if target_return is not None:
        mean_returns = returns.mean()
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ mean_returns - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(
        lambda w: np.sqrt(w @ cov_matrix @ w),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

# Build efficient frontiers
print("\nBuilding efficient frontiers...")
mean_returns = returns.mean() * 252

min_return = mean_returns.min()
max_return = mean_returns.max() * 0.85
target_returns = np.linspace(min_return, max_return, 12)

mv_portfolios = []
msv_portfolios = []

for target_ret in target_returns:
    # Mean-Variance
    mv_weights = optimize_mean_variance(returns, target_ret/252)
    mv_ret = mv_weights @ returns.mean() * 252
    mv_vol = np.sqrt(mv_weights @ returns.cov() @ mv_weights) * np.sqrt(252)
    mv_dd = portfolio_downside_deviation(mv_weights, returns, 0) * np.sqrt(252)
    mv_sortino = (mv_ret - 0.02) / mv_dd if mv_dd > 0 else np.nan
    
    mv_portfolios.append({
        'return': mv_ret,
        'volatility': mv_vol,
        'downside_dev': mv_dd,
        'sortino': mv_sortino,
        'weights': mv_weights
    })
    
    # Mean-Semivariance
    msv_weights = optimize_mean_semivariance(returns, target_ret/252, 0)
    msv_ret = msv_weights @ returns.mean() * 252
    msv_vol = np.sqrt(msv_weights @ returns.cov() @ msv_weights) * np.sqrt(252)
    msv_dd = portfolio_downside_deviation(msv_weights, returns, 0) * np.sqrt(252)
    msv_sortino = (msv_ret - 0.02) / msv_dd if msv_dd > 0 else np.nan
    
    msv_portfolios.append({
        'return': msv_ret,
        'volatility': msv_vol,
        'downside_dev': msv_dd,
        'sortino': msv_sortino,
        'weights': msv_weights
    })

mv_df = pd.DataFrame(mv_portfolios)
msv_df = pd.DataFrame(msv_portfolios)

# Compare specific portfolios
print("\n" + "="*110)
print("PORTFOLIO COMPARISON: Mean-Variance vs Mean-Semivariance (Target Return: 10%)")
print("="*110)

target_idx = np.argmin(np.abs(mv_df['return'] - 0.10))
mv_port = mv_df.iloc[target_idx]
msv_port = msv_df.iloc[target_idx]

comparison = pd.DataFrame({
    'Mean-Variance': [
        mv_port['return'],
        mv_port['volatility'],
        mv_port['downside_dev'],
        mv_port['sortino'],
        (mv_port['return'] - 0.02) / mv_port['volatility']
    ],
    'Mean-Semivariance': [
        msv_port['return'],
        msv_port['volatility'],
        msv_port['downside_dev'],
        msv_port['sortino'],
        (msv_port['return'] - 0.02) / msv_port['volatility']
    ]
}, index=['Return', 'Volatility', 'Downside Dev', 'Sortino', 'Sharpe'])

print(comparison.round(4))

print("\n" + "="*110)
print("WEIGHT ALLOCATION COMPARISON")
print("="*110)
weight_comp = pd.DataFrame({
    'Mean-Variance': mv_port['weights'],
    'Mean-Semivariance': msv_port['weights']
}, index=tickers)
print(weight_comp.round(4))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficient Frontier (Return vs Volatility)
asset_returns = returns.mean() * 252
asset_vols = returns.std() * np.sqrt(252)

axes[0, 0].scatter(asset_vols, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 0].annotate(ticker, (asset_vols[i], asset_returns[i]), fontsize=8, ha='right')

axes[0, 0].plot(mv_df['volatility'], mv_df['return'], 'b-o',
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 0].plot(msv_df['volatility'], msv_df['return'], 'r-s',
               label='Mean-Semivariance', alpha=0.7, markersize=4)

axes[0, 0].set_xlabel('Volatility (Annual)')
axes[0, 0].set_ylabel('Expected Return (Annual)')
axes[0, 0].set_title('Efficient Frontiers: Mean-Variance vs Mean-Semivariance')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Return vs Downside Deviation
asset_dds = [downside_deviation(returns[t], 0) * np.sqrt(252) for t in tickers]

axes[0, 1].scatter(asset_dds, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 1].annotate(ticker, (asset_dds[i], asset_returns[i]), fontsize=8, ha='right')

axes[0, 1].plot(mv_df['downside_dev'], mv_df['return'], 'b-o',
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 1].plot(msv_df['downside_dev'], msv_df['return'], 'r-s',
               label='Mean-Semivariance', alpha=0.7, markersize=4)

axes[0, 1].set_xlabel('Downside Deviation (Annual)')
axes[0, 1].set_ylabel('Expected Return (Annual)')
axes[0, 1].set_title('Mean-Downside Risk Frontier')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Sharpe vs Sortino Comparison
sharpes = metrics_df['Sharpe Ratio']
sortinos = metrics_df['Sortino Ratio']

axes[1, 0].scatter(sharpes, sortinos, s=100, alpha=0.7)
for ticker in tickers:
    axes[1, 0].annotate(ticker, 
                       (metrics_df.loc[ticker, 'Sharpe Ratio'],
                        metrics_df.loc[ticker, 'Sortino Ratio']),
                       fontsize=8, ha='right')

# Add diagonal line
max_val = max(sharpes.max(), sortinos.max())
axes[1, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal Line')

axes[1, 0].set_xlabel('Sharpe Ratio')
axes[1, 0].set_ylabel('Sortino Ratio')
axes[1, 0].set_title('Sharpe vs Sortino: Capturing Asymmetric Risk')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Upside vs Downside Capture
upside_caps = metrics_df['Upside Capture'].values
downside_caps = metrics_df['Downside Capture'].values

axes[1, 1].scatter(downside_caps, upside_caps, s=100, alpha=0.7)
for ticker in tickers:
    if ticker != 'SPY':  # SPY is (100, 100) - the benchmark
        axes[1, 1].annotate(ticker,
                           (metrics_df.loc[ticker, 'Downside Capture'],
                            metrics_df.loc[ticker, 'Upside Capture']),
                           fontsize=8, ha='right')

# Add reference lines
axes[1, 1].axhline(100, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].axvline(100, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].plot([0, 150], [0, 150], 'k--', alpha=0.3, label='Equal Capture')

# Ideal quadrant (upper left)
axes[1, 1].fill_between([0, 100], 100, 150, alpha=0.1, color='green', label='Ideal')

axes[1, 1].set_xlabel('Downside Capture (%)')
axes[1, 1].set_ylabel('Upside Capture (%)')
axes[1, 1].set_title('Upside vs Downside Capture Ratios (vs SPY)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim([50, 130])
axes[1, 1].set_ylim([50, 130])

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "="*110)
print("KEY INSIGHTS: SEMI-VARIANCE AND DOWNSIDE RISK")
print("="*110)
print("1. Semi-variance focuses on investor-relevant risk (losses, not gains)")
print("2. Sortino Ratio > Sharpe when returns positively skewed (upside volatility)")
print("3. Mean-semivariance optimization produces different allocations than mean-variance")
print("4. Downside capture < 100% ideal (participate less in market declines)")
print("5. Upside capture > 100% ideal (participate more in market advances)")
print("6. Downside beta measures crash risk specifically (systematic downside)")
print("7. Lower Partial Moments generalize downside risk (different orders)")
print("8. Semi-variance especially useful for asymmetric assets (hedge funds, options)")

# Asymmetry analysis
print("\n" + "="*110)
print("ASYMMETRY ANALYSIS: Variance Decomposition")
print("="*110)
print(f"{'Ticker':>8} {'Variance':>10} {'Downside':>11} {'Upside':>11} {'Skewness':>11} {'Down%':>8}")
print("-"*110)

for ticker in tickers:
    ret = returns[ticker]
    total_var = ret.var() * 252
    down_var = semivariance(ret, ret.mean(), below=True) * 252
    up_var = semivariance(ret, ret.mean(), below=False) * 252
    skew = ret.skew()
    down_pct = down_var / total_var * 100 if total_var > 0 else 0
    
    print(f"{ticker:>8} {total_var:>10.4f} {down_var:>11.4f} {up_var:>11.4f} {skew:>11.3f} {down_pct:>7.1f}%")

print("\nFor symmetric returns: Downside% ≈ 50%")
print("Downside% > 50% indicates negative skewness (fat left tail)")
print("Downside% < 50% indicates positive skewness (fat right tail)")
```

## 6. Challenge Round
When is semi-variance more useful than variance?
- Asymmetric returns: Hedge funds, options, emerging markets (skewed distributions)
- Loss aversion: Behavioral finance shows investors fear losses more than value gains
- Downside protection: Conservative investors (retirees) care about downside specifically
- Performance evaluation: Sortino better than Sharpe for strategies with upside skew
- Risk budgeting: Allocate based on harmful volatility only

Semi-variance limitations:
- Data requirements: Fewer observations contribute (only below-target returns)
- Estimation error: Tail estimation less stable, needs longer history
- Target choice: Subjective (mean? 0%? risk-free rate? custom MAR?)
- Symmetric distributions: No advantage over variance if returns truly normal
- Diversification: Benefits may appear smaller (upside correlation ignored)

How does semi-variance relate to other downside measures?
- VaR: Semi-variance is distributional, VaR is single quantile
- CVaR: Both focus on downside, CVaR is tail-specific (beyond threshold)
- LPM: Semi-variance is LPM of order 2; LPM generalizes to other powers
- Drawdown: Semi-variance is single-period, drawdown is path-dependent
- Omega ratio: Ratio of upside to downside moments (gain/loss ratio)

## 7. Key References
- [Markowitz, H. (1959) "Portfolio Selection: Efficient Diversification of Investments"](https://www.jstor.org/stable/2975974) - Original discussion of semi-variance
- [Sortino, F. & Price, L. (1994) "Performance Measurement in a Downside Risk Framework"](https://www.jstor.org/stable/4479185)
- [Estrada, J. (2002) "Systematic Risk in Emerging Markets: The D-CAPM"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=277531)
- [Investopedia - Sortino Ratio](https://www.investopedia.com/terms/s/sortinoratio.asp)

---
**Status:** Asymmetric risk measure for loss-averse investors | **Complements:** Sortino Ratio, Downside Beta, LPM
