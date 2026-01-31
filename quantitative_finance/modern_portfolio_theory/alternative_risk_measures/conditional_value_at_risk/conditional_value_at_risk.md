# Conditional Value at Risk (CVaR / Expected Shortfall)

## 1. Concept Skeleton
**Definition:** Average loss beyond VaR threshold; expected loss conditional on exceeding VaR; tail risk measure  
**Purpose:** Capture severity of tail losses, coherent risk measure, optimization objective  
**Prerequisites:** VaR, quantiles, conditional expectation, probability distributions

## 2. Comparative Framing
| Risk Measure | VaR | CVaR (ES) | Standard Deviation | Max Drawdown | Semi-Variance |
|--------------|-----|-----------|-------------------|--------------|---------------|
| **Focus** | Threshold | Tail average | Total volatility | Worst decline | Downside only |
| **Tail Info** | None beyond | Full tail | None | Single path | Below mean |
| **Coherent** | No (fails sub-additivity) | Yes (all properties) | Yes | No | Yes |
| **Optimization** | Non-convex (difficult) | Convex (tractable) | Convex | Non-convex | Convex |
| **Interpretation** | "Max loss at α" | "Avg loss when bad" | "Typical deviation" | "Worst experienced" | "Bad outcomes only" |

## 3. Examples + Counterexamples

**Simple Example:**  
95% VaR = $2M, 95% CVaR = $3.5M  
Interpretation: If loss exceeds $2M (5% of days), average loss on those days is $3.5M

**Why CVaR > VaR:**  
VaR = 5th percentile, CVaR = average of worst 5%  
CVaR always ≥ VaR (equality only if all tail losses identical)

**Coherence Advantage:**  
Portfolio A: VaR = $10M, Portfolio B: VaR = $10M  
Combined VaR could be $22M (diversification penalty!)  
But CVaR(A+B) ≤ CVaR(A) + CVaR(B) always (sub-additive)

## 4. Layer Breakdown
```
Conditional Value at Risk Framework:
├─ Mathematical Definition:
│   ├─ CVaRα = E[L | L ≥ VaRα]
│   ├─ Expected Shortfall (ES): Same concept, different name
│   ├─ Tail Conditional Expectation (TCE): Continuous version
│   └─ For continuous distributions: CVaRα = (1/(1-α)) ∫[α,1] VaRu du
├─ Intuitive Interpretation:
│   ├─ VaR asks: "How bad can it get?"
│   ├─ CVaR asks: "When it gets bad, how bad on average?"
│   ├─ 95% CVaR: Average of worst 5% of outcomes
│   └─ Always ≥ VaR (captures tail severity)
├─ Coherence Properties (Artzner et al. 1999):
│   ├─ 1. Monotonicity: If X ≤ Y, then ρ(X) ≤ ρ(Y)
│   ├─ 2. Sub-additivity: ρ(X+Y) ≤ ρ(X) + ρ(Y) ← VaR fails this!
│   ├─ 3. Positive homogeneity: ρ(λX) = λρ(X) for λ > 0
│   ├─ 4. Translation invariance: ρ(X+c) = ρ(X) - c
│   └─ CVaR satisfies all 4; encourages diversification
├─ Calculation Methods:
│   ├─ Historical CVaR:
│   │   ├─ Sort returns, find VaR threshold
│   │   ├─ Average all returns beyond VaR
│   │   ├─ Simple but limited by history
│   │   └─ CVaR = mean(returns[returns ≤ -VaR])
│   ├─ Parametric CVaR (Normal):
│   │   ├─ CVaR95% = μ - σ × φ(z0.95) / (1-0.95)
│   │   ├─ φ(·) = standard normal PDF
│   │   ├─ z0.95 = 1.645 (normal quantile)
│   │   └─ CVaR ≈ VaR + σ × λ (λ ≈ 0.4 for 95%)
│   ├─ Parametric CVaR (t-distribution):
│   │   ├─ Better for fat tails
│   │   ├─ CVaR = μ + [(df + z²)/(df-1)] × σ × ft(z) / α
│   │   └─ ft(·) = t-distribution PDF
│   └─ Monte Carlo CVaR:
│       ├─ Generate N scenarios
│       ├─ Find VaR from simulations
│       ├─ Average worst (1-α)×N scenarios
│       └─ Flexible for complex portfolios
├─ CVaR Optimization:
│   ├─ Rockafellar-Uryasev (2000) Formulation:
│   │   ├─ min CVaRα(w'X) subject to constraints
│   │   ├─ Equivalent to: min t + (1/(1-α))E[(L-t)⁺]
│   │   ├─ Convex optimization problem (LP for discrete)
│   │   └─ Can use standard solvers (CVXPY, Gurobi)
│   ├─ Advantages vs Mean-Variance:
│   │   ├─ Better tail control
│   │   ├─ No normality assumption
│   │   ├─ Convex (globally optimal solutions)
│   │   └─ Captures downside risk specifically
│   └─ Mean-CVaR Frontier:
│       ├─ Similar to efficient frontier
│       ├─ Trade-off: E[R] vs CVaR
│       └─ Tangency portfolio: max (E[R] - rf) / CVaR
├─ Portfolio Applications:
│   ├─ Risk Budgeting: Allocate CVaR across assets
│   ├─ Component CVaR: Each asset's contribution to portfolio CVaR
│   ├─ Marginal CVaR: Change in CVaR from position change
│   └─ CVaR Parity: Equal CVaR contribution from each asset
├─ CVaR vs VaR:
│   ├─ VaR: Single quantile, ignores tail
│   ├─ CVaR: Entire tail, averages extremes
│   ├─ VaR: Can encourage risk concentration
│   ├─ CVaR: Encourages diversification (coherent)
│   ├─ VaR: Non-convex optimization (hard)
│   └─ CVaR: Convex optimization (tractable)
└─ Regulatory & Practical:
    ├─ Basel III: Moving from VaR to Expected Shortfall
    ├─ FRTB (Fundamental Review): ES for market risk
    ├─ Backtesting: Harder than VaR (tail estimation uncertainty)
    └─ Estimation Error: Requires more data for stable estimates
```

**Interaction:** CVaR extends VaR by quantifying tail severity, enabling coherent risk management

## 5. Mini-Project
Implement CVaR optimization and compare with mean-variance framework:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats

# Download data
tickers = ['SPY', 'AGG', 'TLT', 'GLD', 'VNQ', 'EEM', 'DBC']
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
returns = data.pct_change().dropna()

def historical_cvar(returns, weights, alpha=0.95):
    """
    Calculate CVaR using historical method
    """
    portfolio_returns = (returns @ weights)
    var_threshold = -np.percentile(portfolio_returns, (1-alpha)*100)
    
    # Losses exceeding VaR
    tail_losses = -portfolio_returns[portfolio_returns < -var_threshold]
    
    if len(tail_losses) > 0:
        cvar = tail_losses.mean()
    else:
        cvar = var_threshold
    
    return cvar

def parametric_cvar_normal(returns, weights, alpha=0.95):
    """
    Calculate CVaR assuming normal distribution
    """
    portfolio_returns = returns @ weights
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    # For normal: CVaR = μ - σ × φ(z_α)/(1-α)
    z_alpha = stats.norm.ppf(alpha)
    phi_z = stats.norm.pdf(z_alpha)
    
    cvar = -(mu - sigma * phi_z / (1 - alpha))
    return cvar

def parametric_cvar_t(returns, weights, alpha=0.95):
    """
    Calculate CVaR assuming t-distribution
    """
    portfolio_returns = returns @ weights
    
    # Fit t-distribution
    params = stats.t.fit(portfolio_returns)
    df, loc, scale = params
    
    # t-distribution CVaR formula
    t_alpha = stats.t.ppf(alpha, df)
    f_t = stats.t.pdf(t_alpha, df)
    
    cvar = -(loc - scale * (df + t_alpha**2) / (df - 1) * f_t / (1 - alpha))
    return cvar

def calculate_var_cvar(returns, weights, alpha=0.95):
    """
    Calculate both VaR and CVaR
    """
    portfolio_returns = returns @ weights
    
    # VaR
    var = -np.percentile(portfolio_returns, (1-alpha)*100)
    
    # CVaR
    cvar = historical_cvar(returns, weights, alpha)
    
    return var, cvar

# CVaR Optimization (Rockafellar-Uryasev formulation)
def cvar_optimization_objective(weights, returns, alpha=0.95):
    """
    Minimize CVaR using historical simulation
    For discrete distribution: CVaR_α = min_t {t + 1/(1-α) * mean([L-t]⁺)}
    """
    portfolio_returns = returns @ weights
    losses = -portfolio_returns
    
    # Optimize over auxiliary variable t (VaR estimate)
    def objective_with_t(t):
        # [L - t]⁺ = max(L - t, 0)
        excess_losses = np.maximum(losses - t, 0)
        cvar = t + np.mean(excess_losses) / (1 - alpha)
        return cvar
    
    # Find optimal t
    result = minimize(objective_with_t, x0=np.percentile(losses, alpha*100), 
                     method='BFGS')
    
    return result.fun

def optimize_cvar_portfolio(returns, target_return=None, alpha=0.95):
    """
    Find portfolio minimizing CVaR with optional return constraint
    """
    n_assets = returns.shape[1]
    
    # Initial guess (equal weight)
    w0 = np.ones(n_assets) / n_assets
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
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
        lambda w: cvar_optimization_objective(w, returns, alpha),
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

def mean_variance_optimize(returns, target_return=None):
    """
    Traditional mean-variance optimization for comparison
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    def portfolio_volatility(weights):
        return np.sqrt(weights @ cov_matrix @ weights)
    
    w0 = np.ones(n_assets) / n_assets
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: w @ mean_returns - target_return
        })
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(
        portfolio_volatility,
        x0=w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    return result.x if result.success else w0

# Compute efficient frontiers
print("\nComputing efficient frontiers...")
mean_returns = returns.mean() * 252  # Annualized
n_points = 15

# Range of target returns
min_return = mean_returns.min()
max_return = mean_returns.max()
target_returns = np.linspace(min_return, max_return * 0.9, n_points)

# Storage
mv_portfolios = []
cvar_portfolios = []

for target_ret in target_returns:
    # Mean-Variance
    mv_weights = mean_variance_optimize(returns, target_ret/252)
    mv_vol = np.sqrt(mv_weights @ returns.cov() @ mv_weights) * np.sqrt(252)
    mv_var, mv_cvar = calculate_var_cvar(returns, mv_weights, 0.95)
    
    mv_portfolios.append({
        'return': target_ret,
        'volatility': mv_vol,
        'var': mv_var * np.sqrt(252),
        'cvar': mv_cvar * np.sqrt(252),
        'weights': mv_weights
    })
    
    # CVaR Optimization
    cvar_weights = optimize_cvar_portfolio(returns, target_ret/252, 0.95)
    cvar_ret = cvar_weights @ returns.mean() * 252
    cvar_vol = np.sqrt(cvar_weights @ returns.cov() @ cvar_weights) * np.sqrt(252)
    cv_var, cv_cvar = calculate_var_cvar(returns, cvar_weights, 0.95)
    
    cvar_portfolios.append({
        'return': cvar_ret,
        'volatility': cvar_vol,
        'var': cv_var * np.sqrt(252),
        'cvar': cv_cvar * np.sqrt(252),
        'weights': cvar_weights
    })

mv_df = pd.DataFrame(mv_portfolios)
cvar_df = pd.DataFrame(cvar_portfolios)

# Individual assets
asset_returns = returns.mean() * 252
asset_vols = returns.std() * np.sqrt(252)
asset_cvars = []
for i in range(len(tickers)):
    weights = np.zeros(len(tickers))
    weights[i] = 1.0
    _, cvar = calculate_var_cvar(returns, weights, 0.95)
    asset_cvars.append(cvar * np.sqrt(252))

# Compare specific portfolios
print("\n" + "="*100)
print("COMPARISON: Mean-Variance vs CVaR Optimization (Target Return: 8%)")
print("="*100)

target_idx = np.argmin(np.abs(mv_df['return'] - 0.08))
mv_port = mv_df.iloc[target_idx]
cvar_port = cvar_df.iloc[target_idx]

comparison = pd.DataFrame({
    'Mean-Variance': [
        mv_port['return'],
        mv_port['volatility'],
        mv_port['var'],
        mv_port['cvar'],
        mv_port['return'] / mv_port['volatility'],
        mv_port['return'] / mv_port['cvar']
    ],
    'CVaR Optimized': [
        cvar_port['return'],
        cvar_port['volatility'],
        cvar_port['var'],
        cvar_port['cvar'],
        cvar_port['return'] / cvar_port['volatility'],
        cvar_port['return'] / cvar_port['cvar']
    ]
}, index=['Annual Return', 'Volatility', '95% VaR', '95% CVaR', 'Sharpe Ratio', 'Return/CVaR'])

print(comparison.round(4))

# Portfolio weights comparison
print("\n" + "="*100)
print("PORTFOLIO WEIGHTS COMPARISON")
print("="*100)
weights_comp = pd.DataFrame({
    'Mean-Variance': mv_port['weights'],
    'CVaR Optimized': cvar_port['weights']
}, index=tickers)
print(weights_comp.round(4))

# CVaR contribution analysis
def component_cvar(returns, weights, alpha=0.95, epsilon=0.0001):
    """
    Calculate marginal and component CVaR
    """
    base_cvar = historical_cvar(returns, weights, alpha)
    
    marginal_cvars = []
    for i in range(len(weights)):
        # Perturb weight
        perturbed_weights = weights.copy()
        perturbed_weights[i] += epsilon
        perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
        
        perturbed_cvar = historical_cvar(returns, perturbed_weights, alpha)
        marginal_cvar = (perturbed_cvar - base_cvar) / epsilon
        marginal_cvars.append(marginal_cvar)
    
    # Component CVaR
    component_cvars = weights * np.array(marginal_cvars)
    
    return {
        'portfolio_cvar': base_cvar,
        'marginal_cvar': marginal_cvars,
        'component_cvar': component_cvars
    }

cvar_comp = component_cvar(returns, cvar_port['weights'], 0.95)

print("\n" + "="*100)
print("CVaR CONTRIBUTION ANALYSIS (CVaR-Optimized Portfolio)")
print("="*100)
contrib_df = pd.DataFrame({
    'Weight': cvar_port['weights'],
    'Component CVaR': cvar_comp['component_cvar'],
    '% of Total': cvar_comp['component_cvar'] / cvar_comp['portfolio_cvar'] * 100
}, index=tickers)
print(contrib_df.round(4))
print(f"\nTotal Portfolio CVaR: {cvar_comp['portfolio_cvar']*np.sqrt(252):.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficient Frontier (Return vs Volatility)
axes[0, 0].scatter(asset_vols, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 0].annotate(ticker, (asset_vols[i], asset_returns[i]), 
                       fontsize=8, ha='right')

axes[0, 0].plot(mv_df['volatility'], mv_df['return'], 'b-o', 
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 0].plot(cvar_df['volatility'], cvar_df['return'], 'r-s',
               label='CVaR Optimized', alpha=0.7, markersize=4)

axes[0, 0].set_xlabel('Volatility (Annual)')
axes[0, 0].set_ylabel('Expected Return (Annual)')
axes[0, 0].set_title('Efficient Frontiers: Mean-Variance vs CVaR')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Return vs CVaR
axes[0, 1].scatter(asset_cvars, asset_returns, s=80, alpha=0.6, label='Individual Assets')
for i, ticker in enumerate(tickers):
    axes[0, 1].annotate(ticker, (asset_cvars[i], asset_returns[i]),
                       fontsize=8, ha='right')

axes[0, 1].plot(mv_df['cvar'], mv_df['return'], 'b-o',
               label='Mean-Variance', alpha=0.7, markersize=4)
axes[0, 1].plot(cvar_df['cvar'], cvar_df['return'], 'r-s',
               label='CVaR Optimized', alpha=0.7, markersize=4)

axes[0, 1].set_xlabel('95% CVaR (Annual)')
axes[0, 1].set_ylabel('Expected Return (Annual)')
axes[0, 1].set_title('Mean-CVaR Frontier')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: VaR vs CVaR comparison
portfolio_returns_mv = returns @ mv_port['weights']
portfolio_returns_cvar = returns @ cvar_port['weights']

axes[1, 0].hist(portfolio_returns_mv * 100, bins=50, alpha=0.5, 
               label='Mean-Variance Portfolio', density=True, color='blue')
axes[1, 0].hist(portfolio_returns_cvar * 100, bins=50, alpha=0.5,
               label='CVaR Portfolio', density=True, color='red')

# Mark VaR and CVaR
mv_var_line = mv_port['var'] / np.sqrt(252) * 100
mv_cvar_line = mv_port['cvar'] / np.sqrt(252) * 100
cvar_var_line = cvar_port['var'] / np.sqrt(252) * 100
cvar_cvar_line = cvar_port['cvar'] / np.sqrt(252) * 100

axes[1, 0].axvline(-mv_var_line, color='blue', linestyle='--', alpha=0.7,
                  label=f'MV VaR: {mv_var_line:.2f}%')
axes[1, 0].axvline(-mv_cvar_line, color='blue', linestyle=':', alpha=0.7,
                  label=f'MV CVaR: {mv_cvar_line:.2f}%')
axes[1, 0].axvline(-cvar_cvar_line, color='red', linestyle=':', alpha=0.7,
                  label=f'CVaR CVaR: {cvar_cvar_line:.2f}%')

axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Return Distributions: MV vs CVaR Portfolios')
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Weight comparison
x = np.arange(len(tickers))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, mv_port['weights'], width, 
                       label='Mean-Variance', alpha=0.8)
bars2 = axes[1, 1].bar(x + width/2, cvar_port['weights'], width,
                       label='CVaR Optimized', alpha=0.8)

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(tickers)
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_title('Portfolio Weight Comparison')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS: CVaR vs VaR")
print("="*100)
print("1. CVaR always ≥ VaR (captures tail severity)")
print("2. CVaR optimization produces more diversified portfolios")
print("3. CVaR is coherent (sub-additive), VaR is not")
print("4. CVaR optimization is convex, enables global optimum")
print("5. Mean-CVaR frontier similar to mean-variance but better tail control")
print("6. CVaR portfolio typically has lower extreme losses")
print("7. Basel III moving from VaR to Expected Shortfall for market risk")
print("8. CVaR estimation requires more data (tail focus)")

# Coherence demonstration
print("\n" + "="*100)
print("COHERENCE PROPERTY DEMONSTRATION")
print("="*100)

# Create two simple portfolios
weights_A = np.array([0.8, 0.2, 0, 0, 0, 0, 0])
weights_B = np.array([0, 0, 0.6, 0.4, 0, 0, 0])
weights_combined = (weights_A + weights_B) / 2

var_A, cvar_A = calculate_var_cvar(returns, weights_A, 0.95)
var_B, cvar_B = calculate_var_cvar(returns, weights_B, 0.95)
var_combined, cvar_combined = calculate_var_cvar(returns, weights_combined, 0.95)

print(f"Portfolio A - VaR: {var_A*np.sqrt(252):.4f}, CVaR: {cvar_A*np.sqrt(252):.4f}")
print(f"Portfolio B - VaR: {var_B*np.sqrt(252):.4f}, CVaR: {cvar_B*np.sqrt(252):.4f}")
print(f"Combined    - VaR: {var_combined*np.sqrt(252):.4f}, CVaR: {cvar_combined*np.sqrt(252):.4f}")
print(f"\nVaR Sub-additivity: VaR(A+B) ≤ VaR(A) + VaR(B)?")
print(f"  {var_combined*np.sqrt(252):.4f} ≤ {(var_A + var_B)*np.sqrt(252):.4f}? {var_combined <= var_A + var_B}")
print(f"\nCVaR Sub-additivity: CVaR(A+B) ≤ CVaR(A) + CVaR(B)?")
print(f"  {cvar_combined*np.sqrt(252):.4f} ≤ {(cvar_A + cvar_B)*np.sqrt(252):.4f}? {cvar_combined <= cvar_A + cvar_B}")
print(f"\nCVaR is coherent (always sub-additive), VaR can fail!")
```

## 6. Challenge Round
When should you prefer CVaR over VaR?
- Portfolio optimization: CVaR enables convex optimization (tractable, global solution)
- Tail risk focus: Need to know severity of extreme losses, not just threshold
- Regulatory reporting: Basel III moved to Expected Shortfall for market risk
- Encourages diversification: CVaR is sub-additive (coherent), VaR can penalize diversification
- Complex portfolios: Options, structured products have asymmetric tails

CVaR limitations and practical challenges:
- Estimation uncertainty: Requires more data than VaR (tail focus, sparse observations)
- Computational cost: Optimization more complex than mean-variance (though tractable)
- Backtesting difficulty: Harder to validate (how to test average beyond threshold?)
- Interpretation: Less intuitive than VaR for non-technical stakeholders
- Spectral risk measures: CVaR equally weights tail, may want more weight on extremes

How does CVaR relate to other risk measures?
- VaR: CVaR ≥ VaR always; CVaR = E[L | L ≥ VaR]
- Standard deviation: CVaR focuses on downside, σ treats up/down equally
- Semi-variance: CVaR is tail-specific, semi-var is all downside
- Spectral measures: CVaR is special case (equal weights in tail, zero elsewhere)
- Drawdown: CVaR is single-period, drawdown is path-dependent

## 7. Key References
- [Rockafellar, R.T. & Uryasev, S. (2000) "Optimization of Conditional Value-at-Risk"](http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf)
- [Artzner et al. (1999) "Coherent Measures of Risk"](https://link.springer.com/article/10.1007/s780050100)
- [Acerbi, C. & Tasche, D. (2002) "Expected Shortfall: A Natural Coherent Alternative to VaR"](https://www.bis.org/bcbs/ca/acertasc.pdf)
- [Basel Committee (2016) "Fundamental Review of the Trading Book"](https://www.bis.org/bcbs/publ/d352.htm)

---
**Status:** Preferred coherent risk measure (Basel III standard) | **Complements:** VaR, Stress Testing, Risk Parity
