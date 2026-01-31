import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EMPIRICAL ANALYSIS: Limits of Diversification")
print("="*70)

# Simulate 100 stocks with factor structure
np.random.seed(42)
periods = 252
n_stocks_max = 100

# Market factor (systematic risk)
market = np.random.normal(0.0005, 0.01, periods)

# Generate 100 stock returns
returns = np.zeros((periods, n_stocks_max))
betas = np.linspace(0.5, 1.5, n_stocks_max)  # Beta heterogeneity
idio_vol = np.random.uniform(0.008, 0.015, n_stocks_max)  # Idiosyncratic vol

for i in range(n_stocks_max):
    idiosyncratic = np.random.normal(0, idio_vol[i], periods)
    returns[:, i] = 0.0001 + betas[i] * market + idiosyncratic

returns_df = pd.DataFrame(returns, columns=[f'Stock_{i+1}' for i in range(n_stocks_max)])

# 1. Average return, variance, correlation by number of stocks
print("\n1. DIVERSIFICATION ANALYSIS: Portfolio Size Effects")
print("-"*70)

portfolio_sizes = np.arange(1, n_stocks_max + 1, 5)
results = []

for n in portfolio_sizes:
    # Select first n stocks (or random sample)
    subset_returns = returns_df.iloc[:, :n]
    
    # Equal-weight portfolio
    w = np.ones(n) / n
    
    # Portfolio statistics
    mean_return = w @ subset_returns.mean() * 252
    variance = w @ subset_returns.cov().values @ w * 252
    volatility = np.sqrt(variance)
    
    # Systematic vs idiosyncratic
    portfolio_betas = w @ betas[:n]
    systematic_vol = portfolio_betas * np.std(market) * np.sqrt(252)
    idiosyncratic_variance = variance - systematic_vol**2
    idiosyncratic_vol = np.sqrt(max(0, idiosyncratic_variance))
    
    # Average correlation in subset
    corr_matrix = subset_returns.corr()
    avg_corr = (corr_matrix.values.sum() - n) / (n * (n - 1))
    
    # Effective number of bets
    w_weights = w ** 2
    herfindahl = w_weights.sum()
    effective_n = 1 / herfindahl if herfindahl > 0 else 0
    
    results.append({
        'portfolio_size': n,
        'mean_return': mean_return,
        'total_volatility': volatility,
        'systematic_vol': systematic_vol,
        'idiosyncratic_vol': idiosyncratic_vol,
        'avg_correlation': avg_corr,
        'sharpe': mean_return / (volatility + 1e-10),
        'effective_n': effective_n,
        'herfindahl': herfindahl
    })

results_df = pd.DataFrame(results)

print("\nPortfolio Statistics by Size (Equal-Weight):")
print(results_df[['portfolio_size', 'total_volatility', 'systematic_vol', 
                   'idiosyncratic_vol', 'avg_correlation']].round(4).to_string(index=False))

# 2. Diversification curve (risk vs number of assets)
print("\n2. DIVERSIFICATION CURVE ANALYSIS")
print("-"*70)

# Find where benefit plateaus (derivative small)
vols = results_df['total_volatility'].values
n_assets = results_df['portfolio_size'].values
marginal_benefit = -np.diff(vols) / np.diff(n_assets)

# Find approximately where marginal benefit < 0.0001
plateau_idx = np.where(marginal_benefit < 0.0001)[0]
if len(plateau_idx) > 0:
    plateau_n = n_assets[plateau_idx[0] + 1]
    print(f"\nDiversification plateau approximately at: {plateau_n} stocks")
    print(f"Plateau volatility: {vols[plateau_idx[0] + 1]:.4f}")

# 3. Risk attribution
print("\n3. RISK ATTRIBUTION")
print("-"*70)

print(f"\nInitial portfolio (1 stock):")
print(f"  Total volatility: {results_df.iloc[0]['total_volatility']:.4f}")
print(f"  Systematic component: {results_df.iloc[0]['systematic_vol']:.4f}")
print(f"  Idiosyncratic component: {results_df.iloc[0]['idiosyncratic_vol']:.4f}")

print(f"\nFull diversification (100 stocks):")
print(f"  Total volatility: {results_df.iloc[-1]['total_volatility']:.4f}")
print(f"  Systematic component: {results_df.iloc[-1]['systematic_vol']:.4f}")
print(f"  Idiosyncratic component: {results_df.iloc[-1]['idiosyncratic_vol']:.4f}")

diversification_reduction = (results_df.iloc[0]['total_volatility'] - results_df.iloc[-1]['total_volatility']) / results_df.iloc[0]['total_volatility']
print(f"\nTotal risk reduction: {diversification_reduction*100:.1f}%")

systematic_pct_initial = results_df.iloc[0]['systematic_vol'] / results_df.iloc[0]['total_volatility']
systematic_pct_final = results_df.iloc[-1]['systematic_vol'] / results_df.iloc[-1]['total_volatility']

print(f"Systematic risk (% of total):")
print(f"  1 stock: {systematic_pct_initial*100:.1f}%")
print(f"  100 stocks: {systematic_pct_final*100:.1f}%")

# 4. Optimal portfolio size (considering costs)
print("\n4. OPTIMAL PORTFOLIO SIZE (With Transaction Costs)")
print("-"*70)

# Assume rebalancing cost per stock
cost_per_rebalancing = 0.001  # 0.1% per asset per rebalance
rebalance_frequency = 1  # Annual

# "Net benefit" = risk reduction benefit - transaction costs
net_benefit = []

for i, row in results_df.iterrows():
    risk_reduction_value = 1 - row['total_volatility'] / results_df.iloc[0]['total_volatility']
    transaction_cost_impact = row['portfolio_size'] * cost_per_rebalancing * rebalance_frequency
    net = risk_reduction_value - transaction_cost_impact
    net_benefit.append(net)

optimal_idx = np.argmax(net_benefit)
optimal_size = results_df.iloc[optimal_idx]['portfolio_size']

print(f"\nOptimal portfolio size: {optimal_size} stocks")
print(f"Net benefit at optimum: {net_benefit[optimal_idx]:.4f}")
print(f"  Risk reduction: {(1 - results_df.iloc[optimal_idx]['total_volatility'] / results_df.iloc[0]['total_volatility']):.4f}")
print(f"  Transaction cost: {results_df.iloc[optimal_idx]['portfolio_size'] * cost_per_rebalancing:.4f}")

# 5. Correlation structure analysis
print("\n5. CORRELATION BREAKDOWN UNDER STRESS")
print("-"*70)

# Simulate correlation increase in crisis
normal_corr = results_df.iloc[-1]['avg_correlation']
crisis_corrs = [0.3, 0.5, 0.7, 0.9]

print(f"\nNormal average correlation: {normal_corr:.4f}")

for crisis_corr in crisis_corrs:
    # Approximate portfolio vol under higher correlation
    # σₚ² ≈ σ²̄/n + ρ·σ²̄·(1-1/n)
    avg_var = np.mean([np.var(returns_df.iloc[:, i]) for i in range(n_stocks_max)]) * 252
    vol_crisis = np.sqrt(avg_var / n_stocks_max + crisis_corr * avg_var * (1 - 1/n_stocks_max))
    
    diversification_ratio = results_df.iloc[-1]['total_volatility'] / vol_crisis
    print(f"  Crisis correlation ρ={crisis_corr:.2f}: Vol = {vol_crisis:.4f}, " +
          f"Diversification loss: {(1-diversification_ratio)*100:.1f}%")

# 6. Concentration vs diversification tradeoff
print("\n6. CONCENTRATION VS. PERFORMANCE")
print("-"*70)

# Minimum variance portfolio
n_test = 50
test_returns = returns_df.iloc[:, :n_test]
cov_mat = test_returns.cov().values * 252

inv_cov = np.linalg.inv(cov_mat)
ones = np.ones(n_test)
w_minvar = inv_cov @ ones / (ones @ inv_cov @ ones)

# Equal weight
w_equal = np.ones(n_test) / n_test

# Compute metrics
vol_minvar = np.sqrt(w_minvar @ cov_mat @ w_minvar)
vol_equal = np.sqrt(w_equal @ cov_mat @ w_equal)

hhi_minvar = (w_minvar ** 2).sum()
hhi_equal = (w_equal ** 2).sum()

neff_minvar = 1 / hhi_minvar
neff_equal = 1 / hhi_equal

print(f"\nPortfolio comparison ({n_test} stocks):")
print(f"  Equal-weight:")
print(f"    Volatility: {vol_equal:.4f}")
print(f"    Concentration (HHI): {hhi_equal:.4f}")
print(f"    Effective assets: {neff_equal:.1f}")

print(f"\n  Minimum Variance:")
print(f"    Volatility: {vol_minvar:.4f} (-{(1-vol_minvar/vol_equal)*100:.1f}%)")
print(f"    Concentration (HHI): {hhi_minvar:.4f}")
print(f"    Effective assets: {neff_minvar:.1f}")
print(f"    Risk reduction cost: {(hhi_minvar - hhi_equal)*100:.1f}% concentration increase")

# 7. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Total, systematic, idiosyncratic volatility
ax = axes[0, 0]
ax.plot(results_df['portfolio_size'], results_df['total_volatility'], 'o-', 
        linewidth=2, markersize=6, label='Total')
ax.plot(results_df['portfolio_size'], results_df['systematic_vol'], 's--', 
        linewidth=2, markersize=5, label='Systematic')
ax.plot(results_df['portfolio_size'], results_df['idiosyncratic_vol'], '^:', 
        linewidth=2, markersize=5, label='Idiosyncratic')
ax.set_xlabel('Portfolio Size (# of stocks)')
ax.set_ylabel('Volatility (annualized)')
ax.set_title('Risk Decomposition vs Portfolio Size')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Marginal diversification benefit
ax = axes[0, 1]
marginal_benefit_all = np.diff(results_df['total_volatility'].values)
portfolio_sizes_all = results_df['portfolio_size'].values[:-1]
ax.bar(portfolio_sizes_all, -marginal_benefit_all, width=4, alpha=0.7)
ax.axhline(y=0.0001, color='r', linestyle='--', alpha=0.5, label='Threshold')
ax.set_xlabel('Portfolio Size')
ax.set_ylabel('Risk Reduction per Stock Added')
ax.set_title('Marginal Diversification Benefit')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Net benefit with costs
ax = axes[0, 2]
ax.plot(results_df['portfolio_size'], net_benefit, 'o-', linewidth=2, markersize=6)
ax.axvline(x=optimal_size, color='r', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_size}')
ax.set_xlabel('Portfolio Size')
ax.set_ylabel('Net Benefit')
ax.set_title('Risk Reduction - Transaction Costs')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Average correlation
ax = axes[1, 0]
ax.plot(results_df['portfolio_size'], results_df['avg_correlation'], 'o-', 
        linewidth=2, markersize=6, color='green')
ax.set_xlabel('Portfolio Size')
ax.set_ylabel('Average Correlation')
ax.set_title('Average Pairwise Correlation vs Size')
ax.grid(alpha=0.3)

# Plot 5: Crisis vs normal correlation impact
ax = axes[1, 1]
crisis_corrs_plot = [0.3, 0.5, 0.7, 0.9]
vols_crisis = []
for crisis_corr in crisis_corrs_plot:
    avg_var = np.mean([np.var(returns_df.iloc[:, i]) for i in range(n_stocks_max)]) * 252
    vol_crisis = np.sqrt(avg_var / n_stocks_max + crisis_corr * avg_var * (1 - 1/n_stocks_max))
    vols_crisis.append(vol_crisis)

normal_vol = results_df[results_df['portfolio_size'] == n_stocks_max]['total_volatility'].values[0]
ax.plot(crisis_corrs_plot, vols_crisis, 'o-', linewidth=2, markersize=8, label='Crisis volatility')
ax.axhline(y=normal_vol, color='g', linestyle='--', alpha=0.7, label=f'Normal vol: {normal_vol:.4f}')
ax.set_xlabel('Average Correlation')
ax.set_ylabel('Portfolio Volatility')
ax.set_title('Portfolio Volatility Under Different Correlations')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Effective number of bets
ax = axes[1, 2]
ax.plot(results_df['portfolio_size'], results_df['effective_n'], 'o-', 
        linewidth=2, markersize=6, color='purple')
ax.plot(results_df['portfolio_size'], results_df['portfolio_size'], 'r--', 
        linewidth=1, alpha=0.5, label='Ideal (no concentration)')
ax.set_xlabel('Nominal Number of Stocks')
ax.set_ylabel('Effective Number (1/HHI)')
ax.set_title('Effective vs Nominal Diversification')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diversification_limits.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("SUMMARY & IMPLICATIONS")
print("="*70)
print(f"""
Key Findings:

1. Diversification Ceiling:
   → Risk reduction plateaus around {plateau_n} stocks
   → Residual systematic risk: {results_df.iloc[-1]['systematic_vol']:.4f}
   → Further diversification: Minimal additional benefit

2. Risk Decomposition:
   → 1 stock: {systematic_pct_initial*100:.1f}% systematic + {(1-systematic_pct_initial)*100:.1f}% idiosyncratic
   → 100 stocks: {systematic_pct_final*100:.1f}% systematic + {(1-systematic_pct_final)*100:.1f}% idiosyncratic
   → Diversification eliminates idiosyncratic only

3. Optimal Portfolio Size:
   → With costs: {optimal_size} stocks optimal
   → Risk reduction benefit = transaction cost impact
   → Beyond optimum: Costs exceed benefits

4. Correlation Breakdown Risk:
   → Normal (ρ≈0.3): Diversification most effective
   → Crisis (ρ≈0.9): Diversification benefit {(1-vols_crisis[-1]/normal_vol)*100:.0f}% reduction
   → "Crisis when diversification fails most needed"

5. Practical Recommendations:
   → Domestic equity: 30-40 stocks sufficient
   → Global equity: 50-100 stocks optimal
   → Multi-asset: 15-20 total (better diversification via factors)
   → Factor diversification: Often superior to asset diversification
""")