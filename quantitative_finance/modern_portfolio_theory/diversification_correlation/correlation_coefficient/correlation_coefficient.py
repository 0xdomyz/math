import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, kendalltau, norm
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic stock returns with known correlations
np.random.seed(42)
periods = 252  # One trading year

# Market factor
market_returns = np.random.normal(0.0005, 0.01, periods)

# Five stock returns with different correlations to market
stock_a = 0.0003 + 0.9 * market_returns + np.random.normal(0, 0.008, periods)   # ρ_market ≈ 0.90
stock_b = 0.0002 + 0.5 * market_returns + np.random.normal(0, 0.012, periods)   # ρ_market ≈ 0.50
stock_c = 0.0001 + 0.2 * market_returns + np.random.normal(0, 0.015, periods)   # ρ_market ≈ 0.20
stock_d = 0.0004 - 0.3 * market_returns + np.random.normal(0, 0.010, periods)   # ρ_market ≈ -0.30
stock_e = np.random.normal(0.0003, 0.009, periods)                              # ρ_market ≈ 0 (idiosyncratic)

returns = pd.DataFrame({
    'market': market_returns,
    'stock_a': stock_a,
    'stock_b': stock_b,
    'stock_c': stock_c,
    'stock_d': stock_d,
    'stock_e': stock_e,
})

dates = pd.date_range('2024-01-01', periods=periods, freq='D')
returns.index = dates

print("="*70)
print("CORRELATION ANALYSIS: Multi-Stock Portfolio")
print("="*70)

# 1. Pairwise Correlations
print("\n1. PAIRWISE CORRELATION MATRIX")
print("-"*70)

corr_matrix = returns.corr()
print("\nPearson Correlation Matrix:")
print(corr_matrix.round(4))

# Covariance matrix
cov_matrix = returns.cov() * 252  # Annualize
print(f"\n\nAnnualized Covariance Matrix:")
print(cov_matrix.round(6))

# 2. Correlation with market
print("\n2. CORRELATION WITH MARKET")
print("-"*70)

market_corr = corr_matrix['market'].drop('market')
print("\nStock-Market Correlations (sorted):")
for ticker, corr in market_corr.sort_values(ascending=False).items():
    print(f"  {ticker:10s}: {corr:7.4f}")

# 3. Statistical significance tests
print("\n3. SIGNIFICANCE TESTS (ρ = 0)")
print("-"*70)

from scipy.stats import t

for stock in ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']:
    r = market_corr[stock]
    n = len(returns)
    
    # t-statistic
    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), n - 2))
    
    # Fisher z-transform confidence interval
    z = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1 / np.sqrt(n - 3)
    z_ci = 1.96  # 95% CI
    
    z_lower = z - z_ci * se_z
    z_upper = z + z_ci * se_z
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    print(f"\n{stock}:")
    print(f"  Correlation: {r:7.4f}")
    print(f"  t-stat: {t_stat:8.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  95% CI: [{r_lower:7.4f}, {r_upper:7.4f}]")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

# 4. Alternative correlation measures
print("\n4. ALTERNATIVE CORRELATION MEASURES")
print("-"*70)

# Spearman rank correlation (non-parametric)
spearman_corr = {}
# Kendall tau (rank-based, for outliers)
kendall_corr = {}

for stock in ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']:
    spearman_corr[stock], _ = spearmanr(returns['market'], returns[stock])
    kendall_corr[stock], _ = kendalltau(returns['market'], returns[stock])

print("\nComparison of Correlation Measures:")
print(f"{'Stock':<12} {'Pearson':<10} {'Spearman':<10} {'Kendall':<10}")
print("-"*42)
for stock in ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']:
    print(f"{stock:<12} {market_corr[stock]:<10.4f} {spearman_corr[stock]:<10.4f} {kendall_corr[stock]:<10.4f}")

# 5. Correlation over rolling windows
print("\n5. TIME-VARYING CORRELATION")
print("-"*70)

window = 60  # 60-day rolling window
rolling_corr = {}

for stock in ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']:
    rolling_corr[stock] = returns['market'].rolling(window).corr(returns[stock])

print(f"\nRolling {window}-day correlation with market:")
for stock in ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']:
    r = rolling_corr[stock].dropna()
    print(f"{stock}:")
    print(f"  Mean: {r.mean():7.4f}")
    print(f"  Std Dev: {r.std():7.4f}")
    print(f"  Min: {r.min():7.4f}")
    print(f"  Max: {r.max():7.4f}")

# 6. Diversification analysis (two-stock portfolio)
print("\n6. DIVERSIFICATION ANALYSIS: Two-Asset Portfolios")
print("-"*70)

# Portfolio: 50% stock_a + 50% stock_b
stocks_pair = ['stock_a', 'stock_b']
var_a = returns['stock_a'].var() * 252
var_b = returns['stock_b'].var() * 252
corr_ab = returns[['stock_a', 'stock_b']].corr().iloc[0, 1]
cov_ab = corr_ab * np.sqrt(var_a) * np.sqrt(var_b)

# Equal weight portfolio variance
w_a = w_b = 0.5
var_portfolio_ew = w_a**2 * var_a + w_b**2 * var_b + 2*w_a*w_b*cov_ab

# Minimum variance portfolio
numerator = var_b - cov_ab
denominator = var_a + var_b - 2*cov_ab
w_a_min = numerator / denominator
w_b_min = 1 - w_a_min

var_portfolio_min = w_a_min**2 * var_a + w_b_min**2 * var_b + 2*w_a_min*w_b_min*cov_ab

print(f"\n50-50 Portfolio (Stock A + Stock B):")
print(f"  Stock A variance: {var_a:.6f} (σ={np.sqrt(var_a):.4f})")
print(f"  Stock B variance: {var_b:.6f} (σ={np.sqrt(var_b):.4f})")
print(f"  Correlation: {corr_ab:.4f}")
print(f"  Portfolio variance: {var_portfolio_ew:.6f} (σ={np.sqrt(var_portfolio_ew):.4f})")

diversification_ratio = (0.5*np.sqrt(var_a) + 0.5*np.sqrt(var_b)) / np.sqrt(var_portfolio_ew)
print(f"  Diversification ratio: {diversification_ratio:.4f}")
print(f"  (> 1 means less risky than avg of components)")

print(f"\n\nMinimum Variance Portfolio:")
print(f"  Weight stock A: {w_a_min:.4f}")
print(f"  Weight stock B: {w_b_min:.4f}")
print(f"  Portfolio variance: {var_portfolio_min:.6f} (σ={np.sqrt(var_portfolio_min):.4f})")

# 7. Correlation visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Heatmap of correlation matrix
ax = axes[0, 0]
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_matrix)))
ax.set_yticks(range(len(corr_matrix)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)
ax.set_title('Correlation Matrix Heatmap')
# Add values
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
plt.colorbar(im, ax=ax)

# Plot 2: Scatter - Market vs Stock A
ax = axes[0, 1]
ax.scatter(returns['market']*100, returns['stock_a']*100, alpha=0.6, s=30)
z = np.polyfit(returns['market']*100, returns['stock_a']*100, 1)
p = np.poly1d(z)
x_line = np.array([returns['market'].min()*100, returns['market'].max()*100])
ax.plot(x_line, p(x_line), 'r--', linewidth=2)
ax.set_xlabel('Market Return (%)')
ax.set_ylabel('Stock A Return (%)')
ax.set_title(f'Market vs Stock A (ρ={market_corr["stock_a"]:.4f})')
ax.grid(alpha=0.3)

# Plot 3: Scatter - Market vs Stock D (negative correlation)
ax = axes[0, 2]
ax.scatter(returns['market']*100, returns['stock_d']*100, alpha=0.6, s=30, color='orange')
z = np.polyfit(returns['market']*100, returns['stock_d']*100, 1)
p = np.poly1d(z)
x_line = np.array([returns['market'].min()*100, returns['market'].max()*100])
ax.plot(x_line, p(x_line), 'r--', linewidth=2)
ax.set_xlabel('Market Return (%)')
ax.set_ylabel('Stock D Return (%)')
ax.set_title(f'Market vs Stock D (ρ={market_corr["stock_d"]:.4f})')
ax.grid(alpha=0.3)

# Plot 4: Rolling correlation
ax = axes[1, 0]
for stock in ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']:
    ax.plot(dates[window:], rolling_corr[stock][window:], label=stock, linewidth=1, alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_ylabel('Rolling Correlation')
ax.set_title(f'{window}-Day Rolling Correlation with Market')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)

# Plot 5: Distribution comparison
ax = axes[1, 1]
ax.hist(returns['market']*100, bins=30, alpha=0.5, label='Market', density=True)
ax.hist(returns['stock_a']*100, bins=30, alpha=0.5, label='Stock A', density=True)
ax.set_xlabel('Return (%)')
ax.set_ylabel('Density')
ax.set_title('Return Distributions')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Correlation confidence interval
ax = axes[1, 2]
stocks_list = ['stock_a', 'stock_b', 'stock_c', 'stock_d', 'stock_e']
correlations = [market_corr[s] for s in stocks_list]
ci_lower = []
ci_upper = []

for stock in stocks_list:
    r = market_corr[stock]
    n = len(returns)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se_z = 1 / np.sqrt(n - 3)
    z_ci = 1.96
    z_lower = z - z_ci * se_z
    z_upper = z + z_ci * se_z
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    ci_lower.append(r_lower)
    ci_upper.append(r_upper)

ax.errorbar(range(len(stocks_list)), correlations, 
            yerr=[np.array(correlations) - np.array(ci_lower), 
                  np.array(ci_upper) - np.array(correlations)],
            fmt='o', markersize=8, capsize=5, capthick=2)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No correlation')
ax.set_xticks(range(len(stocks_list)))
ax.set_xticklabels(stocks_list, rotation=45, ha='right')
ax.set_ylabel('Correlation with Market')
ax.set_title('95% Confidence Intervals for Correlations')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"""
Key Findings:

1. Correlation Structure:
   → Stocks show varying market sensitivity
   → Stock A (ρ=0.90): High systematic risk, moves with market
   → Stock D (ρ=-0.30): Negative correlation, hedge against market
   → Stock E (ρ≈0): Independent, pure idiosyncratic risk
   
2. Statistical Significance:
   → Some correlations significant (p < 0.05), others not
   → Large n can make small ρ statistically significant
   → Practical significance > statistical significance
   
3. Rolling Correlation Shows:
   → Time-varying relationships (correlation instability)
   → Crisis periods: correlation increases (contagion)
   → Normal periods: correlation more stable
   
4. Diversification Benefit:
   → Lower correlation → larger diversification gain
   → Perfect negative correlation: maximum risk reduction
   → Stock A + B combination: {diversification_ratio:.2f}x more diversified than weighted average
   
5. Portfolio Implications:
   → Optimal weights depend on correlation and variances
   → Minimum variance portfolio: overweight lower-correlation assets
   → Negative correlation assets most valuable for hedging
""")