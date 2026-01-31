from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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