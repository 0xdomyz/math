import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def capm_regression(asset_returns, market_returns):
    """
    Simple CAPM regression
    """
    common_dates = asset_returns.index.intersection(market_returns.index)
    y = asset_returns.loc[common_dates]
    X = sm.add_constant(market_returns.loc[common_dates])
    
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    
    return {
        'alpha': results.params['const'],
        'beta': results.params.iloc[1],
        'rsquared': results.rsquared
    }

print("Running CAPM regressions for comparison...")
capm_results = {}
for ticker in tickers:
    if ticker != 'SPY':
        capm_results[ticker] = capm_regression(returns[ticker], returns['SPY'])

# Comparison table
comparison_data = []
for ticker in tickers:
    if ticker == 'SPY':
        continue
    
    capm = capm_results[ticker]
    apt_m = apt_macro_results[ticker]
    apt_s = apt_stat_results[ticker]
    
    comparison_data.append({
        'Ticker': ticker,
        'CAPM Alpha (%)': capm['alpha'] * 12 * 100,
        'CAPM Beta': capm['beta'],
        'CAPM R²': capm['rsquared'],
        'APT-Macro Alpha (%)': apt_m['alpha'] * 12 * 100,
        'APT-Macro R²': apt_m['rsquared'],
        'APT-Stat Alpha (%)': apt_s['alpha'] * 12 * 100,
        'APT-Stat R²': apt_s['rsquared']
    })

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "="*100)
print("MODEL COMPARISON: CAPM vs APT (Macro) vs APT (Statistical)")
print("="*100)
print(comparison_df.round(4).to_string(index=False))

# Factor loadings for one example (QQQ - tech-heavy)
print("\n" + "="*100)
print("DETAILED FACTOR LOADINGS: QQQ (Nasdaq ETF)")
print("="*100)

print("\nMacroeconomic Factors:")
qqq_macro = apt_macro_results['QQQ']
for factor_name, loading, tval in zip(macro_factors.columns, 
                                      qqq_macro['loadings'], 
                                      qqq_macro['tvalues']):
    sig = '***' if abs(tval) > 2.576 else ('**' if abs(tval) > 1.96 else ('*' if abs(tval) > 1.645 else ''))
    print(f"  {factor_name:>20}: {loading:>8.4f}  (t = {tval:>6.2f}) {sig}")

print("\nStatistical Factors (PCA):")
qqq_stat = apt_stat_results['QQQ']
for factor_name, loading, tval in zip([f'PC{i+1}' for i in range(5)],
                                     qqq_stat['loadings'],
                                     qqq_stat['tvalues']):
    sig = '***' if abs(tval) > 2.576 else ('**' if abs(tval) > 1.96 else ('*' if abs(tval) > 1.645 else ''))
    print(f"  {factor_name:>20}: {loading:>8.4f}  (t = {tval:>6.2f}) {sig}")

print("\n* p<0.10, ** p<0.05, *** p<0.01")

# Cross-sectional regression (estimate risk premiums λ)
def cross_sectional_regression(returns_avg, factor_loadings_matrix):
    """
    Second-pass cross-sectional regression: E[R] = λ0 + β·λ
    """
    X = sm.add_constant(factor_loadings_matrix)
    y = returns_avg
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results

# Prepare data for cross-sectional regression (macro factors)
avg_returns = returns.mean() * 12 * 100  # Annualized %
loading_matrix_macro = pd.DataFrame({
    factor: [apt_macro_results[ticker]['loadings'][factor] for ticker in tickers]
    for factor in macro_factors.columns
}, index=tickers)

print("\n" + "="*100)
print("CROSS-SECTIONAL REGRESSION: Risk Premiums (λ)")
print("="*100)
cross_sect_results = cross_sectional_regression(avg_returns, loading_matrix_macro)
print(cross_sect_results.summary())

risk_premiums = cross_sect_results.params[1:]  # Exclude intercept
print("\nEstimated Risk Premiums (% per year):")
for factor, premium in risk_premiums.items():
    print(f"  {factor:>20}: {premium:>8.2f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: R² comparison
models = ['CAPM', 'APT-Macro', 'APT-Stat']
r2_means = [
    comparison_df['CAPM R²'].mean(),
    comparison_df['APT-Macro R²'].mean(),
    comparison_df['APT-Stat R²'].mean()
]

bars = axes[0, 0].bar(models, r2_means, alpha=0.7, color=['blue', 'green', 'red'])
axes[0, 0].set_ylabel('Average R²')
axes[0, 0].set_title('Model Explanatory Power (Average R²)')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, r2_means):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')

# Plot 2: Factor loadings heatmap (macro factors)
loading_heatmap = loading_matrix_macro.T
im = axes[0, 1].imshow(loading_heatmap, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

axes[0, 1].set_yticks(range(len(loading_heatmap.index)))
axes[0, 1].set_yticklabels(loading_heatmap.index)
axes[0, 1].set_xticks(range(len(loading_heatmap.columns)))
axes[0, 1].set_xticklabels(loading_heatmap.columns, rotation=45, ha='right')
axes[0, 1].set_title('Factor Loadings: Macro Factors')
plt.colorbar(im, ax=axes[0, 1])

# Plot 3: Alpha comparison across models
x = np.arange(len(comparison_df))
width = 0.25

bars1 = axes[1, 0].bar(x - width, comparison_df['CAPM Alpha (%)'], width, 
                       label='CAPM', alpha=0.8)
bars2 = axes[1, 0].bar(x, comparison_df['APT-Macro Alpha (%)'], width,
                       label='APT-Macro', alpha=0.8)
bars3 = axes[1, 0].bar(x + width, comparison_df['APT-Stat Alpha (%)'], width,
                       label='APT-Stat', alpha=0.8)

axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(comparison_df['Ticker'], rotation=45, ha='right')
axes[1, 0].set_ylabel('Alpha (% per year)')
axes[1, 0].set_title('Alpha Comparison Across Models')
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Cumulative variance explained by PCs
cum_var = np.cumsum(pca_model.explained_variance_ratio_) * 100
axes[1, 1].plot(range(1, len(cum_var)+1), cum_var, marker='o', linewidth=2)
axes[1, 1].set_xlabel('Number of Principal Components')
axes[1, 1].set_ylabel('Cumulative Variance Explained (%)')
axes[1, 1].set_title('PCA: Cumulative Variance Explained')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xticks(range(1, len(cum_var)+1))

# Add percentage labels
for i, val in enumerate(cum_var):
    axes[1, 1].text(i+1, val, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Arbitrage test
print("\n" + "="*100)
print("ARBITRAGE PORTFOLIO TEST")
print("="*100)
print("Concept: If two assets have similar factor exposures but different returns,")
print("         arbitrage opportunity exists (long high return, short low return)")

# Find assets with similar loadings on first 2 macro factors
loading_similarity = {}
for i, ticker1 in enumerate(tickers):
    for ticker2 in tickers[i+1:]:
        load1 = apt_macro_results[ticker1]['loadings'][:2]
        load2 = apt_macro_results[ticker2]['loadings'][:2]
        
        # Euclidean distance in loading space
        distance = np.sqrt(((load1 - load2)**2).sum())
        loading_similarity[(ticker1, ticker2)] = distance

# Find most similar pair
most_similar = min(loading_similarity, key=loading_similarity.get)
ticker_a, ticker_b = most_similar

print(f"\nMost similar factor exposures: {ticker_a} and {ticker_b}")
print(f"Loading distance: {loading_similarity[most_similar]:.4f}")

ret_a = avg_returns[ticker_a]
ret_b = avg_returns[ticker_b]

print(f"\nAverage Returns:")
print(f"  {ticker_a}: {ret_a:.2f}%")
print(f"  {ticker_b}: {ret_b:.2f}%")
print(f"  Difference: {abs(ret_a - ret_b):.2f}%")

if abs(ret_a - ret_b) > 2:  # Threshold for "significant" difference
    print(f"\nPotential arbitrage: Long {ticker_a if ret_a > ret_b else ticker_b}, "
          f"Short {ticker_b if ret_a > ret_b else ticker_a}")
else:
    print("\nNo clear arbitrage opportunity (returns similar given factor exposures)")

# Key insights
print("\n" + "="*100)
print("KEY INSIGHTS: ARBITRAGE PRICING THEORY")
print("="*100)
print("1. APT more flexible than CAPM (multiple risk factors vs single market factor)")
print("2. Statistical factors (PCA) capture latent risk sources in returns")
print("3. Macro factors provide economic interpretation of risk exposures")
print("4. APT typically increases R² by 5-20% over CAPM")
print("5. First principal component often resembles market factor (systematic risk)")
print("6. Factor loadings stable over time → useful for risk management")
print("7. APT doesn't specify factors → practitioner must choose (flexibility + ambiguity)")
print("8. Cross-sectional regression estimates risk premiums (price of factor risk)")

print("\n" + "="*100)
print("APT vs CAPM: Key Differences")
print("="*100)
print("• Assumptions: APT weaker (no market portfolio, no M-V utility required)")
print("• Factors: CAPM single (market), APT multiple (unspecified)")
print("• Derivation: CAPM equilibrium, APT arbitrage (no-arbitrage condition)")
print("• Empirics: APT better fit but requires factor identification")
print("• Testability: CAPM more specific, APT more ambiguous")