# Commonality in Liquidity

## 1. Concept Skeleton
**Definition:** Comovement of liquidity across assets, indicating systematic liquidity risk factors beyond idiosyncratic asset characteristics  
**Purpose:** Identifies liquidity crises, informs portfolio risk management, explains asset pricing anomalies  
**Prerequisites:** Liquidity measures, factor models, correlation analysis, systematic risk

## 2. Comparative Framing
| Dimension | Commonality in Liquidity | Commonality in Returns | Liquidity Beta | Funding Liquidity |
|-----------|-------------------------|------------------------|----------------|-------------------|
| **Focus** | Cross-asset liquidity comovement | Price comovement | Exposure to liquidity factor | Capital availability |
| **Measure** | Correlation of liquidity | Correlation of returns | β_liquidity | Broker balance sheets |
| **Driver** | Market-wide shocks | Fundamentals, sentiment | Sensitivity to aggregate liquidity | Credit conditions |
| **Impact** | Systematic liquidity risk | Diversification | Risk premium | Funding constraints |

## 3. Examples + Counterexamples

**Positive Commonality:**  
2008 crisis: Bid-ask spreads widen across all stocks simultaneously (correlation >0.7) → systematic liquidity shock

**Idiosyncratic Liquidity:**  
Earnings announcement for AAPL: Spreads widen for AAPL only, other stocks unaffected → firm-specific event

**Edge Case:**  
Flight to quality: Small-cap spreads widen while Treasury spreads narrow → negative commonality (opposite liquidity movement)

## 4. Layer Breakdown
```
Commonality in Liquidity Framework:
├─ Chordia et al (2000) Findings:
│   ├─ Discovery: Individual stock liquidity correlates with market liquidity
│   ├─ Method: Regress stock-level liquidity on market-level liquidity
│   │   L_{i,t} = α + β × L_{M,t} + ε_{i,t}
│   ├─ Results: Significant positive β for spread, depth, volume
│   └─ Interpretation: Systematic component to liquidity risk
├─ Sources of Commonality:
│   ├─ Supply-Side (Market Maker Constraints):
│   │   - Capital constraints: Limited dealer capacity
│   │   - Risk management: VaR limits bind across portfolio
│   │   - Funding liquidity: Prime broker credit tightens
│   │   - Inventory management: Correlated position unwinding
│   ├─ Demand-Side (Investor Behavior):
│   │   - Correlated trading: Mutual fund flows, index rebalancing
│   │   - Flight to quality: Risk-off selling pressure
│   │   - Margin calls: Forced liquidations across assets
│   │   - Herding: Correlated order flow from sentiment
│   └─ Information Asymmetry:
│       - Market-wide information events (Fed, macro)
│       - Contagion: News from one sector spreads
│       - Uncertainty shocks: VIX spikes widen spreads
├─ Measurement Approaches:
│   ├─ Principal Components Analysis:
│   │   - First PC of liquidity measures captures commonality
│   │   - Proportion of variance explained by PC1
│   ├─ Cross-Sectional Correlation:
│   │   - Average pairwise correlation of liquidity
│   │   - Time-varying correlation (rolling window)
│   ├─ Factor Model:
│   │   - Market liquidity factor (equal/value-weighted)
│   │   - Liquidity beta: sensitivity to market liquidity
│   └─ Copula Models:
│       - Capture non-linear tail dependence
│       - Asymmetric commonality in crises
├─ Time Variation:
│   ├─ Normal Periods: Low commonality (~0.1-0.3)
│   ├─ Stress Periods: High commonality (>0.7)
│   ├─ Crisis Amplification: Positive feedback loops
│   └─ Flight to Quality: Divergence in liquidity
└─ Implications:
    ├─ Systematic Risk: Liquidity risk not diversifiable
    ├─ Asset Pricing: Liquidity beta earns risk premium
    ├─ Portfolio Construction: Allocate to low liquidity beta assets
    └─ Risk Management: Stress test liquidity risk
```

**Interaction:** Market shock → dealer constraint → liquidity withdrawal → correlated spread widening → feedback loop

## 5. Mini-Project
Measure commonality in liquidity across assets:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import pandas as pd

np.random.seed(42)

# Simulate multiple stocks with commonality in liquidity
n_stocks = 30
n_days = 500

# Market-wide liquidity factor (systematic component)
market_liquidity_factor = np.cumsum(np.random.normal(0, 0.1, n_days))

# Volatility regime (affects liquidity)
volatility_regime = np.abs(np.sin(np.linspace(0, 4*np.pi, n_days))) + 0.5

# Individual stock liquidity (systematic + idiosyncratic)
stock_liquidity = np.zeros((n_stocks, n_days))
liquidity_betas = np.random.uniform(0.3, 1.5, n_stocks)  # Sensitivity to market factor
idiosyncratic_vol = np.random.uniform(0.5, 2.0, n_stocks)

for i in range(n_stocks):
    # Stock liquidity = beta × market_factor + idiosyncratic
    systematic_component = liquidity_betas[i] * market_liquidity_factor
    idiosyncratic_component = np.cumsum(np.random.normal(0, idiosyncratic_vol[i] * 0.1, n_days))
    
    # Add volatility regime effect (spreads widen in high vol)
    regime_effect = volatility_regime * 0.5
    
    stock_liquidity[i, :] = systematic_component + idiosyncratic_component + regime_effect

# Convert to bid-ask spreads (positive values)
spreads = np.exp(stock_liquidity) * 0.01  # Spread in dollars

# Calculate returns (for comparison)
stock_returns = np.random.multivariate_normal(
    mean=np.zeros(n_stocks),
    cov=0.3 * np.eye(n_stocks) + 0.7 * np.ones((n_stocks, n_stocks)),  # Correlated returns
    size=n_days
).T

# Analysis
# 1. Cross-sectional correlation of liquidity
spread_changes = np.diff(spreads, axis=1)
corr_matrix_liquidity = np.corrcoef(spread_changes)
avg_correlation_liquidity = (corr_matrix_liquidity.sum() - n_stocks) / (n_stocks * (n_stocks - 1))

# Compare to return correlation
corr_matrix_returns = np.corrcoef(stock_returns)
avg_correlation_returns = (corr_matrix_returns.sum() - n_stocks) / (n_stocks * (n_stocks - 1))

# 2. Principal Components Analysis
pca_liquidity = PCA()
pca_liquidity.fit(spread_changes.T)
variance_explained = pca_liquidity.explained_variance_ratio_

# 3. Market liquidity factor (equal-weighted average)
market_spread = spreads.mean(axis=0)
market_spread_changes = np.diff(market_spread)

# 4. Liquidity betas (regression of individual on market)
estimated_betas = []
r_squareds = []

for i in range(n_stocks):
    individual_changes = spread_changes[i, :]
    if len(individual_changes) > 0 and len(market_spread_changes) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            market_spread_changes, individual_changes
        )
        estimated_betas.append(slope)
        r_squareds.append(r_value ** 2)

estimated_betas = np.array(estimated_betas)
r_squareds = np.array(r_squareds)

# 5. Time-varying commonality (rolling correlation)
window = 50
rolling_correlations = []

for t in range(window, n_days - 1):
    window_data = spread_changes[:, t-window:t]
    corr_matrix_window = np.corrcoef(window_data)
    avg_corr_window = (corr_matrix_window.sum() - n_stocks) / (n_stocks * (n_stocks - 1))
    rolling_correlations.append(avg_corr_window)

rolling_correlations = np.array(rolling_correlations)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Individual spreads and market average
sample_stocks = [0, 5, 10, 15, 20]
colors = plt.cm.viridis(np.linspace(0, 1, len(sample_stocks)))

for idx, stock_idx in enumerate(sample_stocks):
    axes[0, 0].plot(spreads[stock_idx, :], alpha=0.5, linewidth=1, 
                   color=colors[idx], label=f'Stock {stock_idx+1}')

axes[0, 0].plot(market_spread, 'r-', linewidth=2.5, alpha=0.8, 
               label='Market Average')
axes[0, 0].set_xlabel('Day')
axes[0, 0].set_ylabel('Bid-Ask Spread ($)')
axes[0, 0].set_title('Individual Stock Spreads vs Market Average')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

print("Commonality in Liquidity Analysis:")
print("=" * 70)
print(f"\nCross-Sectional Correlations:")
print(f"Average Liquidity Correlation: {avg_correlation_liquidity:.4f}")
print(f"Average Return Correlation: {avg_correlation_returns:.4f}")
print(f"Liquidity/Return Correlation Ratio: {avg_correlation_liquidity/avg_correlation_returns:.2f}")

# Plot 2: PCA variance explained
cumsum_variance = np.cumsum(variance_explained)

axes[0, 1].bar(range(1, 11), variance_explained[:10], alpha=0.7, 
              label='Individual PC')
axes[0, 1].plot(range(1, 11), cumsum_variance[:10], 'r-o', linewidth=2, 
               markersize=6, label='Cumulative')

# Add text for PC1
axes[0, 1].text(1, variance_explained[0] + 0.02, 
               f'{variance_explained[0]*100:.1f}%', 
               ha='center', fontsize=10, fontweight='bold')

axes[0, 1].set_xlabel('Principal Component')
axes[0, 1].set_ylabel('Proportion of Variance Explained')
axes[0, 1].set_title('PCA of Liquidity Changes')
axes[0, 1].set_xticks(range(1, 11))
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

print(f"\nPrincipal Components Analysis:")
print(f"PC1 explains {variance_explained[0]*100:.2f}% of variance")
print(f"PC1-3 explain {cumsum_variance[2]*100:.2f}% of variance")
print("(High PC1 variance indicates strong commonality)")

# Plot 3: Liquidity beta distribution
axes[1, 0].hist(estimated_betas, bins=20, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(estimated_betas.mean(), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {estimated_betas.mean():.2f}')
axes[1, 0].axvline(1.0, color='g', linestyle='--', linewidth=2, 
                  label='Beta = 1.0')
axes[1, 0].set_xlabel('Liquidity Beta')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Liquidity Betas')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

print(f"\nLiquidity Beta Statistics:")
print(f"Mean Beta: {estimated_betas.mean():.3f}")
print(f"Median Beta: {np.median(estimated_betas):.3f}")
print(f"Std Dev: {estimated_betas.std():.3f}")
print(f"Min/Max: {estimated_betas.min():.3f} / {estimated_betas.max():.3f}")
print(f"Mean R-squared: {r_squareds.mean():.3f}")
print("(R² measures how much variation explained by market factor)")

# Plot 4: Time-varying commonality
time_axis = np.arange(window, n_days - 1)

axes[1, 1].plot(time_axis, rolling_correlations, linewidth=2, label='Rolling Correlation')
axes[1, 1].axhline(avg_correlation_liquidity, color='r', linestyle='--', 
                  linewidth=2, label=f'Average: {avg_correlation_liquidity:.3f}')

# Shade high-stress periods (high volatility regime)
stress_threshold = np.percentile(volatility_regime, 75)
stress_periods = volatility_regime[window:-1] > stress_threshold

axes[1, 1].fill_between(time_axis, 0, 1, where=stress_periods, 
                        alpha=0.2, color='red', label='High Volatility Period')

axes[1, 1].set_xlabel('Day')
axes[1, 1].set_ylabel('Average Pairwise Correlation')
axes[1, 1].set_title(f'Time-Varying Commonality ({window}-day window)')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Calculate correlation during stress vs calm
calm_periods = ~stress_periods
if stress_periods.sum() > 0 and calm_periods.sum() > 0:
    corr_stress = rolling_correlations[stress_periods].mean()
    corr_calm = rolling_correlations[calm_periods].mean()
    
    print(f"\nTime-Varying Commonality:")
    print(f"Correlation during Calm periods: {corr_calm:.4f}")
    print(f"Correlation during Stress periods: {corr_stress:.4f}")
    print(f"Stress/Calm Ratio: {corr_stress/corr_calm:.2f}x")
    print("(Higher during stress indicates systematic liquidity risk)")

plt.tight_layout()
plt.show()

# Statistical test: Is commonality significant?
# Test if average correlation significantly different from zero
t_stat = avg_correlation_liquidity * np.sqrt(n_stocks * (n_days - 2)) / \
         np.sqrt(1 - avg_correlation_liquidity**2)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_days-2))

print(f"\nStatistical Significance Test:")
print(f"H0: No commonality (correlation = 0)")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.6f}")
if p_value < 0.01:
    print("Result: Reject null - significant commonality exists")

# Chordia et al (2000) style regression
print(f"\nChordia et al (2000) Regression Results:")
print(f"(Individual spread change = α + β × Market spread change + ε)")
print(f"\nAverage across {n_stocks} stocks:")
print(f"  Mean Beta: {estimated_betas.mean():.3f}")
print(f"  % with significant beta (R²>0.1): {(r_squareds > 0.1).mean()*100:.1f}%")
print(f"  Mean R-squared: {r_squareds.mean():.3f}")

# Implications for diversification
print(f"\nDiversification Implications:")
diversification_benefit = 1 - avg_correlation_liquidity
print(f"Liquidity Diversification Benefit: {diversification_benefit*100:.1f}%")
print("(Lower correlation → better diversification)")

# Compare true vs estimated betas
if len(liquidity_betas) == len(estimated_betas):
    corr_betas = np.corrcoef(liquidity_betas, estimated_betas)[0, 1]
    print(f"\nModel Validation:")
    print(f"True vs Estimated Beta Correlation: {corr_betas:.3f}")
    print("(High correlation validates factor model)")
```

## 6. Challenge Round
Why does commonality in liquidity intensify during crises, and what are the policy implications?
- **Funding liquidity spirals**: Broker-dealers face margin calls → withdraw market making across all assets → correlated spread widening → fire sales → further losses → feedback loop
- **Flight to quality**: Risk-off triggers selling of illiquid assets, buying Treasuries → negative commonality (divergence) but still systematic
- **Regulatory capital constraints**: Basel III risk-weighted assets bind simultaneously across portfolios → synchronized liquidity provision cuts
- **Policy interventions**: Fed's 2008 Primary Dealer Credit Facility directly addressed dealer funding liquidity → broke feedback loop, restored commonality to normal levels
- **Circuit breakers**: Trading halts prevent cascading illiquidity but may worsen commonality by synchronizing restarts

## 7. Key References
- [Chordia et al (2000) - Common Determinants of Liquidity and Trading](https://www.jstor.org/stable/222555)
- [Hasbrouck & Seppi (2001) - Common Factors in Prices, Order Flows, and Liquidity](https://www.sciencedirect.com/science/article/abs/pii/S0304405X00000910)
- [Brunnermeier & Pedersen (2009) - Market Liquidity and Funding Liquidity](https://academic.oup.com/rfs/article-abstract/22/6/2201/1585686)
- [Kamara et al (2008) - Why is the Stock Market More Volatile Today?](https://www.sciencedirect.com/science/article/abs/pii/S0304405X08000251)

---
**Status:** Systematic liquidity risk | **Complements:** Liquidity Measures, Liquidity Crises, Asset Pricing
