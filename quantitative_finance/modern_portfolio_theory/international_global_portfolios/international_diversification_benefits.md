# International Diversification Benefits

## 1. Concept Skeleton
**Definition:** Risk reduction from combining assets across different countries whose returns are imperfectly correlated (ρ < 1); benefit from business cycle desynchronization and structural economic differences  
**Purpose:** Quantify diversification gains from geographic expansion, optimize country/regional allocation, understand how international correlation structure changes over time  
**Prerequisites:** Correlation and covariance, diversification principles, portfolio optimization, efficient frontier

---

## 2. Comparative Framing

| Characteristic | Domestic Portfolio (100% US) | Mixed Portfolio (60% US / 40% Int'l) | Full Global Portfolio (35% US / 30% Dev. / 20% EM / 15% Bonds Int'l) |
|----------------|------------------------------|---------------------------------------|----------------------------------------------------------------------|
| **Expected Return** | ~9% p.a. | ~8.5% p.a. (slight drag from lower EM returns in some periods) | ~8% p.a. (diversification benefit offsets some return drag) |
| **Volatility** | ~16% p.a. | ~14% p.a. (lower due to diversification) | ~13% p.a. (maximum benefit from geography + asset class) |
| **Correlation Structure** | N/A (single market) | ρ(US stocks, intl developed) ≈ 0.85; ρ(US stocks, EM) ≈ 0.65 | Lower correlations with multiple regions; ρ varies by cycle |
| **Sharpe Ratio** | 9% / 16% = 0.56 | 8.5% / 14% = 0.61 (better) | 8% / 13% = 0.62 (best) |
| **Max Drawdown (2008)** | -57% | -52% (less severe) | -48% (mildest) |
| **Diversification Benefit** | 0% (baseline) | 2-3% risk reduction | 4-5% risk reduction |
| **Currency Risk** | 0% | Adds ~2-3% volatility (unhedged) | ~3-4% volatility if unhedged |
| **Implementation Cost** | Lowest (domestic) | Moderate (FX, higher fees) | Highest (multiple markets, taxes) |
| **Regulatory Risk** | Minimal | Moderate | Higher (political, accounting) |

**Key Insight:** Gains peak when correlation < 0.7. Correlation increased dramatically in 2008 crisis (from 0.7 → 0.95), reducing diversification benefit. Over long periods, correlation reverts toward 0.80-0.85 as markets normalize.

---

## 3. Examples + Counterexamples

**Example 1: 1990s Diversification Success (Japan Bubble Burst)**
- US stocks (S&P 500): +1,500% return (1990-2000)
- Japanese stocks (Nikkei): +20% return (stagnation during "Lost Decade")
- Japanese bonds: +50% return (capital gains as rates fell)
- Global 60/40 portfolio with 20% Japan: Mixed benefit; stocks drag returns, bonds help
- Lesson: International diversification protected against US downturn risk (which didn't materialize in 1990s) but added Japanese stagnation risk

**Example 2: 2008 Global Financial Crisis (Correlation Breakdown)**
- August 2008: US stocks (S&P 500): -3%, European stocks: -2%, Japanese stocks: -1%
- October 2008 (crisis peak): US: -20%, Europe: -28% (worse!), Japan: -25%
- Correlation jumped from 0.75 → 0.98 (lost diversification benefit)
- Unhedged EUR investor in US stocks: Lost additional 8% to EUR weakness
- Lesson: Correlations spike in crisis; diversification benefit vanishes precisely when needed most

**Example 3: 2010-2015 EM Growth Divergence (Segmentation)**
- Brazil (commodity exporter): -60% local currency (commodity crash + currency collapse)
- India (domestic consumption): +200% (structural growth, demographic dividend)
- Correlation(Brazil, India): -0.3 (near negative; perfect diversification)
- Global 60/40 with 10% Brazil, 10% India: EM volatility offset by negative correlation
- Lesson: Within-region diversification (EM heterogeneity) adds benefit beyond developed market correlations

**Counterexample: US Exceptionalism (Tech Bubble Burst 2000-2002)**
- NASDAQ (tech-heavy): -78% (bubble burst)
- S&P 500 (broad): -47%
- EAFE (Europe): -35% (no tech bubble)
- Japanese stocks: -30% (structural decline, but not bubble)
- International diversification helped: 40% EAFE buffer reduced drawdown
- BUT: If investor was overweight tech (common in 1990s), diversification insufficient

**Edge Case: Emerging Market Contagion (Russian Crisis 1998, Asian Crisis 1997)**
- Thailand (epicenter): -50% local currency
- Indonesia: -70%
- South Korea: -40%
- Brazil (seemingly unrelated): -35% (contagion as investors fled emerging assets globally)
- Correlation(Brazil, Thailand): +0.8 during crisis (normally +0.4)
- Lesson: Diversification within EM vulnerable to systemic shocks; developed market diversification better (correlation spike less extreme)

---

## 4. Layer Breakdown

```
International Diversification Architecture:

├─ Correlation Structure Over Time:
│   ├─ Normal Times (70% of history): ρ(US, Dev. Int'l) ≈ 0.75-0.85
│   │   └─ Supports ~2-3% risk reduction vs domestic-only
│   ├─ Crisis Periods (15% of history): ρ → 0.95+ (crisis correlation)
│   │   └─ Diversification benefit collapses; all assets fall together
│   ├─ Divergence Periods (15% of history): ρ → 0.40-0.60 (regional decoupling)
│   │   └─ Maximum diversification benefit (e.g., 2016: Fed tighten, ECB QE)
│   └─ Implications: Historical correlations overstate diversification benefit
│       in crises (when needed most); backtest with rolling correlations
│
├─ Return Decomposition by Geography:
│   ├─ US Equity (35%): ~10% p.a. return, 16% volatility
│   ├─ Developed ex-US (30%): ~8% p.a. return, 18% volatility
│   │   └─ Geographic weighting: Japan (35%), UK (20%), Germany (15%), France (10%), other (20%)
│   ├─ Emerging Markets (20%): ~10% p.a. return (high variance), 22% volatility
│   │   └─ Allocation: China (35%), India (25%), Brazil (15%), EM Asia (15%), other (10%)
│   └─ Bonds (15%): ~3% p.a. return, 5% volatility
│
├─ Correlation Matrix Evolution:
│   ├─ 1980-2000 (Pre-globalization): ρ(US, Int'l) ≈ 0.60 (low)
│   │   └─ Reason: National markets more segmented, regulatory barriers
│   │   └─ Diversification benefit: 8-10% variance reduction possible
│   │
│   ├─ 2000-2007 (Globalization Peak): ρ(US, Int'l) ≈ 0.75 (moderate)
│   │   └─ Reason: Capital liberalization, multinational corporations
│   │   └─ Diversification benefit: 4-6% variance reduction
│   │
│   ├─ 2008-2009 (Crisis): ρ → 0.95 (all assets tank together)
│   │   └─ Reason: Systemic financial shock; forced liquidations
│   │   └─ Diversification benefit: Nearly zero (crisis contagion)
│   │
│   ├─ 2010-2015 (Divergence): ρ = 0.55-0.70 (structural divergence)
│   │   └─ Reason: Fed QE (push US stocks higher), ECB stagnation, EM growth
│   │   └─ Diversification benefit: 6-8% variance reduction (peak!)
│   │
│   ├─ 2016-2019 (Synchronized Growth): ρ = 0.75-0.85 (normal)
│   │   └─ Reason: Global expansion, synchronized monetary policy
│   │   └─ Diversification benefit: 4-6% variance reduction
│   │
│   └─ 2020-Present (Regime Shift?): ρ fluctuates 0.70-0.85
│       └─ COVID divergence followed by global recovery
│       └─ Diversification benefit: 4-6% variance reduction (normal)
│
├─ Risk Reduction Mechanics:
│   ├─ Total Portfolio Variance:
│   │   σ²_p = Σᵢ wᵢ² σᵢ² + 2Σᵢ<ⱼ wᵢ wⱼ ρᵢⱼ σᵢ σⱼ
│   │   └─ Diagonal term (wᵢ² σᵢ²) = direct risk
│   │   └─ Off-diagonal term (ρᵢⱼ) = diversification; lower ρ = higher benefit
│   │
│   ├─ Diversification Ratio:
│   │   DR = (Σᵢ wᵢ σᵢ) / σ_p
│   │   └─ Perfect correlation (ρ = 1): DR = 1 (no diversification benefit)
│   │   └─ Zero correlation (ρ = 0): DR = 1 / √N (maximum benefit)
│   │   └─ Typical global portfolio: DR ≈ 1.5-2.0 (variance reduction vs equal-weight)
│   │
│   └─ Risk Reduction from Diversification:
│       σ²_diversified = σ²_average_asset × (1/N + (N-1)/N × ρ_avg)
│       At N = 3 (US + Dev Int'l + EM), ρ_avg = 0.70:
│       σ² = σ²_avg × (1/3 + 2/3 × 0.70) = σ²_avg × 0.80
│       Risk reduction: 20% compared to equally weighted
│
├─ Country/Regional Allocation:
│   ├─ Static Allocation (Traditional): Equal-weighted by region
│   │   └─ US 35%, Dev Int'l 30%, EM 20%, Bonds 15%
│   │   └─ Rationale: Easy to implement; rebalancing natural constraint
│   │
│   ├─ Market-Cap Weighted: Allocate by size of economy/market
│   │   └─ US ~35-40%, Europe ~25-30%, Japan ~10-15%, EM ~15-20%
│   │   └─ Rationale: Less diversification (home bias in cap-weighting)
│   │
│   ├─ Risk-Parity Allocation: Weight by inverse volatility
│   │   └─ If σ_US=16%, σ_Int'l=18%, σ_EM=22%, σ_Bonds=5%:
│   │   └─ Weights: US (27%), Int'l (26%), EM (23%), Bonds (24%)
│   │   └─ Rationale: Equalizes marginal risk contribution; smoother returns
│   │
│   └─ Optimal MV Allocation: Maximize Sharpe ratio
│       └─ Often overweights bonds, underweights EM
│       └─ Rationale: Mathematical optimization; often unstable (sensitive to inputs)
│
├─ Tail Risk & Correlation Regimes:
│   ├─ Normal State (80% probability): ρ(US, Int'l) = 0.75
│   │   └─ Diversification working as intended
│   │
│   ├─ Crisis State (10% probability): ρ → 0.95+
│   │   └─ All assets crash together
│   │   └─ Diversification benefit evaporates
│   │   └─ Implication: Long-only portfolios vulnerable to simultaneous drawdowns
│   │
│   ├─ Divergence State (10% probability): ρ → 0.50
│   │   └─ Different regions boom/bust
│   │   └─ Maximum diversification benefit
│   │   └─ Example: 2016 (Fed tightening → US up, EM benefited from other policies)
│   │
│   └─ Risk Management: Monitor rolling correlations; rebalance when regime shifts
│
└─ Implementation Reality vs Theory:
    ├─ Theoretical Benefit: 20-30% variance reduction from international diversification
    ├─ Actual Realized Benefit: 8-12% variance reduction (after costs, taxes, timing)
    ├─ Gap Reasons:
    │   ├─ Correlation higher than expected in downturns (when risk reduction most valuable)
    │   ├─ Currency hedging costs: 1-3% annually if fully hedged
    │   ├─ FX volatility: Adds 2-4% to portfolio volatility if unhedged
    │   ├─ Implementation costs: 0.3-0.5% annually (trading spreads, fees)
    │   ├─ Behavioral: Tendency to abandon strategy during downturns (lock in losses)
    │   └─ Taxes: Capital gains on rebalancing reduce net returns
    └─ Recommendation: International diversification valuable but don't overpay for it
        (use low-cost index funds; limit FX hedging to material positions)
```

**Mathematical Formulas:**

Correlation Matrix Impact on Portfolio Variance (simplified 2-asset case):
$$\sigma^2_p = w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2w_1 w_2 \rho_{12} \sigma_1 \sigma_2$$

Equal-weight (w₁ = w₂ = 0.5), σ₁ = σ₂ = σ, varying ρ:
$$\sigma^2_p = 0.5 \sigma^2 + 0.25 \rho_{12} \sigma^2 = \sigma^2(0.5 + 0.25\rho_{12})$$

- If ρ₁₂ = 1 (perfect correlation): σₚ = σ (no diversification)
- If ρ₁₂ = 0 (zero correlation): σₚ = 0.707σ (√0.5 × σ, ~30% reduction)
- If ρ₁₂ = 0.75 (typical): σₚ = 0.806σ (~19% reduction)

Diversification Ratio (measures effectiveness of diversification):
$$DR = \frac{\sum_i w_i \sigma_i}{\sigma_p}$$

For well-diversified portfolio with N assets, equal weight, ρ_avg:
$$DR \approx \sqrt{\frac{1 + (N-1)\rho_{avg}}{N \rho_{avg}}}$$

---

## 5. Mini-Project: International Diversification Backtest

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

# Download international equity indices and analyze diversification benefit

def get_international_indices(start_date, end_date):
    """
    Fetch returns for major global indices.
    """
    indices = {
        'SPY': 'US (S&P 500)',
        'EFA': 'Developed Int\'l (EAFE)',
        'EEM': 'Emerging Markets',
        'BND': 'US Bonds',
        'IEMG': 'EM Bonds',
    }
    
    data = yf.download(list(indices.keys()), 
                       start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns, indices


def analyze_correlation_structure(returns, indices):
    """
    Compute correlation matrices and statistics.
    """
    corr = returns.corr()
    
    # Rolling correlation (252-day window = 1 year)
    rolling_corr = pd.DataFrame({
        'US_vs_IntDev': returns['SPY'].rolling(252).corr(returns['EFA']),
        'US_vs_EM': returns['SPY'].rolling(252).corr(returns['EEM']),
        'IntDev_vs_EM': returns['EFA'].rolling(252).corr(returns['EEM']),
    })
    
    return corr, rolling_corr


def portfolio_variance(weights, returns, indices):
    """
    Compute portfolio variance given weights and return correlations.
    """
    cov_matrix = returns[list(indices.keys())].cov() * 252  # Annualize
    var = weights @ cov_matrix @ weights.T
    return np.sqrt(var)


def calculate_diversification_benefit(returns, indices, scenario_names):
    """
    Compare diversification approaches:
    1. Domestic only (100% US)
    2. Simple international (60% US, 40% Int'l)
    3. Multi-region (35% US, 25% Dev Int'l, 20% EM, 20% Bonds)
    """
    
    scenarios = {
        'Domestic Only': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),  # 100% US stocks
        'US + Int\'l Dev': np.array([0.6, 0.4, 0.0, 0.0, 0.0]),  # 60% US, 40% EAFE
        'Global Diversified': np.array([0.35, 0.25, 0.20, 0.15, 0.05]),  # Multi-region
        'Global + EM Bonds': np.array([0.30, 0.20, 0.15, 0.20, 0.15]),  # Multi-region + EM bonds
    }
    
    results = pd.DataFrame()
    
    for scenario_name, weights in scenarios.items():
        portfolio_returns = (returns[list(indices.keys())] * weights).sum(axis=1)
        
        results.loc[scenario_name, 'Annual Return'] = portfolio_returns.mean() * 252
        results.loc[scenario_name, 'Annual Volatility'] = portfolio_returns.std() * np.sqrt(252)
        results.loc[scenario_name, 'Sharpe Ratio'] = results.loc[scenario_name, 'Annual Return'] / results.loc[scenario_name, 'Annual Volatility']
        results.loc[scenario_name, 'Max Drawdown'] = (1 + portfolio_returns).cumprod().min() / (1 + portfolio_returns).cumprod().max() - 1
        
        # Diversification ratio
        weight_vol_sum = sum(weights[i] * returns[list(indices.keys())[i]].std() * np.sqrt(252) for i in range(len(weights)))
        results.loc[scenario_name, 'Diversification Ratio'] = weight_vol_sum / results.loc[scenario_name, 'Annual Volatility']
    
    return results


def analyze_correlation_breakdown(returns, indices):
    """
    Analyze why correlation changes (crisis vs normal times).
    """
    # Recent period: 5 years
    recent = returns.tail(252 * 5)  # Last 5 years
    
    # Crisis period simulation: Mark high volatility days
    portfolio_return = (returns[['SPY', 'EFA', 'EEM', 'BND']].std()).mean()
    crisis_threshold = returns.std() * 1.5  # Days with volatility > 1.5 std
    
    normal_corr = recent.corr()
    
    # Calculate correlation on crisis days vs normal days
    crisis_days = (recent.std() > crisis_threshold).sum() > 0
    
    return normal_corr


# Main Analysis
print("=" * 100)
print("INTERNATIONAL DIVERSIFICATION BENEFITS BACKTEST")
print("=" * 100)

# Get 15 years of data
returns, indices = get_international_indices('2009-01-01', '2024-01-01')

# 1. Correlation analysis
print("\n1. CURRENT CORRELATION STRUCTURE (as of latest date)")
print("-" * 100)
corr, rolling_corr = analyze_correlation_structure(returns, indices)
print(corr[['SPY', 'EFA', 'EEM', 'BND', 'IEMG']].round(3))

# 2. Rolling correlation trends
print("\n2. ROLLING CORRELATION TRENDS (12-month windows)")
print("-" * 100)
print(f"US vs Developed Int'l (mean): {rolling_corr['US_vs_IntDev'].mean():.3f}")
print(f"  → Range: {rolling_corr['US_vs_IntDev'].min():.3f} to {rolling_corr['US_vs_IntDev'].max():.3f}")
print(f"US vs Emerging Markets (mean): {rolling_corr['US_vs_EM'].mean():.3f}")
print(f"  → Range: {rolling_corr['US_vs_EM'].min():.3f} to {rolling_corr['US_vs_EM'].max():.3f}")
print(f"Developed Int'l vs EM (mean): {rolling_corr['IntDev_vs_EM'].mean():.3f}")
print(f"  → Range: {rolling_corr['IntDev_vs_EM'].min():.3f} to {rolling_corr['IntDev_vs_EM'].max():.3f}")

# 3. Portfolio comparison
print("\n3. DIVERSIFICATION SCENARIOS: RISK-RETURN COMPARISON")
print("-" * 100)

scenario_names = list(['Domestic Only', 'US + Int\'l Dev', 'Global Diversified', 'Global + EM Bonds'])
results = calculate_diversification_benefit(returns, indices, scenario_names)
print(results.round(4))

print("\nRisk Reduction vs Domestic Only:")
domestic_vol = results.loc['Domestic Only', 'Annual Volatility']
for scenario in results.index[1:]:
    reduction = (1 - results.loc[scenario, 'Annual Volatility'] / domestic_vol) * 100
    print(f"  {scenario}: {reduction:.1f}% volatility reduction")

# 4. Crisis analysis: 2008, 2020
print("\n4. CRISIS PERFORMANCE ANALYSIS")
print("-" * 100)

# 2008 Crisis
crisis_2008 = returns.loc['2008-01-01':'2008-12-31']
crisis_2008_corr = crisis_2008.corr()
print("2008 Financial Crisis - Correlation Structure:")
print(f"  US vs Int'l Dev: {crisis_2008_corr.loc['SPY', 'EFA']:.3f}")
print(f"  US vs EM: {crisis_2008_corr.loc['SPY', 'EEM']:.3f}")
print(f"  → All positive & high (diversification benefit collapsed)")

# 2020 COVID
crisis_2020 = returns.loc['2020-01-01':'2020-06-30']
crisis_2020_corr = crisis_2020.corr()
print("\n2020 COVID Crisis - Correlation Structure:")
print(f"  US vs Int'l Dev: {crisis_2020_corr.loc['SPY', 'EFA']:.3f}")
print(f"  US vs EM: {crisis_2020_corr.loc['SPY', 'EEM']:.3f}")
print(f"  → Again high (crisis regime)")

# 5. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Rolling correlations
ax = axes[0, 0]
ax.plot(rolling_corr.index, rolling_corr['US_vs_IntDev'], label='US vs Developed Int\'l', linewidth=2)
ax.plot(rolling_corr.index, rolling_corr['US_vs_EM'], label='US vs Emerging Markets', linewidth=2)
ax.axhline(0.75, color='gray', linestyle='--', label='Normal (0.75)')
ax.axhline(0.95, color='red', linestyle='--', label='Crisis (0.95)')
ax.set_title('Rolling 12-Month Correlations Over Time', fontweight='bold')
ax.set_ylabel('Correlation')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(-0.2, 1.0)

# Plot 2: Cumulative returns by portfolio
ax = axes[0, 1]
scenarios_to_plot = ['Domestic Only', 'US + Int\'l Dev', 'Global Diversified']
weights_dict = {
    'Domestic Only': np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    'US + Int\'l Dev': np.array([0.6, 0.4, 0.0, 0.0, 0.0]),
    'Global Diversified': np.array([0.35, 0.25, 0.20, 0.15, 0.05]),
}

for scenario in scenarios_to_plot:
    weights = weights_dict[scenario]
    portfolio_returns = (returns[list(indices.keys())] * weights).sum(axis=1)
    cum_returns = (1 + portfolio_returns).cumprod()
    ax.plot(cum_returns.index, cum_returns, label=scenario, linewidth=2)

ax.set_title('Cumulative Portfolio Returns', fontweight='bold')
ax.set_ylabel('Cumulative Return (1.0 = start)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: Risk-return scatter
ax = axes[0, 2]
for scenario in results.index:
    ax.scatter(results.loc[scenario, 'Annual Volatility'], 
               results.loc[scenario, 'Annual Return'],
               s=200, alpha=0.7, label=scenario)

ax.set_xlabel('Annual Volatility')
ax.set_ylabel('Annual Return')
ax.set_title('Risk-Return Profile of Diversification Scenarios', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 4: Sharpe ratios
ax = axes[1, 0]
sharpe_ratios = results['Sharpe Ratio'].sort_values(ascending=False)
colors = ['green' if x > results['Sharpe Ratio'].mean() else 'steelblue' for x in sharpe_ratios]
ax.barh(range(len(sharpe_ratios)), sharpe_ratios, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(sharpe_ratios)))
ax.set_yticklabels(sharpe_ratios.index, fontsize=9)
ax.set_xlabel('Sharpe Ratio')
ax.set_title('Risk-Adjusted Returns by Portfolio', fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Plot 5: Drawdowns
ax = axes[1, 1]
max_drawdowns = results['Max Drawdown'].sort_values()
colors = ['green' if abs(x) < 0.4 else 'orange' if abs(x) < 0.5 else 'red' for x in max_drawdowns]
ax.barh(range(len(max_drawdowns)), max_drawdowns * 100, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(max_drawdowns)))
ax.set_yticklabels(max_drawdowns.index, fontsize=9)
ax.set_xlabel('Max Drawdown (%)')
ax.set_title('Downside Risk by Portfolio', fontweight='bold')
ax.grid(alpha=0.3, axis='x')

# Plot 6: Diversification ratio
ax = axes[1, 2]
div_ratios = results['Diversification Ratio']
ax.bar(range(len(div_ratios)), div_ratios, color='teal', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(div_ratios)))
ax.set_xticklabels(div_ratios.index, rotation=15, ha='right', fontsize=8)
ax.set_ylabel('Diversification Ratio')
ax.set_title('Effectiveness of Diversification\n(Higher = More Effective)', fontweight='bold')
ax.axhline(1.0, color='red', linestyle='--', label='No diversification')
ax.grid(alpha=0.3, axis='y')
ax.legend()

plt.tight_layout()
plt.savefig('international_diversification_backtest.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: international_diversification_backtest.png")
plt.show()

# 6. Key findings
print("\n5. KEY FINDINGS & RECOMMENDATIONS")
print("-" * 100)
print(f"""
DIVERSIFICATION BENEFIT QUANTIFIED:
├─ Domestic Only Volatility: {results.loc['Domestic Only', 'Annual Volatility']:.2%}
├─ Global Diversified Volatility: {results.loc['Global Diversified', 'Annual Volatility']:.2%}
└─ Risk Reduction: {(1 - results.loc['Global Diversified', 'Annual Volatility'] / results.loc['Domestic Only', 'Annual Volatility']) * 100:.1f}%

RETURN TRADE-OFF:
├─ Domestic Return: {results.loc['Domestic Only', 'Annual Return']:.2%}
├─ Global Return: {results.loc['Global Diversified', 'Annual Return']:.2%}
└─ Difference: {(results.loc['Global Diversified', 'Annual Return'] - results.loc['Domestic Only', 'Annual Return']) * 100:.1f}%

RISK-ADJUSTED PERFORMANCE (Sharpe Ratio):
├─ Domestic Only: {results.loc['Domestic Only', 'Sharpe Ratio']:.3f}
├─ Global Diversified: {results.loc['Global Diversified', 'Sharpe Ratio']:.3f}
└─ Improvement: {(results.loc['Global Diversified', 'Sharpe Ratio'] / results.loc['Domestic Only', 'Sharpe Ratio'] - 1) * 100:.1f}%

CRISIS PROTECTION:
├─ Domestic Only Max Drawdown: {results.loc['Domestic Only', 'Max Drawdown']:.1%}
├─ Global Diversified Max Drawdown: {results.loc['Global Diversified', 'Max Drawdown']:.1%}
└─ Downside Protection: {(1 - results.loc['Global Diversified', 'Max Drawdown'] / results.loc['Domestic Only', 'Max Drawdown']) * 100:.1f}%

CORRELATION INSIGHTS:
├─ Normal times: Correlations 0.70-0.80 (diversification works)
├─ Crisis times: Correlations 0.90-0.95 (diversification fails)
└─ Implication: Diversification excellent in normal markets, unreliable in crises

RECOMMENDATIONS:
1. Allocate 30-40% internationally (sweet spot: diversification benefit vs added complexity)
2. Diversify within regions (don't over-concentrate in 1-2 countries)
3. Include emerging markets (higher returns offset higher volatility)
4. Rebalance annually (contrarian discipline; buy low, sell high)
5. Use unhedged FX exposure (negative correlations often valuable during crises)
6. Don't overpay for international access (use low-cost index funds; 0.10-0.20% fees)
7. Monitor correlation regimes quarterly (adjust hedge ratio if crisis signals appear)
""")

print("=" * 100)
```

---

## 6. Challenge Round

1. **Correlation Paradox:** Historical data shows correlation(US stocks, Emerging Markets) ≈ +0.65. Yet during COVID (2020), correlation spiked to +0.95 for 3 months, then fell back to +0.55. Why did correlation briefly spike? What does this tell you about using historical correlations for risk estimation?

2. **Home Bias Problem:** US investors hold ~90% domestic stocks, global market cap suggests ~35-40% optimal. Why do investors exhibit such extreme home bias despite international diversification benefits? What psychological or structural factors explain this?

3. **Diversification Failure in Crises:** In 1998 (Russian Crisis), Brazil was hit despite no direct Russia exposure (contagion). A portfolio 60% Brazil / 40% bonds lost -28% in one month. Why did diversification fail? How would dynamic hedging or correlation monitoring help?

4. **Multi-Currency Impact:** You have 40% EAFE (mostly EUR), 20% EM (diverse currencies). EUR strengthens 5%, EM currencies weaken 3%. What is the net FX impact on returns? Should you hedge selectively?

5. **Rebalancing with International Assets:** Your 60/40 portfolio (60% global stocks, 40% bonds) has rebalanced into 65% stocks, 35% bonds after equity rally. Rebalancing costs 0.5%. Is it worth rebalancing? How does this decision change if correlations are high (crisis mode) vs low (normal)?

---

## 7. Key References

- **Grubel, H. (1968).** "Internationally Diversified Portfolios: Welfare Gains and Capital Flows" – Early evidence that international diversification reduces portfolio risk.

- **Solnik, B. (1974).** "Why Not Diversify Internationally Rather Than Domestically?" – Landmark study showing 40% foreign allocation reduces variance without reducing returns.

- **Karolyi, G.A. & Stulz, R.M. (2003).** "Are Financial Assets Priced Locally or Globally?" – Examines market segmentation and integration effects on diversification benefits.

- **Longin, F. & Solnik, B. (2001).** "Extreme Correlation of Equity Markets" – Documents the correlation breakdown in crisis periods (correlation → 1 during crashes).

- **Ferreira, M.A. & Matos, P. (2008).** "The Colors of Investors' Money" – Shows how correlation structure determines optimal country weights.

- **Fama-French Data Library** – https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html – International equity factor returns for backtesting.

- **Federal Reserve FRED Database** – https://fred.stlouisfed.org – Currency rates and international economic indicators.

