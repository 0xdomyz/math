# Concentration Risk

## 1. Concept Skeleton
**Definition:** Risk arising from uneven exposure distribution; large exposures to few borrowers/sectors amplify losses when they default  
**Purpose:** Quantify portfolio vulnerability to idiosyncratic shocks, measure diversification effectiveness, set exposure limits  
**Prerequisites:** Portfolio statistics, Herfindahl index, exposure limits, granularity analysis

## 2. Comparative Framing
| Metric | Calculation | Interpretation | Use |
|--------|------------|-----------------|-----|
| **Single-Name Limit** | Max exposure per borrower | Direct size constraint | Risk management |
| **Herfindahl Index** | ∑(w_i²) | Overall concentration (0-1) | Portfolio comparison |
| **Gini Coefficient** | Lorenz curve area | Inequality in exposures | Regulation |
| **Numbers Equivalent** | 1/HHI | Effective # independent exposures | Diversification measure |

## 3. Examples + Counterexamples

**Simple Example:**  
Portfolio A: $1B spread over 1000 loans; Portfolio B: $1B with 10 loans. Portfolio B has 100x higher concentration risk

**Failure Case:**  
Lehman Brothers: Major bank concentration risk went unnoticed; default wiped out counterparties' capital

**Edge Case:**  
Perfectly diversified portfolio with correlated defaults (crisis); concentration looks low but tail risk extremely high

## 4. Layer Breakdown
```
Concentration Risk Framework:
├─ Concentration Metrics:
│   ├─ Single-name exposure: Largest exposure as % of capital
│   ├─ Sector exposure: Total exposure to one industry
│   ├─ Geographic exposure: Regional or country concentration
│   ├─ Counterparty concentration: Derivatives counterparty risk
│   └─ Collateral concentration: Similar security types
├─ Herfindahl-Hirschman Index (HHI):
│   ├─ Definition: HHI = ∑ᵢ (Wᵢ)² where Wᵢ = weight of exposure i
│   ├─ Range: 1/n (perfect diversification) to 1 (single exposure)
│   ├─ Interpretation: HHI > 0.25 typically considered concentrated
│   └─ Regulatory: Used in merger analysis, capital rules
├─ Gini Coefficient:
│   ├─ Definition: Based on Lorenz curve (cumulative % exposures)
│   ├─ Range: 0 (equal distribution) to 1 (single exposure)
│   ├─ Formula: G = 1 - 2 ∑ᵢ (1/n) × Lorenz(i)
│   └─ Interpretation: Higher = more concentrated
├─ Numbers Equivalent (N_eq):
│   ├─ N_eq = 1 / ∑ᵢ (wᵢ)² = 1 / HHI
│   ├─ Interpretation: Portfolio equivalent to N_eq equally-sized exposures
│   ├─ Example: N_eq = 50 means portfolio = 50 diversified exposures
│   └─ Regulatory use: Basel III granularity adjustment
├─ Large Exposure Limits:
│   ├─ Single-name: Typically 10-15% of capital
│   ├─ Sector cap: 25-50% of capital
│   ├─ Related parties: Aggregate all connections
│   ├─ Collateral haircuts: Value adjustments for concentration
│   └─ Stress scenarios: Apply losses to concentrated positions
├─ Concentration Risk Sources:
│   ├─ Business model: Bank lending inherently concentrated
│   ├─ Portfolio drift: Active management changes concentrations
│   ├─ Correlation clustering: Related defaults cluster
│   ├─ Maturity mismatch: All exposures mature together
│   └─ Crisis amplification: Concentration effects magnify in downturns
├─ Diversification Benefits:
│   ├─ Naive: Portfolio risk ↓ as exposures increase
│   ├─ Granularity: Diminishing returns after N_eq ≈ 50-100
│   ├─ Correlation benefit: Decreases in crisis
│   └─ Ultimate limit: Systematic risk cannot be diversified away
└─ Regulatory Framework:
    ├─ Standardized approach: Single-name limits (25% of capital)
    ├─ IRB approach: Granularity adjustment (pg)
    ├─ Large exposures: EU limit 25% on single name
    └─ Systemic: Systemically important institution limits
```

## 5. Mini-Project
Analyze portfolio concentration:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("=== Portfolio Concentration Risk Analysis ===")

# Create three portfolios with different concentration levels
print("\n=== Portfolio Configurations ===")

# Portfolio 1: Highly concentrated (10 large exposures)
portfolio_1_size = 10
portfolio_1 = pd.DataFrame({
    'Portfolio': 'Concentrated',
    'Exposure': np.random.lognormal(16, 0.5, portfolio_1_size),
    'Borrower': [f'Large_{i}' for i in range(portfolio_1_size)]
})

# Portfolio 2: Moderate concentration (100 medium exposures)
portfolio_2_size = 100
portfolio_2 = pd.DataFrame({
    'Portfolio': 'Moderate',
    'Exposure': np.random.lognormal(12, 1.0, portfolio_2_size),
    'Borrower': [f'Medium_{i}' for i in range(portfolio_2_size)]
})

# Portfolio 3: Well-diversified (1000 small exposures)
portfolio_3_size = 1000
portfolio_3 = pd.DataFrame({
    'Portfolio': 'Diversified',
    'Exposure': np.random.lognormal(8, 1.5, portfolio_3_size),
    'Borrower': [f'Small_{i}' for i in range(portfolio_3_size)]
})

# Combine and normalize
portfolios = pd.concat([portfolio_1, portfolio_2, portfolio_3], ignore_index=True)

# Scale so total exposure = $1B for each
for portfolio_name in portfolios['Portfolio'].unique():
    mask = portfolios['Portfolio'] == portfolio_name
    total = portfolios.loc[mask, 'Exposure'].sum()
    portfolios.loc[mask, 'Exposure'] = portfolios.loc[mask, 'Exposure'] / total * 1e9

print("Portfolio Summary:")
for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    print(f"\n{portfolio_name}:")
    print(f"  Number of exposures: {len(subset)}")
    print(f"  Total exposure: ${subset['Exposure'].sum()/1e9:.1f}B")
    print(f"  Average exposure: ${subset['Exposure'].mean()/1e6:.1f}M")
    print(f"  Largest exposure: ${subset['Exposure'].max()/1e6:.1f}M")
    print(f"  Smallest exposure: ${subset['Exposure'].min()/1e6:.1f}M")

# Calculate concentration metrics
print("\n=== Concentration Metrics ===")

metrics_list = []
for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    
    # Weights
    weights = subset['Exposure'].values / subset['Exposure'].sum()
    
    # Herfindahl-Hirschman Index
    hhi = np.sum(weights**2)
    
    # Numbers Equivalent
    n_eq = 1 / hhi if hhi > 0 else np.inf
    
    # Gini Coefficient
    sorted_exposures = np.sort(subset['Exposure'].values)
    cumsum = np.cumsum(sorted_exposures)
    n = len(sorted_exposures)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_exposures)) / (n * cumsum[-1]) - (n + 1) / n
    
    # Lorenz curve
    lorenz_x = np.arange(0, n+1) / n
    lorenz_y = np.concatenate([[0], cumsum / cumsum[-1]])
    
    # Largest exposure as % of total
    top1_pct = weights.max() * 100
    top10_pct = weights[np.argsort(weights)[-10:] if len(weights) >= 10 else :].sum() * 100
    
    metrics_list.append({
        'Portfolio': portfolio_name,
        'HHI': hhi,
        'N_Equivalent': n_eq,
        'Gini': gini,
        'Top 1 %': top1_pct,
        'Top 10 %': top10_pct
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df.to_string(index=False))

# Loss scenarios by concentration
print("\n=== Loss Scenarios: Concentration Impact ===")

# Assume PDs: large borrowers 2%, medium 3%, small 5%
loss_scenarios = []

for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    
    if portfolio_name == 'Concentrated':
        pd_vals = np.full(len(subset), 0.02)
    elif portfolio_name == 'Moderate':
        pd_vals = np.full(len(subset), 0.03)
    else:
        pd_vals = np.full(len(subset), 0.05)
    
    lgd = 0.40  # Uniform LGD
    
    # Single default scenario: largest exposure defaults
    largest_idx = subset['Exposure'].idxmax()
    loss_largest = subset.loc[largest_idx, 'Exposure'] * lgd
    loss_largest_pct = loss_largest / 1e9 * 100
    
    # Top 5 defaults scenario
    top_5_idx = subset.nlargest(5, 'Exposure').index
    loss_top5 = (subset.loc[top_5_idx, 'Exposure'] * lgd).sum()
    loss_top5_pct = loss_top5 / 1e9 * 100
    
    # 5% default rate scenario (uniform)
    loss_5pct = (subset['Exposure'] * pd_vals * lgd).sum()
    loss_5pct_pct = loss_5pct / 1e9 * 100
    
    loss_scenarios.append({
        'Portfolio': portfolio_name,
        'Largest Default': f'${loss_largest_pct:.2f}%',
        'Top 5 Default': f'${loss_top5_pct:.2f}%',
        '5% Default Rate': f'${loss_5pct_pct:.2f}%'
    })

loss_df = pd.DataFrame(loss_scenarios)
print(loss_df.to_string(index=False))

# Regulatory limits and exposure utilization
print("\n=== Regulatory Capital Utilization ===")

capital = 1e9  # $1B capital
large_exposure_limit = 0.25 * capital  # 25% limit

for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    
    # How many exposures exceed single-name limit?
    exceeding = (subset['Exposure'] > large_exposure_limit).sum()
    
    # Cumulative exposure of top N exposures
    for n in [1, 5, 10]:
        if len(subset) >= n:
            top_n_exp = subset.nlargest(n, 'Exposure')['Exposure'].sum()
            top_n_pct = top_n_exp / 1e9 * 100
            print(f"{portfolio_name}: Top {n} exposures = {top_n_pct:.1f}% of portfolio")

# Granularity adjustment (Basel III)
print("\n=== Basel III Granularity Adjustment ===")

# Granularity effect: pg = (1 - exp(-2×HHI)) / (2×HHI)
for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    hhi = metrics_df[metrics_df['Portfolio'] == portfolio_name]['HHI'].values[0]
    
    if hhi > 0:
        pg = (1 - np.exp(-2 * hhi)) / (2 * hhi)
    else:
        pg = 0
    
    # Capital increase due to granularity
    # K = K_uniform + pg (simplified)
    capital_add = pg * 100  # As percentage of capital
    print(f"{portfolio_name}: Granularity adjustment = {capital_add:.2f}pp of capital")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Exposure distribution
ax1 = axes[0, 0]
for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    ax1.hist(subset['Exposure']/1e6, bins=30, alpha=0.5, label=portfolio_name, edgecolor='black')

ax1.set_xlabel('Exposure ($M)')
ax1.set_ylabel('Frequency')
ax1.set_title('Exposure Size Distribution')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Lorenz curve
ax2 = axes[0, 1]
for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    weights = subset['Exposure'].values / subset['Exposure'].sum()
    sorted_exposures = np.sort(subset['Exposure'].values)
    cumsum = np.cumsum(sorted_exposures)
    n = len(sorted_exposures)
    lorenz_x = np.arange(0, n+1) / n
    lorenz_y = np.concatenate([[0], cumsum / cumsum[-1]])
    ax2.plot(lorenz_x, lorenz_y, linewidth=2, label=portfolio_name, marker='o', markersize=3)

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect equality')
ax2.set_xlabel('Cumulative % of Exposures')
ax2.set_ylabel('Cumulative % of Total Exposure')
ax2.set_title('Lorenz Curve (Concentration)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: HHI and N_Equivalent
ax3 = axes[0, 2]
x_pos = np.arange(len(metrics_df))
ax3_1 = ax3
ax3_2 = ax3.twinx()

bars = ax3_1.bar(x_pos - 0.2, metrics_df['HHI'], 0.4, label='HHI', alpha=0.7, edgecolor='black')
ax3_2.plot(x_pos, metrics_df['N_Equivalent'], 'ro-', linewidth=2, markersize=8, label='N_Equivalent')

ax3_1.set_ylabel('Herfindahl Index (HHI)')
ax3_2.set_ylabel('Numbers Equivalent (N_eq)')
ax3_1.set_xticks(x_pos)
ax3_1.set_xticklabels(metrics_df['Portfolio'])
ax3.set_title('Concentration Metrics')
ax3_1.grid(True, alpha=0.3, axis='y')

# Plot 4: Loss scenarios
ax4 = axes[1, 0]
x = np.arange(len(loss_scenarios))
width = 0.25

# Extract numeric values from loss scenarios
loss_largest_vals = [float(loss_scenarios[i]['Largest Default'].replace('$', '').replace('%', '')) for i in range(len(loss_scenarios))]
loss_top5_vals = [float(loss_scenarios[i]['Top 5 Default'].replace('$', '').replace('%', '')) for i in range(len(loss_scenarios))]

ax4.bar(x - width, loss_largest_vals, width, label='Largest Default', alpha=0.7, edgecolor='black')
ax4.bar(x, loss_top5_vals, width, label='Top 5 Default', alpha=0.7, edgecolor='black')
ax4.set_ylabel('Loss (% of $1B Portfolio)')
ax4.set_title('Stress Scenarios: Impact of Concentration')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_df['Portfolio'])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Top exposures cumulative %
ax5 = axes[1, 1]
for portfolio_name in ['Concentrated', 'Moderate', 'Diversified']:
    subset = portfolios[portfolios['Portfolio'] == portfolio_name]
    top_n_exposures = np.arange(1, min(51, len(subset)+1))
    top_n_pct = []
    
    for n in top_n_exposures:
        top_n = subset.nlargest(n, 'Exposure')['Exposure'].sum() / 1e9 * 100
        top_n_pct.append(top_n)
    
    ax5.plot(top_n_exposures, top_n_pct, linewidth=2, label=portfolio_name, marker='o', markersize=3)

ax5.set_xlabel('Number of Top Exposures')
ax5.set_ylabel('Cumulative % of Portfolio')
ax5.set_title('Cumulative Exposure Concentration')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Gini Coefficient comparison
ax6 = axes[1, 2]
bars = ax6.bar(metrics_df['Portfolio'], metrics_df['Gini'], 
              color=['red', 'yellow', 'green'], alpha=0.7, edgecolor='black')
ax6.set_ylabel('Gini Coefficient')
ax6.set_ylim([0, 1])
ax6.set_title('Portfolio Inequality\n(Gini: 0=equal, 1=concentrated)')
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n=== Concentration Risk Summary ===")
print(f"Concentration risk quantifies portfolio vulnerability to idiosyncratic shocks")
print(f"HHI, Gini, and N_eq provide complementary perspectives on diversification")
print(f"Regulatory limits and granularity adjustments address concentration in capital rules")
```

## 6. Challenge Round
When is concentration analysis problematic?
- **Correlation masking**: Two "independent" exposures may default together (sector, geographic)
- **Wrong aggregation**: Single-name limit prevents individual concentrations but allows sector concentration
- **Collateral correlation**: Diversified borrowers but same collateral type (real estate); concentrated collateral risk
- **Liquidity**: Diversified portfolio hard to liquidate if all exposures illiquid; fire-sale losses
- **Dynamic effects**: Concentration changes over time as positions mature, new loans added, prepayments occur

## 7. Key References
- [Herfindahl Index](https://en.wikipedia.org/wiki/Herfindahl_index) - Standard concentration metric
- [Basel III Granularity Adjustment](https://www.bis.org/basel_framework/chapter/CRE/20.htm) - Regulatory treatment
- [Concentration Limits](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32015L0035) - EU Large Exposures Regulation

---
**Status:** Key portfolio risk driver, fundamental to diversification | **Complements:** Credit VaR, Correlation, Portfolio limits
