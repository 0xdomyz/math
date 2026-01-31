"""
Granularity Adjustment (Pg) - Basel III Capital Add-on

Capital add-on accounting for finite portfolio effects; adjusts from infinite (granular) 
portfolio assumption to realistic finite portfolio with large exposures.

References:
- Basel III IRB pg Formula: https://www.bis.org/basel_framework/chapter/CRE/20.htm
- Granularity Effect Theory: https://www.bis.org/publ/work155.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

np.random.seed(42)

print("=== Granularity Adjustment (pg) Calculation ===")

# Create portfolios with varying concentration
print("\n=== Portfolio Granularity Analysis ===")

portfolios_config = {
    'Highly Granular': {
        'n_exposures': 1000,
        'size_distribution': 'lognormal',
        'params': (8, 1.5)
    },
    'Moderately Granular': {
        'n_exposures': 100,
        'size_distribution': 'lognormal',
        'params': (12, 1.0)
    },
    'Concentrated': {
        'n_exposures': 20,
        'size_distribution': 'lognormal',
        'params': (14, 0.8)
    },
    'Highly Concentrated': {
        'n_exposures': 5,
        'size_distribution': 'lognormal',
        'params': (15, 0.5)
    }
}

results = []

for portfolio_name, config in portfolios_config.items():
    n = config['n_exposures']
    
    # Generate exposures
    exposures = np.random.lognormal(config['params'][0], config['params'][1], n)
    exposures = exposures / exposures.sum() * 1e9  # Normalize to $1B
    
    # Calculate concentration metrics
    weights = exposures / exposures.sum()
    hhi = np.sum(weights**2)
    n_eq = 1 / hhi if hhi > 0 else np.inf
    
    # Basel III granularity adjustment formula
    if hhi > 0:
        pg_basel = (1 - np.exp(-2 * hhi)) / (2 * hhi)
    else:
        pg_basel = 0
    
    # Simplified linear approximation (for comparison)
    pg_linear = 0.5 * hhi
    
    # Add more sophisticated adjustment (accounts for correlation)
    rho = 0.30  # Assumed correlation
    pg_merton = hhi * (1 + rho * np.sqrt(1 - rho**2))  # Approximation
    
    results.append({
        'Portfolio': portfolio_name,
        'N_Exposures': n,
        'N_Equivalent': n_eq,
        'HHI': hhi,
        'Largest_Exposure_%': weights.max() * 100,
        'pg_Basel': pg_basel,
        'pg_Linear': pg_linear,
        'pg_Merton': pg_merton,
        'Top_5_%': weights[np.argsort(weights)[-5:] if n >= 5 else :].sum() * 100
    })

results_df = pd.DataFrame(results)

print("\nGranularity Adjustment Results:")
print(results_df[['Portfolio', 'N_Exposures', 'N_Equivalent', 'HHI', 'pg_Basel']].to_string(index=False))

print("\n=== pg Comparison Across Methods ===")
print(results_df[['Portfolio', 'pg_Basel', 'pg_Linear', 'pg_Merton']].to_string(index=False))

# Capital impact calculation
print("\n=== Capital Impact of Granularity Adjustment ===")

base_capital_ratio = 0.08  # 8% minimum capital ratio
total_exposure = 1e9  # $1B

for idx, row in results_df.iterrows():
    # Capital without adjustment
    capital_no_adj = total_exposure * base_capital_ratio
    
    # Capital with adjustment
    pg_adjustment = row['pg_Basel'] / 100  # Convert from % to decimal
    capital_with_adj = capital_no_adj + (total_exposure * pg_adjustment)
    
    # Effective capital ratio
    effective_ratio = capital_with_adj / total_exposure
    
    # Capital increase %
    capital_increase_pct = (capital_with_adj / capital_no_adj - 1) * 100
    
    print(f"\n{row['Portfolio']}:")
    print(f"  Base capital required: ${capital_no_adj/1e6:.1f}M ({base_capital_ratio*100:.1f}%)")
    print(f"  Granularity adjustment: ${total_exposure * pg_adjustment/1e6:.1f}M ({pg_adjustment*100:.2f}pp)")
    print(f"  Total capital required: ${capital_with_adj/1e6:.1f}M ({effective_ratio*100:.2f}%)")
    print(f"  Capital increase: {capital_increase_pct:.1f}%")

# Sensitivity analysis: pg vs HHI
print("\n=== Sensitivity Analysis: pg vs Concentration Metrics ===")

hhi_range = np.linspace(0.001, 0.2, 50)
pg_range = []

for hhi in hhi_range:
    pg = (1 - np.exp(-2 * hhi)) / (2 * hhi)
    pg_range.append(pg)

# Find typical thresholds
hhi_granular = 0.01  # Granular threshold
pg_granular = (1 - np.exp(-2 * hhi_granular)) / (2 * hhi_granular)

hhi_moderate = 0.05  # Moderate threshold
pg_moderate = (1 - np.exp(-2 * hhi_moderate)) / (2 * hhi_moderate)

print(f"\nTypical Thresholds:")
print(f"Granular portfolio (HHI ≤ {hhi_granular}): pg ≈ {pg_granular*100:.2f}%")
print(f"Moderate portfolio (HHI ≈ {hhi_moderate}): pg ≈ {pg_moderate*100:.2f}%")

# Multi-period granularity adjustment
print("\n=== Maturity Effect on Granularity Adjustment ===")

maturities = [1, 3, 5, 10]  # Years
maturity_effect = 1.0  # Baseline at 1 year

print(f"Maturity | pg Factor | pg Increase")
print("-" * 40)

for t in maturities:
    # Granularity increases with maturity (more opportunity for correlation to manifest)
    maturity_factor = np.sqrt(t)  # Simplification: √t adjustment
    pg_adjusted = results_df.iloc[0]['pg_Basel'] * maturity_factor
    
    print(f"{t:2d} years | {maturity_factor:9.2f}x | {pg_adjusted:10.2%}")

# Scenario analysis: Correlation dependency
print("\n=== Granularity Adjustment vs Default Correlation ===")

hhi_fixed = 0.02  # Fixed concentration
correlations = np.linspace(0, 0.5, 10)

print(f"Correlation | pg Adjustment")
print("-" * 35)

for rho in correlations:
    # Merton adjustment: pg includes correlation effect
    pg_adjusted = hhi_fixed * (1 + rho * np.sqrt(1 - rho**2))
    print(f"{rho:10.2f} | {pg_adjusted:13.2%}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: pg vs HHI
ax1 = axes[0, 0]
ax1.plot(hhi_range, np.array(pg_range)*100, linewidth=2)
ax1.axvline(hhi_granular, color='g', linestyle='--', alpha=0.5, label=f'Granular (HHI={hhi_granular})')
ax1.axvline(hhi_moderate, color='orange', linestyle='--', alpha=0.5, label=f'Moderate (HHI={hhi_moderate})')
ax1.fill_between(hhi_range, 0, np.array(pg_range)*100, alpha=0.2)
ax1.set_xlabel('Herfindahl Index (HHI)')
ax1.set_ylabel('Granularity Adjustment pg (%)')
ax1.set_title('Basel III pg Formula\nincreases with concentration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: pg comparison across portfolios
ax2 = axes[0, 1]
x_pos = np.arange(len(results_df))
width = 0.25

ax2.bar(x_pos - width, results_df['pg_Basel']*100, width, label='Basel III', alpha=0.7, edgecolor='black')
ax2.bar(x_pos, results_df['pg_Linear']*100, width, label='Linear', alpha=0.7, edgecolor='black')
ax2.bar(x_pos + width, results_df['pg_Merton']*100, width, label='Merton', alpha=0.7, edgecolor='black')

ax2.set_ylabel('pg (%)')
ax2.set_title('Granularity Adjustment by Method')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['Portfolio'], rotation=45, ha='right', fontsize=9)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Capital impact
ax3 = axes[0, 2]
capital_no_adj = results_df['HHI'] * 0  # Baseline
capital_with_adj = results_df['pg_Basel']

ax3.scatter(results_df['HHI']*100, capital_with_adj*100, s=200, alpha=0.6, edgecolors='black', linewidth=2)

for idx, row in results_df.iterrows():
    ax3.annotate(row['Portfolio'], 
                (row['HHI']*100, row['pg_Basel']*100),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax3.set_xlabel('HHI (%)')
ax3.set_ylabel('pg (%)')
ax3.set_title('Capital Add-on vs Concentration')
ax3.grid(True, alpha=0.3)

# Plot 4: N_Equivalent vs pg
ax4 = axes[1, 0]
n_eq_range = results_df['N_Equivalent'].values
pg_vals = results_df['pg_Basel'].values * 100

ax4.scatter(n_eq_range, pg_vals, s=200, alpha=0.6, edgecolors='black', linewidth=2)

for idx, row in results_df.iterrows():
    ax4.annotate(row['Portfolio'], 
                (row['N_Equivalent'], row['pg_Basel']*100),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.set_xlabel('Numbers Equivalent (N_eq)')
ax4.set_ylabel('pg (%)')
ax4.set_title('Granularity Adjustment vs Diversification')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

# Plot 5: Capital ratio impact
ax5 = axes[1, 1]
base_ratios = np.ones(len(results_df)) * 8.0
adjusted_ratios = base_ratios + results_df['pg_Basel']*100

x_port = np.arange(len(results_df))
width = 0.35

ax5.bar(x_port - width/2, base_ratios, width, label='Base (8%)', alpha=0.7, edgecolor='black')
ax5.bar(x_port + width/2, adjusted_ratios, width, label='With pg Adjustment', alpha=0.7, edgecolor='black')

ax5.set_ylabel('Capital Ratio (%)')
ax5.set_title('Effective Capital Ratio with Granularity')
ax5.set_xticks(x_port)
ax5.set_xticklabels(results_df['Portfolio'], rotation=45, ha='right', fontsize=9)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Maturity effect
ax6 = axes[1, 2]
maturity_years = np.array(maturities)
maturity_factors = np.sqrt(maturity_years)
pg_maturity_adjusted = results_df.iloc[0]['pg_Basel'] * maturity_factors

ax6.plot(maturity_years, pg_maturity_adjusted*100, 'o-', linewidth=2, markersize=8)
ax6.fill_between(maturity_years, 0, pg_maturity_adjusted*100, alpha=0.2)
ax6.set_xlabel('Maturity (Years)')
ax6.set_ylabel('Adjusted pg (%)')
ax6.set_title('Maturity Effect on pg\n(higher pg for longer horizons)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Granularity Adjustment Summary ===")
print(f"pg = (1 - exp(-2×HHI)) / (2×HHI)")
print(f"pg captures finite portfolio effects not in granular models")
print(f"Regulatory capital = Base capital + pg × exposure")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Granularity Adjustment Analysis Complete")
    print("="*70)
