# Auto-extracted from markdown file
# Source: expected_loss.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# Portfolio construction
print("=== Portfolio Expected Loss Analysis ===")
n_loans = 500

portfolio = pd.DataFrame({
    'Loan_ID': np.arange(n_loans),
    'Segment': np.random.choice(['Corporate', 'SME', 'Retail'], n_loans, p=[0.3, 0.4, 0.3]),
    'Size': np.random.lognormal(12, 1.5, n_loans),
})

# Assign risk parameters by segment
segment_params = {
    'Corporate': {'PD': 0.015, 'LGD': 0.35, 'Volatility': 0.08},
    'SME': {'PD': 0.035, 'LGD': 0.45, 'Volatility': 0.12},
    'Retail': {'PD': 0.025, 'LGD': 0.40, 'Volatility': 0.10}
}

portfolio['PD'] = portfolio['Segment'].map(lambda x: segment_params[x]['PD'])
portfolio['LGD'] = portfolio['Segment'].map(lambda x: segment_params[x]['LGD'])
portfolio['Volatility'] = portfolio['Segment'].map(lambda x: segment_params[x]['Volatility'])

# EAD = loan size (simplified)
portfolio['EAD'] = portfolio['Size']

# Calculate individual expected loss
portfolio['EL_Individual'] = portfolio['PD'] * portfolio['LGD'] * portfolio['EAD']

print(f"\nPortfolio size: ${portfolio['Size'].sum()/1e6:.1f}M")
print(f"Number of loans: {len(portfolio)}")
print(f"Average loan size: ${portfolio['Size'].mean()/1e3:.0f}K")

# Summary by segment
print("\n=== Expected Loss by Segment ===")
for segment in ['Corporate', 'SME', 'Retail']:
    subset = portfolio[portfolio['Segment'] == segment]
    print(f"\n{segment}:")
    print(f"  Count: {len(subset)}")
    print(f"  Portfolio: ${subset['Size'].sum()/1e6:.1f}M")
    print(f"  Avg PD: {subset['PD'].mean():.2%}")
    print(f"  Avg LGD: {subset['LGD'].mean():.1%}")
    print(f"  Total EL: ${subset['EL_Individual'].sum()/1e3:.0f}K")
    print(f"  EL% of Portfolio: {subset['EL_Individual'].sum() / subset['Size'].sum():.2%}")

# Portfolio-level EL (simple sum - assumes independence)
portfolio_el_independent = portfolio['EL_Individual'].sum()

# Portfolio EL with correlation (simplified Vasicek model)
# EL_correlated = EL_independent × (1 + correlation adjustment)
# More precise: Use correlation matrix

correlation_within_segment = 0.30
correlation_cross_segment = 0.10

# Calculate correlated portfolio EL using systematic risk factor
def calculate_correlated_el(portfolio_data, within_corr, cross_corr):
    """
    Simplified calculation using Vasicek model
    EL_portfolio ≈ EL_independent × √(1 + ρ(n-1))
    For portfolio with multiple segments, weight by segment
    """
    total_el_ind = portfolio_data['EL_Individual'].sum()
    
    # Adjust for correlation - conservatively assume some systematic risk
    correlation_factor = 1 + within_corr * 0.5  # 50% of within-segment correlation
    el_correlated = total_el_ind * correlation_factor
    
    return el_correlated

portfolio_el_correlated = calculate_correlated_el(portfolio, correlation_within_segment, correlation_cross_segment)
el_increase_pct = (portfolio_el_correlated - portfolio_el_independent) / portfolio_el_independent * 100

print(f"\n=== Expected Loss Aggregation ===")
print(f"Sum of individual ELs: ${portfolio_el_independent/1e3:.0f}K")
print(f"Correlated portfolio EL: ${portfolio_el_correlated/1e3:.0f}K")
print(f"Correlation effect: +{el_increase_pct:.1f}%")

# Multi-year expected loss (cumulative)
print("\n=== Multi-Year Expected Loss ===")
years = [1, 3, 5]
for year in years:
    # Assume PD is annual, cumulative over years
    cumulative_pd = 1 - (1 - portfolio['PD']) ** year
    annual_el = (cumulative_pd * portfolio['LGD'] * portfolio['EAD']).sum()
    print(f"{year}-year cumulative EL: ${annual_el/1e3:.0f}K")

# Stress scenario analysis
print("\n=== Stress Scenario EL ===")
scenarios = {
    'Base Case': {'pd_mult': 1.0, 'lgd_mult': 1.0},
    'Mild Stress': {'pd_mult': 1.5, 'lgd_mult': 1.1},
    'Severe Stress': {'pd_mult': 3.0, 'lgd_mult': 1.3},
    'Crisis': {'pd_mult': 5.0, 'lgd_mult': 1.5}
}

stress_results = []
for scenario_name, multipliers in scenarios.items():
    stressed_pd = portfolio['PD'] * multipliers['pd_mult']
    stressed_lgd = np.minimum(portfolio['LGD'] * multipliers['lgd_mult'], 1.0)
    stressed_el = (stressed_pd * stressed_lgd * portfolio['EAD']).sum()
    stress_results.append({
        'Scenario': scenario_name,
        'Total EL': stressed_el,
        'EL % of Portfolio': stressed_el / portfolio['Size'].sum() * 100,
        'Capital Required (8% min)': stressed_el / 0.08
    })

stress_df = pd.DataFrame(stress_results)
print(stress_df.to_string(index=False))

# Concentration analysis
print("\n=== Top 10 Loans by EL Contribution ===")
top_loans = portfolio.nlargest(10, 'EL_Individual')[['Loan_ID', 'Segment', 'Size', 'PD', 'LGD', 'EL_Individual']]
top_loans['EL_Contribution_%'] = (top_loans['EL_Individual'] / portfolio_el_independent * 100).round(2)
print(top_loans.to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: EL by segment
ax1 = axes[0, 0]
segment_el = portfolio.groupby('Segment')['EL_Individual'].sum()
segment_names = segment_el.index
segment_values = segment_el.values
colors = ['steelblue', 'orange', 'green']
ax1.bar(segment_names, segment_values/1e3, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Expected Loss ($K)')
ax1.set_title('EL by Segment')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: PD vs LGD scatter
ax2 = axes[0, 1]
for segment, color in zip(['Corporate', 'SME', 'Retail'], colors):
    subset = portfolio[portfolio['Segment'] == segment]
    ax2.scatter(subset['PD']*100, subset['LGD']*100, s=subset['Size']/5000, 
               alpha=0.5, label=segment, color=color, edgecolors='black')
ax2.set_xlabel('PD (%)')
ax2.set_ylabel('LGD (%)')
ax2.set_title('Risk Profile by Loan\n(Bubble size = loan size)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: EL correlation impact
ax3 = axes[0, 2]
correlations = np.linspace(0, 0.5, 20)
el_by_corr = []
for corr in correlations:
    el_corr = calculate_correlated_el(portfolio, corr, corr*0.5)
    el_by_corr.append(el_corr)

ax3.plot(correlations, np.array(el_by_corr)/1e3, linewidth=2, label='Portfolio EL')
ax3.axhline(portfolio_el_independent/1e3, color='r', linestyle='--', 
           linewidth=2, label='Independent (ρ=0)')
ax3.set_xlabel('Correlation')
ax3.set_ylabel('Expected Loss ($K)')
ax3.set_title('EL Sensitivity to Correlation\n(Higher correlation → Higher EL)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: EL contributions
ax4 = axes[1, 0]
portfolio_sorted = portfolio.sort_values('EL_Individual', ascending=False)
cumulative_el = np.cumsum(portfolio_sorted['EL_Individual']) / portfolio['EL_Individual'].sum() * 100
ax4.plot(range(len(portfolio_sorted)), cumulative_el, linewidth=2)
ax4.axhline(80, color='r', linestyle='--', alpha=0.5, label='80% EL')
ax4.axhline(95, color='orange', linestyle='--', alpha=0.5, label='95% EL')
ax4.set_xlabel('Loans (ranked by EL)')
ax4.set_ylabel('Cumulative % of Total EL')
ax4.set_title('EL Concentration\n(Pareto effect)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Stress scenario comparison
ax5 = axes[1, 1]
x_stress = np.arange(len(stress_df))
width = 0.6
colors_stress = ['green', 'yellow', 'orange', 'red']
bars = ax5.bar(stress_df['Scenario'], stress_df['Total EL']/1e3, 
              color=colors_stress, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Expected Loss ($K)')
ax5.set_title('EL Under Stress Scenarios')
ax5.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.0f}K', ha='center', va='bottom')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Multi-year EL
ax6 = axes[1, 2]
years_range = np.arange(1, 11)
el_multiyear = []
for year in years_range:
    cumul_pd = 1 - (1 - portfolio['PD']) ** year
    annual_el = (cumul_pd * portfolio['LGD'] * portfolio['EAD']).sum()
    el_multiyear.append(annual_el)

ax6.plot(years_range, np.array(el_multiyear)/1e3, 'o-', linewidth=2, markersize=6)
ax6.fill_between(years_range, 0, np.array(el_multiyear)/1e3, alpha=0.2)
ax6.set_xlabel('Years')
ax6.set_ylabel('Cumulative Expected Loss ($K)')
ax6.set_title('Multi-Year Expected Loss\n(Increases with horizon)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

