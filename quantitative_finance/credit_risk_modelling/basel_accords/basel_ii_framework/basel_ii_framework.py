# Auto-extracted from markdown file
# Source: basel_ii_framework.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bank portfolio (simplified)
portfolio = [
    {'description': 'US Treasury Bonds', 'exposure': 50, 'rating': 'AAA', 'rw': 0.00},
    {'description': 'Corporate Bonds (A-rated)', 'exposure': 100, 'rating': 'A', 'rw': 0.50},
    {'description': 'Corporate Bonds (BBB-rated)', 'exposure': 80, 'rating': 'BBB', 'rw': 1.00},
    {'description': 'Mortgages (Prime)', 'exposure': 200, 'rating': 'Secured', 'rw': 0.35},
    {'description': 'Mortgages (Subprime)', 'exposure': 150, 'rating': 'Secured (Low LTV)', 'rw': 0.75},
    {'description': 'Business Loans (Small firms)', 'exposure': 120, 'rating': 'Unrated', 'rw': 1.00},
    {'description': 'Derivatives (Credit exposure)', 'exposure': 30, 'rating': 'BBB', 'rw': 1.00},
    {'description': 'Equity Holdings', 'exposure': 60, 'rating': 'Unrated', 'rw': 1.50},
]

df = pd.DataFrame(portfolio)

# Calculate risk-weighted assets
df['risk_weighted_exposure'] = df['exposure'] * df['rw']

# Minimum capital ratio (Pillar I)
min_capital_ratio = 0.08

# Calculate capital requirements
df['capital_required'] = df['risk_weighted_exposure'] * min_capital_ratio

# Summary calculations
total_exposure = df['exposure'].sum()
total_rwa = df['risk_weighted_exposure'].sum()
total_capital_required = df['capital_required'].sum()

print("="*100)
print("BASEL II STANDARDIZED APPROACH - CAPITAL REQUIREMENT CALCULATION")
print("="*100)
print(f"\nBank Portfolio (Millions USD):\n")
print(df[['description', 'exposure', 'rating', 'rw', 'risk_weighted_exposure', 'capital_required']].to_string(index=False))

print(f"\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"Total Exposure: ${total_exposure:.1f}M")
print(f"Total Risk-Weighted Assets (RWA): ${total_rwa:.1f}M")
print(f"Risk-Weighted Ratio: {total_rwa / total_exposure * 100:.1f}% (overall risk weight)")
print(f"Minimum Capital Requirement (8% of RWA): ${total_capital_required:.1f}M")
print(f"Required Capital Ratio: {total_capital_required / total_exposure * 100:.1f}% of total exposure")

# Scenario: Adding Pillar II buffer
pillar_ii_buffer = 0.025  # 2.5% for supervisory discretion
pillar_ii_capital = (min_capital_ratio + pillar_ii_buffer) * total_rwa

print(f"\nWith Pillar II Guidance Buffer ({pillar_ii_buffer*100:.1f}%): ${pillar_ii_capital:.1f}M")
print(f"Capital Ratio (Pillar I + Pillar II): {(min_capital_ratio + pillar_ii_buffer)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Risk weight distribution
ax = axes[0, 0]
colors_rw = plt.cm.RdYlGn_r(df['rw'] / df['rw'].max())
ax.barh(df['description'], df['rw'], color=colors_rw, alpha=0.8)
ax.set_xlabel('Risk Weight')
ax.set_title('Risk Weight by Asset Type (Basel II Standardized)')
ax.grid(alpha=0.3, axis='x')

# Plot 2: Risk-weighted assets breakdown
ax = axes[0, 1]
colors_assets = plt.cm.Set3(np.linspace(0, 1, len(df)))
ax.pie(df['risk_weighted_exposure'], labels=df['description'], autopct='%1.1f%%',
      colors=colors_assets, startangle=90)
ax.set_title('Risk-Weighted Assets (RWA) Composition')

# Plot 3: Capital requirement by asset
ax = axes[1, 0]
ax.bar(range(len(df)), df['capital_required'], color=colors_assets, alpha=0.8)
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['description'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Capital Required ($M)')
ax.set_title('Capital Requirement by Asset Type')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Exposure vs Risk-Weighted Assets
ax = axes[1, 1]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['exposure'], width, label='Total Exposure', alpha=0.8)
ax.bar(x + width/2, df['risk_weighted_exposure'], width, label='Risk-Weighted Assets', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df['description'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Amount ($M)')
ax.set_title('Exposure vs Risk-Weighted Assets')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Sensitivity: What if risk weights change?
print(f"\n" + "="*100)
print("SENSITIVITY ANALYSIS: Impact of Risk Weight Changes")
print("="*100)

scenarios = {
    'Baseline': 1.0,
    'Conservative (+50% RW)': 1.5,
    'Aggressive (-30% RW)': 0.7,
}

for scenario_name, multiplier in scenarios.items():
    rwa_scenario = (df['rw'] * multiplier * df['exposure']).sum()
    capital_scenario = rwa_scenario * min_capital_ratio
    print(f"{scenario_name}: RWA = ${rwa_scenario:.1f}M, Capital = ${capital_scenario:.1f}M")

