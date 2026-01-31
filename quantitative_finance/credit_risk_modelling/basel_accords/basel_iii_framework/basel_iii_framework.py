# Auto-extracted from markdown file
# Source: basel_iii_framework.md

# --- Code Block 1 ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bank starting position (Basel III)
tier1_capital = 50  # $50B
tier2_capital = 20  # $20B
rwa = 400  # $400B risk-weighted assets
total_assets = 1000  # $1000B total assets

# Baseline ratios
baseline = {
    'CET1': tier1_capital / rwa,
    'Tier1': tier1_capital / rwa,
    'Total': (tier1_capital + tier2_capital) / rwa,
    'Leverage': tier1_capital / total_assets,
}

print("="*100)
print("BASEL III CAPITAL RATIO STRESS TEST")
print("="*100)
print(f"\nBank Starting Position:")
print(f"  Tier 1 Capital: ${tier1_capital}B")
print(f"  Tier 2 Capital: ${tier2_capital}B")
print(f"  Risk-Weighted Assets: ${rwa}B")
print(f"  Total Assets: ${total_assets}B")

print(f"\nBaseline Ratios:")
for metric, value in baseline.items():
    print(f"  {metric}: {value*100:.2f}%")

# Basel III Minimums
minimums = {
    'CET1': 0.045,
    'Tier1': 0.065,
    'Total': 0.105,
    'Leverage': 0.03,
}

# Stress scenarios
scenarios = {
    'Mild Downturn': {
        'tier1_loss': -5,  # -$5B
        'rwa_increase': 50,  # +$50B (asset quality deteriorates)
        'tier2_loss': -2,  # -$2B
    },
    'Moderate Recession': {
        'tier1_loss': -15,
        'rwa_increase': 150,
        'tier2_loss': -5,
    },
    'Severe Crisis': {
        'tier1_loss': -25,
        'rwa_increase': 250,
        'tier2_loss': -15,
    },
}

results = {}

for scenario_name, shocks in scenarios.items():
    t1 = tier1_capital + shocks['tier1_loss']
    t2 = tier2_capital + shocks['tier2_loss']
    rwa_stressed = rwa + shocks['rwa_increase']
    
    ratios = {
        'CET1': t1 / rwa_stressed,
        'Tier1': t1 / rwa_stressed,
        'Total': (t1 + t2) / rwa_stressed,
        'Leverage': t1 / total_assets,
    }
    
    # Check vs minimums
    passes = {k: ratios[k] >= minimums[k] for k in minimums}
    
    results[scenario_name] = {
        'ratios': ratios,
        'passes': passes,
        'capital': t1,
        'rwa': rwa_stressed,
    }

# Print results
print(f"\n" + "="*100)
print("STRESS TEST RESULTS")
print("="*100)

for scenario_name, scenario_results in results.items():
    print(f"\n{scenario_name}:")
    print(f"  Tier 1 Capital: ${scenario_results['capital']:.1f}B")
    print(f"  Risk-Weighted Assets: ${scenario_results['rwa']:.1f}B")
    for metric in minimums.keys():
        stressed_ratio = scenario_results['ratios'][metric]
        minimum_ratio = minimums[metric]
        status = "✓ PASS" if scenario_results['passes'][metric] else "✗ FAIL"
        shortfall = (minimum_ratio - stressed_ratio) * scenario_results['rwa']
        print(f"    {metric}: {stressed_ratio*100:.2f}% (min: {minimum_ratio*100:.2f}%) {status}")
        if not scenario_results['passes'][metric]:
            print(f"      Capital shortfall: ${shortfall:.1f}B")

# Add buffers (Capital Conservation Buffer, CyCB)
print(f"\n" + "="*100)
print("WITH REGULATORY BUFFERS (CCB + CyCB)")
print("="*100)

buffers = {
    'Capital Conservation Buffer (CCB)': 0.025,
    'Countercyclical Buffer (CyCB)': 0.010,
    'G-SIB Buffer': 0.015,
}

total_buffer = sum(buffers.values())
print(f"\nTotal Buffer: {total_buffer*100:.2f}%")

# Check vs total buffer requirement
print(f"\nSevere Crisis Scenario with Buffers:")
stressed_tier1 = results['Severe Crisis']['capital']
stressed_rwa = results['Severe Crisis']['rwa']
total_min_with_buffers = minimums['Total'] + total_buffer
stressed_total_ratio = (stressed_tier1 + tier2_capital + shocks['tier2_loss']) / stressed_rwa

print(f"  Total Capital Ratio (stressed): {stressed_total_ratio*100:.2f}%")
print(f"  Total Requirement (with buffers): {total_min_with_buffers*100:.2f}%")
print(f"  Status: {'✓ PASS' if stressed_total_ratio >= total_min_with_buffers else '✗ FAIL'}")

if stressed_total_ratio < total_min_with_buffers:
    shortfall_amount = (total_min_with_buffers - stressed_total_ratio) * stressed_rwa
    print(f"  Capital raise needed: ${shortfall_amount:.1f}B")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Capital ratios across scenarios
ax = axes[0, 0]
scenarios_list = list(results.keys())
ratios_tier1 = [results[s]['ratios']['Tier1']*100 for s in scenarios_list]
ratios_total = [results[s]['ratios']['Total']*100 for s in scenarios_list]

x = np.arange(len(scenarios_list))
width = 0.35
ax.bar(x - width/2, ratios_tier1, width, label='Tier 1', alpha=0.8)
ax.bar(x + width/2, ratios_total, width, label='Total', alpha=0.8)
ax.axhline(y=minimums['Tier1']*100, color='blue', linestyle='--', linewidth=1, label='Min Tier1')
ax.axhline(y=minimums['Total']*100, color='orange', linestyle='--', linewidth=1, label='Min Total')
ax.set_ylabel('Capital Ratio (%)')
ax.set_title('Capital Ratios Under Stress Scenarios')
ax.set_xticks(x)
ax.set_xticklabels(scenarios_list, rotation=15)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 2: Pass/Fail matrix
ax = axes[0, 1]
pass_fail_matrix = []
for scenario_name in scenarios_list:
    scenario_results = results[scenario_name]
    pass_fail_matrix.append([1 if scenario_results['passes'][m] else 0 for m in minimums.keys()])

im = ax.imshow(pass_fail_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(np.arange(len(minimums)))
ax.set_yticks(np.arange(len(scenarios_list)))
ax.set_xticklabels(list(minimums.keys()))
ax.set_yticklabels(scenarios_list)
ax.set_title('Pass/Fail Regulatory Requirements')
for i in range(len(scenarios_list)):
    for j in range(len(minimums)):
        text_color = 'white' if pass_fail_matrix[i][j] == 1 else 'black'
        text = '✓' if pass_fail_matrix[i][j] == 1 else '✗'
        ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')

# Plot 3: Leverage ratio
ax = axes[1, 0]
leverage_ratios = [results[s]['ratios']['Leverage']*100 for s in scenarios_list]
ax.bar(scenarios_list, leverage_ratios, color=['green' if r >= minimums['Leverage']*100 else 'red' for r in leverage_ratios], alpha=0.7)
ax.axhline(y=minimums['Leverage']*100, color='black', linestyle='--', linewidth=2, label=f"Min ({minimums['Leverage']*100:.1f}%)")
ax.set_ylabel('Leverage Ratio (%)')
ax.set_title('Leverage Ratio Under Stress (Non-Risk-Weighted Floor)')
ax.tick_params(axis='x', rotation=15)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Capital shortfall/surplus
ax = axes[1, 1]
shortfalls = []
for scenario_name in scenarios_list:
    stressed_total = results[scenario_name]['ratios']['Total']
    min_total = minimums['Total']
    if stressed_total < min_total:
        shortfall = (min_total - stressed_total) * results[scenario_name]['rwa']
        shortfalls.append(-shortfall)
    else:
        shortfalls.append(0)

colors_shortfall = ['red' if s < 0 else 'green' for s in shortfalls]
ax.bar(scenarios_list, shortfalls, color=colors_shortfall, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Capital Surplus/(Shortfall) ($B)')
ax.set_title('Capital Shortfall/(Surplus) vs Minimum Requirement')
ax.tick_params(axis='x', rotation=15)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

