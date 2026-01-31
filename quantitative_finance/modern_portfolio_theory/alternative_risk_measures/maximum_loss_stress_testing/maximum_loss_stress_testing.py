import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Portfolio definition
portfolio = {
    'US Equities': {'weight': 0.60, 'volatility': 0.15},
    'Bonds': {'weight': 0.30, 'volatility': 0.05},
    'Commodities': {'weight': 0.10, 'volatility': 0.12}
}

# Historical scenarios (actual crises + worst days)
historical_scenarios = {
    'Black Monday 1987': {
        'US Equities': -0.226,
        'Bonds': 0.02,
        'Commodities': -0.05
    },
    'LTCM Crisis 1998': {
        'US Equities': -0.15,
        'Bonds': -0.02,
        'Commodities': -0.10
    },
    '2008 Financial Crisis': {
        'US Equities': -0.37,
        'Bonds': 0.05,
        'Commodities': -0.35
    },
    'COVID Crash March 2020': {
        'US Equities': -0.34,
        'Bonds': 0.02,
        'Commodities': -0.20
    },
    'Dot-com Peak (2000-2002)': {
        'US Equities': -0.78,
        'Bonds': 0.12,
        'Commodities': -0.10
    }
}

# Hypothetical stress scenarios
hypothetical_scenarios = {
    'Mild Correction': {
        'US Equities': -0.10,
        'Bonds': 0.01,
        'Commodities': -0.05
    },
    'Moderate Downturn': {
        'US Equities': -0.20,
        'Bonds': 0.03,
        'Commodities': -0.15
    },
    'Severe Crisis (All Down)': {
        'US Equities': -0.40,
        'Bonds': -0.05,
        'Commodities': -0.30
    },
    'Stagflation Scenario': {
        'US Equities': -0.25,
        'Bonds': -0.20,
        'Commodities': +0.40
    },
    'Rate Shock (+200 bps)': {
        'US Equities': -0.15,
        'Bonds': -0.12,
        'Commodities': +0.05
    },
    'Geopolitical Crisis': {
        'US Equities': -0.18,
        'Bonds': +0.02,
        'Commodities': +0.25
    }
}

# Calculate portfolio impact for each scenario
def calculate_portfolio_loss(portfolio, scenario):
    """Calculate portfolio loss under scenario"""
    loss = 0
    for asset, asset_info in portfolio.items():
        weight = asset_info['weight']
        asset_move = scenario[asset]
        loss += weight * asset_move
    return loss

# Evaluate all scenarios
all_scenarios = {**historical_scenarios, **hypothetical_scenarios}
results = []

for scenario_name, moves in all_scenarios.items():
    loss = calculate_portfolio_loss(portfolio, moves)
    scenario_type = 'Historical' if scenario_name in historical_scenarios else 'Hypothetical'
    results.append({
        'Scenario': scenario_name,
        'Type': scenario_type,
        'Portfolio Loss': loss,
        'Loss %': f'{loss*100:.2f}%',
        'USD Loss ($M)': loss * 100  # Assuming $100M AUM
    })

results_df = pd.DataFrame(results).sort_values('Portfolio Loss')

# Print results
print("="*100)
print("STRESS TEST RESULTS")
print("="*100)
print(f"\nPortfolio Composition: ", end='')
for asset, info in portfolio.items():
    print(f"{asset} {info['weight']*100:.0f}%, ", end='')
print("\n")

print(results_df.to_string(index=False))

print(f"\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)
print(f"Worst-case loss: {results_df['Portfolio Loss'].min()*100:.2f}% ({results_df['Scenario'].iloc[0]})")
print(f"Best-case loss: {results_df['Portfolio Loss'].max()*100:.2f}% ({results_df['Scenario'].iloc[-1]})")
print(f"Average loss (all scenarios): {results_df['Portfolio Loss'].mean()*100:.2f}%")
print(f"Median loss: {results_df['Portfolio Loss'].median()*100:.2f}%")
print(f"Standard deviation: {results_df['Portfolio Loss'].std()*100:.2f}%")

# Calculate VaR approximation for comparison
daily_vol_portfolio = np.sqrt(
    sum(portfolio[asset]['weight']**2 * portfolio[asset]['volatility']**2 
        for asset in portfolio)
)
var_95 = daily_vol_portfolio * 1.645  # 95% confidence
var_99 = daily_vol_portfolio * 2.326  # 99% confidence

print(f"\nTraditional Risk Metrics (Daily):")
print(f"  Portfolio Volatility: {daily_vol_portfolio*100:.2f}%")
print(f"  VaR (95%, 1-day): {var_95*100:.2f}%")
print(f"  VaR (99%, 1-day): {var_99*100:.2f}%")
print(f"\nMax Loss vs VaR Gap:")
print(f"  Worst Stress Loss: {results_df['Portfolio Loss'].min()*100:.2f}%")
print(f"  VaR(99%): {var_99*100:.2f}%")
print(f"  Gap: {(results_df['Portfolio Loss'].min() - (-var_99))*100:.2f}% (stress worse)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss comparison (historical vs hypothetical)
ax = axes[0, 0]
colors = ['red' if t == 'Historical' else 'orange' for t in results_df['Type']]
sorted_results = results_df.sort_values('Portfolio Loss')
ax.barh(range(len(sorted_results)), sorted_results['Portfolio Loss']*100, color=colors, alpha=0.7)
ax.set_yticks(range(len(sorted_results)))
ax.set_yticklabels(sorted_results['Scenario'], fontsize=9)
ax.axvline(x=-var_95*100, color='blue', linestyle='--', linewidth=2, label=f'VaR(95%)={-var_95*100:.1f}%')
ax.axvline(x=-var_99*100, color='green', linestyle='--', linewidth=2, label=f'VaR(99%)={-var_99*100:.1f}%')
ax.set_xlabel('Portfolio Loss (%)')
ax.set_title('Stress Test Results: All Scenarios')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Plot 2: Historical vs Hypothetical box plot
ax = axes[0, 1]
hist_losses = results_df[results_df['Type'] == 'Historical']['Portfolio Loss'].values * 100
hyp_losses = results_df[results_df['Type'] == 'Hypothetical']['Portfolio Loss'].values * 100
ax.boxplot([hist_losses, hyp_losses], labels=['Historical', 'Hypothetical'])
ax.set_ylabel('Portfolio Loss (%)')
ax.set_title('Loss Distribution: Historical vs Hypothetical')
ax.grid(alpha=0.3, axis='y')

# Plot 3: Asset contribution to losses
ax = axes[1, 0]
scenario_worst = historical_scenarios['2008 Financial Crisis']
contributions = []
assets_list = list(portfolio.keys())
for asset in assets_list:
    weight = portfolio[asset]['weight']
    move = scenario_worst[asset]
    contribution = weight * move * 100  # Convert to percentage
    contributions.append(contribution)

colors_contrib = ['red' if c < 0 else 'green' for c in contributions]
ax.bar(assets_list, contributions, color=colors_contrib, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Contribution to Loss (%)')
ax.set_title('Asset Contribution: 2008 Crisis Scenario')
ax.grid(alpha=0.3, axis='y')

# Plot 4: Risk metrics comparison
ax = axes[1, 1]
risk_metrics = {
    'Daily Vol': daily_vol_portfolio * 100,
    'VaR(95%)': var_95 * 100,
    'VaR(99%)': var_99 * 100,
    'Worst Stress': -results_df['Portfolio Loss'].min() * 100,
    'Avg Stress': -results_df['Portfolio Loss'].mean() * 100,
}
x = np.arange(len(risk_metrics))
ax.bar(x, list(risk_metrics.values()), color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(list(risk_metrics.keys()), rotation=45, ha='right')
ax.set_ylabel('Loss (%)')
ax.set_title('Risk Metrics Summary')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()