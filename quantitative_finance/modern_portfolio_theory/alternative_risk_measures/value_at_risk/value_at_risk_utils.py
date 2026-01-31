import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def backtest_var(returns, var_values, confidence_level=0.95):
    """
    Backtest VaR model - count exceedances
    """
    losses = -returns
    exceedances = np.sum(losses > var_values)
    expected_exceedances = len(returns) * (1 - confidence_level)
    
    # Kupiec test (likelihood ratio test)
    p = exceedances / len(returns)
    p0 = 1 - confidence_level
    
    if exceedances > 0 and p < 1:
        lr_stat = -2 * (np.log((1-p0)**(len(returns)-exceedances) * p0**exceedances) - 
                       np.log((1-p)**(len(returns)-exceedances) * p**exceedances))
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    else:
        lr_stat = np.nan
        p_value = np.nan
    
    return {
        'exceedances': exceedances,
        'expected': expected_exceedances,
        'exceedance_rate': p,
        'lr_statistic': lr_stat,
        'p_value': p_value
    }

# Backtest historical VaR
backtest_results = {}
for conf in confidence_levels:
    # Use expanding window to avoid look-ahead bias
    var_series = []
    
    for i in range(252, len(portfolio_returns)):  # Start after 1 year
        historical_window = portfolio_returns.iloc[:i]
        var_value = historical_var(historical_window, conf)
        var_series.append(var_value)
    
    var_series = np.array(var_series)
    test_returns = portfolio_returns.iloc[252:]
    
    backtest_results[conf] = backtest_var(test_returns, var_series, conf)

# Component VaR
def component_var(returns, weights, confidence_level=0.95):
    """
    Calculate contribution of each asset to portfolio VaR
    """
    # Portfolio VaR
    portfolio_rets = (returns * weights).sum(axis=1)
    port_var = historical_var(portfolio_rets, confidence_level)
    
    # Marginal VaR for each asset
    marginal_vars = []
    component_vars = []
    
    for i in range(len(weights)):
        # Correlation with portfolio
        corr = returns.iloc[:, i].corr(portfolio_rets)
        asset_std = returns.iloc[:, i].std()
        port_std = portfolio_rets.std()
        
        # Marginal VaR
        marginal_var = corr * asset_std / port_std * port_var
        marginal_vars.append(marginal_var)
        
        # Component VaR
        component_var = weights[i] * marginal_var
        component_vars.append(component_var)
    
    return {
        'portfolio_var': port_var,
        'marginal_var': marginal_vars,
        'component_var': component_vars,
        'sum_components': sum(component_vars)
    }

comp_var = component_var(returns, weights, 0.95)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: VaR comparison across methods
methods = list(var_results[0.95].keys())
var_95 = [var_results[0.95][m] * 100 for m in methods]

bars = axes[0, 0].bar(range(len(methods)), var_95, alpha=0.7, color='red')
axes[0, 0].set_xticks(range(len(methods)))
axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
axes[0, 0].set_ylabel('VaR (% of portfolio)')
axes[0, 0].set_title('95% VaR: Comparison of Methods')
axes[0, 0].grid(axis='y', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{var_95[i]:.2f}%', ha='center', va='bottom', fontsize=9)

# Plot 2: VaR across confidence levels
conf_labels = ['90%', '95%', '99%']
x_pos = np.arange(len(conf_labels))
width = 0.15

for i, method in enumerate(['Historical', 'Parametric (Normal)', 'CVaR']):
    values = [var_results[c][method] * 100 for c in confidence_levels]
    axes[0, 1].bar(x_pos + i*width, values, width, label=method, alpha=0.8)

axes[0, 1].set_xticks(x_pos + width)
axes[0, 1].set_xticklabels(conf_labels)
axes[0, 1].set_xlabel('Confidence Level')
axes[0, 1].set_ylabel('VaR (% of portfolio)')
axes[0, 1].set_title('VaR Across Confidence Levels')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Return distribution with VaR levels
_, simulated_returns = monte_carlo_var(portfolio_returns, 0.95, n_simulations=10000)

axes[1, 0].hist(portfolio_returns * 100, bins=50, alpha=0.5, 
               label='Historical Returns', density=True, color='blue')
axes[1, 0].hist(simulated_returns * 100, bins=50, alpha=0.3,
               label='Simulated (t-dist)', density=True, color='green')

# Mark VaR levels
var_95_hist = historical_var(portfolio_returns, 0.95) * 100
var_99_hist = historical_var(portfolio_returns, 0.99) * 100
cvar_95 = conditional_var(portfolio_returns, 0.95) * 100

axes[1, 0].axvline(-var_95_hist, color='orange', linestyle='--', linewidth=2,
                  label=f'95% VaR: {var_95_hist:.2f}%')
axes[1, 0].axvline(-var_99_hist, color='red', linestyle='--', linewidth=2,
                  label=f'99% VaR: {var_99_hist:.2f}%')
axes[1, 0].axvline(-cvar_95, color='darkred', linestyle=':', linewidth=2,
                  label=f'95% CVaR: {cvar_95:.2f}%')

axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Return Distribution with VaR Thresholds')
axes[1, 0].legend(fontsize=7)
axes[1, 0].grid(alpha=0.3)

# Plot 4: Component VaR
component_contributions = np.array(comp_var['component_var']) * 100
component_pct = component_contributions / component_contributions.sum() * 100

bars = axes[1, 1].bar(range(len(tickers)), component_contributions, alpha=0.7)
axes[1, 1].set_xticks(range(len(tickers)))
axes[1, 1].set_xticklabels(tickers)
axes[1, 1].set_ylabel('Component VaR (% contribution)')
axes[1, 1].set_title('Component VaR: Risk Contribution by Asset')
axes[1, 1].grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (bar, pct) in enumerate(zip(bars, component_pct)):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Print backtesting results
print("\n" + "=" * 90)
print("VaR BACKTESTING (Historical Method)")
print("=" * 90)
print(f"{'Confidence':>12} {'Exceedances':>15} {'Expected':>12} {'Rate':>10} {'p-value':>10} {'Result':>12}")
print("-" * 90)

for conf in confidence_levels:
    bt = backtest_results[conf]
    result = "Pass" if bt['p_value'] > 0.05 else "Fail"
    print(f"{conf:>11.0%} {bt['exceedances']:>15.0f} {bt['expected']:>12.1f} "
          f"{bt['exceedance_rate']:>9.2%} {bt['p_value']:>10.4f} {result:>12}")

print("\nInterpretation: p-value > 0.05 suggests VaR model is accurate")

# Component VaR analysis
print("\n" + "=" * 90)
print("COMPONENT VaR DECOMPOSITION (95% Confidence)")
print("=" * 90)
print(f"{'Asset':>8} {'Weight':>10} {'Component VaR':>15} {'% of Total VaR':>18}")
print("-" * 90)

for i, ticker in enumerate(tickers):
    comp_var_val = comp_var['component_var'][i] * 100
    pct_contribution = comp_var_val / (comp_var['portfolio_var'] * 100) * 100
    print(f"{ticker:>8} {weights[i]:>9.1%} {comp_var_val:>14.3f}% {pct_contribution:>17.1f}%")

print(f"\nPortfolio VaR: {comp_var['portfolio_var'] * 100:.3f}%")
print(f"Sum of Components: {sum(comp_var['component_var']) * 100:.3f}%")
print(f"Diversification Benefit: {((sum(comp_var['component_var']) - comp_var['portfolio_var'])/comp_var['portfolio_var'])*100:.1f}%")

# Key insights
print("\n" + "=" * 90)
print("KEY INSIGHTS")
print("=" * 90)
print(f"1. Normal VaR often underestimates risk (fat tails in real data)")
print(f"2. CVaR > VaR always (captures tail severity beyond threshold)")
print(f"3. Historical VaR: model-free but limited by historical period")
print(f"4. Monte Carlo: flexible but computationally intensive")
print(f"5. VaR should be complemented with stress testing and CVaR")
print(f"6. Component VaR shows which positions contribute most to risk")