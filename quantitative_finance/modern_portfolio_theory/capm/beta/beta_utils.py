import numpy as np
import matplotlib.pyplot as plt

def portfolio_beta(weights, betas):
    """Calculate portfolio beta as weighted average"""
    return np.dot(weights, betas)

# Example portfolios
portfolio_compositions = {
    'Conservative': {'PG': 0.4, 'KO': 0.3, 'WMT': 0.3},
    'Balanced': {'PG': 0.25, 'SPY': 0.50, 'TSLA': 0.25},
    'Aggressive': {'TSLA': 0.4, 'NVDA': 0.4, 'AMD': 0.2},
}

portfolio_betas = {}
for port_name, composition in portfolio_compositions.items():
    weights = []
    betas = []
    for ticker, weight in composition.items():
        if ticker in beta_results:
            weights.append(weight)
            betas.append(beta_results[ticker]['beta'])
    
    portfolio_betas[port_name] = {
        'beta': portfolio_beta(np.array(weights), np.array(betas)),
        'composition': composition
    }

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Beta distribution by category
categories = []
beta_values = []
colors_map = {'Defensive': 'green', 'Market-Like': 'gray', 
              'Aggressive': 'red', 'Leveraged': 'darkred'}

for category, ticker_list in tickers.items():
    for ticker in ticker_list:
        if ticker in beta_results:
            categories.append(f"{ticker}\n({category})")
            beta_values.append(beta_results[ticker]['beta'])

x_pos = np.arange(len(categories))
colors = []
for cat in categories:
    for key, color in colors_map.items():
        if key in cat:
            colors.append(color)
            break

bars = axes[0, 0].barh(x_pos, beta_values, color=colors, alpha=0.7)
axes[0, 0].set_yticks(x_pos)
axes[0, 0].set_yticklabels(categories, fontsize=8)
axes[0, 0].axvline(1.0, color='black', linestyle='--', linewidth=2, label='Market (β=1)')
axes[0, 0].axvline(0.0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_xlabel('Beta')
axes[0, 0].set_title('Beta Values by Stock Category')
axes[0, 0].legend()
axes[0, 0].grid(axis='x', alpha=0.3)

# Add confidence intervals
for i, ticker in enumerate([t.split('\n')[0] for t in categories]):
    if ticker in beta_results and not np.isnan(beta_results[ticker]['se_beta']):
        beta = beta_results[ticker]['beta']
        se = beta_results[ticker]['se_beta']
        axes[0, 0].errorbar(beta, i, xerr=1.96*se, fmt='none', 
                           ecolor='black', capsize=3, alpha=0.5)

# Plot 2: Rolling beta over time
for ticker in example_stocks:
    if ticker in rolling_betas:
        axes[0, 1].plot(rolling_betas[ticker].index, rolling_betas[ticker].values,
                       linewidth=2, label=ticker, alpha=0.8)

axes[0, 1].axhline(1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Market')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Beta (1-year rolling window)')
axes[0, 1].set_title('Time-Varying Beta')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Beta vs R² scatter
betas_plot = [beta_results[t]['beta'] for t in beta_results.keys()]
r_squareds = [beta_results[t]['r_squared'] for t in beta_results.keys()]
tickers_plot = list(beta_results.keys())

scatter = axes[1, 0].scatter(betas_plot, r_squareds, s=200, alpha=0.6, c=betas_plot,
                             cmap='RdYlGn_r', vmin=0, vmax=2)

for i, ticker in enumerate(tickers_plot):
    axes[1, 0].annotate(ticker, (betas_plot[i], r_squareds[i]),
                       fontsize=8, ha='center', va='bottom')

axes[1, 0].axvline(1.0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Beta')
axes[1, 0].set_ylabel('R² (Systematic Risk %)')
axes[1, 0].set_title('Beta vs Explanatory Power')
axes[1, 0].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Beta Value')

# Plot 4: Portfolio beta composition
port_names = list(portfolio_betas.keys())
port_beta_values = [portfolio_betas[p]['beta'] for p in port_names]

bars = axes[1, 1].bar(range(len(port_names)), port_beta_values, alpha=0.7)
for i, bar in enumerate(bars):
    if port_beta_values[i] < 1:
        bar.set_color('green')
    elif port_beta_values[i] > 1:
        bar.set_color('red')
    else:
        bar.set_color('gray')

axes[1, 1].set_xticks(range(len(port_names)))
axes[1, 1].set_xticklabels(port_names, rotation=45, ha='right')
axes[1, 1].axhline(1.0, color='black', linestyle='--', linewidth=2, label='Market')
axes[1, 1].set_ylabel('Portfolio Beta')
axes[1, 1].set_title('Portfolio Beta (Weighted Average)')
axes[1, 1].legend()
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (name, value) in enumerate(zip(port_names, port_beta_values)):
    axes[1, 1].text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Print detailed portfolio analysis
print("\n" + "=" * 100)
print("PORTFOLIO BETA ANALYSIS")
print("=" * 100)

for port_name in port_names:
    port_beta_val = portfolio_betas[port_name]['beta']
    composition = portfolio_betas[port_name]['composition']
    
    print(f"\n{port_name} Portfolio (β = {port_beta_val:.3f}):")
    print("-" * 50)
    
    for ticker, weight in composition.items():
        if ticker in beta_results:
            ticker_beta = beta_results[ticker]['beta']
            contribution = weight * ticker_beta
            print(f"  {ticker:6s}: {weight:>6.1%} × β={ticker_beta:>5.2f} = {contribution:>6.3f}")

# CAPM expected returns
print("\n" + "=" * 100)
print("CAPM EXPECTED RETURNS")
print("=" * 100)

market_return = excess_returns[market_ticker].mean() * 252 + rf_annual
market_premium = market_return - rf_annual

print(f"Risk-Free Rate:      {rf_annual:.2%}")
print(f"Market Return:       {market_return:.2%}")
print(f"Market Risk Premium: {market_premium:.2%}")
print(f"\n{'Ticker':<8} {'Beta':>8} {'CAPM E[R]':>12} {'Actual R':>12} {'Alpha':>12}")
print("-" * 100)

for ticker in beta_results.keys():
    beta = beta_results[ticker]['beta']
    capm_expected = rf_annual + beta * market_premium
    actual_return = (excess_returns[ticker].mean() * 252 + rf_annual) if ticker in excess_returns.columns else np.nan
    alpha = actual_return - capm_expected if not np.isnan(actual_return) else np.nan
    
    print(f"{ticker:<8} {beta:>8.3f} {capm_expected:>11.2%} {actual_return:>11.2%} {alpha:>11.2%}")

# Statistical significance of beta
print("\n" + "=" * 100)
print("BETA STATISTICAL SIGNIFICANCE")
print("=" * 100)
print(f"{'Ticker':<8} {'Beta':>8} {'Std Error':>12} {'t-stat':>10} {'p-value':>10} {'Significant':>12}")
print("-" * 100)

for ticker in beta_results.keys():
    beta = beta_results[ticker]['beta']
    se = beta_results[ticker]['se_beta']
    t_stat = beta_results[ticker]['t_statistic']
    p_val = beta_results[ticker]['p_value']
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "No"
    
    print(f"{ticker:<8} {beta:>8.3f} {se:>12.4f} {t_stat:>10.2f} {p_val:>10.4f} {sig:>12}")

print("\n*** p<0.01, ** p<0.05, * p<0.1")