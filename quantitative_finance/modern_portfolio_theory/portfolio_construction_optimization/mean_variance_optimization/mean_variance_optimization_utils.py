import numpy as np
import matplotlib.pyplot as plt

def perturb_returns(mean_returns, perturbation_std=0.02):
    """Add noise to expected returns"""
    noise = np.random.normal(0, perturbation_std, len(mean_returns))
    return mean_returns + noise

num_simulations = 100
simulated_mvp_weights = []
simulated_ms_weights = []

np.random.seed(42)
for _ in range(num_simulations):
    perturbed_returns = perturb_returns(mean_returns)
    
    try:
        mvp_w = min_variance(perturbed_returns, cov_matrix)
        ms_w = max_sharpe_portfolio(perturbed_returns, cov_matrix)
        simulated_mvp_weights.append(mvp_w)
        simulated_ms_weights.append(ms_w)
    except:
        pass

simulated_mvp_weights = np.array(simulated_mvp_weights)
simulated_ms_weights = np.array(simulated_ms_weights)

# Box plot of weight distributions
positions = np.arange(len(tickers))
axes[1, 0].boxplot([simulated_mvp_weights[:, i] for i in range(len(tickers))],
                   positions=positions - 0.2, widths=0.3,
                   patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7),
                   label='Min Variance')
axes[1, 0].boxplot([simulated_ms_weights[:, i] for i in range(len(tickers))],
                   positions=positions + 0.2, widths=0.3,
                   patch_artist=True,
                   boxprops=dict(facecolor='gold', alpha=0.7),
                   label='Max Sharpe')

axes[1, 0].set_xticks(positions)
axes[1, 0].set_xticklabels(tickers)
axes[1, 0].set_ylabel('Weight')
axes[1, 0].set_title('Weight Sensitivity to Return Estimates (±2% noise)')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Sharpe ratio along efficient frontier
sharpe_ratios = (ef_returns - rf_rate) / ef_volatilities
axes[1, 1].plot(ef_volatilities, sharpe_ratios, 'b-', linewidth=2)
axes[1, 1].axhline((ms_return - rf_rate) / ms_vol, color='gold', 
                   linestyle='--', linewidth=2, label='Maximum Sharpe')
axes[1, 1].axvline(ms_vol, color='gold', linestyle='--', alpha=0.5)
axes[1, 1].scatter(ms_vol, (ms_return - rf_rate) / ms_vol, 
                  c='gold', marker='*', s=300, edgecolors='black', linewidths=2)

axes[1, 1].set_xlabel('Portfolio Volatility')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].set_title('Sharpe Ratio Along Efficient Frontier')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Comparison with naive strategies
equal_weight = np.array([1/len(tickers)] * len(tickers))
eq_return, eq_vol = portfolio_stats(equal_weight, mean_returns, cov_matrix)

print("\n" + "=" * 80)
print("COMPARISON WITH NAIVE STRATEGIES")
print("=" * 80)
print(f"{'Strategy':<25} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
print("-" * 80)
print(f"{'Minimum Variance':<25} {mvp_return:>9.2%} {mvp_vol:>11.2%} {(mvp_return-0.03)/mvp_vol:>9.3f}")
print(f"{'Maximum Sharpe':<25} {ms_return:>9.2%} {ms_vol:>11.2%} {(ms_return-0.03)/ms_vol:>9.3f}")
print(f"{'Equal Weight (1/n)':<25} {eq_return:>9.2%} {eq_vol:>11.2%} {(eq_return-0.03)/eq_vol:>9.3f}")

# Individual assets
for ticker in tickers:
    asset_return = mean_returns[ticker]
    asset_vol = np.sqrt(cov_matrix.loc[ticker, ticker])
    asset_sharpe = (asset_return - 0.03) / asset_vol
    print(f"{ticker + ' (100%)':<25} {asset_return:>9.2%} {asset_vol:>11.2%} {asset_sharpe:>9.3f}")