import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def rolling_alpha(fund_returns, market_returns, window=252):
    """Calculate rolling alpha over time"""
    alphas = []
    betas = []
    dates = []
    
    for i in range(window, len(fund_returns)):
        window_fund = fund_returns.iloc[i-window:i]
        window_market = market_returns.iloc[i-window:i]
        
        aligned = pd.concat([window_fund, window_market], axis=1).dropna()
        if len(aligned) > 50:
            X = sm.add_constant(aligned.iloc[:, 1])
            model = sm.OLS(aligned.iloc[:, 0], X).fit()
            
            alpha = model.params[0] * 252  # Annualize
            beta = model.params[1]
            
            alphas.append(alpha)
            betas.append(beta)
            dates.append(fund_returns.index[i])
    
    return pd.DataFrame({'alpha': alphas, 'beta': betas}, index=dates)

# Calculate rolling alpha for selected funds
rolling_results = {}
example_funds = ['Berkshire Hathaway', 'ARK Innovation', 'Vanguard 500']
for name in example_funds:
    if name in alpha_results:
        ticker = funds[name]
        rolling_results[name] = rolling_alpha(
            excess_returns[ticker], market_excess, window=252
        )

# Simulate luck vs skill
def simulate_managers(n_managers=1000, n_periods=60, market_return=0.08/12, market_vol=0.15/np.sqrt(12)):
    """
    Simulate random managers with no skill (α=0)
    Show how many appear to have alpha by chance
    """
    np.random.seed(42)
    
    # Generate market returns
    market_returns = np.random.normal(market_return, market_vol, n_periods)
    
    manager_alphas = []
    manager_t_stats = []
    
    for _ in range(n_managers):
        # True alpha = 0, but random beta
        beta = np.random.uniform(0.8, 1.2)
        
        # Generate manager returns = beta * market + noise (no true alpha)
        idiosyncratic_vol = 0.05 / np.sqrt(12)
        noise = np.random.normal(0, idiosyncratic_vol, n_periods)
        manager_returns = beta * market_returns + noise
        
        # Estimate alpha
        X = sm.add_constant(market_returns)
        model = sm.OLS(manager_returns, X).fit()
        
        alpha = model.params[0] * 12  # Annualize
        t_stat = model.tvalues[0]
        
        manager_alphas.append(alpha)
        manager_t_stats.append(t_stat)
    
    return np.array(manager_alphas), np.array(manager_t_stats)

# Run simulation
simulated_alphas, simulated_tstats = simulate_managers(n_managers=1000, n_periods=60)

# Count false positives
false_positives_5pct = np.sum(np.abs(simulated_tstats) > 1.96) / len(simulated_tstats)
apparent_skill = np.sum(simulated_alphas > 0.02) / len(simulated_alphas)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Alpha comparison with confidence intervals
fund_names = alpha_df.index
alphas = alpha_df['alpha_annual'].values
alpha_ses = alpha_df['alpha_se_annual'].values
colors = ['green' if p < 0.05 and a > 0 else 'red' if p < 0.05 and a < 0 
          else 'gray' for a, p in zip(alphas, alpha_df['p_value'])]

y_pos = np.arange(len(fund_names))
bars = axes[0, 0].barh(y_pos, alphas, color=colors, alpha=0.6)
axes[0, 0].errorbar(alphas, y_pos, xerr=1.96*alpha_ses,
                    fmt='none', ecolor='black', capsize=5)

axes[0, 0].set_yticks(y_pos)
axes[0, 0].set_yticklabels(fund_names)
axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Jensen\'s Alpha (Annual %)')
axes[0, 0].set_title('Alpha with 95% Confidence Intervals')
axes[0, 0].grid(axis='x', alpha=0.3)

# Plot 2: Rolling alpha over time
for name in rolling_results.keys():
    if name in rolling_results:
        axes[0, 1].plot(rolling_results[name].index, 
                       rolling_results[name]['alpha'],
                       linewidth=2, label=name, alpha=0.8)

axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Rolling 1-Year Alpha')
axes[0, 1].set_title('Time-Varying Alpha')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Luck vs Skill simulation
axes[1, 0].hist(simulated_alphas * 100, bins=50, alpha=0.7, 
               edgecolor='black', color='steelblue')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, 
                  label='True Alpha = 0')
axes[1, 0].axvline(2, color='orange', linestyle='--', linewidth=2,
                  label=f'{apparent_skill:.1%} appear > 2%')

# Mark significant alphas
sig_threshold = 1.96 * np.std(simulated_alphas)
axes[1, 0].axvline(sig_threshold * 100, color='green', linestyle=':', linewidth=2, alpha=0.5)
axes[1, 0].axvline(-sig_threshold * 100, color='green', linestyle=':', linewidth=2, alpha=0.5)

axes[1, 0].set_xlabel('Estimated Alpha (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Luck vs Skill: 1000 Managers with True α=0\n{false_positives_5pct:.1%} false positives at 5% level')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Information Ratio vs Alpha
ir_values = alpha_df['information_ratio'].values
alpha_values = alpha_df['alpha_annual'].values

scatter = axes[1, 1].scatter(ir_values, alpha_values, s=300, alpha=0.6,
                            c=alpha_df['p_value'], cmap='RdYlGn_r',
                            vmin=0, vmax=0.1)

for i, name in enumerate(fund_names):
    axes[1, 1].annotate(name.split()[0], 
                       (ir_values[i], alpha_values[i]),
                       fontsize=8, ha='center', va='bottom')

axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1, 1].axvline(0.5, color='orange', linestyle='--', linewidth=1, 
                  alpha=0.5, label='IR = 0.5 (Good)')
axes[1, 1].axvline(1.0, color='green', linestyle='--', linewidth=1,
                  alpha=0.5, label='IR = 1.0 (Excellent)')

axes[1, 1].set_xlabel('Information Ratio')
axes[1, 1].set_ylabel('Alpha (Annual %)')
axes[1, 1].set_title('Alpha Magnitude vs Consistency (color = p-value)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='p-value')

plt.tight_layout()
plt.show()

# Detailed performance attribution
print("\n" + "=" * 110)
print("PERFORMANCE ATTRIBUTION")
print("=" * 110)
print(f"{'Fund':<25} {'Actual Return':>15} {'Expected (CAPM)':>18} {'Alpha':>10} {'Attribution':>15}")
print("-" * 110)

for name in alpha_df.index:
    row = alpha_df.loc[name]
    actual = row['actual_return']
    expected = expected_return = rf_annual + row['beta'] * (market_excess.mean() * 252)
    alpha = row['alpha_annual']
    
    # Attribution
    rf_contrib = rf_annual
    beta_contrib = row['beta'] * (market_excess.mean() * 252)
    alpha_contrib = alpha
    
    print(f"{name:<25} {actual:>14.2%} {expected:>17.2%} {alpha:>9.2%}")
    print(f"{'  Breakdown:':<25} rf={rf_contrib:>5.2%} + β×MRP={beta_contrib:>5.2%} + α={alpha_contrib:>5.2%}")

# Alpha persistence test
print("\n" + "=" * 110)
print("ALPHA PERSISTENCE TEST (First half vs Second half)")
print("=" * 110)

for name, ticker in list(funds.items())[:3]:  # Sample funds
    if ticker in excess_returns.columns:
        fund_rets = excess_returns[ticker]
        n = len(fund_rets)
        midpoint = n // 2
        
        # First half
        first_half = calculate_alpha(fund_rets.iloc[:midpoint], 
                                     market_excess.iloc[:midpoint])
        # Second half
        second_half = calculate_alpha(fund_rets.iloc[midpoint:],
                                      market_excess.iloc[midpoint:])
        
        correlation = "Positive" if (first_half['alpha_annual'] > 0 and 
                                     second_half['alpha_annual'] > 0) else "Negative"
        
        print(f"\n{name}:")
        print(f"  First half:  α = {first_half['alpha_annual']:>6.2%} (p={first_half['p_value']:.4f})")
        print(f"  Second half: α = {second_half['alpha_annual']:>6.2%} (p={second_half['p_value']:.4f})")
        print(f"  Persistence: {correlation}")

print("\n" + "=" * 110)
print("KEY INSIGHTS")
print("=" * 110)
print(f"1. Statistically significant alpha is rare (requires |t| > 2, typically)")
print(f"2. In simulation, {false_positives_5pct:.1%} of zero-alpha managers appear significant")
print(f"3. Information Ratio > 0.5 suggests consistent skill, not just luck")
print(f"4. Alpha persistence is key test: does it continue in out-of-sample periods?")
print(f"5. Gross alpha may exist, but net alpha (after fees) is harder to achieve")