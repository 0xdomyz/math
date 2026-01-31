from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def compute_capital_allocation_line(returns, rf_rate=0.02):
    """
    Compute CAL using tangency portfolio.
    CAL: E[Rp] = rf + (E[Rm] - rf) / σm × σp
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Objective: maximize Sharpe ratio
    def neg_sharpe(w):
        portfolio_return = np.sum(mean_returns * w)
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)
        sharpe = (portfolio_return - rf_rate) / portfolio_vol
        return -sharpe
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([1 / n_assets] * n_assets)
    
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    tangency_weights = result.x
    tangency_return = np.sum(mean_returns * tangency_weights)
    tangency_vol = np.sqrt(tangency_weights @ cov_matrix @ tangency_weights)
    tangency_sharpe = (tangency_return - rf_rate) / tangency_vol
    
    return tangency_weights, tangency_return, tangency_vol, tangency_sharpe


# Main Analysis
print("=" * 100)
print("MEAN-VARIANCE UTILITY & RISK AVERSION COEFFICIENT ESTIMATION")
print("=" * 100)

# 1. Risk aversion questionnaire
print("\n1. INVESTOR QUESTIONNAIRE & LAMBDA ESTIMATION")
print("-" * 100)

investor_profiles = {
    'Conservative (Retiree)': {
        'age': 72,
        'loss_comfort': 1,
        'portfolio_risk': 30,
        'gambling': 1
    },
    'Moderate (Mid-Career)': {
        'age': 45,
        'loss_comfort': 3,
        'portfolio_risk': 60,
        'gambling': 2
    },
    'Aggressive (Young Professional)': {
        'age': 32,
        'loss_comfort': 4,
        'portfolio_risk': 85,
        'gambling': 4
    },
}

lambda_estimates = {}
for profile_name, responses in investor_profiles.items():
    lambda_val = estimate_lambda_from_questionnaire(responses)
    lambda_estimates[profile_name] = lambda_val
    print(f"\n{profile_name}:")
    print(f"  Age: {responses['age']}, Loss comfort: {responses['loss_comfort']}/5, Portfolio risk: {responses['portfolio_risk']}%")
    print(f"  Estimated λ: {lambda_val:.2f}")

# 2. Get market data
print("\n2. ASSET CLASS DATA")
print("-" * 100)

returns, tickers = get_market_data('2015-01-01', '2024-01-01')

print(f"\nAsset Classes:")
for asset, ticker in tickers.items():
    mean_ret = returns[ticker].mean() * 252
    volatility = returns[ticker].std() * np.sqrt(252)
    print(f"  {asset}: Expected return {mean_ret:.2%}, Volatility {volatility:.2%}")

# 3. Optimal allocation for each investor
print("\n3. OPTIMAL PORTFOLIO ALLOCATION BY INVESTOR TYPE")
print("-" * 100)

allocations = {}
for profile_name, lambda_val in lambda_estimates.items():
    weights, utility = optimize_portfolio_mean_variance(returns, lambda_val)
    allocations[profile_name] = weights
    
    portfolio_return = np.sum(weights * (returns.mean() * 252))
    portfolio_vol = np.sqrt(weights @ (returns.cov() * 252) @ weights)
    
    print(f"\n{profile_name} (λ = {lambda_val:.2f}):")
    print(f"  Expected return: {portfolio_return:.2%}")
    print(f"  Volatility: {portfolio_vol:.2%}")
    print(f"  Utility: {utility:.4f}")
    print(f"  Allocation:")
    for i, (asset, ticker) in enumerate(tickers.items()):
        print(f"    {asset}: {weights[i]:5.1%}")

# 4. Capital Allocation Line
print("\n4. TANGENCY PORTFOLIO & CAPITAL ALLOCATION LINE (CAL)")
print("-" * 100)

rf_rate = 0.02
tangency_weights, tangency_return, tangency_vol, tangency_sharpe = compute_capital_allocation_line(returns, rf_rate)

print(f"Risk-free rate: {rf_rate:.2%}")
print(f"\nTangency Portfolio (Highest Sharpe Ratio):")
print(f"  Expected return: {tangency_return:.2%}")
print(f"  Volatility: {tangency_vol:.2%}")
print(f"  Sharpe ratio: {tangency_sharpe:.3f}")
print(f"  Allocation:")
for i, (asset, ticker) in enumerate(tickers.items()):
    print(f"    {asset}: {tangency_weights[i]:5.1%}")

print(f"\nCAL Equation: E[Rp] = {rf_rate:.2%} + {(tangency_return - rf_rate)/tangency_vol:.3f} × σp")

# 5. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Utility function illustration
ax = axes[0, 0]
sigma_range = np.linspace(0, 0.25, 100)
for profile_name, lambda_val in lambda_estimates.items():
    fixed_return = 0.08
    utilities = fixed_return - (lambda_val / 2) * sigma_range**2
    ax.plot(sigma_range * 100, utilities, label=f'{profile_name} (λ={lambda_val:.1f})', linewidth=2)

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Utility (for fixed 8% return)')
ax.set_title('Mean-Variance Utility by Risk Aversion Coefficient', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Optimal allocations by investor
ax = axes[0, 1]
profile_names = list(allocations.keys())
asset_names = list(tickers.keys())
allocation_matrix = np.array([allocations[p] for p in profile_names]).T

x = np.arange(len(profile_names))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, asset in enumerate(asset_names):
    offset = (i - 1.5) * width
    ax.bar(x + offset, allocation_matrix[i], width, label=asset, color=colors[i], alpha=0.8)

ax.set_ylabel('Allocation (%)')
ax.set_title('Optimal Portfolio Allocation by Investor Type', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(profile_names, rotation=15, ha='right', fontsize=9)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Efficient frontier
ax = axes[1, 0]
lambda_range = np.logspace(-1, 1, 50)  # λ from 0.1 to 10
frontier = compute_efficient_frontier(returns, lambda_range, rf_rate)

ax.plot(frontier['volatility'] * 100, frontier['expected_return'] * 100, 
        linewidth=2.5, color='darkblue', label='Efficient Frontier')
ax.scatter([tangency_vol * 100], [tangency_return * 100], 
          s=300, color='red', marker='*', label='Tangency Portfolio', zorder=5)
ax.scatter([rf_rate * 100], [rf_rate * 100], 
          s=200, color='green', marker='o', label='Risk-Free Asset', zorder=5)

# Add CAL line
cal_vol_range = np.linspace(0, 0.25, 100)
cal_return = rf_rate + (tangency_return - rf_rate) / tangency_vol * cal_vol_range
ax.plot(cal_vol_range * 100, cal_return * 100, 'r--', linewidth=2, label='CAL', alpha=0.7)

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Expected Return (%)')
ax.set_title('Efficient Frontier & Capital Allocation Line', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Risk-return for different λ values
ax = axes[1, 1]
for profile_name, lambda_val in lambda_estimates.items():
    profile_data = frontier[frontier['lambda'] == lambda_val].iloc[0]
    ax.scatter(profile_data['volatility'] * 100, profile_data['expected_return'] * 100,
              s=200, alpha=0.7, label=f'{profile_name} (λ={lambda_val:.1f})')

ax.set_xlabel('Volatility (%)')
ax.set_ylabel('Expected Return (%)')
ax.set_title('Optimal Portfolios on Efficient Frontier', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('mean_variance_utility_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: mean_variance_utility_analysis.png")
plt.show()

# 6. Key insights
print("\n5. KEY INSIGHTS & RECOMMENDATIONS")
print("-" * 100)
print(f"""
RISK AVERSION COEFFICIENT (λ) INTERPRETATION:
├─ Low λ (0.5-1.0): Risk-seeking; willing to accept 2-4% volatility for 1% return
├─ Moderate λ (1.5-3.0): Standard investor; willing to accept 1-2% vol for 1% return
├─ High λ (5-10): Risk-averse; willing to accept <0.5% vol for 1% return
└─ Very high λ (>10): Conservative/retiree; prioritize capital preservation

ESTIMATED INVESTOR PROFILES:
├─ Conservative (λ ≈ {lambda_estimates['Conservative (Retiree)']:.1f}): Retirement focused; capital preservation paramount
├─ Moderate (λ ≈ {lambda_estimates['Moderate (Mid-Career)']:.1f}): Long-term growth; can accept volatility
└─ Aggressive (λ ≈ {lambda_estimates['Aggressive (Young Professional)']:.1f}): Growth-oriented; high risk tolerance

IMPLICATIONS:
├─ Allocation changes dramatically with λ (even 1-unit difference → 10-15% weight shift)
├─ Estimation uncertainty: ±20% in λ → suboptimal allocation costs ~0.5-1% annually
├─ Rebalancing frequency: Conservative investors benefit from quarterly rebalancing (lock in diversification benefit)
├─ Leverage: Aggressive investors can improve utility through leverage (borrow at rf, invest in risky)
└─ Time horizon: λ increases with age (shorter horizon → higher risk aversion)

PRACTICAL RECOMMENDATIONS:
├─ Use questionnaires for initial λ estimate, then refine based on behavior
├─ Consider revealed preferences: What allocation would this investor actually choose?
├─ Build in constraints: Min 20% bonds (retirees), max 50% volatility (safety)
├─ Annual review: Recalibrate λ as life circumstances change (retirement, inheritance, etc.)
└─ Use robust optimization: Assume λ range rather than point estimate; hedge uncertainty
""")

print("=" * 100)