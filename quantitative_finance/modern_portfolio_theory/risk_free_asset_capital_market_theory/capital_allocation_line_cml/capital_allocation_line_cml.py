import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Visualize Capital Allocation Line, Capital Market Line, optimal allocation

def get_market_data(start_date, end_date):
    """
    Fetch market and risk-free rate data.
    """
    # Market proxy (S&P 500)
    spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
    market_returns = spy_data.pct_change().dropna()
    
    # Risk-free rate approximation (T-bills, use constant for simplicity)
    rf = 0.04  # 4% current approximate rate
    
    return market_returns, rf


def compute_market_parameters(market_returns):
    """
    Estimate market expected return and volatility.
    """
    annual_return = market_returns.mean() * 252
    annual_vol = market_returns.std() * np.sqrt(252)
    
    return annual_return, annual_vol


def compute_cal(rf, risky_return, risky_vol, vol_range):
    """
    Compute Capital Allocation Line: E[Rp] = rf + (E[R_risky] - rf) / σ_risky × σp
    """
    sharpe_ratio = (risky_return - rf) / risky_vol
    cal_returns = rf + sharpe_ratio * vol_range
    return cal_returns, sharpe_ratio


def compute_indifference_curves(lambda_values, utility_levels, vol_range):
    """
    Compute indifference curves for different risk aversions.
    E[Rp] = U + (λ/2) σp²
    """
    curves = {}
    for lambda_coeff in lambda_values:
        curve_dict = {}
        for u_level in utility_levels:
            returns = u_level + (lambda_coeff / 2) * vol_range ** 2
            curve_dict[f'U={u_level:.3f}'] = returns
        curves[f'λ={lambda_coeff}'] = curve_dict
    return curves


def optimal_allocation(rf, market_return, market_vol, lambda_coeff):
    """
    Compute optimal allocation on CAL for given λ.
    Tangency: λσp* = (E[Rm] - rf) / σm
    """
    market_premium = market_return - rf
    optimal_vol = market_premium / (lambda_coeff * market_vol)
    optimal_return = rf + market_premium / market_vol * optimal_vol
    
    # Weights
    w_market = optimal_vol / market_vol
    w_rf = 1 - w_market
    
    return optimal_vol, optimal_return, w_market, w_rf


def compute_efficient_frontier_simple(rf, market_return, market_vol, num_points=50):
    """
    Efficient frontier approximated by CML.
    """
    vol_range = np.linspace(0, 0.4, num_points)
    cal_returns, _ = compute_cal(rf, market_return, market_vol, vol_range)
    return vol_range, cal_returns


# Main Analysis
print("=" * 100)
print("CAPITAL ALLOCATION LINE & CAPITAL MARKET LINE")
print("=" * 100)

# 1. Data
print("\n1. MARKET DATA & PARAMETERS")
print("-" * 100)

market_returns, rf = get_market_data('2015-01-01', '2024-01-01')
market_return, market_vol = compute_market_parameters(market_returns)

print(f"Risk-free rate (assumed): {rf*100:.2f}%")
print(f"Market expected return: {market_return*100:.2f}%")
print(f"Market volatility: {market_vol*100:.2f}%")
print(f"Market Sharpe ratio: {(market_return - rf) / market_vol:.3f}")

# 2. CAL slope (Sharpe ratio)
print("\n2. CAPITAL ALLOCATION LINE (CAL)")
print("-" * 100)

_, sharpe_ratio = compute_cal(rf, market_return, market_vol, np.array([0.15]))
print(f"\nCML equation: E[Rp] = {rf*100:.2f}% + {sharpe_ratio:.3f} × σp")
print(f"\nInterpretation: Every 1% volatility accepted → {sharpe_ratio*100:.2f}% return gained")

# Points on CAL
print(f"\nPoints on CML:")
print(f"{'Volatility':<15} {'Expected Return':<20} {'Allocation':<30}")
print("-" * 65)

for w_market in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    vol = w_market * market_vol
    ret = rf + (market_return - rf) * w_market
    w_rf = 1 - w_market
    allocation = f"{w_market*100:.0f}% market, {w_rf*100:.0f}% rf"
    if w_rf < 0:
        allocation = f"{w_market*100:.0f}% market, {abs(w_rf)*100:.0f}% borrowed"
    
    print(f"{vol*100:<15.2f} {ret*100:<20.2f} {allocation:<30}")

# 3. Optimal allocation for different λ values
print("\n3. OPTIMAL ALLOCATION BY RISK AVERSION")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
optimal_results = {}

for lambda_coeff in lambda_values:
    opt_vol, opt_return, w_market, w_rf = optimal_allocation(rf, market_return, market_vol, lambda_coeff)
    utility = opt_return - (lambda_coeff / 2) * opt_vol ** 2
    optimal_results[lambda_coeff] = {
        'vol': opt_vol,
        'return': opt_return,
        'w_market': w_market,
        'w_rf': w_rf,
        'utility': utility
    }
    
    print(f"\nλ = {lambda_coeff} (Risk aversion):")
    print(f"  Optimal volatility: {opt_vol*100:.2f}%")
    print(f"  Optimal return: {opt_return*100:.2f}%")
    print(f"  Utility: {utility:.4f}")
    print(f"  Allocation: {w_market*100:.1f}% market, {w_rf*100:.1f}% risk-free")
    if w_rf < 0:
        print(f"             (Leverage: borrow {abs(w_rf)*100:.1f}% to buy market)")

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: CAL and CML with indifference curves
ax = axes[0, 0]

vol_range = np.linspace(0, 0.35, 200)
cal_returns, _ = compute_cal(rf, market_return, market_vol, vol_range)

# CML
ax.plot(vol_range * 100, cal_returns * 100, 'k-', linewidth=3.5, label='Capital Market Line (CML)', zorder=3)

# Indifference curves
colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
for i, lambda_coeff in enumerate(lambda_values):
    opt_vol, opt_return, _, _ = optimal_allocation(rf, market_return, market_vol, lambda_coeff)
    utility = opt_return - (lambda_coeff / 2) * opt_vol ** 2
    
    indiff_returns = utility + (lambda_coeff / 2) * vol_range ** 2
    ax.plot(vol_range * 100, indiff_returns * 100, '--', color=colors[i], linewidth=2,
            label=f'Indifference (λ={lambda_coeff})', alpha=0.7)
    
    # Optimal point
    ax.scatter(opt_vol * 100, opt_return * 100, s=250, color=colors[i], marker='*', zorder=5)

# Risk-free and market
ax.scatter([0], [rf * 100], s=300, marker='o', color='green', label='Risk-free', zorder=5)
ax.scatter([market_vol * 100], [market_return * 100], s=300, marker='s', color='red',
          label='Market Portfolio', zorder=5)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Capital Market Line & Indifference Curves', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim([0, 35])
ax.set_ylim([2, 20])

# Plot 2: Allocation by λ
ax = axes[0, 1]

lambdas = list(optimal_results.keys())
market_weights = [optimal_results[l]['w_market'] for l in lambdas]
rf_weights = [optimal_results[l]['w_rf'] for l in lambdas]

x = np.arange(len(lambdas))
width = 0.35

ax.bar(x - width/2, [w * 100 for w in market_weights], width, label='Market Portfolio', 
      color='#3498db', alpha=0.8)
ax.bar(x + width/2, [w * 100 for w in rf_weights], width, label='Risk-Free', 
      color='#e74c3c', alpha=0.8)

ax.set_ylabel('Allocation (%)', fontsize=12)
ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_title('Optimal Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'λ={l}' for l in lambdas])
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.axhline(y=0, color='k', linewidth=0.8)

# Plot 3: Risk-return by λ
ax = axes[1, 0]

vols = [optimal_results[l]['vol'] for l in lambdas]
rets = [optimal_results[l]['return'] for l in lambdas]

ax.plot(lambdas, [v * 100 for v in vols], 'o-', linewidth=2.5, markersize=8, 
       label='Volatility', color='#e74c3c')
ax_twin = ax.twinx()
ax_twin.plot(lambdas, [r * 100 for r in rets], 's-', linewidth=2.5, markersize=8,
            label='Expected Return', color='#27ae60')

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Volatility (%)', fontsize=12, color='#e74c3c')
ax_twin.set_ylabel('Expected Return (%)', fontsize=12, color='#27ae60')
ax.set_title('Optimal Portfolio Risk-Return by λ', fontweight='bold', fontsize=13)
ax.tick_params(axis='y', labelcolor='#e74c3c')
ax_twin.tick_params(axis='y', labelcolor='#27ae60')
ax.grid(alpha=0.3)

# Plot 4: Utility by λ
ax = axes[1, 1]

utilities = [optimal_results[l]['utility'] for l in lambdas]

ax.plot(lambdas, utilities, 'o-', linewidth=2.5, markersize=10, color='#2ecc71')
ax.fill_between(lambdas, utilities, alpha=0.3, color='#2ecc71')

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Utility', fontsize=12)
ax.set_title('Optimal Utility by Risk Aversion', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cal_cml_optimal_allocation.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: cal_cml_optimal_allocation.png")
plt.show()

# 5. Comparison with constrained optimization
print("\n4. IMPACT OF LEVERAGE CONSTRAINT")
print("-" * 100)

for lambda_coeff in [1.0, 2.0]:
    opt_vol, opt_return, w_market, w_rf = optimal_allocation(rf, market_return, market_vol, lambda_coeff)
    
    if w_market > 1.0:
        # Constraint binds
        w_market_constrained = 1.0
        w_rf_constrained = 0.0
        vol_constrained = w_market_constrained * market_vol
        ret_constrained = rf + (market_return - rf) * w_market_constrained
        utility_unconstrained = opt_return - (lambda_coeff / 2) * opt_vol ** 2
        utility_constrained = ret_constrained - (lambda_coeff / 2) * vol_constrained ** 2
        
        print(f"\nλ = {lambda_coeff}:")
        print(f"  Unconstrained optimal:")
        print(f"    w_market = {w_market*100:.1f}% (leverage), w_rf = {w_rf*100:.1f}%")
        print(f"    Utility = {utility_unconstrained:.4f}")
        print(f"  Constrained optimal (max leverage = 0%):")
        print(f"    w_market = {w_market_constrained*100:.1f}%, w_rf = {w_rf_constrained*100:.1f}%")
        print(f"    Utility = {utility_constrained:.4f}")
        print(f"  Welfare loss from constraint: {(utility_unconstrained - utility_constrained)*100:.2f} bp")

print("\n" + "=" * 100)