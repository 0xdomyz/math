import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import yfinance as yf

# Compute and visualize indifference curves; find optimal allocation graphically

def get_efficient_frontier_data(start_date, end_date):
    """
    Fetch market data to estimate frontier and CAL.
    """
    tickers = ['SPY', 'BND']  # Stocks and Bonds
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns


def estimate_cov_matrix(returns):
    """
    Estimate covariance matrix and mean returns.
    """
    annual_returns = returns.mean() * 252
    annual_cov = returns.cov() * 252
    
    return annual_returns, annual_cov


def compute_efficient_frontier_points(annual_returns, annual_cov, n_points=50):
    """
    Compute efficient frontier from 0% to 100% in risky asset.
    """
    results = []
    
    for w_risky in np.linspace(0, 1.5, n_points):  # Allow leverage up to 150%
        w = np.array([w_risky, 1 - w_risky])
        
        portfolio_return = np.sum(w * annual_returns)
        portfolio_var = w @ annual_cov @ w
        portfolio_vol = np.sqrt(portfolio_var)
        
        results.append({
            'w_risky': w_risky,
            'return': portfolio_return,
            'volatility': portfolio_vol
        })
    
    return pd.DataFrame(results)


def compute_indifference_curves(lambda_coeff, utility_levels, vol_range):
    """
    For each utility level, compute E[Rp] = U + (λ/2) σp²
    """
    curves = {}
    
    for u_level in utility_levels:
        returns = u_level + (lambda_coeff / 2) * vol_range**2
        curves[f'U={u_level:.3f}'] = returns
    
    return curves


def find_tangency_portfolio(annual_returns, annual_cov, rf_rate):
    """
    Find portfolio with highest Sharpe ratio (tangent to CAL).
    """
    n_assets = len(annual_returns)
    
    def neg_sharpe(w):
        p_return = np.sum(w * annual_returns)
        p_var = w @ annual_cov @ w
        p_vol = np.sqrt(p_var)
        sharpe = (p_return - rf_rate) / p_vol
        return -sharpe
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([0.6, 0.4])
    
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    w_tangent = result.x
    p_return = np.sum(w_tangent * annual_returns)
    p_vol = np.sqrt(w_tangent @ annual_cov @ w_tangent)
    sharpe = (p_return - rf_rate) / p_vol
    
    return w_tangent, p_return, p_vol, sharpe


def compute_cal(rf_rate, tangent_return, tangent_vol, vol_range):
    """
    Capital Allocation Line: E[Rp] = rf + (E[Rm] - rf)/σm × σp
    """
    sharpe = (tangent_return - rf_rate) / tangent_vol
    cal_returns = rf_rate + sharpe * vol_range
    return cal_returns


def optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate):
    """
    Find optimal allocation on CAL for given λ.
    """
    sharpe = (tangent_return - rf_rate) / tangent_vol
    
    # Tangency condition: λ σp = sharpe
    opt_vol = sharpe / lambda_coeff
    opt_return = rf_rate + sharpe * opt_vol
    
    return opt_vol, opt_return


# Main Analysis
print("=" * 100)
print("INDIFFERENCE CURVES & OPTIMAL PORTFOLIO SELECTION")
print("=" * 100)

# 1. Data
print("\n1. MARKET DATA & PARAMETERS")
print("-" * 100)

returns = get_efficient_frontier_data('2015-01-01', '2024-01-01')
annual_returns, annual_cov = estimate_cov_matrix(returns)
rf_rate = 0.025

print(f"Risk-free rate: {rf_rate:.2%}")
print(f"Stocks (SPY): {annual_returns['SPY']:.2%} return, {np.sqrt(annual_cov.iloc[0, 0]):.2%} vol")
print(f"Bonds (BND):  {annual_returns['BND']:.2%} return, {np.sqrt(annual_cov.iloc[1, 1]):.2%} vol")
print(f"Correlation: {annual_cov.iloc[0, 1] / (np.sqrt(annual_cov.iloc[0, 0]) * np.sqrt(annual_cov.iloc[1, 1])):.2f}")

# 2. Tangency portfolio
print("\n2. TANGENCY PORTFOLIO (Highest Sharpe Ratio)")
print("-" * 100)

w_tangent, tangent_return, tangent_vol, tangent_sharpe = find_tangency_portfolio(
    annual_returns, annual_cov, rf_rate
)

print(f"Expected return: {tangent_return:.2%}")
print(f"Volatility: {tangent_vol:.2%}")
print(f"Sharpe ratio: {tangent_sharpe:.3f}")
print(f"Allocation: {w_tangent[0]:.1%} stocks, {w_tangent[1]:.1%} bonds")

# 3. Optimal allocations for different λ values
print("\n3. OPTIMAL ALLOCATION BY RISK AVERSION COEFFICIENT")
print("-" * 100)

lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
optimal_allocations = {}

for lambda_coeff in lambda_values:
    opt_vol, opt_return = optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate)
    
    w_risky = opt_vol / tangent_vol
    w_rf = 1 - w_risky
    
    optimal_allocations[lambda_coeff] = {
        'vol': opt_vol,
        'return': opt_return,
        'w_risky': w_risky,
        'w_rf': w_rf
    }
    
    print(f"\nλ = {lambda_coeff}:")
    print(f"  Optimal vol: {opt_vol:.2%}")
    print(f"  Optimal return: {opt_return:.2%}")
    print(f"  Allocation: {w_risky:5.1%} risky portfolio + {w_rf:5.1%} risk-free")
    print(f"  Utility: {opt_return - (lambda_coeff/2) * opt_vol**2:.4f}")

# 4. Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Indifference curves and CAL
ax = axes[0, 0]

vol_range = np.linspace(0, 0.40, 200)

# CAL
cal_returns = compute_cal(rf_rate, tangent_return, tangent_vol, vol_range)
ax.plot(vol_range * 100, cal_returns * 100, 'k-', linewidth=3, label='Capital Allocation Line (CAL)')

# Indifference curves for different λ
colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
for i, lambda_coeff in enumerate(lambda_values):
    opt_vol, opt_return = optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate)
    utility = opt_return - (lambda_coeff / 2) * opt_vol**2
    
    indiff_returns = utility + (lambda_coeff / 2) * vol_range**2
    ax.plot(vol_range * 100, indiff_returns * 100, '--', color=colors[i], linewidth=2, 
            label=f'Indifference (λ={lambda_coeff}), U={utility:.4f}')
    
    # Mark optimal point
    ax.scatter(opt_vol * 100, opt_return * 100, s=200, color=colors[i], marker='*', zorder=5)

# Risk-free asset
ax.scatter([0], [rf_rate * 100], s=300, marker='o', color='green', label='Risk-free asset', zorder=5)

# Tangency portfolio
ax.scatter([tangent_vol * 100], [tangent_return * 100], s=300, marker='s', color='red', 
          label='Tangency portfolio', zorder=5)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Expected Return (%)', fontsize=12)
ax.set_title('Indifference Curves & Capital Allocation Line', fontweight='bold', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim([0, 40])
ax.set_ylim([0, 15])

# Plot 2: Optimal allocation vs λ
ax = axes[0, 1]

lambdas = list(optimal_allocations.keys())
risky_weights = [optimal_allocations[l]['w_risky'] for l in lambdas]
rf_weights = [optimal_allocations[l]['w_rf'] for l in lambdas]

x = np.arange(len(lambdas))
width = 0.35

ax.bar(x - width/2, [w * 100 for w in risky_weights], width, label='Risky portfolio', color='#3498db', alpha=0.8)
ax.bar(x + width/2, [w * 100 for w in rf_weights], width, label='Risk-free', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Allocation (%)', fontsize=12)
ax.set_title('Optimal Allocation by Risk Aversion', fontweight='bold', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f'λ={l}' for l in lambdas])
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.axhline(y=0, color='k', linewidth=0.8)

# Plot 3: Utility by λ
ax = axes[1, 0]

utilities = [optimal_allocations[l]['return'] - (l / 2) * optimal_allocations[l]['vol']**2 
            for l in lambdas]

ax.plot(lambdas, utilities, 'o-', linewidth=2.5, markersize=8, color='#2ecc71')
ax.fill_between(lambdas, utilities, alpha=0.3, color='#2ecc71')

ax.set_xlabel('Risk Aversion Coefficient (λ)', fontsize=12)
ax.set_ylabel('Utility', fontsize=12)
ax.set_title('Optimal Utility by Risk Aversion', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Plot 4: Tangency condition illustration
ax = axes[1, 1]

vol_range = np.linspace(0, 0.40, 200)
lambda_coeff = 4.0

# Indifference curve for λ=4
opt_vol, opt_return = optimize_for_lambda(lambda_coeff, tangent_vol, tangent_return, rf_rate)
utility = opt_return - (lambda_coeff / 2) * opt_vol**2
indiff_returns = utility + (lambda_coeff / 2) * vol_range**2

# MRS along indifference curve
mrs = lambda_coeff * vol_range

# CAL slope (constant)
cal_slope = (tangent_return - rf_rate) / tangent_vol

ax.plot(vol_range * 100, mrs * 100, 'b-', linewidth=2.5, label=f'MRS (λ={lambda_coeff})')
ax.axhline(y=cal_slope * 100, color='r', linestyle='--', linewidth=2.5, label=f'CAL slope = {cal_slope:.3f}')

# Tangency point
ax.scatter([opt_vol * 100], [lambda_coeff * opt_vol * 100], s=300, marker='*', 
          color='green', label='Optimal (MRS = CAL slope)', zorder=5)

ax.set_xlabel('Volatility (%)', fontsize=12)
ax.set_ylabel('Marginal Rate of Substitution (dE[R]/dσ)', fontsize=12)
ax.set_title('Tangency Condition: MRS = CAL Slope', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 40])

plt.tight_layout()
plt.savefig('indifference_curves_optimal_selection.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: indifference_curves_optimal_selection.png")
plt.show()

# 5. Key insights
print("\n4. KEY INSIGHTS & INTERPRETATION")
print("-" * 100)
print(f"""
INDIFFERENCE CURVE GEOMETRY:
├─ Parabolic shape: E[Rp] = U₀ + (λ/2)σp²
├─ Curvature increases with λ (risk-averse investors have more curved preferences)
├─ Higher indifference curves represent higher utility (farther from origin)
└─ Curves never intersect (transitive preferences)

MARGINAL RATE OF SUBSTITUTION (MRS):
├─ Formula: MRS = λ × σp (slope of indifference curve)
├─ Interpretation: How much return willing to sacrifice per unit volatility reduction
├─ Increases with σp: Steeper slopes at higher volatility (flattens demand for more vol)
├─ Higher λ: Steeper curve at all σp (more risk-averse; demands higher return per vol)
└─ Example: At σp=10% with λ=4, MRS=0.4 (willing to sacrifice 0.4% return per 1% vol cut)

CAPITAL ALLOCATION LINE (CAL):
├─ Linear opportunity set: E[Rp] = {rf_rate:.2%} + {cal_slope:.3f} × σp
├─ Tangent portfolio composition: {w_tangent[0]:.1%} stocks, {w_tangent[1]:.1%} bonds
├─ Slope (Sharpe ratio): {cal_slope:.3f} (risk premium per unit risk)
├─ All investors lie on CAL; choice determined by λ (how much risk to take)
└─ Leverage extends CAL rightward; leverage is controlled by borrowing

OPTIMAL PORTFOLIO SELECTION:
├─ Tangency condition: MRS = CAL slope at optimum
├─ Optimization: Max utility subject to feasibility on CAL
├─ Solution: Higher λ → move left on CAL (lower vol, lower return)
├─ Solution: Lower λ → move right on CAL (higher vol, higher return, may lever)
└─ Example: λ=2 investor holds {optimal_allocations[2.0]['w_risky']:.1%} risky + {optimal_allocations[2.0]['w_rf']:.1%} risk-free

TWO-FUND SEPARATION:
├─ Implication: All investors hold combination of risk-free + one risky portfolio
├─ Market portfolio = optimal risky portfolio (for investor with market's λ)
├─ Institutional practice: Index funds (market proxy) + bonds/bills
├─ Theoretical: Explains why beating market hard (everyone agrees on asset pricing)
└─ Violation: Home bias, behavioral preferences → people don't follow pure two-fund

PRACTICAL RECOMMENDATIONS:
├─ Estimate λ from questionnaire or revealed preference (historical allocation)
├─ Find tangency portfolio (market portfolio often serves this)
├─ Locate optimal point on CAL based on your λ
├─ Allocate: w_risky = {cal_slope:.3f} / (λ × {tangent_vol:.2%}) = {cal_slope:.3f} / ({tangent_vol:.2%}λ)
├─ Rebalance periodically (quarterly) to maintain allocation
└─ Adjust λ if life changes: retirement, inheritance, income shock, time horizon shift
""")

print("=" * 100)