import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Historical annual volatility and correlation estimates
assets = ['US Equities', 'US Bonds', 'Commodities']
vols = np.array([0.15, 0.05, 0.12])  # std dev
corr_matrix = np.array([
    [1.00, -0.20, 0.50],
    [-0.20, 1.00, -0.10],
    [0.50, -0.10, 1.00]
])

# Build covariance matrix
cov_matrix = np.outer(vols, vols) * corr_matrix

# Risk Parity Optimization
def risk_parity_weights(cov_matrix, target_leverage=1.0):
    """Calculate risk parity weights"""
    n = cov_matrix.shape[0]
    vols = np.sqrt(np.diag(cov_matrix))
    
    # Initial inverse volatility weights
    w_init = 1 / vols
    w_init = w_init / w_init.sum()
    
    # Refine: iterate to achieve equal risk
    def risk_contribution(w):
        """Return risk contributions"""
        port_vol = np.sqrt(w @ cov_matrix @ w)
        mrc = (cov_matrix @ w) / port_vol
        rc = w * mrc
        return rc
    
    def objective(w):
        """Minimize variance in risk contributions"""
        rc = risk_contribution(w)
        target_rc = 1.0 / n  # target: equal risk
        return np.sum((rc - target_rc)**2)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, None) for _ in range(n)]
    
    result = minimize(objective, w_init, method='SLSQP', 
                     constraints=constraints, bounds=bounds)
    
    w_rp = result.x
    
    # Apply leverage if needed
    port_vol_rp = np.sqrt(w_rp @ cov_matrix @ w_rp)
    w_rp_leveraged = w_rp * target_leverage / port_vol_rp
    
    return w_rp, w_rp_leveraged

# Comparison portfolios
w_market_cap = np.array([0.70, 0.25, 0.05])  # typical market cap weight
w_equal = np.array([1/3, 1/3, 1/3])
w_inverse_vol = 1 / vols
w_inverse_vol = w_inverse_vol / w_inverse_vol.sum()

w_rp, w_rp_lev = risk_parity_weights(cov_matrix, target_leverage=1.3)

portfolios = {
    'Market Cap': w_market_cap,
    'Equal Weight': w_equal,
    'Inverse Vol': w_inverse_vol,
    'Risk Parity': w_rp,
    'RP Leveraged': w_rp_lev
}

# Calculate portfolio metrics
results = {}
for pname, w in portfolios.items():
    port_vol = np.sqrt(w @ cov_matrix @ w)
    
    # Risk contributions
    mrc = (cov_matrix @ w) / port_vol
    rc = w * mrc
    
    # Gross exposure
    gross_exposure = np.sum(np.abs(w))
    
    results[pname] = {
        'weights': w,
        'port_vol': port_vol,
        'risk_contrib': rc,
        'gross_exposure': gross_exposure,
    }

# Print results
print("="*90)
print("RISK PARITY PORTFOLIO ANALYSIS")
print("="*90)
print(f"\nAsset Volatilities: {dict(zip(assets, vols))}")
print(f"\nCorrelation Matrix:\n{corr_matrix}")

print(f"\n{'Portfolio':<20} ", end='')
for asset in assets:
    print(f"{asset:<15} ", end='')
print(f"{'Gross Exp':<10} {'Vol':<8}")
print("-"*90)

for pname, metrics in results.items():
    w = metrics['weights']
    vol = metrics['port_vol']
    gross = metrics['gross_exposure']
    print(f"{pname:<20} ", end='')
    for i in range(len(assets)):
        print(f"{w[i]:<15.2%} ", end='')
    print(f"{gross:<10.2f}x {vol:<8.2%}")

print(f"\n{'Portfolio':<20} ", end='')
for asset in assets:
    print(f"RC: {asset:<12} ", end='')
print()
print("-"*90)

for pname, metrics in results.items():
    rc = metrics['risk_contrib']
    print(f"{pname:<20} ", end='')
    for i in range(len(assets)):
        print(f"{rc[i]:<15.2%} ", end='')
    print()

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Weight comparison
ax = axes[0, 0]
x = np.arange(len(assets))
width = 0.15
for i, (pname, metrics) in enumerate(results.items()):
    ax.bar(x + i*width, metrics['weights'], width, label=pname, alpha=0.8)
ax.set_ylabel('Weight')
ax.set_title('Portfolio Weights Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(assets)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 2: Risk contributions
ax = axes[0, 1]
for i, (pname, metrics) in enumerate(results.items()):
    ax.bar(x + i*width, metrics['risk_contrib'], width, label=pname, alpha=0.8)
ax.axhline(y=1/3, color='red', linestyle='--', linewidth=2, label='Equal Risk Target')
ax.set_ylabel('Risk Contribution')
ax.set_title('Risk Contributions (RP Target: 33.3% each)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(assets)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# Plot 3: Portfolio volatility comparison
ax = axes[0, 2]
vols_port = [results[pname]['port_vol'] for pname in results]
colors = ['red' if results[pname]['gross_exposure'] > 1 else 'blue' for pname in results]
bars = ax.bar(results.keys(), vols_port, color=colors, alpha=0.7)
ax.set_ylabel('Portfolio Volatility')
ax.set_title('Portfolio Volatility (Red=Leveraged)')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Risk decomposition (stacked)
ax = axes[1, 0]
rc_matrix = np.array([results[pname]['risk_contrib'] for pname in results]).T
rc_matrix_pct = rc_matrix * 100
bottom = np.zeros(len(results))
colors_stack = ['steelblue', 'orange', 'green']
for i, asset in enumerate(assets):
    ax.bar(results.keys(), rc_matrix_pct[i], bottom=bottom, label=asset, 
          color=colors_stack[i], alpha=0.8)
    bottom += rc_matrix_pct[i]
ax.set_ylabel('Risk Contribution (%)')
ax.set_title('Risk Decomposition by Asset')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 5: Correlation matrix heatmap
ax = axes[1, 1]
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(assets)))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(assets, rotation=45, ha='right')
ax.set_yticklabels(assets)
for i in range(len(assets)):
    for j in range(len(assets)):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10)
ax.set_title('Asset Correlation Matrix')
plt.colorbar(im, ax=ax)

# Plot 6: Gross exposure summary
ax = axes[1, 2]
gross_exps = [results[pname]['gross_exposure'] for pname in results]
colors_exp = ['red' if g > 1 else 'green' for g in gross_exps]
ax.bar(results.keys(), gross_exps, color=colors_exp, alpha=0.7)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No Leverage')
ax.set_ylabel('Gross Exposure (Leverage Multiple)')
ax.set_title('Gross Exposure Summary')
ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*90)
print("KEY INSIGHTS:")
print("="*90)
print(f"Risk Parity trades off leverage for diversification:")
print(f"  - Market Cap weight: High equity (70%), low leverage")
print(f"  - Risk Parity: Lower equity (careful RP calcs), may need leverage")
print(f"  - Benefit: Equal risk contribution across assets")
print(f"  - Tradeoff: Requires rebalancing, transaction costs")