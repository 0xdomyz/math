import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Generate 3 assets with different skewness profiles
np.random.seed(42)
n = 1000

# Asset 1: Normal returns (symmetric)
asset1 = np.random.normal(0.05, 0.12, n)

# Asset 2: Positive skew (long call strategy)
asset2_normal = np.random.normal(0.04, 0.10, n)
asset2_crashes = np.random.normal(-0.30, 0.05, 20)
asset2 = np.concatenate([asset2_normal[:-20], asset2_crashes])

# Asset 3: Negative skew (short premium)
asset3_normal = np.random.normal(0.06, 0.08, n)
asset3_crashes = np.random.normal(-0.25, 0.08, 15)
asset3 = np.concatenate([asset3_normal[:-15], asset3_crashes])

returns = np.column_stack([asset1, asset2, asset3])
asset_names = ['Asset 1\n(Symmetric)', 'Asset 2\n(Positive Skew)', 'Asset 3\n(Negative Skew)']

# Calculate statistics
means = returns.mean(axis=0)
stds = returns.std(axis=0)
tau = means.mean()  # threshold = average of all means

# Calculate semi-variance (below threshold)
def semi_variance(returns_col, threshold):
    downside = np.maximum(0, threshold - returns_col)
    return np.mean(downside ** 2)

def downside_deviation(returns_col, threshold):
    return np.sqrt(semi_variance(returns_col, threshold))

semi_vars = [semi_variance(returns[:, i], tau) for i in range(3)]
semi_devs = [downside_deviation(returns[:, i], tau) for i in range(3)]

# Covariance and semi-covariance matrices
cov_matrix = np.cov(returns.T)

# Semi-covariance (downside co-movement)
def semi_cov_matrix(returns, threshold):
    n = returns.shape[1]
    semi_cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            downside_i = np.maximum(0, threshold - returns[:, i])
            downside_j = np.maximum(0, threshold - returns[:, j])
            semi_cov[i, j] = np.mean(downside_i * downside_j)
    return semi_cov

semi_cov = semi_cov_matrix(returns, tau)

# Portfolio optimization: Mean-Variance
def portfolio_variance(w, cov):
    return w @ cov @ w

def portfolio_return(w, means):
    return w @ means

target_return = 0.045
constraints_mv = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w: portfolio_return(w, means) - target_return}
]
bounds = [(0, 1) for _ in range(3)]
x0 = np.array([1/3, 1/3, 1/3])

result_mv = minimize(lambda w: portfolio_variance(w, cov_matrix),
                     x0, method='SLSQP', bounds=bounds, constraints=constraints_mv)
w_mv = result_mv.x

# Portfolio optimization: Semi-Variance
def portfolio_semi_variance(w, returns, threshold):
    p_returns = returns @ w
    downside = np.maximum(0, threshold - p_returns)
    return np.mean(downside ** 2)

result_sv = minimize(lambda w: portfolio_semi_variance(w, returns, tau),
                     x0, method='SLSQP', bounds=bounds, constraints=constraints_mv)
w_sv = result_sv.x

# Calculate results
port_ret_mv = portfolio_return(w_mv, means)
port_vol_mv = np.sqrt(portfolio_variance(w_mv, cov_matrix))
port_semi_dev_mv = downside_deviation(returns @ w_mv, tau)

port_ret_sv = portfolio_return(w_sv, means)
port_vol_sv = np.sqrt(portfolio_variance(w_sv, cov_matrix))
port_semi_dev_sv = downside_deviation(returns @ w_sv, tau)

# Print comparison
print("="*80)
print("VARIANCE vs SEMI-VARIANCE PORTFOLIO OPTIMIZATION")
print("="*80)
print(f"\nAsset Statistics:")
print(f"{'Asset':<20} {'Mean':<10} {'Std Dev':<12} {'Semi-Dev':<12} {'Skewness':<10}")
print("-"*80)
for i, name in enumerate(asset_names):
    skew = pd.Series(returns[:, i]).skew()
    print(f"{name:<20} {means[i]:<10.4f} {stds[i]:<12.4f} {semi_devs[i]:<12.4f} {skew:<10.3f}")

print(f"\nPortfolio Weights (Target Return = {target_return:.2%}):")
print(f"{'Portfolio':<20} {'Asset 1':<10} {'Asset 2':<10} {'Asset 3':<10}")
print("-"*80)
print(f"{'Mean-Variance':<20} {w_mv[0]:<10.3f} {w_mv[1]:<10.3f} {w_mv[2]:<10.3f}")
print(f"{'Semi-Variance':<20} {w_sv[0]:<10.3f} {w_sv[1]:<10.3f} {w_sv[2]:<10.3f}")

print(f"\nPortfolio Performance:")
print(f"{'Metric':<20} {'Mean-Var':<15} {'Semi-Var':<15}")
print("-"*80)
print(f"{'Expected Return':<20} {port_ret_mv:<15.4f} {port_ret_sv:<15.4f}")
print(f"{'Std Deviation':<20} {port_vol_mv:<15.4f} {port_vol_sv:<15.4f}")
print(f"{'Downside Dev':<20} {port_semi_dev_mv:<15.4f} {port_semi_dev_sv:<15.4f}")
print(f"{'Sortino Ratio':<20} {(port_ret_mv - tau) / port_semi_dev_mv:<15.4f} {(port_ret_sv - tau) / port_semi_dev_sv:<15.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Return distributions
for i, name in enumerate(asset_names):
    ax = axes[0, i]
    ax.hist(returns[:, i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=means[i], color='red', linestyle='--', linewidth=2, label=f'Mean={means[i]:.3f}')
    ax.axvline(x=tau, color='green', linestyle='--', linewidth=2, label=f'Threshold={tau:.3f}')
    ax.set_title(name)
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

# Plot 2: Comparison bars (bottom)
ax = axes[1, 0]
metrics = ['Std Dev', 'Semi-Dev']
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, stds, width, label='Std Dev', alpha=0.8)
ax.bar(x + width/2, semi_devs, width, label='Semi-Dev', alpha=0.8)
ax.set_ylabel('Risk Measure')
ax.set_title('Variance vs Semi-Variance by Asset')
ax.set_xticks(x)
ax.set_xticklabels(asset_names)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Efficient frontier comparison
ax = axes[1, 1]
w_range = np.linspace(0, 1, 50)
frontier_mv_vol = []
frontier_mv_semi = []
frontier_sv_vol = []
frontier_sv_semi = []

for w1 in w_range:
    w = np.array([w1, (1-w1)/2, (1-w1)/2])
    vol = np.sqrt(portfolio_variance(w, cov_matrix))
    semi_dev = downside_deviation(returns @ w, tau)
    frontier_mv_vol.append(vol)
    frontier_mv_semi.append(semi_dev)

ax.scatter(frontier_mv_vol, [portfolio_return(np.array([w1, (1-w1)/2, (1-w1)/2]), means) 
           for w1 in w_range], alpha=0.5, s=30, label='Equal Weight Frontier', color='gray')
ax.scatter([port_vol_mv], [port_ret_mv], s=200, marker='*', color='blue', 
          label=f'MV Optimal', zorder=5)
ax.scatter([port_vol_sv], [port_ret_sv], s=200, marker='o', color='red',
          label=f'Semi-Var Optimal', zorder=5)
ax.set_xlabel('Volatility (Std Dev)')
ax.set_ylabel('Expected Return')
ax.set_title('Portfolio Optimization: MV vs Semi-Variance')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Weight comparison
ax = axes[1, 2]
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, w_mv, width, label='Mean-Variance', alpha=0.8)
ax.bar(x + width/2, w_sv, width, label='Semi-Variance', alpha=0.8)
ax.set_ylabel('Weight')
ax.set_title('Optimal Weights Comparison')
ax.set_xticks(x)
ax.set_xticklabels(asset_names)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()