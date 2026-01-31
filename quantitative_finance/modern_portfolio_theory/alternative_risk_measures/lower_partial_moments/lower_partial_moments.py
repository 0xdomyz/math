import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Generate return distributions with different tail profiles
np.random.seed(42)
n_sims = 10000

# Distribution 1: Normal (symmetric, light tails)
returns_normal = np.random.normal(0.05, 0.12, n_sims)

# Distribution 2: Student-t (fat tails, symmetric)
returns_fat = np.random.standard_t(5) * 0.12 * np.sqrt(5/3) + 0.05

# Distribution 3: Skewed (hedge fund-like: rare crashes)
normal_part = np.random.normal(0.06, 0.08, int(n_sims * 0.98))
crash_part = np.random.normal(-0.30, 0.05, int(n_sims * 0.02))
returns_skewed = np.concatenate([normal_part, crash_part])

distributions = {
    'Normal': returns_normal,
    'Fat-Tailed (t)': returns_fat,
    'Skewed (Crashes)': returns_skewed
}

# Calculate LPM for different orders and thresholds
lpm_orders = [0, 1, 2, 3, 4]
thresholds = [0, np.mean(returns_normal)]

def calculate_lpm(returns, tau, n):
    """Calculate LPM(n, tau)"""
    downside = np.maximum(0, tau - returns)
    if n == 0:
        return (downside > 0).mean()  # Probability of loss
    else:
        return np.mean(downside ** n) ** (1 / n)  # Root of n-th moment

results = {}
for dist_name, returns in distributions.items():
    results[dist_name] = {}
    for order in lpm_orders:
        for threshold_label, threshold in [('0%', 0), ('Mean', returns.mean())]:
            lpm_val = calculate_lpm(returns, threshold, order)
            key = f'LPM({order}, τ={threshold_label})'
            results[dist_name][key] = lpm_val

# Print results
print("="*100)
print("LOWER PARTIAL MOMENTS COMPARISON")
print("="*100)

for dist_name, dist_results in results.items():
    print(f"\n{dist_name} Distribution:")
    print(f"  Mean: {distributions[dist_name].mean():.4f}")
    print(f"  Std Dev: {distributions[dist_name].std():.4f}")
    print(f"  Skewness: {skew(distributions[dist_name]):.4f}")
    print(f"  Kurtosis: {kurtosis(distributions[dist_name]):.4f}")
    print(f"\n  {'LPM Metric':<30} {'Value':<12}")
    print(f"  {'-'*42}")
    for metric, value in dist_results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric:<30} {value:<12.4f}")

# Risk-adjusted returns (Sortino-like)
print("\n" + "="*100)
print("RISK-ADJUSTED RETURN (EXCESS RETURN / LPM)")
print("="*100)

for dist_name in distributions.keys():
    mean_ret = distributions[dist_name].mean()
    print(f"\n{dist_name}:")
    for order in [1, 2, 3, 4]:
        lpm_val = calculate_lpm(distributions[dist_name], 0, order)
        ratio = mean_ret / lpm_val if lpm_val > 0 else np.inf
        print(f"  Kappa({order}, τ=0): Return/LPM({order}) = {ratio:.4f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Distributions
for i, (dist_name, returns) in enumerate(distributions.items()):
    ax = axes[0, i]
    ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=returns.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero')
    ax.set_title(f'{dist_name}\n(Skew={skew(returns):.2f}, Kurt={kurtosis(returns):.2f})')
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

# Plot 2: LPM by order
ax = axes[1, 0]
threshold = 0
lpm_values = {dist: [] for dist in distributions.keys()}
for order in lpm_orders:
    for dist in distributions.keys():
        lpm_val = calculate_lpm(distributions[dist], threshold, order)
        lpm_values[dist].append(lpm_val)

for dist in distributions.keys():
    ax.plot(lpm_orders, lpm_values[dist], marker='o', linewidth=2, label=dist)
ax.set_xlabel('LPM Order (n)')
ax.set_ylabel('LPM Value')
ax.set_title(f'LPM by Order (τ = {threshold:.1%})')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Tail focus comparison
ax = axes[1, 1]
tail_ratios = {}
for dist in distributions.keys():
    lpm1 = calculate_lpm(distributions[dist], 0, 1)
    lpm2 = calculate_lpm(distributions[dist], 0, 2)
    lpm3 = calculate_lpm(distributions[dist], 0, 3)
    tail_ratios[dist] = [lpm1, lpm2, lpm3]

x = np.arange(len(distributions))
width = 0.25
for i, order in enumerate([1, 2, 3]):
    values = [tail_ratios[dist][i] for dist in distributions.keys()]
    ax.bar(x + i*width, values, width, label=f'LPM({order})', alpha=0.8)

ax.set_ylabel('LPM Value')
ax.set_title('LPM by Distribution (Higher Orders = Tail Focus)')
ax.set_xticks(x + width)
ax.set_xticklabels(distributions.keys())
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Kappa ratios
ax = axes[1, 2]
kappas = {}
for dist_name in distributions.keys():
    mean_ret = distributions[dist_name].mean()
    kappas[dist_name] = []
    for order in [1, 2, 3, 4]:
        lpm_val = calculate_lpm(distributions[dist_name], 0, order)
        kappa = mean_ret / lpm_val if lpm_val > 0 else np.inf
        kappas[dist_name].append(kappa)

for dist in distributions.keys():
    ax.plot([1, 2, 3, 4], kappas[dist], marker='s', linewidth=2, label=dist)
ax.set_xlabel('LPM Order (n)')
ax.set_ylabel('Kappa(n) = Return / LPM(n)')
ax.set_title('Risk-Adjusted Return by LPM Order')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate portfolio selection impact
print("\n" + "="*100)
print("PORTFOLIO SELECTION: WHICH DISTRIBUTION TO PREFER?")
print("="*100)
print("\nChoice depends on LPM order (investor loss aversion):")
print("  n=1: Average downside → Fat-tailed worse (high avg loss)")
print("  n=2: Downside variance → Normal vs Fat-tail comparable")
print("  n=3: Tail severity → Skewed worse (crashes cubed)")
print("  n=4: Extreme tail → Skewed vastly worse")
print("\nInvestor risk tolerance determines order selection!")