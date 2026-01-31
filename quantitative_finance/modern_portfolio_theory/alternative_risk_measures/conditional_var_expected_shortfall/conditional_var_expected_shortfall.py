import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize

# Generate sample returns from different distributions
np.random.seed(42)
n_samples = 10000
alpha = 0.05  # 5% confidence level

# Distribution 1: Normal (thin tails)
returns_normal = np.random.normal(0.0005, 0.01, n_samples)

# Distribution 2: Student's t (fat tails)
df = 5  # degrees of freedom
returns_fat = np.random.standard_t(df) * 0.01 / np.sqrt(df/(df-2)) + 0.0005

# Distribution 3: Mixture (normal + rare crashes)
normal_prob = 0.99
crash_prob = 0.01
returns_mixture = np.where(np.random.random(n_samples) < normal_prob,
                           np.random.normal(0.0005, 0.01, n_samples),
                           np.random.normal(-0.05, 0.02, n_samples))

distributions = {
    'Normal': returns_normal,
    'Fat-Tailed (t)': returns_fat,
    'Mixture (Crashes)': returns_mixture
}

# Calculate VaR and CVaR
results = {}

for name, returns in distributions.items():
    # VaR: α-th percentile
    var = np.percentile(returns, alpha * 100)
    
    # CVaR: average of returns ≤ VaR
    cvar = returns[returns <= var].mean()
    
    # Parametric CVaR (assuming normal)
    z_alpha = norm.ppf(alpha)
    mu, sigma = returns.mean(), returns.std()
    cvar_param = mu - sigma * norm.pdf(z_alpha) / alpha
    
    results[name] = {
        'VaR': var,
        'CVaR': cvar,
        'CVaR_Param': cvar_param,
        'Excess': cvar - var,
    }

# Print comparison
print("="*70)
print("VaR vs CVaR COMPARISON (95% confidence, α=0.05)")
print("="*70)
print(f"\n{'Distribution':<20} {'VaR':<12} {'CVaR':<12} {'Excess':<12} {'Excess %':<12}")
print("-"*70)

for name, stats in results.items():
    var = stats['VaR']
    cvar = stats['CVaR']
    excess = stats['Excess']
    excess_pct = (excess / var * 100) if var != 0 else 0
    print(f"{name:<20} {var:<12.4f} {cvar:<12.4f} {excess:<12.4f} {excess_pct:<12.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution comparison (histograms)
ax = axes[0, 0]
for name, returns in distributions.items():
    ax.hist(returns, bins=50, alpha=0.5, label=name, density=True)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax.set_title('Return Distributions')
ax.set_xlabel('Return')
ax.set_ylabel('Density')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Tail comparison (zoomed)
ax = axes[0, 1]
quantiles = np.linspace(0, 0.1, 50)
for name, returns in distributions.items():
    tail_values = [np.percentile(returns, q * 100) for q in quantiles]
    ax.plot(quantiles * 100, tail_values, linewidth=2, marker='o', markersize=3, label=name)
ax.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='α=5%')
ax.set_title('Left Tail Quantiles (Worst Returns)')
ax.set_xlabel('Percentile (%)')
ax.set_ylabel('Return Value')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: VaR vs CVaR bar chart
ax = axes[1, 0]
names_list = list(results.keys())
vars_vals = [results[name]['VaR'] for name in names_list]
cvars_vals = [results[name]['CVaR'] for name in names_list]

x = np.arange(len(names_list))
width = 0.35
ax.bar(x - width/2, vars_vals, width, label='VaR (5%)', color='steelblue')
ax.bar(x + width/2, cvars_vals, width, label='CVaR (5%)', color='darkred')
ax.set_ylabel('Loss (Return)')
ax.set_title('VaR vs CVaR Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names_list, rotation=15, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Excess over VaR (tail severity)
ax = axes[1, 1]
excesses = [results[name]['Excess'] for name in names_list]
ax.bar(names_list, excesses, color=['green' if e > 0 else 'red' for e in excesses], alpha=0.7)
ax.set_ylabel('CVaR - VaR (Tail Severity)')
ax.set_title('Excess Loss Beyond VaR')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Additional: Coherence check (subadditivity)
print("\n" + "="*70)
print("COHERENCE CHECK: Diversification Reduces CVaR")
print("="*70)

# Create portfolio of Normal + Fat-tailed
weights = [0.5, 0.5]
portfolio_returns = weights[0] * (returns_normal / returns_normal.std()) + \
                   weights[1] * (returns_fat / returns_fat.std())

portfolio_var = np.percentile(portfolio_returns, alpha * 100)
portfolio_cvar = portfolio_returns[portfolio_returns <= portfolio_var].mean()

weighted_cvar = weights[0] * results['Normal']['CVaR'] + \
                weights[1] * results['Fat-Tailed (t)']['CVaR']

print(f"Normal Distribution CVaR: {results['Normal']['CVaR']:.4f}")
print(f"Fat-Tailed Distribution CVaR: {results['Fat-Tailed (t)']['CVaR']:.4f}")
print(f"Weighted Sum (should be ≥ Portfolio): {weighted_cvar:.4f}")
print(f"Portfolio CVaR (50/50 blend): {portfolio_cvar:.4f}")
print(f"Subadditivity holds: {portfolio_cvar <= weighted_cvar}")