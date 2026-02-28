# %% [markdown]
# # portfolio construction
#
# This interactive script follows the quantitative finance topic template.

# %% [markdown]
# ## Section 1 - Overview & Setup
# Define imports, reproducibility settings, and runtime configuration.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
print("Setup complete for topic: portfolio construction")

# %% [markdown]
# ## Section 2 - Data Generation
# Generate or load data used by the modeling workflow.

# %%
if 'df' not in globals():
    t = np.arange(500)
    base = np.sin(t / 20.0)
    noise = np.random.normal(0, 0.2, size=len(t))
    df = pd.DataFrame({'t': t, 'signal': base + noise})
print("Data shape:", df.shape)
print(df.head(3))

# %% [markdown]
# ## Section 3 - Model Implementation
# Core topic logic and implementation details.

# %%
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Simulate 10 assets: expected returns and covariance
np.random.seed(42)
n_assets = 10
asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

# Expected annual returns (random 5%-15%)
expected_returns = np.random.uniform(0.05, 0.15, n_assets)

# Covariance matrix (realistic correlation structure)
correlation = np.random.uniform(0.1, 0.6, (n_assets, n_assets))
np.fill_diagonal(correlation, 1.0)
correlation = (correlation + correlation.T) / 2  # Symmetrize
volatilities = np.random.uniform(0.10, 0.30, n_assets)  # 10%-30% annual vol
cov_matrix = np.outer(volatilities, volatilities) * correlation

# Portfolio optimization: maximize Sharpe ratio
# Sharpe = (w'mu - rf) / sqrt(w'Sigma w)
# Equivalent: minimize -Sharpe or minimize variance for target return

def portfolio_stats(weights, returns, cov):
    port_return = np.dot(weights, returns)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

def negative_sharpe(weights, returns, cov):
    _, _, sharpe = portfolio_stats(weights, returns, cov)
    return -sharpe

# Constraints: weights sum to 1, long-only, max 20% per asset
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
]
bounds = [(0, 0.20) for _ in range(n_assets)]  # Long-only, max 20% per asset

# Initial guess: equal weight
initial_weights = np.ones(n_assets) / n_assets

# Optimize
result = minimize(
    negative_sharpe,
    initial_weights,
    args=(expected_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
opt_return, opt_vol, opt_sharpe = portfolio_stats(optimal_weights, expected_returns, cov_matrix)

# Compare with equal-weight portfolio
equal_weights = np.ones(n_assets) / n_assets
eq_return, eq_vol, eq_sharpe = portfolio_stats(equal_weights, expected_returns, cov_matrix)

# Display results
print("=" * 60)
print("PORTFOLIO CONSTRUCTION RESULTS")
print("=" * 60)
print("\nOptimal Portfolio Weights:")
for asset, weight in zip(asset_names, optimal_weights):
    if weight > 0.01:  # Show only meaningful positions
        print(f"  {asset:12s}: {weight:>6.2%}")

print(f"\nOptimal Portfolio:")
print(f"  Expected Return: {opt_return:>6.2%}")
print(f"  Volatility:      {opt_vol:>6.2%}")
print(f"  Sharpe Ratio:    {opt_sharpe:>6.2f}")

print(f"\nEqual-Weight Portfolio:")
print(f"  Expected Return: {eq_return:>6.2%}")
print(f"  Volatility:      {eq_vol:>6.2%}")
print(f"  Sharpe Ratio:    {eq_sharpe:>6.2f}")

print(f"\nImprovement:")
print(f"  Sharpe Gain:     {opt_sharpe - eq_sharpe:>+6.2f} ({(opt_sharpe/eq_sharpe - 1)*100:>+.1f}%)")

# Risk decomposition (marginal contribution to risk)
marginal_risk = np.dot(cov_matrix, optimal_weights) / opt_vol
risk_contribution = optimal_weights * marginal_risk
print(f"\nTop 3 Risk Contributors:")
top_contributors = np.argsort(risk_contribution)[-3:][::-1]
for idx in top_contributors:
    print(f"  {asset_names[idx]:12s}: {risk_contribution[idx]/opt_vol:>6.2%} of portfolio risk")

# %% [markdown]
# ## Section 4 - Training & Evaluation
# Compute basic evaluation metrics for demonstration.

# %%
if 'df' in globals() and 'signal' in df.columns:
    eval_mean = float(df['signal'].mean())
    eval_std = float(df['signal'].std())
    print(f"Evaluation summary -> mean: {eval_mean:.4f}, std: {eval_std:.4f}")
else:
    print("Evaluation note: custom topic code executed; add topic-specific metrics as needed.")

# %% [markdown]
# ## Section 5 - Visualization & Interpretation
# Visualize outputs to support interpretation.

# %%
if 'df' in globals() and 'signal' in df.columns:
    ax = df.plot(x='t', y='signal', figsize=(10, 4), title='portfolio construction - Signal View')
    ax.set_ylabel('signal')
    plt.tight_layout()
    plt.show()
else:
    print("Visualization note: add topic-specific plots from model outputs.")

# %% [markdown]
# ## Section 6 - Summary & Deployment
#
# Key takeaways:
# - End-to-end workflow scaffold is in place.
# - Replace placeholder evaluation/visualization with production metrics and controls.
# - Validate assumptions under realistic transaction costs, latency, and liquidity constraints.
