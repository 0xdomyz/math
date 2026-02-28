# %% [markdown]
# # monte carlo simulation
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
print("Setup complete for topic: monte carlo simulation")

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
import matplotlib.pyplot as plt

# Simulate strategy returns (500 daily returns)
np.random.seed(321)
n_days = 500
true_mean = 0.0008  # 0.08% daily
true_vol = 0.015    # 1.5% daily
strategy_returns = np.random.normal(true_mean, true_vol, n_days)

# Calculate historical Sharpe
historical_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
print(f"Historical Sharpe Ratio: {historical_sharpe:.2f}")

# Monte Carlo Bootstrap (10,000 simulations)
n_simulations = 10000
bootstrap_sharpes = []

for sim in range(n_simulations):
    # Resample returns with replacement
    resampled_returns = np.random.choice(strategy_returns, size=n_days, replace=True)
    sharpe = resampled_returns.mean() / resampled_returns.std() * np.sqrt(252)
    bootstrap_sharpes.append(sharpe)

bootstrap_sharpes = np.array(bootstrap_sharpes)

# Calculate confidence intervals
ci_5th = np.percentile(bootstrap_sharpes, 5)
ci_50th = np.percentile(bootstrap_sharpes, 50)
ci_95th = np.percentile(bootstrap_sharpes, 95)
mean_sharpe = bootstrap_sharpes.mean()
std_sharpe = bootstrap_sharpes.std()

print(f"\nMonte Carlo Results (n={n_simulations:,}):")
print(f"Mean Sharpe:          {mean_sharpe:.2f}")
print(f"Median Sharpe:        {ci_50th:.2f}")
print(f"Std Dev of Sharpe:    {std_sharpe:.2f}")
print(f"95% Confidence Interval: [{ci_5th:.2f}, {ci_95th:.2f}]")

# Probability of negative Sharpe
prob_negative = (bootstrap_sharpes < 0).mean()
print(f"Probability of Sharpe < 0: {prob_negative:.1%}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_sharpes, bins=50, alpha=0.7, edgecolor='black', density=True)
plt.axvline(historical_sharpe, color='red', linestyle='--', linewidth=2, label=f'Historical Sharpe ({historical_sharpe:.2f})')
plt.axvline(ci_5th, color='orange', linestyle=':', label=f'5th Percentile ({ci_5th:.2f})')
plt.axvline(ci_95th, color='orange', linestyle=':', label=f'95th Percentile ({ci_95th:.2f})')
plt.xlabel('Sharpe Ratio')
plt.ylabel('Density')
plt.title('Monte Carlo Bootstrap: Sharpe Ratio Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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
    ax = df.plot(x='t', y='signal', figsize=(10, 4), title='monte carlo simulation - Signal View')
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
