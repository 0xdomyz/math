# %% [markdown]
# # walk forward analysis
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
print("Setup complete for topic: walk forward analysis")

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

# Generate synthetic price data
np.random.seed(123)
n_days = 1500
prices = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.008))
df = pd.DataFrame({'price': prices}, index=pd.date_range('2018-01-01', periods=n_days, freq='D'))

# Walk-forward parameters
is_days = 252  # In-sample: 1 year
oos_days = 63  # Out-of-sample: 3 months
ma_fast_range = [10, 20, 30, 40, 50]
ma_slow_range = [100, 150, 200]

# Storage for results
oos_results = []

# Walk-forward loop
start_idx = 0
while start_idx + is_days + oos_days <= len(df):
    # Define IS and OOS windows
    is_end = start_idx + is_days
    oos_end = is_end + oos_days
    is_data = df.iloc[start_idx:is_end]
    oos_data = df.iloc[is_end:oos_end]
    
    # Optimize on IS window (grid search)
    best_sharpe = -np.inf
    best_params = None
    
    for ma_fast in ma_fast_range:
        for ma_slow in ma_slow_range:
            if ma_fast >= ma_slow:
                continue
            
            # Compute signals on IS data
            is_data_copy = is_data.copy()
            is_data_copy['ma_fast'] = is_data_copy['price'].rolling(ma_fast).mean()
            is_data_copy['ma_slow'] = is_data_copy['price'].rolling(ma_slow).mean()
            is_data_copy['signal'] = (is_data_copy['ma_fast'] > is_data_copy['ma_slow']).astype(int)
            is_data_copy['returns'] = is_data_copy['price'].pct_change()
            is_data_copy['strategy_returns'] = is_data_copy['signal'].shift(1) * is_data_copy['returns']
            
            # IS Sharpe ratio
            is_sharpe = is_data_copy['strategy_returns'].mean() / is_data_copy['strategy_returns'].std() * np.sqrt(252)
            
            if is_sharpe > best_sharpe:
                best_sharpe = is_sharpe
                best_params = (ma_fast, ma_slow)
    
    # Test best params on OOS window
    ma_fast, ma_slow = best_params
    oos_data_copy = oos_data.copy()
    oos_data_copy['ma_fast'] = oos_data_copy['price'].rolling(ma_fast).mean()
    oos_data_copy['ma_slow'] = oos_data_copy['price'].rolling(ma_slow).mean()
    oos_data_copy['signal'] = (oos_data_copy['ma_fast'] > oos_data_copy['ma_slow']).astype(int)
    oos_data_copy['returns'] = oos_data_copy['price'].pct_change()
    oos_data_copy['strategy_returns'] = oos_data_copy['signal'].shift(1) * oos_data_copy['returns']
    
    oos_sharpe = oos_data_copy['strategy_returns'].mean() / oos_data_copy['strategy_returns'].std() * np.sqrt(252)
    oos_cum_return = (1 + oos_data_copy['strategy_returns']).prod() - 1
    
    oos_results.append({
        'is_start': is_data.index[0],
        'oos_start': oos_data.index[0],
        'best_params': best_params,
        'is_sharpe': best_sharpe,
        'oos_sharpe': oos_sharpe,
        'oos_cum_return': oos_cum_return
    })
    
    # Move to next window (rolling)
    start_idx += oos_days

# Aggregate OOS performance
oos_df = pd.DataFrame(oos_results)
print("Walk-Forward Analysis Results:\n")
print(oos_df[['oos_start', 'best_params', 'is_sharpe', 'oos_sharpe', 'oos_cum_return']])
print(f"\nAverage IS Sharpe: {oos_df['is_sharpe'].mean():.2f}")
print(f"Average OOS Sharpe: {oos_df['oos_sharpe'].mean():.2f}")
print(f"OOS Sharpe Std Dev: {oos_df['oos_sharpe'].std():.2f}")
print(f"Fraction Positive OOS Returns: {(oos_df['oos_cum_return'] > 0).mean():.1%}")

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
    ax = df.plot(x='t', y='signal', figsize=(10, 4), title='walk forward analysis - Signal View')
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
