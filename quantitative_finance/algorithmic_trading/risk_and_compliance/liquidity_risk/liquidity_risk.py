# %% [markdown]
# # liquidity risk
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
print("Setup complete for topic: liquidity risk")

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

# Market parameters
np.random.seed(42)
initial_price = 100.0
n_steps = 100  # Time steps (minutes)

# Order to execute
order_size = 50000  # shares
order_side = 'sell'  # Selling pressure

# Scenario 1: Normal liquidity
normal_spread = 0.02  # 2 cents
normal_depth = 10000  # shares per price level
normal_resilience = 0.90  # 90% of liquidity replenishes each period

# Scenario 2: Stressed liquidity
stressed_spread = 0.20  # 20 cents (10Ã— wider)
stressed_depth = 1000  # shares (90% reduction)
stressed_resilience = 0.50  # 50% replenishment (slower recovery)

def simulate_execution(order_size, spread, depth, resilience, n_steps):
    """Simulate order execution with market impact"""
    prices = [initial_price]
    remaining = order_size
    cumulative_cost = 0
    available_liquidity = depth
    
    for t in range(n_steps):
        if remaining <= 0:
            prices.append(prices[-1])
            continue
        
        # Determine execution size (min of remaining, available liquidity, or 1% of order)
        execution_size = min(remaining, available_liquidity, order_size * 0.01)
        
        # Market impact: temporary (proportional to size / depth) + spread cost
        temporary_impact = (execution_size / depth) * 0.50  # 50 cent max impact
        spread_cost = spread / 2  # Pay half-spread on average
        
        # Price moves down (selling pressure) by temporary impact
        new_price = prices[-1] - temporary_impact - spread_cost
        
        # Permanent impact (10% of temporary impact persists)
        permanent_impact = temporary_impact * 0.10
        
        # Update state
        remaining -= execution_size
        cumulative_cost += execution_size * (initial_price - new_price)
        prices.append(new_price - permanent_impact)
        
        # Liquidity replenishes partially
        available_liquidity = depth * resilience + (depth - available_liquidity) * (1 - resilience)
    
    # Average execution price
    avg_price = initial_price - (cumulative_cost / order_size)
    slippage = (initial_price - avg_price) / initial_price * 10000  # bps
    
    return prices, avg_price, slippage

# Run simulations
prices_normal, avg_price_normal, slippage_normal = simulate_execution(
    order_size, normal_spread, normal_depth, normal_resilience, n_steps
)

prices_stressed, avg_price_stressed, slippage_stressed = simulate_execution(
    order_size, stressed_spread, stressed_depth, stressed_resilience, n_steps
)

# Display results
print("=" * 80)
print("LIQUIDITY RISK: MARKET IMPACT SIMULATION")
print("=" * 80)
print(f"Order: SELL {order_size:,} shares")
print(f"Initial Price: ${initial_price:.2f}")
print()

print("SCENARIO 1: NORMAL LIQUIDITY")
print("-" * 80)
print(f"  Bid-Ask Spread:           ${normal_spread:.2f}  ({normal_spread/initial_price*10000:.0f} bps)")
print(f"  Order Book Depth:         {normal_depth:,} shares per level")
print(f"  Liquidity Resilience:     {normal_resilience:.0%} replenishment")
print(f"\n  Average Execution Price:  ${avg_price_normal:.4f}")
print(f"  Slippage:                 {slippage_normal:.1f} bps")
print(f"  Total Cost:               ${(initial_price - avg_price_normal) * order_size:,.0f}")
print()

print("SCENARIO 2: STRESSED LIQUIDITY (Crisis)")
print("-" * 80)
print(f"  Bid-Ask Spread:           ${stressed_spread:.2f}  ({stressed_spread/initial_price*10000:.0f} bps)  [10Ã— wider]")
print(f"  Order Book Depth:         {stressed_depth:,} shares per level  [90% reduction]")
print(f"  Liquidity Resilience:     {stressed_resilience:.0%} replenishment  [slower recovery]")
print(f"\n  Average Execution Price:  ${avg_price_stressed:.4f}")
print(f"  Slippage:                 {slippage_stressed:.1f} bps")
print(f"  Total Cost:               ${(initial_price - avg_price_stressed) * order_size:,.0f}")
print()

print("COMPARISON")
print("-" * 80)
print(f"  Slippage Increase:        {slippage_stressed / slippage_normal:.1f}Ã— higher in crisis")
print(f"  Additional Cost:          ${(avg_price_normal - avg_price_stressed) * order_size:,.0f}")
print("=" * 80)

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price path
axes[0].plot(prices_normal, label='Normal Liquidity', linewidth=2, color='blue')
axes[0].plot(prices_stressed, label='Stressed Liquidity', linewidth=2, color='red')
axes[0].axhline(initial_price, color='black', linestyle='--', linewidth=1, label='Initial Price')
axes[0].axhline(avg_price_normal, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].axhline(avg_price_stressed, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
axes[0].set_ylabel('Price ($)', fontsize=12, fontweight='bold')
axes[0].set_title('Liquidity Risk: Market Impact During Order Execution', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# Slippage accumulation
cumulative_slippage_normal = (initial_price - np.array(prices_normal)) / initial_price * 10000
cumulative_slippage_stressed = (initial_price - np.array(prices_stressed)) / initial_price * 10000

axes[1].plot(cumulative_slippage_normal, label='Normal Liquidity', linewidth=2, color='blue')
axes[1].plot(cumulative_slippage_stressed, label='Stressed Liquidity', linewidth=2, color='red')
axes[1].set_ylabel('Cumulative Slippage (bps)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('liquidity_risk_simulation.png', dpi=150)
plt.show()

# Liquidity stress test summary
print("\n" + "=" * 80)
print("LIQUIDITY STRESS TEST SUMMARY")
print("=" * 80)
print("Portfolio: $50M position to liquidate")
print()
print(f"{'Scenario':<25} {'Slippage (bps)':<20} {'Cost ($)':<20} {'Days to Complete':<20}")
print("-" * 80)
print(f"{'Normal (5-day VWAP)':<25} {slippage_normal:<20.1f} ${(initial_price - avg_price_normal) * order_size:<19,.0f} {'5 days':<20}")
print(f"{'Stressed (Urgent)':<25} {slippage_stressed:<20.1f} ${(initial_price - avg_price_stressed) * order_size:<19,.0f} {'1 day (forced)':<20}")
print(f"{'Crisis (Firesale)':<25} {slippage_stressed * 2:<20.1f} ${(initial_price - avg_price_stressed) * order_size * 2:<19,.0f} {'Immediate (minutes)':<20}")
print("=" * 80)

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
    ax = df.plot(x='t', y='signal', figsize=(10, 4), title='liquidity risk - Signal View')
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
