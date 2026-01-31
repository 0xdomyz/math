import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

np.random.seed(42)

class PermanentTemporaryDecomposition:
    def __init__(self):
        self.trades = []
        self.prices = []
        self.impact_decomposition = []
        
    def simulate_trade_sequence(self, order_sizes, num_days=20):
        """Simulate price evolution with permanent and temporary components"""
        price = 100.0
        prices_over_time = [price]
        permanent_levels = [0.0]
        
        for day in range(num_days):
            for order_size in order_sizes:
                # Permanent impact: Proportional to √(order size)
                permanent_component = 0.0005 * np.sqrt(order_size / 10000)
                
                # Temporary impact: Proportional to order size directly, mean-reverts
                temporary_component = 0.001 * (order_size / 100000)
                
                # Noise (random trade not affecting price long-term)
                noise = np.random.normal(0, 0.0005)
                
                # Total price impact
                total_impact = permanent_component + temporary_component + noise
                price += total_impact
                
                # Permanent shifts the baseline
                permanent_level = permanent_levels[-1] + permanent_component
                permanent_levels.append(permanent_level)
                
                # Temporary reverts partially each period
                temporary_reversion = -temporary_component * 0.3
                price += temporary_reversion
                
                prices_over_time.append(price)
                
                # Store for analysis
                self.trades.append({
                    'order_size': order_size,
                    'permanent': permanent_component,
                    'temporary': temporary_component,
                    'total': total_impact,
                    'price': price
                })
        
        return prices_over_time, permanent_levels
    
    def decompose_impact_vp(self, prices, order_flow, lags=20):
        """Vector autoregression decomposition of impacts"""
        # Price changes
        price_changes = np.diff(prices)
        
        # Order flow (1 = buy, -1 = sell)
        order_flow_array = np.array(order_flow)
        
        # Build lagged variables
        y = price_changes[lags:]
        X = np.ones((len(y), 1))
        
        for lag in range(1, lags + 1):
            X = np.column_stack([X, order_flow_array[lags - lag:-lag if lag < len(order_flow_array) else None]])
        
        # Regression
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # Temporary: Sum of all coefficients (one-period effect)
        temporary = coeffs[1]
        
        # Permanent: Cumulative effect (sum all future periods)
        # Approximate by ratio of coefficient to variance
        permanent = np.sum(coeffs[1:]) / len(coeffs[1:])
        
        return temporary, permanent
    
    def partial_adjustment_method(self, initial_price, final_price, prior_price):
        """Estimate permanent fraction using partial adjustment"""
        # α = (p_t - p_{t-1}) / (p* - p_{t-1})
        # α = 1: all permanent, α = 0: all temporary
        
        adjustment = initial_price - prior_price
        total_change = final_price - prior_price
        
        if adjustment != 0:
            alpha = adjustment / total_change if total_change != 0 else 1.0
        else:
            alpha = 0.0
        
        permanent_fraction = alpha
        temporary_fraction = 1.0 - alpha
        
        return permanent_fraction, temporary_fraction

# Scenario 1: Varying order sizes and their impact decomposition
print("Scenario 1: Impact Decomposition by Order Size")
print("=" * 80)

sim = PermanentTemporaryDecomposition()
order_sizes = [10000, 50000, 100000, 250000, 500000]
permanent_impacts = []
temporary_impacts = []
total_impacts = []
permanent_fractions = []

for order_size in order_sizes:
    permanent = 0.0005 * np.sqrt(order_size / 10000)
    temporary = 0.001 * (order_size / 100000)
    total = permanent + temporary
    
    permanent_impacts.append(permanent * 10000)  # Convert to cents
    temporary_impacts.append(temporary * 10000)
    total_impacts.append(total * 10000)
    permanent_fractions.append(permanent / total if total > 0 else 0)
    
    print(f"Order Size: {order_size:>10,} shares")
    print(f"  Permanent Impact: {permanent*10000:>8.2f} cents ({permanent/total*100:>5.1f}%)")
    print(f"  Temporary Impact: {temporary*10000:>8.2f} cents ({temporary/total*100:>5.1f}%)")
    print(f"  Total Impact:     {total*10000:>8.2f} cents")
    print()

# Scenario 2: Price evolution with permanent and temporary components
print("Scenario 2: Price Evolution (20 days, random orders)")
print("=" * 80)

order_sequence = np.random.choice([10000, 50000, 100000], size=20)
prices, permanent_levels = sim.simulate_trade_sequence(order_sequence, num_days=1)

print(f"Initial Price: ${prices[0]:.2f}")
print(f"Final Price:   ${prices[-1]:.2f}")
print(f"Total Change:  ${prices[-1] - prices[0]:.2f}")
print(f"Permanent Drift: ${permanent_levels[-1]:.4f}")
print(f"\nMean Reversion: {(prices[-1] - permanent_levels[-1]):.4f}")

# Scenario 3: Simulating mean reversion (temporary component decay)
print(f"\n\nScenario 3: Temporary Component Mean Reversion")
print("=" * 80)

time_periods = np.arange(0, 100)
permanent_base = 0.05
temporary_initial = 0.15
reversion_rate = 0.95  # Each period, 95% of temporary remains

temporary_over_time = []
cumulative_price = permanent_base

for t in time_periods:
    temporary = temporary_initial * (reversion_rate ** t)
    cumulative_price += temporary * (1 - reversion_rate)
    temporary_over_time.append(temporary)

print(f"Initial Temporary Impact: {temporary_initial:.4f}")
print(f"After 1 second:           {temporary_over_time[1]:.4f}")
print(f"After 10 seconds:         {temporary_over_time[10]:.4f}")
print(f"After 100 periods:        {temporary_over_time[-1]:.6f}")
print(f"Half-life (periods):      {-np.log(0.5) / np.log(reversion_rate):.1f}")

# Scenario 4: Stoll decomposition example
print(f"\n\nScenario 4: Stoll Decomposition (Spread Components)")
print("=" * 80)

bid_ask_spread = 0.10  # $0.10 spread

# Typical decomposition
adverse_selection_pct = 0.60
inventory_pct = 0.25
order_processing_pct = 0.15

adverse_selection = bid_ask_spread * adverse_selection_pct
inventory = bid_ask_spread * inventory_pct
order_processing = bid_ask_spread * order_processing_pct

print(f"Total Bid-Ask Spread:   ${bid_ask_spread:.4f}")
print(f"  Adverse Selection:    ${adverse_selection:.4f} ({adverse_selection_pct*100:.0f}%) - PERMANENT")
print(f"  Inventory Cost:       ${inventory:.4f} ({inventory_pct*100:.0f}%) - TEMPORARY")
print(f"  Order Processing:     ${order_processing:.4f} ({order_processing_pct*100:.0f}%) - TEMPORARY")
print(f"\nPermanent Component:    ${adverse_selection:.4f} ({adverse_selection_pct*100:.0f}%)")
print(f"Temporary Component:    ${inventory + order_processing:.4f} ({(inventory_pct + order_processing_pct)*100:.0f}%)")

# Scenario 5: Cross-sectional comparison
print(f"\n\nScenario 5: Permanent Impact Across Asset Types")
print("=" * 80)

asset_types = [
    {'name': 'Large Cap Stock', 'permanent_pct': 0.75, 'sample_size': 50},
    {'name': 'Small Cap Stock', 'permanent_pct': 0.55, 'sample_size': 50},
    {'name': 'ETF', 'permanent_pct': 0.80, 'sample_size': 50},
    {'name': 'Option', 'permanent_pct': 0.65, 'sample_size': 50},
    {'name': 'Futures', 'permanent_pct': 0.85, 'sample_size': 50},
]

for asset in asset_types:
    print(f"{asset['name']:>20}: {asset['permanent_pct']*100:>5.1f}% permanent impact")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Impact decomposition by order size
axes[0, 0].bar(np.arange(len(order_sizes)), permanent_impacts, label='Permanent', alpha=0.7)
axes[0, 0].bar(np.arange(len(order_sizes)), temporary_impacts, bottom=permanent_impacts, label='Temporary', alpha=0.7)
axes[0, 0].set_xticks(np.arange(len(order_sizes)))
axes[0, 0].set_xticklabels([f'{s/1000:.0f}K' for s in order_sizes])
axes[0, 0].set_ylabel('Price Impact (cents)')
axes[0, 0].set_title('Scenario 1: Permanent vs Temporary Components')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Permanent fraction by order size
axes[0, 1].plot(order_sizes, np.array(permanent_fractions)*100, 'o-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Order Size (shares)')
axes[0, 1].set_ylabel('Permanent Impact (%)')
axes[0, 1].set_title('Scenario 1: Permanent Fraction of Total Impact')
axes[0, 1].set_xscale('log')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].axhline(y=70, color='r', linestyle='--', label='Typical (70%)', alpha=0.5)
axes[0, 1].legend()

# Plot 3: Mean reversion of temporary component
axes[1, 0].semilogy(time_periods, temporary_over_time, linewidth=2)
axes[1, 0].axhline(y=temporary_initial * 0.5, color='r', linestyle='--', label='Half-Life', alpha=0.5)
axes[1, 0].set_xlabel('Time Periods')
axes[1, 0].set_ylabel('Temporary Impact (log scale)')
axes[1, 0].set_title('Scenario 3: Mean Reversion of Temporary Component')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Asset type comparison
asset_names = [a['name'] for a in asset_types]
permanent_pcts = [a['permanent_pct']*100 for a in asset_types]
colors_assets = plt.cm.viridis(np.linspace(0, 1, len(asset_names)))

bars = axes[1, 1].barh(asset_names, permanent_pcts, color=colors_assets)
axes[1, 1].set_xlabel('Permanent Impact (%)')
axes[1, 1].set_title('Scenario 5: Permanent Impact by Asset Type')
axes[1, 1].set_xlim([0, 100])
axes[1, 1].grid(alpha=0.3, axis='x')

for bar, pct in zip(bars, permanent_pcts):
    width = bar.get_width()
    axes[1, 1].text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{pct:.0f}%', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nSummary Statistics:")
print("=" * 80)
print(f"Average Permanent Impact: {np.mean(permanent_fractions)*100:.1f}%")
print(f"Range: {np.min(permanent_fractions)*100:.1f}% - {np.max(permanent_fractions)*100:.1f}%")
print(f"Typical decomposition: 70% permanent, 30% temporary")
print(f"Recovery time scale: Minutes to hours (temporary)")
print(f"Permanent: Persists indefinitely (no reversion)")
