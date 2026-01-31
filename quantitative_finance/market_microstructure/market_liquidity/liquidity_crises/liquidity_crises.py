import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Flash Crash Simulation with Circuit Breaker
class MarketCrashSimulator:
    def __init__(self, n_traders=100, circuit_breaker_threshold=-0.07):
        self.n_traders = n_traders
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.halt_duration = 300  # 5 minutes in seconds
        self.price_history = []
        self.volume_history = []
        self.spread_history = []
        self.depth_history = []
        self.trader_states = np.random.choice(['buyer', 'seller', 'market_maker'], 
                                              n_traders, p=[0.4, 0.4, 0.2])
        
    def simulate_crash(self, n_seconds=1800, shock_time=300, shock_size=5000):
        """
        Simulate flash crash with large sell order shock
        
        Parameters:
        - n_seconds: simulation duration
        - shock_time: when shock occurs
        - shock_size: size of shock sell order
        """
        
        # Initialize
        price = 100.0
        base_spread = 0.02
        base_depth = 10000  # shares per level
        
        imbalance = 0  # Order flow imbalance
        volatility = 0.0001  # Per-second volatility
        
        halted = False
        halt_end_time = 0
        halt_triggered_time = None
        
        results = {
            'time': [],
            'price': [],
            'spread': [],
            'depth': [],
            'volume': [],
            'imbalance': [],
            'halted': [],
            'hft_active': []
        }
        
        for t in range(n_seconds):
            # Check if market is halted
            if halted:
                if t >= halt_end_time:
                    halted = False
                    print(f"Trading resumes at t={t}")
                    # Reset some panic
                    imbalance = imbalance * 0.5
                    volatility = volatility * 0.7
                else:
                    # Market is halted, no trading
                    results['time'].append(t)
                    results['price'].append(price)
                    results['spread'].append(np.nan)
                    results['depth'].append(0)
                    results['volume'].append(0)
                    results['imbalance'].append(imbalance)
                    results['halted'].append(True)
                    results['hft_active'].append(False)
                    continue
            
            # Large sell shock at shock_time
            if t == shock_time:
                imbalance -= shock_size
                print(f"SHOCK at t={t}: Large sell order {shock_size} shares")
            
            # Random order flow
            random_flow = np.random.normal(0, 100)
            imbalance += random_flow
            
            # Stop-loss cascades (if price falling rapidly)
            if t > shock_time and price < 99.0:
                # More sellers as price falls
                cascade_flow = -abs(price - 100) * 200
                imbalance += cascade_flow
            
            # Volatility adjustment (spikes during crisis)
            if abs(imbalance) > 1000:
                volatility = min(0.001, volatility * 1.5)
            else:
                volatility = max(0.0001, volatility * 0.95)
            
            # HFT withdrawal (when uncertainty high)
            uncertainty = abs(imbalance) / 1000
            hft_active_prob = max(0.1, 1 - uncertainty)
            hft_active = np.random.random() < hft_active_prob
            
            # Price impact
            price_change = -imbalance * 0.00005 + np.random.normal(0, volatility)
            price = price * (1 + price_change)
            
            # Spread widening (function of volatility and HFT presence)
            if hft_active:
                spread = base_spread * (1 + volatility * 1000)
            else:
                # Spread explodes when HFT withdraws
                spread = base_spread * (1 + volatility * 10000)
            
            # Order book depth (depletes with imbalance)
            depth = max(100, base_depth - abs(imbalance) * 0.5)
            if not hft_active:
                depth = depth * 0.1  # Depth vanishes
            
            # Volume (higher during crisis)
            volume = abs(random_flow) + abs(imbalance) * 0.01
            
            # Circuit breaker check
            price_drop = (price - 100) / 100
            if price_drop < self.circuit_breaker_threshold and not halted:
                halted = True
                halt_end_time = t + self.halt_duration
                halt_triggered_time = t
                print(f"CIRCUIT BREAKER triggered at t={t}, price=${price:.2f} ({price_drop*100:.1f}%)")
            
            # Decay imbalance
            imbalance = imbalance * 0.95
            
            # Record results
            results['time'].append(t)
            results['price'].append(price)
            results['spread'].append(spread)
            results['depth'].append(depth)
            results['volume'].append(volume)
            results['imbalance'].append(imbalance)
            results['halted'].append(halted)
            results['hft_active'].append(hft_active)
        
        return results, halt_triggered_time

# Run simulations: with and without circuit breaker
print("Flash Crash Simulation")
print("=" * 70)

# Scenario 1: WITH circuit breaker
sim_with_cb = MarketCrashSimulator(circuit_breaker_threshold=-0.07)
results_with_cb, halt_time_with = sim_with_cb.simulate_crash(
    n_seconds=1800, shock_time=300, shock_size=8000
)

print("\n" + "=" * 70)

# Scenario 2: WITHOUT circuit breaker (threshold = -100%)
sim_no_cb = MarketCrashSimulator(circuit_breaker_threshold=-1.0)
results_no_cb, halt_time_no = sim_no_cb.simulate_crash(
    n_seconds=1800, shock_time=300, shock_size=8000
)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Price evolution comparison
axes[0, 0].plot(results_with_cb['time'], results_with_cb['price'], 
               label='With Circuit Breaker', linewidth=2, alpha=0.8)
axes[0, 0].plot(results_no_cb['time'], results_no_cb['price'], 
               label='Without Circuit Breaker', linewidth=2, alpha=0.8)
axes[0, 0].axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 0].axhline(93, color='red', linestyle='--', linewidth=1, alpha=0.5, 
                  label='Circuit Breaker Threshold (-7%)')
if halt_time_with:
    axes[0, 0].axvline(halt_time_with, color='red', linestyle=':', linewidth=2, 
                      label=f'Halt Triggered (t={halt_time_with})')
axes[0, 0].set_xlabel('Time (seconds)')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Flash Crash: Price Evolution')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Analysis
min_price_with = min(results_with_cb['price'])
min_price_no = min(results_no_cb['price'])
final_price_with = results_with_cb['price'][-1]
final_price_no = results_no_cb['price'][-1]

print(f"\nPrice Impact Analysis:")
print(f"With Circuit Breaker:")
print(f"  Min Price: ${min_price_with:.2f} ({(min_price_with-100)/100*100:.2f}%)")
print(f"  Final Price: ${final_price_with:.2f} ({(final_price_with-100)/100*100:.2f}%)")
print(f"\nWithout Circuit Breaker:")
print(f"  Min Price: ${min_price_no:.2f} ({(min_price_no-100)/100*100:.2f}%)")
print(f"  Final Price: ${final_price_no:.2f} ({(final_price_no-100)/100*100:.2f}%)")
print(f"\nCircuit Breaker Impact:")
print(f"  Reduced max drawdown by: {(min_price_no - min_price_with):.2f} ({(min_price_with/min_price_no - 1)*100:.1f}%)")

# Plot 2: Spread dynamics
valid_spreads_with = [s if not np.isnan(s) else None for s in results_with_cb['spread']]
valid_spreads_no = [s for s in results_no_cb['spread']]

axes[0, 1].plot(results_with_cb['time'], valid_spreads_with, 
               label='With CB', linewidth=1.5, alpha=0.7)
axes[0, 1].plot(results_no_cb['time'], valid_spreads_no, 
               label='Without CB', linewidth=1.5, alpha=0.7)
axes[0, 1].set_xlabel('Time (seconds)')
axes[0, 1].set_ylabel('Bid-Ask Spread ($)')
axes[0, 1].set_title('Spread Widening During Crisis')
axes[0, 1].legend()
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# Spread statistics
max_spread_with = max([s for s in valid_spreads_with if s is not None])
max_spread_no = max(valid_spreads_no)
mean_spread_with = np.nanmean(results_with_cb['spread'])
mean_spread_no = np.mean(results_no_cb['spread'])

print(f"\nSpread Analysis:")
print(f"With Circuit Breaker:")
print(f"  Max Spread: ${max_spread_with:.4f} ({max_spread_with/0.02:.1f}x normal)")
print(f"  Mean Spread: ${mean_spread_with:.4f}")
print(f"\nWithout Circuit Breaker:")
print(f"  Max Spread: ${max_spread_no:.4f} ({max_spread_no/0.02:.1f}x normal)")
print(f"  Mean Spread: ${mean_spread_no:.4f}")

# Plot 3: Order book depth
axes[1, 0].plot(results_with_cb['time'], results_with_cb['depth'], 
               label='With CB', linewidth=1.5, alpha=0.7)
axes[1, 0].plot(results_no_cb['time'], results_no_cb['depth'], 
               label='Without CB', linewidth=1.5, alpha=0.7)
axes[1, 0].axhline(10000, color='black', linestyle='--', linewidth=1, 
                  alpha=0.5, label='Normal Depth')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Order Book Depth (shares)')
axes[1, 0].set_title('Liquidity Evaporation')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Depth statistics
min_depth_with = min(results_with_cb['depth'])
min_depth_no = min(results_no_cb['depth'])

print(f"\nLiquidity Analysis:")
print(f"With Circuit Breaker:")
print(f"  Min Depth: {min_depth_with:.0f} shares ({min_depth_with/10000*100:.1f}% of normal)")
print(f"\nWithout Circuit Breaker:")
print(f"  Min Depth: {min_depth_no:.0f} shares ({min_depth_no/10000*100:.1f}% of normal)")

# Plot 4: HFT participation
hft_participation_with = np.array([int(h) for h in results_with_cb['hft_active']])
hft_participation_no = np.array([int(h) for h in results_no_cb['hft_active']])

# Rolling window average
window = 60
hft_rolling_with = np.convolve(hft_participation_with, 
                               np.ones(window)/window, mode='valid')
hft_rolling_no = np.convolve(hft_participation_no, 
                             np.ones(window)/window, mode='valid')

axes[1, 1].plot(range(len(hft_rolling_with)), hft_rolling_with * 100, 
               label='With CB', linewidth=2, alpha=0.8)
axes[1, 1].plot(range(len(hft_rolling_no)), hft_rolling_no * 100, 
               label='Without CB', linewidth=2, alpha=0.8)
axes[1, 1].axhline(100, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 1].set_xlabel('Time (seconds)')
axes[1, 1].set_ylabel('HFT Participation Rate (%)')
axes[1, 1].set_title('Market Maker Withdrawal (60-sec rolling avg)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim([0, 105])

# HFT withdrawal stats
hft_active_with = np.mean(hft_participation_with)
hft_active_no = np.mean(hft_participation_no)

# During crisis period (300-600)
crisis_period = slice(300, 600)
hft_crisis_with = np.mean(hft_participation_with[crisis_period])
hft_crisis_no = np.mean(hft_participation_no[crisis_period])

print(f"\nHFT Participation:")
print(f"With Circuit Breaker:")
print(f"  Overall: {hft_active_with*100:.1f}%")
print(f"  During Crisis: {hft_crisis_with*100:.1f}%")
print(f"\nWithout Circuit Breaker:")
print(f"  Overall: {hft_active_no*100:.1f}%")
print(f"  During Crisis: {hft_crisis_no*100:.1f}%")

plt.tight_layout()
plt.show()

# Recovery analysis
print(f"\nRecovery Analysis:")

# Time to recover to within 1% of initial price
recovery_threshold = 99.0
recovery_time_with = None
recovery_time_no = None

for t, p in enumerate(results_with_cb['price']):
    if t > 300 and p >= recovery_threshold and recovery_time_with is None:
        recovery_time_with = t
        break

for t, p in enumerate(results_no_cb['price']):
    if t > 300 and p >= recovery_threshold and recovery_time_no is None:
        recovery_time_no = t
        break

if recovery_time_with:
    print(f"With Circuit Breaker: {recovery_time_with - 300} seconds to recover")
else:
    print(f"With Circuit Breaker: Did not recover within simulation")

if recovery_time_no:
    print(f"Without Circuit Breaker: {recovery_time_no - 300} seconds to recover")
else:
    print(f"Without Circuit Breaker: Did not recover within simulation")

# Market quality metrics
print(f"\nMarket Quality Comparison:")

# Transaction cost (spread)
total_spread_cost_with = np.nansum(results_with_cb['spread']) / len(results_with_cb['spread'])
total_spread_cost_no = np.sum(results_no_cb['spread']) / len(results_no_cb['spread'])

print(f"Average Transaction Cost:")
print(f"  With CB: ${total_spread_cost_with:.4f}")
print(f"  Without CB: ${total_spread_cost_no:.4f}")
print(f"  CB Reduces Cost By: {(1 - total_spread_cost_with/total_spread_cost_no)*100:.1f}%")

# Price volatility
price_returns_with = np.diff(results_with_cb['price']) / results_with_cb['price'][:-1]
price_returns_no = np.diff(results_no_cb['price']) / results_no_cb['price'][:-1]

vol_with = np.std(price_returns_with) * np.sqrt(252 * 6.5 * 3600)  # Annualized
vol_no = np.std(price_returns_no) * np.sqrt(252 * 6.5 * 3600)

print(f"\nPrice Volatility (annualized):")
print(f"  With CB: {vol_with*100:.1f}%")
print(f"  Without CB: {vol_no*100:.1f}%")
print(f"  CB Reduces Volatility By: {(1 - vol_with/vol_no)*100:.1f}%")

# Investor welfare (approximate)
# Welfare loss = sum of squared price deviations from fundamental
welfare_loss_with = sum([(p - 100)**2 for p in results_with_cb['price']])
welfare_loss_no = sum([(p - 100)**2 for p in results_no_cb['price']])

print(f"\nWelfare Loss (price dislocation):")
print(f"  With CB: {welfare_loss_with:.0f}")
print(f"  Without CB: {welfare_loss_no:.0f}")
print(f"  CB Reduces Welfare Loss By: {(1 - welfare_loss_with/welfare_loss_no)*100:.1f}%")
