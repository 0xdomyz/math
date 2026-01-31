import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from scipy.optimize import curve_fit
from dataclasses import dataclass

@dataclass
class ImpactParameters:
    """Market impact parameters"""
    perm_coeff: float      # Permanent impact per share
    temp_coeff: float      # Temporary impact per share
    decay_rate: float      # Exponential decay rate (1/seconds)
    spread: float          # Bid-ask spread
    volatility: float      # Price volatility

class MarketImpactSimulator:
    """Simulate prices with permanent and temporary impact"""
    
    def __init__(self, params: ImpactParameters, dt=1.0):
        self.params = params
        self.dt = dt  # Time step (seconds)
        
        # State
        self.price = 100.0
        self.temp_impact = 0.0
        
        # History
        self.prices = [self.price]
        self.mid_prices = [self.price]
        self.temp_impacts = [0.0]
        self.trades = []
    
    def execute_trade(self, volume, side='buy'):
        """
        Execute trade with permanent and temporary impact
        
        volume: Number of shares (positive)
        side: 'buy' or 'sell'
        """
        direction = 1 if side == 'buy' else -1
        signed_volume = direction * volume
        
        # Permanent impact (information)
        perm_impact = self.params.perm_coeff * signed_volume
        
        # Add to base price
        self.price += perm_impact
        
        # Temporary impact (liquidity)
        # Square-root law plus spread
        temp_impact = (self.params.temp_coeff * np.sqrt(abs(volume)) + 
                      self.params.spread / 2) * direction
        
        self.temp_impact += temp_impact
        
        # Execution price
        execution_price = self.price + self.temp_impact
        
        # Record
        self.trades.append({
            'volume': volume,
            'side': side,
            'execution_price': execution_price,
            'mid_price': self.price,
            'temp_impact': self.temp_impact,
            'perm_impact': perm_impact
        })
        
        return execution_price
    
    def step(self):
        """Advance one time step (temporary impact decays)"""
        # Exponential decay of temporary impact
        decay = np.exp(-self.params.decay_rate * self.dt)
        self.temp_impact *= decay
        
        # Add random volatility to fundamental price
        vol_shock = np.random.normal(0, self.params.volatility * np.sqrt(self.dt))
        self.price += vol_shock
        
        # Record
        self.prices.append(self.price + self.temp_impact)
        self.mid_prices.append(self.price)
        self.temp_impacts.append(self.temp_impact)
    
    def simulate_execution(self, total_volume, n_slices, side='buy'):
        """Simulate sliced execution"""
        slice_size = total_volume // n_slices
        
        for i in range(n_slices):
            # Execute slice
            self.execute_trade(slice_size, side=side)
            
            # Wait a bit
            for _ in range(5):  # 5 time steps between slices
                self.step()
        
        # Continue for monitoring period
        for _ in range(100):  # 100 more steps
            self.step()

def decompose_impact_simple(prices, trade_times, post_trade_horizon=30):
    """
    Simple permanent/temporary decomposition
    
    Permanent = price[t + horizon] - price[t - 1]
    Temporary = price[t] - price[t + horizon]
    """
    results = []
    
    for t in trade_times:
        if t + post_trade_horizon >= len(prices):
            continue
        
        # Pre-trade price
        pre_price = prices[t - 1] if t > 0 else prices[t]
        
        # Execution price (immediate)
        exec_price = prices[t]
        
        # Post-trade price (after horizon)
        post_price = prices[t + post_trade_horizon]
        
        # Decomposition
        total_impact = exec_price - pre_price
        permanent = post_price - pre_price
        temporary = total_impact - permanent
        
        results.append({
            'trade_time': t,
            'total_impact': total_impact,
            'permanent': permanent,
            'temporary': temporary,
            'perm_ratio': permanent / total_impact if total_impact != 0 else 0
        })
    
    return pd.DataFrame(results)

def fit_decay_function(prices, trade_time, max_horizon=100):
    """
    Fit exponential decay to post-trade price path
    
    Model: P(t) = P_perm + ΔP_temp × exp(-λt)
    """
    if trade_time + max_horizon >= len(prices):
        max_horizon = len(prices) - trade_time - 1
    
    # Extract post-trade prices
    post_prices = prices[trade_time:trade_time + max_horizon]
    times = np.arange(len(post_prices))
    
    # Initial price
    P0 = prices[trade_time]
    
    # Fit exponential: P(t) = a + b*exp(-c*t)
    def decay_model(t, a, b, c):
        return a + b * np.exp(-c * t)
    
    try:
        # Initial guess
        p0 = [post_prices[-1], post_prices[0] - post_prices[-1], 0.1]
        
        # Fit
        popt, pcov = curve_fit(decay_model, times, post_prices, p0=p0, maxfev=5000)
        
        permanent_level = popt[0]
        temporary_initial = popt[1]
        decay_rate = popt[2]
        
        # Fitted curve
        fitted = decay_model(times, *popt)
        
        return {
            'permanent': permanent_level - prices[trade_time - 1],
            'temporary': temporary_initial,
            'decay_rate': decay_rate,
            'half_life': np.log(2) / decay_rate if decay_rate > 0 else np.inf,
            'fitted': fitted,
            'R2': 1 - np.sum((post_prices - fitted)**2) / np.sum((post_prices - post_prices.mean())**2)
        }
    except:
        return None

# Simulation
print("="*80)
print("PERMANENT vs TEMPORARY MARKET IMPACT")
print("="*80)

# Parameters
params = ImpactParameters(
    perm_coeff=0.0001,     # $0.01 per 100 shares (permanent)
    temp_coeff=0.05,       # Square-root temp impact
    decay_rate=0.05,       # Decay rate (per time step)
    spread=0.02,           # $0.02 spread
    volatility=0.01        # 1% volatility per sqrt(time)
)

print(f"\nSimulation Parameters:")
print(f"  Permanent coefficient: ${params.perm_coeff:.6f} per share")
print(f"  Temporary coefficient: ${params.temp_coeff:.4f} per √share")
print(f"  Decay rate: {params.decay_rate:.4f} per step")
print(f"  Half-life: {np.log(2)/params.decay_rate:.1f} steps")
print(f"  Bid-ask spread: ${params.spread:.4f}")

# Simulate execution
simulator = MarketImpactSimulator(params, dt=1.0)

print(f"\nSimulating: Buy 10,000 shares in 10 slices (1,000 each)")
initial_price = simulator.price

simulator.simulate_execution(total_volume=10000, n_slices=10, side='buy')

# Analyze
df_trades = pd.DataFrame(simulator.trades)
prices = np.array(simulator.prices)
mid_prices = np.array(simulator.mid_prices)
temp_impacts = np.array(simulator.temp_impacts)

print(f"\nExecution Summary:")
print(f"  Initial price: ${initial_price:.4f}")
print(f"  Final mid-price: ${simulator.price:.4f}")
print(f"  Price change: ${simulator.price - initial_price:.4f}")
print(f"  Average execution: ${df_trades['execution_price'].mean():.4f}")
print(f"  Total slippage: ${df_trades['execution_price'].mean() - initial_price:.4f}")

# Decompose impact for each trade
trade_times = list(range(0, len(df_trades) * 6, 6))  # Every 6 steps (1 trade + 5 wait)
df_decomp = decompose_impact_simple(prices, trade_times[:len(df_trades)], post_trade_horizon=30)

print(f"\nImpact Decomposition (simple method, 30-step horizon):")
print(f"  Average total impact: ${df_decomp['total_impact'].mean():.4f}")
print(f"  Average permanent: ${df_decomp['permanent'].mean():.4f}")
print(f"  Average temporary: ${df_decomp['temporary'].mean():.4f}")
print(f"  Permanent ratio: {df_decomp['perm_ratio'].mean():.1%}")

# Fit decay function to first trade
print(f"\nDecay Analysis (first trade):")
fit_result = fit_decay_function(prices, trade_times[0], max_horizon=80)

if fit_result:
    print(f"  Permanent impact: ${fit_result['permanent']:.4f}")
    print(f"  Initial temporary: ${fit_result['temporary']:.4f}")
    print(f"  Decay rate: {fit_result['decay_rate']:.4f}")
    print(f"  Half-life: {fit_result['half_life']:.1f} steps")
    print(f"  R²: {fit_result['R2']:.3f}")

# Compare different execution strategies
print("\n" + "="*80)
print("STRATEGY COMPARISON: Fast vs Slow Execution")
print("="*80)

strategies = {
    'Aggressive (2 slices)': 2,
    'Moderate (10 slices)': 10,
    'Patient (50 slices)': 50
}

strategy_results = []

for name, n_slices in strategies.items():
    # Reset simulator
    sim = MarketImpactSimulator(params, dt=1.0)
    initial = sim.price
    
    # Execute
    sim.simulate_execution(total_volume=10000, n_slices=n_slices, side='buy')
    
    # Metrics
    trades_df = pd.DataFrame(sim.trades)
    avg_exec = trades_df['execution_price'].mean()
    final_mid = sim.price
    
    total_cost = avg_exec - initial
    perm_cost = final_mid - initial
    temp_cost = total_cost - perm_cost
    
    strategy_results.append({
        'strategy': name,
        'n_slices': n_slices,
        'total_cost': total_cost,
        'permanent': perm_cost,
        'temporary': temp_cost,
        'perm_ratio': perm_cost / total_cost if total_cost > 0 else 0
    })

df_strategies = pd.DataFrame(strategy_results)

print("\nStrategy Comparison:")
print(df_strategies.to_string(index=False))

print(f"\nKey Insight:")
print(f"  Patient execution reduces temporary but increases time risk")
print(f"  Aggressive execution minimizes information leakage but pays liquidity premium")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Price evolution with components
axes[0, 0].plot(mid_prices, label='Mid-price (with permanent)', linewidth=2)
axes[0, 0].plot(prices, label='Market price (mid + temporary)', linewidth=1.5, alpha=0.7)
axes[0, 0].scatter(trade_times[:len(df_trades)], 
                   df_trades['execution_price'], 
                   c='red', s=50, zorder=5, label='Executions')
axes[0, 0].axhline(initial_price, color='green', linestyle='--', label='Initial price')
axes[0, 0].set_title('Price Evolution During Execution')
axes[0, 0].set_xlabel('Time Step')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Temporary impact decay
axes[0, 1].plot(temp_impacts, linewidth=2, color='purple')
axes[0, 1].set_title('Temporary Impact Over Time')
axes[0, 1].set_xlabel('Time Step')
axes[0, 1].set_ylabel('Temporary Impact ($)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Impact decomposition
if not df_decomp.empty:
    width = 0.35
    x = np.arange(len(df_decomp))
    
    axes[1, 0].bar(x - width/2, df_decomp['permanent'], width, 
                   label='Permanent', alpha=0.7, color='blue')
    axes[1, 0].bar(x + width/2, df_decomp['temporary'], width,
                   label='Temporary', alpha=0.7, color='orange')
    
    axes[1, 0].set_title('Impact Decomposition per Trade')
    axes[1, 0].set_xlabel('Trade Number')
    axes[1, 0].set_ylabel('Impact ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

# Plot 4: Decay function fit
if fit_result:
    t0 = trade_times[0]
    horizon = len(fit_result['fitted'])
    
    axes[1, 1].plot(range(horizon), prices[t0:t0+horizon], 'o-', 
                    label='Actual', alpha=0.6, markersize=4)
    axes[1, 1].plot(range(horizon), fit_result['fitted'], 'r--',
                    label=f'Fitted (λ={fit_result["decay_rate"]:.3f})', linewidth=2)
    axes[1, 1].axhline(prices[t0-1] + fit_result['permanent'], 
                       color='green', linestyle='--', label='Permanent level')
    
    axes[1, 1].set_title(f'Exponential Decay Fit (R²={fit_result["R2"]:.3f})')
    axes[1, 1].set_xlabel('Steps After Trade')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

# Plot 5: Strategy comparison - costs
axes[2, 0].bar(df_strategies['strategy'], df_strategies['total_cost'], 
               alpha=0.7, label='Total')
axes[2, 0].bar(df_strategies['strategy'], df_strategies['permanent'],
               alpha=0.7, label='Permanent')
axes[2, 0].set_title('Total Cost by Strategy')
axes[2, 0].set_ylabel('Cost ($)')
axes[2, 0].legend()
axes[2, 0].tick_params(axis='x', rotation=15)
axes[2, 0].grid(axis='y', alpha=0.3)

# Plot 6: Permanent ratio
axes[2, 1].bar(df_strategies['strategy'], df_strategies['perm_ratio']*100, alpha=0.7)
axes[2, 1].set_title('Permanent as % of Total Impact')
axes[2, 1].set_ylabel('Permanent Ratio (%)')
axes[2, 1].tick_params(axis='x', rotation=15)
axes[2, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Permanent impact persists indefinitely, reflects information content")
print(f"2. Temporary impact reverts exponentially (typical half-life: 5-30 minutes)")
print(f"3. Permanent/temporary ratio typically 30-70%, varies by strategy and stock")
print(f"4. Patient execution minimizes temporary but exposes to information leakage risk")
print(f"5. Decomposition critical for TCA and optimal execution strategy design")
print(f"6. Measurement requires sufficient post-trade horizon (avoid noise bias)")
