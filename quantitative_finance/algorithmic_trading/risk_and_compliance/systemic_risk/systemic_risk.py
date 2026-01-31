import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulate order book dynamics with liquidity withdrawal
np.random.seed(42)
n_seconds = 600  # 10-minute window
n_agents = 100

# Price and liquidity arrays
prices = np.zeros(n_seconds)
prices[0] = 100
liquidity = np.ones(n_seconds) * 50  # Order book depth ($50M)
volatility = np.ones(n_seconds) * 0.01

# Agent behavior
def agent_action(price_change, current_volatility, agent_risk_tolerance):
    """Agents sell if volatility exceeds threshold"""
    if current_volatility > agent_risk_tolerance:
        return -1  # Sell
    elif price_change < -0.02:  # Stop-loss trigger
        return -1
    else:
        return 0  # Hold

# Simulate with circuit breaker
circuit_breaker_triggered = False
halt_duration = 30  # 30 seconds
halt_counter = 0

for t in range(1, n_seconds):
    if halt_counter > 0:
        # Market halted
        prices[t] = prices[t-1]
        liquidity[t] = min(50, liquidity[t-1] * 1.1)  # Liquidity returns
        volatility[t] = volatility[t-1] * 0.9  # Vol decays
        halt_counter -= 1
        continue
    
    # Calculate price change
    price_change = (prices[t-1] - prices[max(0, t-10)]) / prices[max(0, t-10)]
    
    # Liquidity withdrawal based on volatility
    if volatility[t-1] > 0.03:
        liquidity[t] = liquidity[t-1] * 0.85  # Rapid withdrawal
    else:
        liquidity[t] = min(50, liquidity[t-1] * 1.02)  # Slow recovery
    
    # Agent actions
    net_order_flow = 0
    for agent in range(n_agents):
        risk_tolerance = 0.02 + 0.01 * np.random.rand()  # Heterogeneous
        action = agent_action(price_change, volatility[t-1], risk_tolerance)
        net_order_flow += action
    
    # Market impact (inverse to liquidity)
    impact = net_order_flow / liquidity[t] * 0.5
    prices[t] = prices[t-1] * (1 + impact + np.random.normal(0, 0.001))
    
    # Update realized volatility
    if t > 20:
        returns = np.diff(prices[t-20:t]) / prices[t-21:t-1]
        volatility[t] = np.std(returns)
    
    # Circuit breaker check (-5% threshold)
    if (prices[t] - prices[max(0, t-60)]) / prices[max(0, t-60)] < -0.05:
        circuit_breaker_triggered = True
        halt_counter = halt_duration
        print(f"Circuit breaker triggered at t={t}, price={prices[t]:.2f}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Price dynamics
axes[0].plot(prices, linewidth=2)
axes[0].axhline(100, color='gray', linestyle='--', alpha=0.5)
axes[0].axhline(95, color='red', linestyle='--', alpha=0.5, label='Circuit breaker (-5%)')
axes[0].set_title('Price Dynamics with Circuit Breaker')
axes[0].set_ylabel('Price ($)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Liquidity
axes[1].plot(liquidity, color='green', linewidth=2)
axes[1].set_title('Order Book Liquidity')
axes[1].set_ylabel('Depth ($M)')
axes[1].grid(alpha=0.3)

# Volatility
axes[2].plot(volatility * 100, color='red', linewidth=2)
axes[2].axhline(3, color='orange', linestyle='--', alpha=0.5, label='Stress threshold')
axes[2].set_title('Realized Volatility')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_ylabel('Volatility (%)')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('systemic_risk_simulation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Minimum price: ${prices.min():.2f} ({(prices.min()-100)/100*100:.1f}% drop)")
print(f"Maximum drawdown: {(prices.min()-100)/100*100:.1f}%")
print(f"Circuit breaker activated: {circuit_breaker_triggered}")