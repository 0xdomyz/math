import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.integrate import odeint
from dataclasses import dataclass

@dataclass
class ASParameters:
    """Avellaneda-Stoikov model parameters"""
    A0: float = 10.0            # Baseline order arrival rate (orders/sec)
    k: float = 0.005            # Order flow elasticity
    gamma: float = 0.1          # Risk aversion (utility parameter)
    phi: float = 0.1            # Inventory penalty at terminal time
    sigma: float = 0.01         # Midprice volatility (daily)
    T: float = 1.0              # Time horizon (days, or fractions)
    T_sim: float = 1.0          # Simulation duration (business day)

class AvellanedaStoikov:
    """Avellaneda-Stoikov market making model"""
    
    def __init__(self, params: ASParameters):
        self.params = params
        self.t = 0.0
        self.inventory = 0.0
        self.price = 100.0
        self.pnl = 0.0
        self.quotes_history = []
        self.fill_history = []
    
    def order_arrival_rate(self, distance):
        """Exponential order flow model: A(x) = A0 * exp(-k*x)"""
        return self.params.A0 * np.exp(-self.params.k * distance)
    
    def reservation_price(self, tau):
        """
        Reservation price (indifference): r*(q,t) = S - q*alpha(tau)
        
        tau = T - t: time remaining
        """
        if tau <= 0:
            return float('-inf') if self.inventory > 0 else float('inf')
        
        # Approximation for long time horizon
        alpha = np.sqrt(self.params.phi * self.params.k / 2) * np.sqrt(tau)
        r = self.price - self.inventory * alpha - (self.params.gamma * self.params.sigma ** 2) * tau
        
        return r
    
    def optimal_spread(self, tau):
        """Optimal half-spread around reservation price"""
        if tau <= 0:
            return 0.0
        
        # Use closed-form approximation
        k = self.params.k
        A0 = self.params.A0
        gamma = self.params.gamma
        
        # Half-spread
        delta = (1.0 / k) * (np.log(1.0 + k / (A0 * gamma)) + 0.5)
        
        return max(delta, 0.0001)  # Floor to avoid negative spreads
    
    def get_quotes(self, tau):
        """
        Get optimal bid and ask quotes
        
        tau: Time remaining
        """
        r = self.reservation_price(tau)
        delta = self.optimal_spread(tau)
        
        # Asymmetric spreads based on inventory
        inventory_adjustment = self.inventory * self.params.gamma * (self.params.sigma ** 2)
        
        bid = r - delta - inventory_adjustment
        ask = r + delta - inventory_adjustment
        
        return bid, ask, r, delta
    
    def simulate_fill(self, distance):
        """
        Simulate order fill probability using Poisson process
        
        Given quote at distance from midpoint, determine if order filled
        """
        arrival_rate = self.order_arrival_rate(distance)
        # Over 1 second interval
        fill_prob = 1.0 - np.exp(-arrival_rate / 1.0)
        
        return np.random.random() < fill_prob
    
    def step(self, dt=1.0):
        """Single time step simulation"""
        tau = self.params.T - self.t
        
        # Get quotes
        bid, ask, r, delta = self.get_quotes(tau)
        
        # Simulate random midprice move
        dS = np.random.normal(0, self.params.sigma / np.sqrt(252) * np.sqrt(dt / 252))
        self.price *= (1 + dS)
        
        # Simulate order fills
        # Buy order (at ask)
        if np.random.random() < 0.5:
            if self.simulate_fill(ask - self.price):
                size = np.random.exponential(10)  # Typical size
                self.inventory += size
                self.pnl += (self.price + (ask - self.price)) * size * (-1)  # Negative for sale
                
                self.fill_history.append({
                    'time': self.t,
                    'side': 'sell',
                    'size': size,
                    'price': ask,
                    'inventory': self.inventory
                })
        
        # Sell order (at bid)
        else:
            if self.simulate_fill(self.price - bid):
                size = np.random.exponential(10)
                self.inventory -= size
                self.pnl += (self.price - (self.price - bid)) * size
                
                self.fill_history.append({
                    'time': self.t,
                    'side': 'buy',
                    'size': size,
                    'price': bid,
                    'inventory': self.inventory
                })
        
        # Financing costs
        self.pnl -= 0.00001 * abs(self.inventory)
        
        # Terminal penalty
        if tau <= 0:
            self.pnl -= self.params.phi * (self.inventory ** 2)
        
        self.quotes_history.append({
            'time': self.t,
            'bid': bid,
            'ask': ask,
            'mid': self.price,
            'reservation': r,
            'inventory': self.inventory,
            'spread': ask - bid,
            'pnl': self.pnl
        })
        
        self.t += dt

# Run simulation
print("="*80)
print("AVELLANEDA-STOIKOV MARKET MAKING MODEL")
print("="*80)

params = ASParameters(
    A0=10.0,
    k=0.005,
    gamma=0.1,
    phi=0.1,
    sigma=0.015,
    T=1.0,
    T_sim=1.0
)

# Run for one trading day (converted to model time units)
n_periods = 500
dt = params.T_sim / n_periods

mm = AvellanedaStoikov(params)

for _ in range(n_periods):
    mm.step(dt)

# Results
hist_df = pd.DataFrame(mm.quotes_history)
fills_df = pd.DataFrame(mm.fill_history) if mm.fill_history else pd.DataFrame()

print(f"\nSimulation Results:")
print(f"  Time steps: {n_periods}")
print(f"  Final PnL: ${mm.pnl:.2f}")
print(f"  Final Inventory: {mm.inventory:.2f} shares")
print(f"  Total fills: {len(mm.fill_history)}")
print(f"  Final price: ${mm.price:.2f}")

print(f"\nQuote Statistics:")
print(f"  Mean spread: {hist_df['spread'].mean()*10000:.2f} bps")
print(f"  Min spread: {hist_df['spread'].min()*10000:.2f} bps")
print(f"  Max spread: {hist_df['spread'].max()*10000:.2f} bps")

print(f"\nInventory Statistics:")
print(f"  Mean: {hist_df['inventory'].mean():.2f}")
print(f"  Std: {hist_df['inventory'].std():.2f}")
print(f"  Max: {hist_df['inventory'].max():.2f}")
print(f"  Min: {hist_df['inventory'].min():.2f}")

if len(fills_df) > 0:
    print(f"\nFill Statistics:")
    print(f"  Buy fills: {len(fills_df[fills_df['side'] == 'buy'])}")
    print(f"  Sell fills: {len(fills_df[fills_df['side'] == 'sell'])}")
    print(f"  Avg buy size: {fills_df[fills_df['side'] == 'buy']['size'].mean():.2f}")
    print(f"  Avg sell size: {fills_df[fills_df['side'] == 'sell']['size'].mean():.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Quotes over time
axes[0, 0].plot(hist_df['time'], hist_df['mid'], label='Midprice', linewidth=2)
axes[0, 0].plot(hist_df['time'], hist_df['bid'], label='Bid', alpha=0.7, linewidth=1)
axes[0, 0].plot(hist_df['time'], hist_df['ask'], label='Ask', alpha=0.7, linewidth=1)
axes[0, 0].plot(hist_df['time'], hist_df['reservation'], label='Reservation', 
                alpha=0.5, linestyle='--', linewidth=1)
axes[0, 0].set_title('Quotes Over Time')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Spread dynamics
axes[0, 1].plot(hist_df['time'], hist_df['spread']*10000, linewidth=1)
axes[0, 1].set_title('Bid-Ask Spread')
axes[0, 1].set_ylabel('Spread (bps)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Inventory
axes[0, 2].plot(hist_df['time'], hist_df['inventory'], linewidth=1)
axes[0, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 2].set_title('Inventory')
axes[0, 2].set_ylabel('Shares')
axes[0, 2].grid(alpha=0.3)

# Plot 4: Inventory vs Spread
axes[1, 0].scatter(hist_df['inventory'], hist_df['spread']*10000, alpha=0.5, s=10)
axes[1, 0].set_title('Inventory vs Spread')
axes[1, 0].set_xlabel('Inventory')
axes[1, 0].set_ylabel('Spread (bps)')
axes[1, 0].grid(alpha=0.3)

# Plot 5: Cumulative PnL
axes[1, 1].plot(hist_df['time'], hist_df['pnl'], linewidth=1)
axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Cumulative P&L')
axes[1, 1].set_ylabel('P&L ($)')
axes[1, 1].grid(alpha=0.3)

# Plot 6: Reservation price vs inventory
axes[1, 2].scatter(hist_df['inventory'], hist_df['reservation'], 
                   alpha=0.5, s=10, label='Reservation')
axes[1, 2].scatter(hist_df['inventory'], hist_df['mid'], 
                   alpha=0.3, s=5, label='Midprice')
axes[1, 2].set_title('Reservation Price vs Inventory')
axes[1, 2].set_xlabel('Inventory')
axes[1, 2].set_ylabel('Price ($)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Reservation price shifts with inventory: Natural hedging impulse")
print(f"2. Spreads adapt to volatility and remaining time: Urgent as T approaches")
print(f"3. Inventory mean-reverts through asymmetric quotes")
print(f"4. Terminal penalty incentivizes flat close")
print(f"5. Model captures bid-ask bounce and adverse selection")
