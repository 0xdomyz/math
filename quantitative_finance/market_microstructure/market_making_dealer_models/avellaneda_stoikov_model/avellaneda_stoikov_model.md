# Avellaneda-Stoikov Market Making Model

## 1. Concept Skeleton
**Definition:** Stochastic control model for optimal bid-ask quote setting under inventory constraints, accounting for adverse selection, inventory penalty, and exponential utility maximization  
**Purpose:** Dynamically adjust spreads and skew quotes to balance profitability, risk management, and inventory control; determine optimal reservation prices  
**Prerequisites:** Stochastic calculus, optimal control, dynamic programming, mean-reversion, utility theory

## 2. Comparative Framing
| Framework | Avellaneda-Stoikov | Stoll (Static) | Kyle Model (Equilibrium) | Optimal Execution |
|-----------|-------------------|----------------|-------------------------|-------------------|
| **Decision Variable** | Quote prices (bid/ask) | Spread only | Depth λ | Execution path |
| **Dynamics** | Continuous, state-dependent | Fixed, inventory-only | Strategic game | Discrete slices |
| **Objective** | Maximize risk-adjusted profit | Minimize adverse selection | Information-driven | Minimize cost |
| **Time Horizon** | Finite (trading day) | Static equilibrium | Single-period | Fixed duration T |
| **Output** | Dynamic quotes (realtime) | Snapshot spreads | Market depth | Execution schedule |

## 3. Examples + Counterexamples

**Simple Example:**  
10:00 AM: MM holds 0 shares, quotes 99.95 bid / 100.05 ask (1 bp spread). Receives buy order → inventory +100.  
10:05 AM: Now holds 100 longs, model recommends 99.90 bid / 100.10 ask (2 bp spread, shifted down 5 bps). Encourages sells to offset.

**Failure Case:**  
Flash crash: Model assumes continuous time and normal volatility. In panic, all parameters break (vol spikes 10x, liquidity vanishes). Model quotes ignored, MM forced to eat massive bid-ask.

**Edge Case:**  
Market close: With 15 minutes left, MM must close position. Reservation price collapses → spread narrowing aggressively to ensure execution. Model naturally adapts to urgency.

## 4. Layer Breakdown
```
Avellaneda-Stoikov Framework:
├─ Core Setup:
│   ├─ State Variables:
│   │   ├─ t: Time (0 ≤ t ≤ T, trading session duration)
│   │   ├─ S(t): Stock price (exogenous, follows geometric Brownian motion)
│   │   ├─ q(t): Inventory (shares held)
│   │   ├─ x: Distance from midpoint for bid/ask quotes
│   │   └─ v(x): Arrival rate of orders (decreasing in x)
│   ├─ Price Process:
│   │   ├─ dS = σ dW (geometric Brownian motion with drift 0)
│   │   ├─ σ: Volatility of midprice
│   │   ├─ W: Standard Brownian motion
│   │   └─ MM quotes relative to S(t)
│   ├─ Order Flow Model:
│   │   ├─ Buy order arrival: Poisson(A(x_a)), at ask = S + x_a
│   │   ├─ Sell order arrival: Poisson(A(x_b)), at bid = S - x_b
│   │   ├─ A(x) = A₀ × e^(-kx): Exponential decay with distance
│   │   ├─ A₀: Baseline arrival rate
│   │   ├─ k: Elasticity (how fast orders dry up with distance)
│   │   └─ Interpretation: Wider quotes → fewer orders arrive
│   └─ Objective Function:
│       ├─ Maximize: E[X(T) - φ q(T)²]
│       ├─ X(T): Cumulative cash collected from trading
│       ├─ q(T): Inventory at time T (final position)
│       ├─ φ: Inventory penalty (risk aversion)
│       └─ Trade-off: Profit (X) vs. terminal position risk (q²)
├─ Continuous-Time Model:
│   ├─ HJB Equation:
│   │   ├─ General form: 0 = max over (x_b, x_a) of:
│   │   ├─  A(x_a)[v(S,q,t|bought) - v(S,q+1,t)] + ...
│   │   ├─  A(x_b)[v(S,q,t|sold) - v(S,q-1,t)] + ...
│   │   ├─  (1/2)σ²S² ∂²v/∂S² + ∂v/∂t
│   │   ├─ v(S,q,t): Value function (optimal profit-to-go)
│   │   └─ Boundary: v(S,q,T) = -φq² (terminal penalty)
│   ├─ Reservation Price:
│   │   ├─ r*(S,q,t) = S - q×γ(t)
│   │   ├─ r*: Indifference price for inventory
│   │   ├─ q: Current inventory
│   │   ├─ γ(t): Decay rate with time-to-close
│   │   ├─ Interpretation: MM willing to trade worse to reduce inventory
│   │   ├─ At t=T: r*→ -∞ if q>0 (must sell) or +∞ if q<0 (must buy)
│   │   └─ At t=0 (far from close): r*≈S (indifferent to inventory)
│   └─ Optimal Quotes:
│       ├─ x*_a = argmax{A(x) × [x + (mid - r*)]}
│       ├─ x*_b = argmax{A(x) × [x + (mid - r*)]}
│       ├─ Trade-off: Wider x → lower A(x) but higher margin per trade
│       ├─ Optimal: Usually 50-70 bps from midpoint in practice
│       └─ Asymmetric: If q>0 (long), ask width reduced (want to sell)
├─ Linear-Exponential Approximation:
│   ├─ Assumptions:
│   │   ├─ Large baseline order flow (A₀ large)
│   │   ├─ Small spreads (competition drives efficiency)
│   │   ├─ Exponential order flow: A(x) = A₀ exp(-kx)
│   │   ├─ Large volatility environment
│   │   └─ Terminal inventory penalty (not intermediate constraints)
│   ├─ Closed-Form Solution:
│   │   ├─ Reservation price: r* = S - qα(t) - β(t)σ²
│   │   ├─ α(t): Inventory decay rate (increases approaching T)
│   │   ├─ β(t): Spread decay rate (reflects uncertainty)
│   │   ├─ Half-spread around r*:
│   │   ├─  δ(t) = (1/k)[log(1 + k/A₀ × γ)]^1/2
│   │   ├─ γ: Urgency parameter
│   │   └─ Result: Bid = r* - δ, Ask = r* + δ
│   ├─ Time Evolution:
│   │   ├─ α(t) = √(πφk/2) × (T-t)^(1/2)
│   │   ├─ β(t) = (1/(2k)) × log((T-t)^(-1)) for t near T
│   │   ├─ As t → T:
│   │   ├─   α(t) → 0 (lose inventory urgency)
│   │   ├─   δ(t) → 0 (spread collapses to close)
│   │   └─ Interpretation: Fleeing inventory, accepting unfavorable fills
│   └─ Parameters:
│       ├─ k: Order elasticity (~0.001-0.01, higher=more elastic)
│       ├─ γ: Risk aversion (~0.1-1.0, higher=more inventory averse)
│       ├─ φ: Terminal penalty rate (~0.001-0.1, higher=risk-averse)
│       ├─ A₀: Baseline flow (~1-100 orders/sec, asset-dependent)
│       └─ σ: Volatility (market-dependent)
├─ Discrete-Time Variant:
│   ├─ Setting: Discrete decision times (e.g., every 1 second)
│   ├─ Bellman Equation (backward induction):
│   │   ├─ V_t(q) = max over (x_b, x_a) of:
│   │   ├─  E[π_t + V_{t+1}(q')]
│   │   ├─ π_t: Profit earned in period [t, t+Δt]
│   │   ├─ q': Inventory after period (q ± 1 or q)
│   │   └─ V_T(q) = -φq²
│   ├─ Computation: Dynamic programming
│   ├─ Advantages: Easier to code, numerically stable
│   ├─ Disadvantages: Curse of dimensionality if discretizing state space
│   └─ Practical: Usually 100-1000 time steps for day-trading
├─ Empirical Calibration:
│   ├─ Data Requirements:
│   │   ├─ High-frequency quote book snapshots
│   │   ├─ Order arrival rates at different distances
│   │   ├─ Volatility estimates
│   │   ├─ Historical market depth
│   │   └─ Own execution data (if MM)
│   ├─ Parameter Estimation:
│   │   ├─ A₀, k: Fit exponential decay to order flow (max likelihood)
│   │   ├─ σ: Realized volatility from price data
│   │   ├─ φ, γ: Calibrate to match observed spreads or fit to profit target
│   │   ├─ Alternative: Infer from bid-ask widths using inverse optimization
│   │   └─ Validation: Compare simulated quotes to actual market
│   ├─ Stock-Specific Adjustments:
│   │   ├─ Large-cap (liquid): A₀ high, k high → tight quotes
│   │   ├─ Small-cap (illiquid): A₀ low, k low → wide quotes
│   │   ├─ High volatility: σ high → spread widens to compensate risk
│   │   └─ Earnings day: Parameters may change (regime shift)
│   └─ Adaptive Learning:
│       ├─ Update parameters intraday based on observed flow
│       ├─ Machine learning: Fit new A(x) periodically
│       ├─ Bayesian: Prior from historical, posterior from current data
│       └─ Result: Model improves throughout day as data accumulates
├─ Extensions & Variants:
│   ├─ Multi-Asset MM:
│   │   ├─ Extend to n assets with correlations
│   │   ├─ Hedging: Can trade other assets to reduce inventory
│   │   ├─ Spillover: Inventory in one asset affects others' quotes
│   │   └─ Complexity: Curse of dimensionality in state space
│   ├─ Regimes:
│   │   ├─ Normal: Low volatility, regular flows
│   │   ├─ Stress: High volatility, flight to safety
│   │   ├─ Switch between regimes based on monitoring
│   │   └─ Parameters (A₀, σ, φ) different per regime
│   ├─ Order Book Effects:
│   │   ├─ Level 2: Other market makers' quotes impact order flow
│   │   ├─ Congestion: When many orders at same level, execution lower
│   │   ├─ Strategic: Other MM behavior endogenous
│   │   └─ Compete on: Speed, price, both
│   ├─ Information Flow:
│   │   ├─ If informed traders present: Spread widens
│   │   ├─ PIN: Probability of informed trading (increase spreads if high)
│   │   ├─ News arrivals: Pre-announcement volatility jump
│   │   └─ Integration: Adjust parameters with PIN estimates
│   └─ Inventory Constraints:
│       ├─ Model assumes can hold arbitrary |q|
│       ├─ Reality: Borrow limits, margin constraints
│       ├─ Modification: Increase φ approaching limits
│       ├─ Hard limit: Reject orders beyond qmax
│       └─ Result: Quotes collapse near limits (forced liquidation mode)
└─ Implementation & Monitoring:
    ├─ System Architecture:
    │   ├─ Real-time: State vector (S, q, t) updated every tick
    │   ├─ Compute: V(S, q, t) or closed-form quotes
    │   ├─ Quote: Submit (x*_b, x*_a) to exchange
    │   ├─ Monitor: Track execution rates, compare to forecast
    │   └─ Latency: Sub-millisecond round-trip critical
    ├─ Parameter Monitoring:
    │   ├─ Watch: A(x) changes (competition, participation)
    │   ├─ σ changes (volatility regimes)
    │   ├─ Track: Realized vs. expected order arrival rates
    │   ├─ Alert: If order flow deviates > 50% from model
    │   └─ Adapt: Recalibrate parameters intraday
    ├─ Risk Controls:
    │   ├─ Hard limit on |q| (override model if exceeded)
    │   ├─ Spread floor (never quote inside natural spread)
    │   ├─ Daily PnL limits (pause if losing exceeds threshold)
    │   └─ Greeks: Monitor delta, gamma, vega of inventory
    ├─ Performance Metrics:
    │   ├─ Hit rate: % of quotes filled (should match forecast)
    │   ├─ Spread efficiency: Actual spread vs. model prediction
    │   ├─ Inventory: Realized |q(T)| vs. target (should be close to 0)
    │   ├─ Sharpe ratio: PnL / volatility
    │   └─ Comparison: Benchmark against passive market maker (fixed spreads)
    └─ Backtesting:
        ├─ Historical simulation: Replay quotes using model
        ├─ Test on: Different volatility regimes, volumes, assets
        ├─ Metrics: PnL, max drawdown, final inventory distribution
        ├─ Sensitivity: How sensitive to parameter errors
        └─ Stress: Performance during flash crashes, earnings
```

**Interaction:** Monitor (S, q, t) → Compute optimal (x_b*, x_a*) → Submit quotes → Fill orders → Update q → Recalculate → Dynamic equilibrium

## 5. Mini-Project
Implement Avellaneda-Stoikov model and backtest:
```python
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
```

## 6. Challenge Round
When does Avellaneda-Stoikov break down?
- **Non-stationary volatility**: If σ changes regime, quotes become suboptimal
- **Informed traders**: Model assumes uninformed flow, fails if smart order flow present
- **Discrete time artifacts**: Real exchanges have latency; model assumes continuous
- **Execution costs**: Model ignores fees; in low-margin environments, may not be worthwhile
- **Liquidity events**: In gaps/halts, model's exponential decay assumption fails completely

How to detect model breakdown?
- **Monitor fill rates**: If actual << forecast, market less liquid than assumed
- **Inventory drift**: If |q| consistently grows (one-way flow), model doesn't balance
- **Profitability**: If realized PnL < theoretical, either bad parameters or market changed
- **Spread compression**: If observed spreads << model's, competition intensified

## 7. Key References
- [Avellaneda & Stoikov (2008): High-Frequency Trading in a Limit Order Book](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf)
- [Cartea & Jaimungal (2016): Algorithmic and High-Frequency Trading](https://arxiv.org/abs/1602.06220)
- [Gueant, Lehalle, Fernandez-Tapia (2012): Optimal Quoting in a Microstructure-Inspired Model](https://arxiv.org/abs/1310.6459)

---
**Status:** Foundational model for active market making | **Complements:** Inventory Management, Order Book Dynamics, HFT Infrastructure
