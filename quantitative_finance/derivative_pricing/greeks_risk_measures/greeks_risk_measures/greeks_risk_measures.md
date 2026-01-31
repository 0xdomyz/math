# Greeks and Risk Measures

## 1. Concept Skeleton
**Definition:** First-order and higher-order partial derivatives of option value with respect to underlying parameters; quantify sensitivity and risk exposure  
**Purpose:** Measure price sensitivity to spot, volatility, time, rates; enable delta hedging, portfolio risk management, and dynamic replication strategies  
**Prerequisites:** Option pricing basics, partial derivatives, Black-Scholes model, portfolio theory, calculus of variations

## 2. Comparative Framing
| Greek | Definition | Interpretation | Hedge Frequency |
|-------|------------|----------------|-----------------|
| **Delta (Δ)** | ∂V/∂S | Shares to hedge | Continuous (practical: daily) |
| **Gamma (Γ)** | ∂²V/∂S² = ∂Δ/∂S | Delta stability | When large (nonlinear) |
| **Vega (ν)** | ∂V/∂σ | Volatility exposure | Weekly/event-driven |
| **Theta (Θ)** | ∂V/∂t | Time decay | Cannot hedge (time passes) |
| **Rho (ρ)** | ∂V/∂r | Interest rate risk | Rarely (stable rates) |

| Risk Measure | Greeks Used | Purpose | Timescale |
|--------------|-------------|---------|-----------|
| **Delta Risk** | Δ | Directional exposure | Intraday |
| **Gamma Risk** | Γ | Curvature, gap risk | Daily |
| **Vega Risk** | ν | Vol surface changes | Weekly |
| **Higher-Order** | Vanna, Volga | Cross-sensitivities | Strategic |

## 3. Examples + Counterexamples

**Simple Example:**  
ATM call Δ≈0.5: Hedge 100 calls with -50 shares. Stock up $1 → call gains ~$50, shares lose $50 → net flat.

**Perfect Fit:**  
Market maker: Sells 1000 calls (Δ=0.6 each), immediately buys 600 shares to delta-hedge. Rebalances daily as Δ changes.

**Gamma Bomb:**  
Large short ATM options near expiry: Γ explodes → Δ changes rapidly → hedging requires massive stock trades → potential losses from slippage.

**Vega Exposure:**  
VIX spike from 15% to 30%: Long straddle (high +vega) profits significantly even if spot unchanged. Short vol position bleeds.

**Theta Decay:**  
OTM option 1 week to expiry: Θ≈-0.05 per day. Each day loses $5 in time value even if spot/vol stable. Cannot hedge, only by offsetting position.

**Poor Fit:**  
Using BS Greeks for deep OTM options near barriers: Model assumes continuous prices, ignores jump risk → Greeks misleading.

## 4. Layer Breakdown
```
Greeks Framework:

├─ First-Order Greeks:
│  ├─ Delta (Δ): ∂V/∂S
│  │   ├─ Call Delta: 0 < Δ < 1 (typically 0.5 ATM)
│  │   ├─ Put Delta: -1 < Δ < 0 (typically -0.5 ATM)
│  │   ├─ Interpretation: Hedge ratio (shares per option)
│  │   ├─ Portfolio Delta: Σ(Δᵢ × Positionᵢ)
│  │   ├─ Delta-neutral: Portfolio Δ = 0
│  │   └─ Delta hedging: Buy/sell stock to maintain Δ=0
│  ├─ Vega (ν): ∂V/∂σ
│  │   ├─ Always positive for vanilla options
│  │   ├─ Maximum for ATM options
│  │   ├─ Interpretation: P&L change per 1% vol shift
│  │   ├─ Vega hedging: Use other options (calendar spreads)
│  │   └─ Volatility smile: Different strikes have different vegas
│  ├─ Theta (Θ): ∂V/∂t (time decay)
│  │   ├─ Negative for long options (lose time value)
│  │   ├─ Positive for short options (earn premium decay)
│  │   ├─ Accelerates near expiry (convex decay)
│  │   ├─ ATM options: Highest absolute theta
│  │   └─ Cannot hedge except with offsetting options
│  └─ Rho (ρ): ∂V/∂r
│      ├─ Call Rho: Positive (higher r → higher call value)
│      ├─ Put Rho: Negative (higher r → lower put value)
│      ├─ Long maturity: Higher sensitivity
│      └─ Rarely hedged in practice (stable rates)
├─ Second-Order Greeks:
│  ├─ Gamma (Γ): ∂²V/∂S² = ∂Δ/∂S
│  │   ├─ Measures delta stability (convexity)
│  │   ├─ Always positive for long options
│  │   ├─ Maximum for ATM options
│  │   ├─ Spikes near expiry for ATM
│  │   ├─ Interpretation: How fast delta changes
│  │   ├─ High Γ → Requires frequent delta rebalancing
│  │   ├─ Gamma P&L: ½Γ(ΔS)² (from realized volatility)
│  │   └─ Gamma risk: Large moves cause hedging losses
│  ├─ Vanna: ∂²V/∂S∂σ = ∂Δ/∂σ = ∂ν/∂S
│  │   ├─ Cross-sensitivity: Spot-vol interaction
│  │   ├─ Important for skew dynamics
│  │   └─ Used in advanced vol surface modeling
│  ├─ Volga (Vomma): ∂²V/∂σ² = ∂ν/∂σ
│  │   ├─ Vega convexity (vega stability)
│  │   ├─ Positive for vanilla options
│  │   └─ Relevant for large vol changes
│  └─ Charm: ∂²V/∂S∂t = ∂Δ/∂t
│      ├─ Delta decay over time
│      └─ Useful for forward delta projections
├─ Greeks in Black-Scholes:
│  ├─ Call Delta: Δ_call = N(d₁)
│  ├─ Put Delta: Δ_put = N(d₁) - 1 = -N(-d₁)
│  ├─ Gamma (same for call/put):
│  │   Γ = φ(d₁) / (S σ √T)
│  │   where φ(x) = (1/√(2π)) exp(-x²/2)
│  ├─ Vega (same for call/put):
│  │   ν = S φ(d₁) √T
│  ├─ Theta:
│  │   Θ_call = -S φ(d₁) σ/(2√T) - rK e^(-rT) N(d₂)
│  │   Θ_put = -S φ(d₁) σ/(2√T) + rK e^(-rT) N(-d₂)
│  └─ Rho:
│      ρ_call = K T e^(-rT) N(d₂)
│      ρ_put = -K T e^(-rT) N(-d₂)
├─ Delta Hedging:
│  ├─ Objective: Eliminate directional risk (Δ=0)
│  ├─ Static hedge: Set at inception, don't adjust
│  │   → Only works for linear payoffs or perfect replication
│  ├─ Dynamic hedge: Continuously rebalance
│  │   ├─ Frequency tradeoff: Transaction costs vs tracking error
│  │   ├─ Discrete rehedging: Gamma risk accumulates
│  │   └─ P&L attribution: Theta + ½Γ(ΔS)²
│  ├─ Delta-hedging portfolio:
│  │   Portfolio = Options + Δ_hedge × Stock
│  │   where Δ_hedge = -Σ(Δ_option × Quantity)
│  ├─ Gamma scalping:
│  │   ├─ Profit from realized vol > implied vol
│  │   ├─ Long gamma (long options): Profit from rebalancing
│  │   └─ Short gamma (short options): Lose from rebalancing
│  └─ Hedging errors:
│      ├─ Discrete rebalancing → Gamma P&L
│      ├─ Model risk (wrong σ) → Vega P&L
│      └─ Transaction costs → Drag on performance
├─ Portfolio-Level Risk:
│  ├─ Net Greeks:
│  │   Δ_portfolio = Σᵢ Δᵢ × Positionᵢ
│  │   Γ_portfolio = Σᵢ Γᵢ × Positionᵢ
│  │   ν_portfolio = Σᵢ νᵢ × Positionᵢ
│  ├─ Risk limits:
│  │   ├─ Delta limit: Maximum directional exposure
│  │   ├─ Gamma limit: Maximum convexity risk
│  │   ├─ Vega limit: Maximum vol exposure
│  │   └─ Aggregated across underlyings, maturities
│  ├─ Greeks by strike/maturity:
│  │   Grid of exposures across vol surface
│  ├─ Greeks ladder (term structure):
│  │   Vega by maturity bucket (1m, 3m, 6m, 1y, etc.)
│  └─ Correlation risk:
│      Greeks across multiple underlyings → correlation matrix
├─ Advanced Topics:
│  ├─ Forward Greeks:
│  │   Greeks at future date (for forward-starting options)
│  ├─ Greeks with dividends:
│  │   Modify formulas for continuous yield q
│  │   Δ_call = e^(-qT) N(d₁)
│  ├─ Greeks for exotic options:
│  │   ├─ Barrier options: Discontinuous delta at barrier
│  │   ├─ Digital options: Dirac delta (infinite gamma)
│  │   └─ Asian options: Path-dependent → weighted Greeks
│  ├─ Model-free Greeks:
│  │   Extract from option prices without assuming model
│  ├─ Greeks hedging with options:
│  │   ├─ Vega hedge: Use different strike/maturity options
│  │   ├─ Gamma hedge: Buy/sell options (cannot hedge with stock)
│  │   └─ Multi-Greek hedging: System of equations
│  └─ Jump risk (beyond Greeks):
│      Greeks assume continuous paths → fail for gaps
└─ Practical Considerations:
   ├─ Bid-ask spreads: Affect hedging costs
   ├─ Discrete rebalancing: Optimal frequency analysis
   ├─ Greeks aggregation: By book, desk, firm-wide
   ├─ Real-time Greeks: Streaming calculations for traders
   ├─ Greeks reporting: Daily risk reports for management
   └─ Regulatory capital: Risk-weighted assets based on Greeks
```

**Interaction:** Delta measures slope, Gamma measures curvature; hedging Delta creates Gamma exposure; Theta compensates Gamma in delta-hedged portfolio (Black-Scholes PDE).

## 5. Mini-Project
Implement Greeks calculation and delta hedging simulation:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("GREEKS AND DELTA HEDGING SIMULATION")
print("="*60)

class BlackScholesGreeks:
    """Black-Scholes option pricing and Greeks"""
    
    def __init__(self, S, K, r, T, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
    
    def d1(self):
        return (np.log(self.S/self.K) + (self.r + 0.5*self.sigma**2)*self.T) / (self.sigma*np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)
    
    def price(self):
        d1, d2 = self.d1(), self.d2()
        if self.option_type == 'call':
            return self.S*norm.cdf(d1) - self.K*np.exp(-self.r*self.T)*norm.cdf(d2)
        else:
            return self.K*np.exp(-self.r*self.T)*norm.cdf(-d2) - self.S*norm.cdf(-d1)
    
    def delta(self):
        if self.option_type == 'call':
            return norm.cdf(self.d1())
        else:
            return norm.cdf(self.d1()) - 1
    
    def gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self):
        return self.S * norm.pdf(self.d1()) * np.sqrt(self.T) / 100  # Per 1% vol change
    
    def theta(self):
        d1, d2 = self.d1(), self.d2()
        term1 = -self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        
        if self.option_type == 'call':
            term2 = -self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365  # Per day
    
    def rho(self):
        d2 = self.d2()
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(d2) / 100  # Per 1% rate change
        else:
            return -self.K * self.T * np.exp(-self.r*self.T) * norm.cdf(-d2) / 100
    
    def all_greeks(self):
        return {
            'price': self.price(),
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }

# Scenario 1: Greeks across spot prices
print("\n" + "="*60)
print("SCENARIO 1: Greeks Profile Across Spot Prices")
print("="*60)

K = 100
r = 0.05
T = 0.25  # 3 months
sigma = 0.2
spot_range = np.linspace(80, 120, 50)

call_deltas = []
call_gammas = []
call_vegas = []
call_thetas = []

for S in spot_range:
    bs = BlackScholesGreeks(S, K, r, T, sigma, 'call')
    greeks = bs.all_greeks()
    call_deltas.append(greeks['delta'])
    call_gammas.append(greeks['gamma'])
    call_vegas.append(greeks['vega'])
    call_thetas.append(greeks['theta'])

# ATM Greeks
bs_atm = BlackScholesGreeks(K, K, r, T, sigma, 'call')
greeks_atm = bs_atm.all_greeks()

print(f"\nATM Call Option (S=K=${K}, T=3m, σ={sigma:.0%}):")
print(f"  Price: ${greeks_atm['price']:.4f}")
print(f"  Delta: {greeks_atm['delta']:.4f}")
print(f"  Gamma: {greeks_atm['gamma']:.6f}")
print(f"  Vega: ${greeks_atm['vega']:.4f} per 1% vol")
print(f"  Theta: ${greeks_atm['theta']:.4f} per day")
print(f"  Rho: ${greeks_atm['rho']:.4f} per 1% rate")

# Scenario 2: Greeks across time to expiry
print("\n" + "="*60)
print("SCENARIO 2: Greeks Evolution Over Time")
print("="*60)

S0 = 100
times = np.linspace(1, 0.01, 50)  # 1 year to near expiry
deltas_time = []
gammas_time = []
thetas_time = []

for t in times:
    bs = BlackScholesGreeks(S0, K, r, t, sigma, 'call')
    greeks = bs.all_greeks()
    deltas_time.append(greeks['delta'])
    gammas_time.append(greeks['gamma'])
    thetas_time.append(greeks['theta'])

print(f"\nATM Call Greeks at Different Times (S=${S0}, K=${K}):")
print(f"{'Time to Expiry':<20} {'Delta':<12} {'Gamma':<12} {'Theta':<12}")
print("-" * 56)
for i, t in enumerate([1.0, 0.5, 0.25, 0.083, 0.02]):  # 1y, 6m, 3m, 1m, 1w
    idx = np.argmin(np.abs(times - t))
    print(f"{t*365:>6.0f} days          {deltas_time[idx]:>10.4f}  {gammas_time[idx]:>10.6f}  {thetas_time[idx]:>10.4f}")

# Scenario 3: Delta hedging simulation
print("\n" + "="*60)
print("SCENARIO 3: Dynamic Delta Hedging Simulation")
print("="*60)

# Parameters
S0 = 100
K = 100
r = 0.05
T = 0.25
sigma_implied = 0.20
sigma_realized = 0.25  # Realized > implied
n_days = int(T * 252)
dt = T / n_days

# Simulate stock path (GBM)
np.random.seed(123)
stock_path = [S0]
for i in range(n_days):
    dW = np.random.normal(0, np.sqrt(dt))
    dS = r * stock_path[-1] * dt + sigma_realized * stock_path[-1] * dW
    stock_path.append(stock_path[-1] + dS)

stock_path = np.array(stock_path)
time_path = np.linspace(T, 0, n_days+1)

# Delta hedging
position_size = 100  # Long 100 call options
hedge_ratios = []
hedge_pnl = []
option_pnl = []
total_pnl = []
transaction_costs = []

cumulative_pnl = 0
cumulative_cost = 0
previous_hedge = 0

for i in range(n_days + 1):
    S_t = stock_path[i]
    T_t = time_path[i]
    
    if T_t > 0:
        bs = BlackScholesGreeks(S_t, K, r, T_t, sigma_implied, 'call')
        greeks = bs.all_greeks()
        delta_t = greeks['delta']
        price_t = greeks['price']
    else:
        # At expiry
        delta_t = 1.0 if S_t > K else 0.0
        price_t = max(S_t - K, 0)
    
    # Hedge ratio (negative because hedging long options)
    hedge_shares = -delta_t * position_size
    hedge_ratios.append(hedge_shares)
    
    # Transaction cost (rebalancing)
    trade_amount = abs(hedge_shares - previous_hedge)
    cost = trade_amount * S_t * 0.001  # 10 bps transaction cost
    cumulative_cost += cost
    transaction_costs.append(cumulative_cost)
    
    previous_hedge = hedge_shares

# Final P&L
initial_option_value = BlackScholesGreeks(S0, K, r, T, sigma_implied, 'call').price() * position_size
final_option_value = max(stock_path[-1] - K, 0) * position_size
option_pnl_total = final_option_value - initial_option_value

# Hedge P&L (simplified - tracking error from discrete rehedging)
# In reality, this would accumulate from each rebalance
hedge_pnl_estimate = -option_pnl_total  # Perfectly hedged would offset

print(f"\nDelta Hedging Simulation ({n_days} days):")
print(f"  Initial Stock: ${S0:.2f}")
print(f"  Final Stock: ${stock_path[-1]:.2f}")
print(f"  Implied Vol: {sigma_implied:.1%}")
print(f"  Realized Vol: {sigma_realized:.1%}")
print(f"  Position: Long {position_size} calls")
print(f"\nP&L Breakdown:")
print(f"  Option P&L: ${option_pnl_total:.2f}")
print(f"  Transaction Costs: -${cumulative_cost:.2f}")
print(f"  Net P&L: ${option_pnl_total - cumulative_cost:.2f}")
print(f"\nGamma Scalping Profit:")
print(f"  Realized > Implied: Profit from rebalancing")
print(f"  Estimated Gamma P&L: ${option_pnl_total - cumulative_cost:.2f}")

# Scenario 4: Greeks heatmap (spot vs vol)
print("\n" + "="*60)
print("SCENARIO 4: Greeks Sensitivity Surface")
print("="*60)

spots = np.linspace(80, 120, 20)
vols = np.linspace(0.10, 0.40, 20)
T_surface = 0.25

delta_surface = np.zeros((len(vols), len(spots)))
gamma_surface = np.zeros((len(vols), len(spots)))
vega_surface = np.zeros((len(vols), len(spots)))

for i, vol in enumerate(vols):
    for j, spot in enumerate(spots):
        bs = BlackScholesGreeks(spot, K, r, T_surface, vol, 'call')
        greeks = bs.all_greeks()
        delta_surface[i, j] = greeks['delta']
        gamma_surface[i, j] = greeks['gamma']
        vega_surface[i, j] = greeks['vega']

print(f"\nGreeks Surface Generated:")
print(f"  Spot range: ${spots.min():.0f} - ${spots.max():.0f}")
print(f"  Vol range: {vols.min():.0%} - {vols.max():.0%}")
print(f"  Strike: ${K}")
print(f"  Maximum Gamma at: S=${spots[np.argmax(gamma_surface.sum(axis=0))]:.0f} (near ATM)")

# Scenario 5: Portfolio Greeks
print("\n" + "="*60)
print("SCENARIO 5: Portfolio-Level Greeks")
print("="*60)

# Portfolio of options
portfolio = [
    {'type': 'call', 'strike': 90, 'quantity': 100},
    {'type': 'call', 'strike': 100, 'quantity': -200},  # Short
    {'type': 'call', 'strike': 110, 'quantity': 100},
    {'type': 'put', 'strike': 95, 'quantity': 50},
]

S_current = 100
T_current = 0.25

portfolio_delta = 0
portfolio_gamma = 0
portfolio_vega = 0
portfolio_theta = 0
portfolio_value = 0

print(f"\nPortfolio Composition (S=${S_current}, T=3m):")
print(f"{'Type':<8} {'Strike':<10} {'Qty':<10} {'Price':<12} {'Delta':<12} {'Gamma':<12}")
print("-" * 74)

for option in portfolio:
    bs = BlackScholesGreeks(S_current, option['strike'], r, T_current, sigma, option['type'])
    greeks = bs.all_greeks()
    
    position_delta = greeks['delta'] * option['quantity']
    position_gamma = greeks['gamma'] * option['quantity']
    position_vega = greeks['vega'] * option['quantity']
    position_theta = greeks['theta'] * option['quantity']
    position_value = greeks['price'] * option['quantity']
    
    portfolio_delta += position_delta
    portfolio_gamma += position_gamma
    portfolio_vega += position_vega
    portfolio_theta += position_theta
    portfolio_value += position_value
    
    print(f"{option['type']:<8} ${option['strike']:<9} {option['quantity']:<9} "
          f"${greeks['price']:<11.4f} {position_delta:<11.2f} {position_gamma:<11.6f}")

print("-" * 74)
print(f"{'TOTAL':<8} {'':<10} {'':<10} ${portfolio_value:<11.2f} "
      f"{portfolio_delta:<11.2f} {portfolio_gamma:<11.6f}")

print(f"\nPortfolio Net Greeks:")
print(f"  Delta: {portfolio_delta:.2f} (shares equivalent)")
print(f"  Gamma: {portfolio_gamma:.6f}")
print(f"  Vega: ${portfolio_vega:.2f} per 1% vol change")
print(f"  Theta: ${portfolio_theta:.2f} per day")
print(f"  Value: ${portfolio_value:.2f}")

print(f"\nRisk Assessment:")
if abs(portfolio_delta) < 10:
    print(f"  ✓ Delta-neutral: Minimal directional risk")
else:
    print(f"  ✗ Delta exposure: {portfolio_delta:.2f} shares")

if portfolio_gamma > 0:
    print(f"  ✓ Positive Gamma: Benefits from volatility")
else:
    print(f"  ✗ Negative Gamma: Loses from large moves")

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

# Plot 1: Delta across spot
ax = axes[0, 0]
ax.plot(spot_range, call_deltas, 'b-', linewidth=2.5, label='Call Delta')
ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='ATM (0.5)')
ax.axvline(K, color='k', linestyle='--', alpha=0.3, label=f'Strike ${K}')
ax.set_xlabel('Spot Price')
ax.set_ylabel('Delta')
ax.set_title('Delta vs Spot Price')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Gamma across spot
ax = axes[0, 1]
ax.plot(spot_range, call_gammas, 'g-', linewidth=2.5)
ax.axvline(K, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Spot Price')
ax.set_ylabel('Gamma')
ax.set_title('Gamma vs Spot Price (Peaks at ATM)')
ax.grid(alpha=0.3)

# Plot 3: Vega across spot
ax = axes[0, 2]
ax.plot(spot_range, call_vegas, 'm-', linewidth=2.5)
ax.axvline(K, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Spot Price')
ax.set_ylabel('Vega ($)')
ax.set_title('Vega vs Spot Price')
ax.grid(alpha=0.3)

# Plot 4: Theta across spot
ax = axes[1, 0]
ax.plot(spot_range, call_thetas, 'r-', linewidth=2.5)
ax.axvline(K, color='k', linestyle='--', alpha=0.3)
ax.axhline(0, color='k', linestyle='-', alpha=0.2)
ax.set_xlabel('Spot Price')
ax.set_ylabel('Theta ($/day)')
ax.set_title('Theta vs Spot Price (Time Decay)')
ax.grid(alpha=0.3)

# Plot 5: Greeks evolution over time
ax = axes[1, 1]
ax.plot(times*365, gammas_time, 'g-', linewidth=2.5, label='Gamma')
ax.set_xlabel('Days to Expiry')
ax.set_ylabel('Gamma')
ax.set_title('Gamma Explosion Near Expiry')
ax.legend()
ax.grid(alpha=0.3)
ax.invert_xaxis()

# Plot 6: Theta evolution
ax = axes[1, 2]
ax.plot(times*365, thetas_time, 'r-', linewidth=2.5)
ax.set_xlabel('Days to Expiry')
ax.set_ylabel('Theta ($/day)')
ax.set_title('Theta Acceleration Near Expiry')
ax.grid(alpha=0.3)
ax.invert_xaxis()

# Plot 7: Stock path and delta hedge
ax = axes[2, 0]
ax.plot(stock_path, 'b-', linewidth=2, alpha=0.7, label='Stock Price')
ax.axhline(K, color='r', linestyle='--', alpha=0.5, label=f'Strike ${K}')
ax.set_xlabel('Days')
ax.set_ylabel('Stock Price ($)')
ax.set_title('Simulated Stock Path')
ax.legend()
ax.grid(alpha=0.3)

# Plot 8: Hedge ratios over time
ax = axes[2, 1]
ax.plot(hedge_ratios, 'g-', linewidth=2)
ax.set_xlabel('Days')
ax.set_ylabel('Hedge Position (shares)')
ax.set_title('Dynamic Delta Hedge Adjustments')
ax.grid(alpha=0.3)

# Plot 9: Delta surface heatmap
ax = axes[2, 2]
im = ax.contourf(spots, vols*100, delta_surface, levels=20, cmap='RdYlGn')
ax.contour(spots, vols*100, delta_surface, levels=[0.5], colors='black', linewidths=2)
ax.set_xlabel('Spot Price')
ax.set_ylabel('Volatility (%)')
ax.set_title('Delta Surface (Black line = 0.5)')
plt.colorbar(im, ax=ax, label='Delta')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
1. **Vanna & Volga:** Implement second-order cross Greeks. How do they change with skew? When are they significant?

2. **Gamma Scalping P&L:** Simulate discrete rehedging (daily vs hourly). Calculate realized Gamma P&L. How does frequency affect profit?

3. **Multi-Greek Hedging:** Build portfolio that is delta-neutral, gamma-neutral, and vega-neutral using 3 different options. Solve system of equations.

4. **Greeks at Barriers:** Calculate Greeks for knock-out option. How does Delta behave near barrier? What's discontinuity at barrier?

5. **Optimal Rehedging:** Find optimal delta-hedge rebalancing frequency minimizing sum of transaction costs and tracking error. Is there a closed-form solution?

## 7. Key References
- [Hull, Options, Futures, and Other Derivatives (Chapter 19)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000006649)
- [Taleb, Dynamic Hedging (Chapter 7-9)](https://www.wiley.com/en-us/Dynamic+Hedging-p-9780471152804)
- [Wilmott, Paul Wilmott on Quantitative Finance (Greeks Section)](https://www.wiley.com/en-us/Paul+Wilmott+on+Quantitative+Finance-p-9781118836798)
- [Haug, The Complete Guide to Option Pricing Formulas (Appendix A)](https://www.mhprofessional.com/the-complete-guide-to-option-pricing-formulas-9780071389976-usa)

---
**Status:** Core risk management tool | **Complements:** Black-Scholes Model, Delta Hedging, Portfolio Risk Management, Option Trading Strategies
