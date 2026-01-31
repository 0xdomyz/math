# Liquidity Crises

## 1. Concept Skeleton
**Definition:** Sudden, severe deterioration in market liquidity characterized by widening spreads, thin order books, and trading halts  
**Purpose:** Understanding systemic market failures helps design circuit breakers, stress tests, and risk management protocols  
**Prerequisites:** Market liquidity, order book depth, market making, volatility dynamics, systemic risk

## 2. Comparative Framing
| Crisis Type | Flash Crash | Market Freeze | Flight to Quality | Forced Liquidation |
|-------------|-------------|---------------|-------------------|-------------------|
| **Duration** | Minutes | Days-weeks | Weeks-months | Hours-days |
| **Cause** | Algorithm cascade | Risk aversion | Macro uncertainty | Margin calls |
| **Spread** | 10-100x | 5-20x | 2-5x | 3-10x |
| **Recovery** | Same day | Weeks | Months | Days |

## 3. Examples + Counterexamples

**Flash Crash (2010-05-06):**  
Dow drops 1000 points (9%) in 5 minutes. E-Mini S&P futures fell 5%, triggering stop-losses → HFT withdrawal → liquidity evaporates. P&G trades @ $39.97 (down 37%)

**COVID Crash (2020-03):**  
Sustained liquidity crisis: VIX → 82, bid-ask spreads widen 5x, Treasury market freezes despite "safe haven" status. Fed intervention required ($700B QE)

**Resilient Market:**  
Brexit vote (2016): GBP drops 8% overnight but orderly trading. Spreads widen 2-3x but market makers remain active. No circuit breaker triggers

## 4. Layer Breakdown
```
Liquidity Crisis Framework:
├─ Crisis Mechanisms:
│   ├─ Flash Crash Dynamics:
│   │   - Trigger: Large sell order + thin order book
│   │   - Amplification: Stop-loss cascades
│   │   - HFT withdrawal: Adverse selection avoidance
│   │   - Price dislocation: Trades 50-90% off true value
│   │   - Recovery: "Stub quotes" filled, trades busted
│   ├─ Market Freeze:
│   │   - Cause: Heightened uncertainty (Lehman 2008)
│   │   - Market maker withdrawal: Risk limits breached
│   │   - Bid-ask spread explosion: 10-20x normal
│   │   - Price discovery failure: No consensus valuation
│   │   - Intervention: Fed liquidity facilities, trading halts
│   ├─ Contagion:
│   │   - Cross-asset: Equity → FX → fixed income
│   │   - Geographic: US → Europe → Asia
│   │   - Institutional: Hedge fund → prime broker → bank
│   │   - Mechanism: Margin calls, forced deleveraging
│   └─ Systemic Spiral:
│       - Volatility spike → VaR limits → position reduction
│       - Sales pressure → price decline → more selling
│       - Liquidity commonality: All assets illiquid simultaneously
├─ 2010 Flash Crash (May 6):
│   ├─ Timeline:
│   │   - 14:32 EST: Waddell & Reed sells 75K E-Mini contracts ($4.1B)
│   │   - 14:42: Dow -600 pts, HFT quote withdrawals accelerate
│   │   - 14:45: Dow -1000 pts, 20K trades execute @ absurd prices
│   │   - 14:47: Trading pauses (5-second halt), prices recover
│   │   - 15:00: Most losses recovered, but 20K trades broken
│   ├─ Anatomy:
│   │   - HFT passed hot potato: Bought/sold same contracts
│   │   - Liquidity illusion: Depth vanished when needed
│   │   - Stub quotes: MM quotes @ $0.01 or $100K (regulatory requirement)
│   │   - ETF arbitrage breakdown: Accenture → $0.01
│   ├─ Regulatory Response:
│   │   - Single-stock circuit breakers (LULD = Limit Up/Limit Down)
│   │   - Market-wide halts: 5-min pause if S&P drops 7%/13%/20%
│   │   - Clearly erroneous trade policy: Bust trades >30% off ref price
│   └─ Lessons:
│       - HFT liquidity is conditional, vanishes in stress
│       - Algorithms can amplify volatility
│       - Need coordinated circuit breakers across markets
├─ COVID Crisis (March 2020):
│   ├─ Symptoms:
│   │   - Treasury market freeze: Bid-ask spreads 10x normal
│   │   - Corporate bond market: No bids for investment-grade debt
│   │   - Repo rates: Spike to 10% (money market stress)
│   │   - Equity spreads: SPY effective spread → 0.05% (vs 0.01% normal)
│   ├─ Causes:
│   │   - Dash for cash: Sell everything for USD liquidity
│   │   - Dealer balance sheet constraints: No capacity to intermediate
│   │   - Risk-off: All correlations → 1 (diversification fails)
│   │   - Margin calls: Forced selling across asset classes
│   ├─ Fed Intervention:
│   │   - $700B QE (Treasury + MBS purchases)
│   │   - Corporate bond facilities (SMCCF, PMCCF)
│   │   - Repo operations: $1T+ daily
│   │   - FX swap lines: Global USD liquidity
│   └─ Market Impact:
│       - 4 circuit breaker triggers (March 9, 12, 16, 18)
│       - VIX → 82 (all-time high)
│       - S&P drops 34% in 23 days (fastest bear market)
│       - Recovery: V-shaped once Fed backstop announced
├─ Circuit Breaker Design:
│   ├─ Market-Wide Halts (S&P 500):
│   │   - Level 1: 7% decline → 15-minute halt (before 3:25pm)
│   │   - Level 2: 13% decline → 15-minute halt
│   │   - Level 3: 20% decline → close for day
│   ├─ Single-Stock LULD:
│   │   - Price bands: ±5% for S&P 500, ±10% for others
│   │   - 5-minute halt if exceeds band
│   │   - Reference price: 5-minute trailing average
│   ├─ Trade-Through Prevention (Reg NMS):
│   │   - Can't trade through better quotes
│   │   - Prevents stub quote executions
│   └─ Market Maker Obligations:
│       - DMMs must maintain quotes (with exceptions)
│       - Wider spreads allowed in volatility
├─ Liquidity Stress Indicators:
│   ├─ Price-Based:
│   │   - Bid-ask spread: Widening > 3x normal
│   │   - Effective spread: Actual cost vs quoted
│   │   - Price impact: $ per share → abnormal
│   ├─ Volume-Based:
│   │   - Order book depth: Top-of-book liquidity decline
│   │   - Hidden liquidity: Dark pool vs lit venue shift
│   │   - Trade size: Avg trade size drops (risk aversion)
│   ├─ Behavioral:
│   │   - HFT withdrawal: Quote cancellation rate spikes
│   │   - Market maker pause: DMM step-aside events
│   │   - ETF premium/discount: Arbitrage breakdown
│   └─ Cross-Asset:
│       - Volatility smile: Skew steepens (tail risk)
│       - Correlation: All assets move together
│       - Safe-haven failure: Treasury liquidity dry-up
└─ Policy Implications:
    ├─ Ex-Ante: Circuit breakers, stress tests, margin buffers
    ├─ Ex-Post: Central bank intervention, trade cancellation
    ├─ Structural: Dealer capital requirements, transparency
    └─ Debate: Do circuit breakers help or signal panic?
```

**Interaction:** Price shock → risk limit breach → position liquidation → further price decline → contagion

## 5. Mini-Project
Simulate flash crash dynamics with circuit breakers:
```python
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
```

## 6. Challenge Round
Why did Treasury markets freeze during COVID despite "flight to quality"?
- **Dealer balance sheet constraints**: Supplementary Leverage Ratio (SLR) rules limit dealer capacity to intermediate, even in Treasuries (risk-free but balance sheet intensive)
- **Forced selling dominates safe-haven demand**: Hedge funds, foreign central banks liquidated Treasuries to raise USD cash → overwhelming sell pressure
- **Basis trade unwind**: Treasury cash-futures arbitrage required deleveraging, adding to selling pressure despite fundamental demand
- **Repo market stress**: Dealers unwilling to provide financing, cutting off leverage → forced sales
- **Fed intervention necessity**: $700B QE to absorb supply, SLR relief to expand dealer capacity → normalcy restored

## 7. Key References
- [Kirilenko et al (2017) - The Flash Crash: High-Frequency Trading in an Electronic Market](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12498)
- [SEC CFTC (2010) - Findings Regarding the Market Events of May 6, 2010](https://www.sec.gov/news/studies/2010/marketevents-report.pdf)
- [Schrimpf et al (2020) - Leverage and Margin Spirals in Fixed Income Markets during the Covid-19 Crisis](https://www.bis.org/publ/bisbull02.pdf)
- [Duffie (2020) - Still the World's Safe Haven? Redesigning the US Treasury Market After the COVID-19 Crisis](https://www.brookings.edu/research/still-the-worlds-safe-haven/)

---
**Status:** Crisis dynamics | **Complements:** Market Liquidity, Circuit Breakers, Systemic Risk
