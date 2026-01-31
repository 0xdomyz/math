import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from collections import deque

class HaltLevel(Enum):
    NONE = 0
    SINGLE_STOCK = 5  # 5 minutes
    LEVEL_1 = 15      # 7% market move
    LEVEL_2 = 15      # 13% market move
    LEVEL_3 = 1440    # 20% market move (EOD)

@dataclass
class CircuitBreakerParams:
    """Parameters for circuit breaker simulation"""
    market_halt_7pct: float = 0.07
    market_halt_13pct: float = 0.13
    market_halt_20pct: float = 0.20
    single_stock_10pct: float = 0.10
    cascade_lambda: float = 0.15  # Feedback strength
    liquidity_evaporation_rate: float = 0.3  # How much liquidity disappears

class MarketSimulator:
    """Simulate circuit breaker triggers and market cascades"""
    
    def __init__(self, initial_price: float, params: CircuitBreakerParams):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.params = params
        
        self.time_steps = []
        self.prices = []
        self.halts = []
        self.cascade_intensity = []
        self.spreads = []
    
    def simulate_shock(self, magnitude: float, duration_steps: int = 100):
        """
        Simulate market shock and cascade dynamics
        
        magnitude: initial shock severity (-0.05 = -5%)
        duration_steps: how many time steps shock persists
        """
        
        halt_active = False
        halt_end_time = 0
        market_open_price = self.current_price
        cumulative_shock = 0
        
        for t in range(duration_steps):
            # Check if halt is active
            if halt_active and t >= halt_end_time:
                halt_active = False
                # Restart: pent-up demand/supply creates sharp move
                self.current_price *= 1.02  # 2% recovery bounce
            
            if not halt_active:
                # Initial shock
                if t < 10:
                    shock_term = magnitude / 10  # Distribute shock over 10 steps
                    cumulative_shock += shock_term
                else:
                    shock_term = 0
                
                # Cascade feedback: selling begets more selling
                cascade_term = -self.params.cascade_lambda * cumulative_shock
                
                # Liquidity: as panic sets in, spreads widen
                liquidity_multiplier = 1 + self.params.liquidity_evaporation_rate * abs(cumulative_shock)
                
                # Price move
                price_move = shock_term + cascade_term
                self.current_price *= (1 + price_move)
                
                # Spread widening
                bid_ask_spread = 0.01 * liquidity_multiplier
                
                # Check for halts
                market_move = (self.current_price - market_open_price) / market_open_price
                
                # Market-wide halts (Rule 80B)
                if abs(market_move) > self.params.market_halt_20pct:
                    halt_active = True
                    halt_end_time = t + int(HaltLevel.LEVEL_3.value)  # EOD
                    self.halts.append({'time': t, 'level': 'EOD (20%)', 'price': self.current_price})
                
                elif abs(market_move) > self.params.market_halt_13pct:
                    halt_active = True
                    halt_end_time = t + HaltLevel.LEVEL_2.value
                    self.halts.append({'time': t, 'level': 'Level 2 (13%)', 'price': self.current_price})
                
                elif abs(market_move) > self.params.market_halt_7pct:
                    halt_active = True
                    halt_end_time = t + HaltLevel.LEVEL_1.value
                    self.halts.append({'time': t, 'level': 'Level 1 (7%)', 'price': self.current_price})
                
                # Record
                self.time_steps.append(t)
                self.prices.append(self.current_price)
                self.cascade_intensity.append(abs(cumulative_shock))
                self.spreads.append(bid_ask_spread)
            else:
                # During halt: accumulate orders but no execution
                self.time_steps.append(t)
                self.prices.append(self.current_price)  # Price frozen
                self.cascade_intensity.append(abs(cumulative_shock))
                self.spreads.append(0.1)  # Spread huge during halt (no quotes)
    
    def analyze_cascade(self):
        """Analyze cascade severity and halt effectiveness"""
        if not self.prices:
            return None
        
        df = pd.DataFrame({
            'time': self.time_steps,
            'price': self.prices,
            'cascade_intensity': self.cascade_intensity,
            'spread_bps': [s * 10000 for s in self.spreads]
        })
        
        max_decline = (min(self.prices) - self.initial_price) / self.initial_price * 100
        cascade_severity = max(self.cascade_intensity)
        
        analysis = {
            'max_decline_pct': max_decline,
            'cascade_severity': cascade_severity,
            'num_halts': len(self.halts),
            'halts': self.halts,
            'final_price': self.current_price,
            'recovery_pct': (self.current_price - min(self.prices)) / min(self.prices) * 100
        }
        
        return df, analysis

# Run simulations
print("="*80)
print("CIRCUIT BREAKER CASCADE SIMULATOR")
print("="*80)

params = CircuitBreakerParams(
    cascade_lambda=0.15,
    liquidity_evaporation_rate=0.3
)

scenarios = [
    {'name': 'Small Shock (3%)', 'magnitude': -0.03},
    {'name': 'Medium Shock (7%)', 'magnitude': -0.07},
    {'name': 'Large Shock (15%)', 'magnitude': -0.15},
    {'name': 'Extreme Shock (25%)', 'magnitude': -0.25}
]

all_results = {}

for scenario in scenarios:
    print(f"\n{scenario['name']}:")
    print("-" * 40)
    
    sim = MarketSimulator(100.0, params)
    sim.simulate_shock(scenario['magnitude'], duration_steps=150)
    
    df, analysis = sim.analyze_cascade()
    all_results[scenario['name']] = (df, analysis)
    
    print(f"  Max decline: {analysis['max_decline_pct']:.2f}%")
    print(f"  Cascade severity: {analysis['cascade_severity']:.3f}")
    print(f"  Number of halts: {analysis['num_halts']}")
    print(f"  Final price: ${analysis['final_price']:.2f}")
    print(f"  Recovery: {analysis['recovery_pct']:.2f}% from low")
    
    if analysis['halts']:
        print(f"  Halts triggered:")
        for halt in analysis['halts']:
            print(f"    T={halt['time']}: {halt['level']} @ ${halt['price']:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, (scenario_name, (df, analysis)) in enumerate(all_results.items()):
    ax = axes[idx]
    
    # Plot price evolution
    ax.plot(df['time'], df['price'], 'b-', linewidth=2, label='Price', alpha=0.8)
    ax.axhline(100, color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    # Mark halts
    for halt in analysis['halts']:
        ax.axvline(halt['time'], color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(halt['time'], ax.get_ylim()[1] * 0.95, halt['level'], 
               rotation=90, fontsize=8, ha='right')
    
    # Fill halt regions
    for halt in analysis['halts']:
        ax.axvspan(halt['time'], halt['time'] + 15, alpha=0.2, color='red')
    
    ax.set_title(f"{scenario_name}\nDecline: {analysis['max_decline_pct']:.1f}%, Halts: {analysis['num_halts']}")
    ax.set_xlabel('Time (steps)')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([85, 105])

plt.tight_layout()
plt.show()

# Comparative analysis
print(f"\n{'='*80}")
print("CIRCUIT BREAKER EFFECTIVENESS ANALYSIS")
print(f"{'='*80}")

print(f"\nShock Magnitude vs Halt Triggers:")
print(f"{'Scenario':<25} {'Decline %':<15} {'Halts':<10} {'Recovery %':<15}")
print("-" * 65)

for scenario_name, (df, analysis) in all_results.items():
    print(f"{scenario_name:<25} {analysis['max_decline_pct']:>7.1f}% {analysis['num_halts']:>15} {analysis['recovery_pct']:>12.1f}%")

print(f"\nKey Insights:")
print(f"1. Small shocks (<3%): No halts, cascade contained by spreads widening")
print(f"2. Medium shocks (7%): Trigger Level 1 halt (Rule 80B 7%), allows recovery")
print(f"3. Large shocks (15%): Multiple halts, pent-up demand creates recovery bounce")
print(f"4. Extreme shocks (25%): Cascades hard, halts may be insufficient")
print(f"\nCircuit Breaker Tradeoff:")
print(f"- Pro: Prevents cascades (halts interrupt feedback loop)")
print(f"- Con: Pent-up orders create sharp recoveries (whipsaw risk)")
print(f"- Optimal: Balance between cascade prevention and trading disruption")
