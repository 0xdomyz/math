import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

class ExecutionModel(Enum):
    AGENT = "Agent"
    PRINCIPAL = "Principal"
    RISKLESS_PRINCIPAL = "Riskless Principal"
    ALGORITHM = "Algorithm"

@dataclass
class BlockTradeParams:
    """Parameters for block trade simulation"""
    market_price: float
    volume: int
    volatility: float
    time_to_execute_hours: float
    dealer_risk_tolerance: float

class BlockTradeSimulator:
    """Simulate block trading mechanics and pricing"""
    
    def __init__(self, params: BlockTradeParams):
        self.params = params
        self.market_price = params.market_price
        self.volume = params.volume
        self.volatility = params.volatility
        self.time_horizon = params.time_to_execute_hours
        
        self.trades = []
        self.pricing_history = []
    
    def estimate_market_impact(self):
        """
        Estimate permanent and temporary market impact
        Using Kyle (1985) model lambda
        """
        # Simplified impact model: Impact ~ sqrt(Volume) / Liquidity
        # For large-cap: ~0.1-0.5 bps per 100k shares
        # For small-cap: ~1-5 bps per 100k shares
        
        # Assume large-cap liquid stock
        liquidity = self.volume / 100000  # Normalize to 100k share units
        
        # Permanent impact: Long-term price shift
        permanent_impact_bps = 0.2 * np.sqrt(liquidity)  # 0.2 bps per sqrt(unit)
        
        # Temporary impact: Bid-ask + liquidity pressure
        temporary_impact_bps = 0.5 * liquidity ** 0.5
        
        # Total market impact
        total_impact_bps = permanent_impact_bps + temporary_impact_bps
        
        return {
            'permanent_impact_bps': permanent_impact_bps,
            'temporary_impact_bps': temporary_impact_bps,
            'total_impact_bps': total_impact_bps
        }
    
    def price_block_trade_agent(self):
        """
        Agent model: Broker finds counterparty, takes commission
        Price: Market price ± commission
        """
        market_impact = self.estimate_market_impact()
        
        # Agent commission: 0.05-0.15% depending on size/urgency
        commission_pct = max(0.0005, 0.0015 - (self.volume / 1000000) * 0.0005)
        commission_bps = commission_pct * 10000
        
        # Price for seller: slightly worse than market
        bid_price = self.market_price * (1 - commission_pct)
        
        execution_cost_bps = commission_bps
        
        return {
            'model': ExecutionModel.AGENT.value,
            'bid_price': bid_price,
            'commission_bps': commission_bps,
            'execution_cost_bps': execution_cost_bps,
            'execution_cost_dollars': bid_price * self.volume * commission_pct,
            'time_to_execute': self.time_horizon + np.random.uniform(0.5, 2.0)  # Extra time to find buyer
        }
    
    def price_block_trade_principal(self):
        """
        Principal model: Broker takes counterparty risk
        Price: Market - spread (dealer profit), but guaranteed
        """
        market_impact = self.estimate_market_impact()
        
        # Dealer spread: Depends on risk tolerance and volatility
        # Base spread: 0.5-2% for large blocks
        base_spread_pct = 0.01 + (self.volatility * 0.02)
        
        # Risk adjustment: Larger positions = larger spread
        risk_adjustment = np.sqrt(self.volume / 100000) * 0.002
        
        total_spread_pct = base_spread_pct + risk_adjustment
        total_spread_bps = total_spread_pct * 10000
        
        # Price for seller: Market price minus spread
        bid_price = self.market_price * (1 - total_spread_pct)
        
        execution_cost_bps = total_spread_bps
        
        return {
            'model': ExecutionModel.PRINCIPAL.value,
            'bid_price': bid_price,
            'spread_bps': total_spread_bps,
            'execution_cost_bps': execution_cost_bps,
            'execution_cost_dollars': self.market_price * self.volume * total_spread_pct,
            'time_to_execute': min(0.5, self.time_horizon),  # Can execute immediately (dealer takes risk)
            'dealer_risk': total_spread_pct  # Dealer's position risk
        }
    
    def price_block_trade_riskless_principal(self):
        """
        Riskless Principal: Broker matches buyer and seller simultaneously
        Price: Spread between matched prices
        """
        market_impact = self.estimate_market_impact()
        
        # Need to find both sides: longer time
        time_to_match = self.time_horizon + np.random.uniform(1.0, 3.0)
        
        # Spread: Smaller than principal model (dealer has no risk)
        # Typically 0.1-0.5% for riskless principal
        riskless_spread_pct = 0.003 + np.random.uniform(-0.0005, 0.0005)
        riskless_spread_bps = riskless_spread_pct * 10000
        
        bid_price = self.market_price * (1 - riskless_spread_pct)
        
        execution_cost_bps = riskless_spread_bps
        
        return {
            'model': ExecutionModel.RISKLESS_PRINCIPAL.value,
            'bid_price': bid_price,
            'spread_bps': riskless_spread_bps,
            'execution_cost_bps': execution_cost_bps,
            'execution_cost_dollars': self.market_price * self.volume * riskless_spread_pct,
            'time_to_execute': time_to_match,
            'dealer_risk': 0  # No risk (matched)
        }
    
    def price_block_trade_algorithm(self):
        """
        Algorithm model: Execute VWAP over time
        Price: VWAP - small commission
        Risk: Market impact (negative)
        """
        
        # VWAP execution: Get average price over execution period
        # Simulate market movement during execution
        time_steps = 100
        prices = []
        
        for t in range(time_steps):
            # Random walk during execution
            price_move = np.random.normal(0, self.volatility / np.sqrt(time_steps))
            prices.append(self.market_price * (1 + price_move))
        
        vwap_price = np.mean(prices)
        
        # Algorithm commission
        algo_commission_pct = 0.0005  # 0.05% for algorithm
        algo_commission_bps = algo_commission_pct * 10000
        
        # Execution cost: Market impact (negative for seller)
        market_impact = self.estimate_market_impact()
        total_impact_bps = market_impact['total_impact_bps']
        
        # Final price: VWAP - commission - impact
        net_price = vwap_price * (1 - algo_commission_pct) * (1 - total_impact_bps / 10000)
        
        execution_cost_bps = algo_commission_bps + total_impact_bps
        
        return {
            'model': ExecutionModel.ALGORITHM.value,
            'bid_price': net_price,
            'vwap': vwap_price,
            'market_impact_bps': total_impact_bps,
            'commission_bps': algo_commission_bps,
            'execution_cost_bps': execution_cost_bps,
            'execution_cost_dollars': self.market_price * self.volume * (execution_cost_bps / 10000),
            'time_to_execute': self.time_horizon
        }
    
    def compare_execution_models(self):
        """Compare all execution models"""
        
        models = [
            self.price_block_trade_agent(),
            self.price_block_trade_principal(),
            self.price_block_trade_riskless_principal(),
            self.price_block_trade_algorithm()
        ]
        
        df = pd.DataFrame(models)
        
        return df

# Run simulations
print("="*80)
print("BLOCK TRADING EXECUTION MODEL COMPARISON")
print("="*80)

scenarios = [
    {
        'name': 'Large-Cap Liquid (1M shares)',
        'params': BlockTradeParams(
            market_price=100.0,
            volume=1000000,
            volatility=0.02,
            time_to_execute_hours=8,
            dealer_risk_tolerance=0.5
        )
    },
    {
        'name': 'Mid-Cap Illiquid (500k shares)',
        'params': BlockTradeParams(
            market_price=50.0,
            volume=500000,
            volatility=0.05,
            time_to_execute_hours=24,
            dealer_risk_tolerance=0.3
        )
    },
    {
        'name': 'Small-Cap Thin (100k shares)',
        'params': BlockTradeParams(
            market_price=15.0,
            volume=100000,
            volatility=0.10,
            time_to_execute_hours=48,
            dealer_risk_tolerance=0.2
        )
    }
]

all_results = {}

for scenario in scenarios:
    print(f"\n{scenario['name']}")
    print("-" * 80)
    
    sim = BlockTradeSimulator(scenario['params'])
    df = sim.compare_execution_models()
    
    all_results[scenario['name']] = df
    
    print(f"\nMarket Price: ${scenario['params'].market_price:.2f}")
    print(f"Volume: {scenario['params'].volume:,} shares")
    print(f"Volatility: {scenario['params'].volatility*100:.1f}%")
    
    print(f"\nExecution Model Comparison:")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Bid Price: ${row['bid_price']:.2f}")
        print(f"  Execution Cost (bps): {row['execution_cost_bps']:.2f}")
        print(f"  Total Cost: ${row['execution_cost_dollars']:,.0f}")
        
        if 'time_to_execute' in row and pd.notna(row['time_to_execute']):
            print(f"  Time to Execute: {row['time_to_execute']:.1f} hours")
        
        if 'dealer_risk' in row and pd.notna(row['dealer_risk']):
            print(f"  Dealer Risk: {row['dealer_risk']*100:.2f}%")

# Visualization
fig, axes = plt.subplots(len(scenarios), 3, figsize=(16, 4*len(scenarios)))

if len(scenarios) == 1:
    axes = [axes]

for idx, scenario_name in enumerate(all_results.keys()):
    df = all_results[scenario_name]
    
    # Plot 1: Execution Cost
    ax = axes[idx][0] if len(scenarios) > 1 else axes[0]
    models = df['model']
    costs = df['execution_cost_bps']
    colors = ['green' if c < 3 else 'orange' if c < 5 else 'red' for c in costs]
    ax.bar(range(len(models)), costs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title(f'{scenario_name}\nExecution Cost (bps)')
    ax.set_ylabel('Cost (bps)')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 2: Dollar Cost
    ax = axes[idx][1] if len(scenarios) > 1 else axes[1]
    dollar_costs = df['execution_cost_dollars']
    ax.bar(range(len(models)), dollar_costs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title(f'{scenario_name}\nTotal Dollar Cost')
    ax.set_ylabel('Cost ($)')
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Model Tradeoffs
    ax = axes[idx][2] if len(scenarios) > 1 else axes[2]
    times = [df.loc[i, 'time_to_execute'] if 'time_to_execute' in df.columns and pd.notna(df.loc[i, 'time_to_execute']) else 0 for i in range(len(df))]
    costs_ax3 = df['execution_cost_bps']
    
    scatter = ax.scatter(times, costs_ax3, s=300, c=range(len(models)), cmap='viridis', alpha=0.7, edgecolor='black')
    for i, model in enumerate(models):
        ax.annotate(model, (times[i], costs_ax3.iloc[i]), fontsize=8, ha='center', va='center')
    
    ax.set_title(f'{scenario_name}\nSpeed vs Cost Tradeoff')
    ax.set_xlabel('Time to Execute (hours)')
    ax.set_ylabel('Execution Cost (bps)')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Principal model: Highest cost but fastest (dealer takes risk)")
print(f"2. Agent model: Moderate cost, slower (need to find counterparty)")
print(f"3. Riskless principal: Low cost, slowest (need both buyer and seller)")
print(f"4. Algorithm: Variable cost, but often best for medium-size trades")
print(f"5. Trade-off: Speed vs Cost (can't optimize both)")
print(f"6. Size matters: Larger blocks → higher cost (market impact)")
print(f"7. Liquidity matters: Illiquid stocks → higher block premiums")
