import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

class ExecutionVenue(Enum):
    LIT_EXCHANGE = "Lit"
    DARK_POOL = "Dark"
    HFT_SCALP = "HFT Scalp"

@dataclass
class Order:
    order_id: str
    side: str  # 'buy' or 'sell'
    size: int
    is_institutional: bool
    submission_time: float

@dataclass
class Trade:
    order_id: str
    venue: ExecutionVenue
    execution_price: float
    execution_size: int
    execution_time: float
    front_run: bool = False

class DarkPoolSimulator:
    """Simulate dark pool execution with predatory practices"""
    
    def __init__(self, lit_price: float, parameters=None):
        self.lit_price = lit_price
        self.orders_dark = []  # Orders in dark pool
        self.orders_lit = []   # Orders in lit venue
        self.trades = []
        self.front_runs = []
        
        # Parameters
        self.p_front_run = parameters.get('p_front_run', 0.3) if parameters else 0.3
        self.front_run_profit_bps = parameters.get('front_run_bps', 1.5) if parameters else 1.5
        self.dark_pool_rebate = parameters.get('rebate', 0.0002) if parameters else 0.0002
        self.hft_participation = parameters.get('hft_participation', 0.4) if parameters else 0.4
        
        self.price_history = [lit_price]
    
    def add_order(self, order: Order, venue: ExecutionVenue):
        """Submit order to venue"""
        if venue == ExecutionVenue.DARK_POOL:
            self.orders_dark.append(order)
        else:
            self.orders_lit.append(order)
    
    def execute_lit_order(self, order: Order, time_step: float):
        """Execute on lit exchange (NBBO-based)"""
        # Lit market has bid-ask spread
        spread_bps = np.random.uniform(0.5, 2.0)  # 0.5-2 bps spread
        spread = self.lit_price * spread_bps / 10000
        
        if order.side == 'buy':
            execution_price = self.lit_price + spread / 2
        else:  # sell
            execution_price = self.lit_price - spread / 2
        
        trade = Trade(
            order_id=order.order_id,
            venue=ExecutionVenue.LIT_EXCHANGE,
            execution_price=execution_price,
            execution_size=order.size,
            execution_time=time_step
        )
        
        self.trades.append(trade)
        
        # Update market price (market impact)
        impact_bps = np.random.uniform(0.5, 2.0) * order.size / 10000
        self.lit_price += impact_bps * (1 if order.side == 'buy' else -1)
        
        return trade
    
    def detect_dark_order(self, order: Order):
        """HFT detects large dark pool order and front-runs"""
        if np.random.random() > self.p_front_run:
            return False  # Not detected
        
        # Front-running strategy
        hft_order = Order(
            order_id=f"HFT_FRONTRUN_{order.order_id}",
            side=order.side,
            size=min(order.size // 2, 5000),  # Partial front-run
            is_institutional=False,
            submission_time=order.submission_time - 0.001  # Slight time advantage
        )
        
        return True
    
    def execute_dark_pool_order(self, order: Order, time_step: float):
        """Execute in dark pool"""
        
        # Check for front-running
        front_run_detected = self.detect_dark_order(order)
        
        # Execution price: typically NBBO +/- small amount
        if front_run_detected:
            # Front-run causes slightly worse execution
            if order.side == 'buy':
                execution_price = self.lit_price + (self.front_run_profit_bps / 10000 * self.lit_price)
            else:
                execution_price = self.lit_price - (self.front_run_profit_bps / 10000 * self.lit_price)
        else:
            # Normal dark pool execution (small improvement over NBBO)
            improvement_bps = self.dark_pool_rebate * 10000
            if order.side == 'buy':
                execution_price = self.lit_price + (improvement_bps - 1) / 10000 * self.lit_price
            else:
                execution_price = self.lit_price - (improvement_bps - 1) / 10000 * self.lit_price
        
        # Possible partial fill (illiquidity in dark pool)
        fill_rate = np.random.uniform(0.7, 1.0)
        execution_size = int(order.size * fill_rate)
        
        trade = Trade(
            order_id=order.order_id,
            venue=ExecutionVenue.DARK_POOL,
            execution_price=execution_price,
            execution_size=execution_size,
            execution_time=time_step,
            front_run=front_run_detected
        )
        
        self.trades.append(trade)
        
        if front_run_detected:
            self.front_runs.append({
                'order_id': order.order_id,
                'side': order.side,
                'size': order.size,
                'front_run_profit_bps': self.front_run_profit_bps,
                'victim_loss': order.size * (self.front_run_profit_bps / 10000 * self.lit_price)
            })
        
        return trade
    
    def simulate_day(self, n_institutional_orders=100, n_retail_orders=500):
        """Simulate one trading day"""
        
        for t in range(n_institutional_orders):
            # Institutional orders: some to dark, some to lit
            if np.random.random() < 0.6:  # 60% to dark pool
                order = Order(
                    order_id=f"INST_{t:04d}",
                    side=np.random.choice(['buy', 'sell']),
                    size=np.random.randint(10000, 100000),
                    is_institutional=True,
                    submission_time=t
                )
                self.add_order(order, ExecutionVenue.DARK_POOL)
                self.execute_dark_pool_order(order, t)
            else:
                order = Order(
                    order_id=f"INST_{t:04d}",
                    side=np.random.choice(['buy', 'sell']),
                    size=np.random.randint(1000, 10000),
                    is_institutional=True,
                    submission_time=t
                )
                self.add_order(order, ExecutionVenue.LIT_EXCHANGE)
                self.execute_lit_order(order, t)
            
            self.price_history.append(self.lit_price)
        
        # Retail orders (mostly lit or PFOF to HFT)
        for t in range(n_institutional_orders, n_institutional_orders + n_retail_orders):
            if np.random.random() < 0.3:  # 30% to lit, rest to PFOF (simulated as worse pricing)
                order = Order(
                    order_id=f"RETAIL_{t:04d}",
                    side=np.random.choice(['buy', 'sell']),
                    size=np.random.randint(100, 1000),
                    is_institutional=False,
                    submission_time=t
                )
                self.add_order(order, ExecutionVenue.LIT_EXCHANGE)
                self.execute_lit_order(order, t)
            else:
                # PFOF: routed to market maker (worse execution)
                order = Order(
                    order_id=f"RETAIL_{t:04d}",
                    side=np.random.choice(['buy', 'sell']),
                    size=np.random.randint(100, 1000),
                    is_institutional=False,
                    submission_time=t
                )
                # Simulate worse execution via PFOF
                spread_bps = np.random.uniform(2.0, 4.0)  # Wider spread for PFOF
                spread = self.lit_price * spread_bps / 10000
                
                if order.side == 'buy':
                    execution_price = self.lit_price + spread / 2
                else:
                    execution_price = self.lit_price - spread / 2
                
                trade = Trade(
                    order_id=order.order_id,
                    venue=ExecutionVenue.HFT_SCALP,
                    execution_price=execution_price,
                    execution_size=order.size,
                    execution_time=t
                )
                self.trades.append(trade)
            
            self.price_history.append(self.lit_price)
    
    def analyze_trades(self):
        """Analyze execution quality"""
        df = pd.DataFrame([{
            'order_id': t.order_id,
            'venue': t.venue.value,
            'execution_price': t.execution_price,
            'size': t.execution_size,
            'front_run': t.front_run
        } for t in self.trades])
        
        # Compute stats by venue
        venue_stats = df.groupby('venue').agg({
            'execution_price': ['mean', 'std'],
            'size': 'mean'
        }).round(4)
        
        return df, venue_stats
    
    def compute_execution_costs(self, benchmark_price=None):
        """Calculate execution costs vs benchmark"""
        if benchmark_price is None:
            benchmark_price = np.mean(self.price_history)
        
        costs = []
        
        for trade in self.trades:
            if trade.side == 'buy':
                cost_bps = (trade.execution_price - benchmark_price) / benchmark_price * 10000
            else:
                cost_bps = (benchmark_price - trade.execution_price) / benchmark_price * 10000
            
            costs.append({
                'order_id': trade.order_id,
                'venue': trade.venue.value,
                'cost_bps': cost_bps,
                'front_run': trade.front_run
            })
        
        return pd.DataFrame(costs)

# Run simulation
print("="*80)
print("DARK POOL EXECUTION & PREDATORY PRACTICES SIMULATOR")
print("="*80)

params = {
    'p_front_run': 0.25,      # 25% of dark pool orders front-run
    'front_run_bps': 1.5,     # Front-running extracts 1.5 bps
    'rebate': 0.0002,         # Dark pool rebate 0.02 bps
    'hft_participation': 0.4  # 40% HFT involvement
}

simulator = DarkPoolSimulator(lit_price=100.0, parameters=params)

print("\nSimulating trading day...")
simulator.simulate_day(n_institutional_orders=200, n_retail_orders=800)

# Analysis
trades_df, venue_stats = simulator.analyze_trades()
costs_df = simulator.compute_execution_costs()

print(f"\nTotal trades: {len(trades_df)}")
print(f"\nTrades by venue:")
print(trades_df['venue'].value_counts())

print(f"\nVenue statistics:")
print(venue_stats)

print(f"\nFront-running incidents: {len(simulator.front_runs)}")
if simulator.front_runs:
    total_victim_loss = sum(fr['victim_loss'] for fr in simulator.front_runs)
    print(f"Total victim losses: ${total_victim_loss:,.0f}")

# Execution costs
print(f"\nExecution Costs by Venue:")
cost_by_venue = costs_df.groupby('venue')['cost_bps'].agg(['mean', 'std', 'min', 'max'])
print(cost_by_venue)

# Front-run impact
front_run_trades = costs_df[costs_df['front_run']]
normal_trades = costs_df[~costs_df['front_run']]

print(f"\nFront-Run Impact:")
if len(front_run_trades) > 0:
    print(f"  Front-run trades (n={len(front_run_trades)}): {front_run_trades['cost_bps'].mean():.2f} bps avg cost")
if len(normal_trades) > 0:
    print(f"  Normal trades (n={len(normal_trades)}): {normal_trades['cost_bps'].mean():.2f} bps avg cost")
print(f"  Additional cost from front-running: {front_run_trades['cost_bps'].mean() - normal_trades['cost_bps'].mean():.2f} bps")

# Institutional vs Retail costs
inst_trades = trades_df[trades_df['order_id'].str.contains('INST')]
retail_trades = trades_df[trades_df['order_id'].str.contains('RETAIL')]

inst_costs = costs_df.loc[costs_df['order_id'].isin(inst_trades['order_id']), 'cost_bps'].mean()
retail_costs = costs_df.loc[costs_df['order_id'].isin(retail_trades['order_id']), 'cost_bps'].mean()

print(f"\nInstitutional vs Retail Execution Quality:")
print(f"  Institutional avg cost: {inst_costs:.2f} bps")
print(f"  Retail avg cost: {retail_costs:.2f} bps")
print(f"  Retail disadvantage: {retail_costs - inst_costs:.2f} bps")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Execution prices by venue
venues_unique = costs_df['venue'].unique()
colors = {'Lit': 'green', 'Dark': 'red', 'HFT Scalp': 'orange'}

for venue in venues_unique:
    venue_data = costs_df[costs_df['venue'] == venue]
    axes[0, 0].hist(venue_data['cost_bps'], bins=20, alpha=0.6, label=venue, color=colors.get(venue, 'blue'))

axes[0, 0].set_title('Execution Cost Distribution by Venue')
axes[0, 0].set_xlabel('Cost (bps)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Front-run impact
front_run_mask = costs_df['front_run']
axes[0, 1].scatter(costs_df.index[~front_run_mask], costs_df.loc[~front_run_mask, 'cost_bps'],
                   alpha=0.5, s=20, label='Normal', color='blue')
axes[0, 1].scatter(costs_df.index[front_run_mask], costs_df.loc[front_run_mask, 'cost_bps'],
                   alpha=0.8, s=40, label='Front-run', color='red', marker='X')
axes[0, 1].set_title('Front-Running Impact on Execution Costs')
axes[0, 1].set_xlabel('Trade #')
axes[0, 1].set_ylabel('Cost (bps)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Institutional vs Retail
client_types = []
costs_list = []
for idx, row in trades_df.iterrows():
    if 'INST' in row['order_id']:
        client_types.append('Institutional')
    else:
        client_types.append('Retail')

trade_costs = costs_df['cost_bps'].values
inst_indices = [i for i, ct in enumerate(client_types) if ct == 'Institutional']
retail_indices = [i for i, ct in enumerate(client_types) if ct == 'Retail']

inst_costs_list = [trade_costs[i] for i in inst_indices if i < len(trade_costs)]
retail_costs_list = [trade_costs[i] for i in retail_indices if i < len(trade_costs)]

axes[1, 0].boxplot([inst_costs_list, retail_costs_list], labels=['Institutional', 'Retail'])
axes[1, 0].set_title('Execution Cost Comparison: Inst vs Retail')
axes[1, 0].set_ylabel('Cost (bps)')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Price evolution
axes[1, 1].plot(simulator.price_history, linewidth=1, alpha=0.7)
axes[1, 1].set_title('Market Price Evolution')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Price')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Dark pools promise NBBO-matching but often worse execution")
print(f"2. Front-running in dark pools extracts 1-3 bps per trade (millions total)")
print(f"3. Retail receives systematically worse execution vs institutional")
print(f"4. HFT participation in dark pools ambiguous (liquidity provider or predator?)")
print(f"5. Information asymmetry: brokers prioritize own dark pools for rebates")
print(f"6. Regulation: Post-trade reporting insufficient (real-time transparency needed)")
print(f"7. Systemic: ~40% of volume in dark pools creates interconnection risk")
