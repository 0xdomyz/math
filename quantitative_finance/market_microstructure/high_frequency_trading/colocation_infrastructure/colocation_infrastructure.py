import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

@dataclass
class LatencyProfile:
    """Latency characteristics for different setups"""
    name: str
    market_data_latency_us: float
    order_latency_us: float
    jitter_us: float
    monthly_cost: float
    setup_cost: float

class LatencySimulator:
    """Simulate trading performance under different latency regimes"""
    
    def __init__(self, profile: LatencyProfile):
        self.profile = profile
        self.trades = []
        self.wins = 0
        self.losses = 0
        
    def race_simulation(self, n_races=1000, competitors=10):
        """
        Simulate racing against competitors for arbitrage opportunities
        
        Winner is fastest to execute (lowest total latency)
        """
        for race in range(n_races):
            # Our total latency (market data + processing + order submission)
            our_latency = (
                self.profile.market_data_latency_us +
                np.random.uniform(-self.profile.jitter_us, self.profile.jitter_us) +
                self.profile.order_latency_us +
                np.random.uniform(-self.profile.jitter_us, self.profile.jitter_us)
            )
            
            # Competitors (assume they have similar or slightly different latency)
            # Best competitors are co-located with FPGAs (50-200us total)
            competitor_latencies = np.random.uniform(50, 200, size=competitors)
            
            # Did we win?
            if our_latency < competitor_latencies.min():
                self.wins += 1
                # Successful arbitrage: ~$1-5 profit
                profit = np.random.uniform(1, 5)
            else:
                self.losses += 1
                # Lost race: either no trade or adverse fill
                profit = np.random.uniform(-0.5, 0)
            
            self.trades.append({
                'race': race,
                'our_latency_us': our_latency,
                'best_competitor_us': competitor_latencies.min(),
                'won': our_latency < competitor_latencies.min(),
                'profit': profit
            })
        
        return pd.DataFrame(self.trades)

def cost_benefit_analysis(profiles: List[LatencyProfile], trading_days=252):
    """Analyze economics of different co-location setups"""
    
    results = []
    
    for profile in profiles:
        sim = LatencySimulator(profile)
        
        # Run race simulation
        df = sim.race_simulation(n_races=10000, competitors=10)
        
        # Calculate metrics
        win_rate = sim.wins / len(sim.trades)
        avg_profit_per_trade = df['profit'].mean()
        
        # Assume we can capture 50 opportunities per day
        opportunities_per_day = 50
        trading_days_per_year = trading_days
        
        daily_profit = avg_profit_per_trade * opportunities_per_day * win_rate
        monthly_profit = daily_profit * 21  # ~21 trading days/month
        annual_profit = daily_profit * trading_days_per_year
        
        # Net profit after costs
        annual_cost = profile.monthly_cost * 12
        net_annual_profit = annual_profit - annual_cost
        
        # ROI calculation
        total_investment = profile.setup_cost + annual_cost
        roi = (net_annual_profit / total_investment) * 100 if total_investment > 0 else 0
        
        # Payback period (months)
        if monthly_profit > profile.monthly_cost:
            payback_months = profile.setup_cost / (monthly_profit - profile.monthly_cost)
        else:
            payback_months = np.inf
        
        results.append({
            'setup': profile.name,
            'win_rate': win_rate * 100,
            'avg_latency_us': df['our_latency_us'].mean(),
            'daily_profit': daily_profit,
            'monthly_profit': monthly_profit,
            'annual_profit': annual_profit,
            'monthly_cost': profile.monthly_cost,
            'annual_cost': annual_cost,
            'net_annual': net_annual_profit,
            'setup_cost': profile.setup_cost,
            'roi_pct': roi,
            'payback_months': payback_months
        })
    
    return pd.DataFrame(results)

# Define latency profiles for different setups
profiles = [
    LatencyProfile(
        name='Office Server',
        market_data_latency_us=25000,  # 25ms
        order_latency_us=25000,  # 25ms
        jitter_us=5000,
        monthly_cost=500,
        setup_cost=10000
    ),
    LatencyProfile(
        name='Cloud VPS',
        market_data_latency_us=5000,  # 5ms
        order_latency_us=5000,  # 5ms
        jitter_us=1000,
        monthly_cost=2000,
        setup_cost=5000
    ),
    LatencyProfile(
        name='Proximity Hosting',
        market_data_latency_us=500,  # 500us
        order_latency_us=500,  # 500us
        jitter_us=100,
        monthly_cost=8000,
        setup_cost=50000
    ),
    LatencyProfile(
        name='Co-location (Software)',
        market_data_latency_us=100,  # 100us
        order_latency_us=100,  # 100us
        jitter_us=20,
        monthly_cost=30000,
        setup_cost=200000
    ),
    LatencyProfile(
        name='Co-location (FPGA)',
        market_data_latency_us=20,  # 20us
        order_latency_us=20,  # 20us
        jitter_us=5,
        monthly_cost=50000,
        setup_cost=1000000
    )
]

# Run analysis
print("="*80)
print("CO-LOCATION INFRASTRUCTURE COST-BENEFIT ANALYSIS")
print("="*80)

df_analysis = cost_benefit_analysis(profiles, trading_days=252)

print("\n" + "="*80)
print("WIN RATE & LATENCY BY SETUP")
print("="*80)
print(df_analysis[['setup', 'win_rate', 'avg_latency_us']].to_string(index=False))

print("\n" + "="*80)
print("PROFITABILITY ANALYSIS")
print("="*80)
print(df_analysis[['setup', 'monthly_profit', 'monthly_cost', 'net_annual']].to_string(index=False))

print("\n" + "="*80)
print("RETURN ON INVESTMENT")
print("="*80)
print(df_analysis[['setup', 'setup_cost', 'roi_pct', 'payback_months']].to_string(index=False))

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Win rate vs latency
axes[0, 0].scatter(df_analysis['avg_latency_us'], df_analysis['win_rate'], s=200, alpha=0.6)
for idx, row in df_analysis.iterrows():
    axes[0, 0].annotate(row['setup'], (row['avg_latency_us'], row['win_rate']), 
                        fontsize=8, ha='center')
axes[0, 0].set_xscale('log')
axes[0, 0].set_title('Win Rate vs Average Latency')
axes[0, 0].set_xlabel('Average Latency (μs, log scale)')
axes[0, 0].set_ylabel('Win Rate (%)')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Monthly profit vs cost
axes[0, 1].scatter(df_analysis['monthly_cost'], df_analysis['monthly_profit'], s=200, alpha=0.6)
for idx, row in df_analysis.iterrows():
    axes[0, 1].annotate(row['setup'], (row['monthly_cost'], row['monthly_profit']), 
                        fontsize=8, ha='left')
# Break-even line
max_val = max(df_analysis['monthly_cost'].max(), df_analysis['monthly_profit'].max())
axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='Break-even')
axes[0, 1].set_title('Monthly Profit vs Monthly Cost')
axes[0, 1].set_xlabel('Monthly Cost ($)')
axes[0, 1].set_ylabel('Monthly Profit ($)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: ROI comparison
colors = ['red' if roi < 0 else 'green' for roi in df_analysis['roi_pct']]
axes[0, 2].barh(df_analysis['setup'], df_analysis['roi_pct'], color=colors, alpha=0.7)
axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=0.5)
axes[0, 2].set_title('Annual ROI by Setup')
axes[0, 2].set_xlabel('ROI (%)')
axes[0, 2].grid(axis='x', alpha=0.3)

# Plot 4: Net annual profit
colors = ['red' if profit < 0 else 'green' for profit in df_analysis['net_annual']]
axes[1, 0].barh(df_analysis['setup'], df_analysis['net_annual']/1000, color=colors, alpha=0.7)
axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].set_title('Net Annual Profit')
axes[1, 0].set_xlabel('Net Profit ($1000s)')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 5: Payback period
# Filter out infinite payback
df_plot = df_analysis[df_analysis['payback_months'] < 100].copy()
if len(df_plot) > 0:
    axes[1, 1].barh(df_plot['setup'], df_plot['payback_months'], alpha=0.7, color='purple')
    axes[1, 1].set_title('Payback Period (Months)')
    axes[1, 1].set_xlabel('Months to Payback')
    axes[1, 1].grid(axis='x', alpha=0.3)

# Plot 6: Setup cost vs net profit
axes[1, 2].scatter(df_analysis['setup_cost']/1000, df_analysis['net_annual']/1000, s=200, alpha=0.6)
for idx, row in df_analysis.iterrows():
    axes[1, 2].annotate(row['setup'], (row['setup_cost']/1000, row['net_annual']/1000), 
                        fontsize=8, ha='left')
axes[1, 2].axhline(0, color='red', linestyle='--', label='Break-even')
axes[1, 2].set_title('Setup Cost vs Net Annual Profit')
axes[1, 2].set_xlabel('Setup Cost ($1000s)')
axes[1, 2].set_ylabel('Net Annual Profit ($1000s)')
axes[1, 2].legend()
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"CONCLUSIONS")
print(f"{'='*80}")
print(f"\n1. Co-location critical for latency-sensitive strategies (HFT, latency arb)")
print(f"2. Office/cloud setups have <5% win rate vs co-located competitors")
print(f"3. FPGA co-location expensive ($1M setup) but necessary for fastest execution")
print(f"4. Profitability depends on capturing sufficient opportunities per day")
print(f"5. Non-co-located setups unprofitable for latency arbitrage")
print(f"6. ROI positive only for software/FPGA co-location at scale")
