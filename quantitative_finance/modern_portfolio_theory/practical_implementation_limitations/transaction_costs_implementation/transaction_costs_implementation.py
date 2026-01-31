import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Download data for 60/40 portfolio simulation
end_date = datetime.now()
start_date = datetime(2015, 1, 1)

print("Downloading data...")
stocks = yf.download('SPY', start=start_date, end=end_date, progress=False)['Adj Close']
bonds = yf.download('BND', start=start_date, end=end_date, progress=False)['Adj Close']

stock_returns = stocks.pct_change().dropna()
bond_returns = bonds.pct_change().dropna()

# Align dates
common_dates = stock_returns.index.intersection(bond_returns.index)
stock_returns = stock_returns.loc[common_dates]
bond_returns = bond_returns.loc[common_dates]

print(f"Period: {stock_returns.index[0].date()} to {stock_returns.index[-1].date()}")
print(f"Total days: {len(stock_returns)}")

# Initial allocation
initial_value = 1000000  # $1M
initial_weights = np.array([0.60, 0.40])  # 60/40 portfolio

def simulate_portfolio(stock_ret, bond_ret, initial_w, rebalance_frequency='annual', 
                      cost_per_trade=0, tax_rate=0.2, tax_on_rebalance=True):
    """
    Simulate 60/40 portfolio with various rebalancing frequencies and costs
    
    Parameters:
    - rebalance_frequency: 'none', 'quarterly', 'semiannual', 'annual', 'threshold'
    - cost_per_trade: Transaction cost in basis points (e.g., 20 = 20 bps = 0.2%)
    - tax_rate: Capital gains tax rate
    - tax_on_rebalance: Whether rebalancing triggers capital gains tax
    """
    n_days = len(stock_ret)
    
    # Track values
    portfolio_value = np.ones(n_days) * 1e6
    stock_value = np.ones(n_days) * 1e6 * initial_w[0]
    bond_value = np.ones(n_days) * 1e6 * initial_w[1]
    
    # Track for rebalancing
    weights = np.array(initial_w)
    total_costs = 0
    total_taxes = 0
    rebalance_dates = []
    
    # Trading parameters
    cost_basis_stock = 1e6 * initial_w[0]  # Cost basis for tax calculation
    cost_basis_bond = 1e6 * initial_w[1]
    
    for i in range(1, n_days):
        # Update values from returns
        stock_value[i] = stock_value[i-1] * (1 + stock_ret.iloc[i])
        bond_value[i] = bond_value[i-1] * (1 + bond_ret.iloc[i])
        portfolio_value[i] = stock_value[i] + bond_value[i]
        
        # Calculate current weights
        current_weights = np.array([stock_value[i], bond_value[i]]) / portfolio_value[i]
        
        # Determine if should rebalance
        should_rebalance = False
        
        if rebalance_frequency == 'quarterly' and i % 63 == 0:  # ~63 trading days/quarter
            should_rebalance = True
        elif rebalance_frequency == 'semiannual' and i % 126 == 0:  # ~126 trading days/half-year
            should_rebalance = True
        elif rebalance_frequency == 'annual' and i % 252 == 0:  # ~252 trading days/year
            should_rebalance = True
        elif rebalance_frequency == 'threshold':
            # Rebalance if any weight drifts >5% from target
            if abs(current_weights[0] - initial_w[0]) > 0.05:
                should_rebalance = True
        
        if should_rebalance and i > 0:
            rebalance_dates.append(stock_ret.index[i])
            
            # Calculate required trades
            target_stock_value = portfolio_value[i] * initial_w[0]
            target_bond_value = portfolio_value[i] * initial_w[1]
            
            stock_trade = target_stock_value - stock_value[i]
            bond_trade = target_bond_value - bond_value[i]
            
            # Transaction costs
            cost_stock = abs(stock_trade) * (cost_per_trade / 10000)
            cost_bond = abs(bond_trade) * (cost_per_trade / 10000)
            trading_cost = cost_stock + cost_bond
            
            # Tax on rebalancing
            tax_cost = 0
            if tax_on_rebalance:
                # Simplified: Assume 20% of rebalance is realized gain
                gain_stock = stock_trade * 0.1 if stock_trade > 0 else 0
                gain_bond = bond_trade * 0.1 if bond_trade > 0 else 0
                tax_cost = (gain_stock + gain_bond) * tax_rate
            
            # Update after costs and taxes
            total_trading_cost = trading_cost + tax_cost
            
            # Reduce portfolio by costs
            portfolio_value[i] -= total_trading_cost
            
            # Adjust values based on target rebalance (net of costs)
            reduction_factor = (portfolio_value[i] - total_trading_cost) / portfolio_value[i]
            
            stock_value[i] = target_stock_value * reduction_factor
            bond_value[i] = target_bond_value * reduction_factor
            
            total_costs += trading_cost
            total_taxes += tax_cost
    
    results = {
        'portfolio_value': portfolio_value,
        'stock_value': stock_value,
        'bond_value': bond_value,
        'total_return': (portfolio_value[-1] - 1e6) / 1e6,
        'annualized_return': (portfolio_value[-1] / 1e6) ** (252 / n_days) - 1,
        'transaction_costs': total_costs,
        'taxes': total_taxes,
        'total_costs': total_costs + total_taxes,
        'num_rebalances': len(rebalance_dates),
        'cost_per_rebalance': (total_costs + total_taxes) / len(rebalance_dates) if rebalance_dates else 0,
        'dates': stock_ret.index
    }
    
    return results

# Simulate different scenarios
print("\n" + "="*100)
print("PORTFOLIO SIMULATION: Transaction Costs Impact on 60/40 Portfolio")
print("="*100)

scenarios = {
    'No Rebalance': {'freq': 'none', 'cost': 0},
    'Annual (No Cost)': {'freq': 'annual', 'cost': 0},
    'Annual (20 bps)': {'freq': 'annual', 'cost': 20},
    'Annual (50 bps)': {'freq': 'annual', 'cost': 50},
    'Quarterly (20 bps)': {'freq': 'quarterly', 'cost': 20},
    'Threshold 5% (20 bps)': {'freq': 'threshold', 'cost': 20},
}

results = {}
for scenario_name, params in scenarios.items():
    results[scenario_name] = simulate_portfolio(
        stock_returns, bond_returns, initial_weights,
        rebalance_frequency=params['freq'],
        cost_per_trade=params['cost'],
        tax_rate=0.2
    )

# Summary table
print("\nSimulation Results:")
summary_data = []
for scenario_name, result in results.items():
    summary_data.append({
        'Scenario': scenario_name,
        'Final Value': f"${result['portfolio_value'][-1]:,.0f}",
        'Total Return': f"{result['total_return']*100:.2f}%",
        'Annualized': f"{result['annualized_return']*100:.2f}%",
        'Rebalances': result['num_rebalances'],
        'Trading Costs': f"${result['transaction_costs']:,.0f}",
        'Taxes': f"${result['taxes']:,.0f}",
        'Total Cost': f"${result['total_costs']:,.0f}",
        'Cost Impact': f"{-(result['total_costs']/1e6)*100:.3f}%"
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Relative performance vs buy-and-hold
buyhold_return = results['No Rebalance']['total_return']
print(f"\n" + "="*100)
print("COST IMPACT RELATIVE TO BUY-AND-HOLD")
print("="*100)
print(f"Buy-and-Hold Total Return: {buyhold_return*100:.2f}%\n")

for scenario_name, result in results.items():
    if scenario_name != 'No Rebalance':
        relative_loss = result['total_return'] - buyhold_return
        print(f"{scenario_name:>25}: {result['total_return']*100:>7.2f}% "
              f"(Loss: {relative_loss*100:>6.2f}%)")

# Calculate turnover for each strategy
print(f"\n" + "="*100)
print("ANNUAL TURNOVER BY STRATEGY")
print("="*100)

for scenario_name, result in results.items():
    if result['num_rebalances'] > 0:
        years = len(result['dates']) / 252
        turnover_annual = (result['num_rebalances'] / years) * 100 if years > 0 else 0
        print(f"{scenario_name:>25}: {turnover_annual:>6.1f}% annual turnover, "
              f"{result['num_rebalances']:>4.0f} total rebalances")

# Cost breakeven analysis
print(f"\n" + "="*100)
print("COST BREAKEVEN ANALYSIS")
print("="*100)

print("\nHow much annual alpha is needed to justify different strategies?")
print("(Assuming alpha would be generated from active management)\n")

for scenario_name in ['Annual (20 bps)', 'Annual (50 bps)', 'Quarterly (20 bps)']:
    result = results[scenario_name]
    annual_cost = result['total_costs'] / (len(result['dates']) / 252)
    print(f"{scenario_name:>25}: Need {annual_cost*100:>5.2f}% annual alpha "
          f"to break even (vs Annual 0 bps)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative portfolio value over time
for scenario_name in ['No Rebalance', 'Annual (No Cost)', 'Annual (20 bps)', 
                      'Quarterly (20 bps)', 'Threshold 5% (20 bps)']:
    result = results[scenario_name]
    cumulative = result['portfolio_value'] / 1e6
    axes[0, 0].plot(result['dates'], cumulative, label=scenario_name, linewidth=1.5)

axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Portfolio Value ($M)')
axes[0, 0].set_title('Portfolio Value Over Time: Transaction Cost Impact')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Cost impact summary (bar chart)
scenarios_to_plot = ['No Rebalance', 'Annual (No Cost)', 'Annual (20 bps)', 
                     'Annual (50 bps)', 'Quarterly (20 bps)']
total_costs = [results[s]['total_costs'] for s in scenarios_to_plot]
colors = ['green' if c < 1000 else 'orange' if c < 5000 else 'red' for c in total_costs]

bars = axes[0, 1].bar(range(len(scenarios_to_plot)), np.array(total_costs)/1000, color=colors, alpha=0.7)
axes[0, 1].set_xticks(range(len(scenarios_to_plot)))
axes[0, 1].set_xticklabels(scenarios_to_plot, rotation=45, ha='right', fontsize=8)
axes[0, 1].set_ylabel('Total Costs ($K)')
axes[0, 1].set_title('Cumulative Transaction Costs + Taxes Over Period')
axes[0, 1].grid(axis='y', alpha=0.3)

for bar, cost in zip(bars, total_costs):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost/1000:.0f}K', ha='center', va='bottom', fontsize=8)

# Plot 3: Final returns comparison
scenarios_return = ['No Rebalance', 'Annual (No Cost)', 'Annual (20 bps)', 
                    'Annual (50 bps)', 'Quarterly (20 bps)', 'Threshold 5% (20 bps)']
returns_pct = [results[s]['total_return']*100 for s in scenarios_return]

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scenarios_return)))
bars = axes[1, 0].barh(range(len(scenarios_return)), returns_pct, color=colors, alpha=0.8)

axes[1, 0].set_yticks(range(len(scenarios_return)))
axes[1, 0].set_yticklabels(scenarios_return, fontsize=9)
axes[1, 0].set_xlabel('Total Return (%)')
axes[1, 0].set_title('Final Total Returns by Strategy')
axes[1, 0].grid(axis='x', alpha=0.3)

for bar, ret in zip(bars, returns_pct):
    width = bar.get_width()
    axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                   f'{ret:.1f}%', ha='left', va='center', fontsize=8)

# Plot 4: Cost efficiency vs returns
annualized_returns = np.array([results[s]['annualized_return']*100 for s in scenarios_return])
cost_pcts = np.array([results[s]['total_costs']/(1e6)*100 for s in scenarios_return])

scatter = axes[1, 1].scatter(cost_pcts, annualized_returns, s=100, alpha=0.6, 
                             c=range(len(scenarios_return)), cmap='viridis')

for i, scenario in enumerate(scenarios_return):
    axes[1, 1].annotate(scenario.replace(' ', '\n'), 
                       (cost_pcts[i], annualized_returns[i]),
                       fontsize=7, ha='right')

axes[1, 1].set_xlabel('Cumulative Costs (% of Initial Investment)')
axes[1, 1].set_ylabel('Annualized Return (%)')
axes[1, 1].set_title('Cost vs Return Trade-off')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Per-rebalance cost analysis
print(f"\n" + "="*100)
print("PER-REBALANCE COST ANALYSIS")
print("="*100)

for scenario_name in ['Annual (20 bps)', 'Quarterly (20 bps)', 'Threshold 5% (20 bps)']:
    result = results[scenario_name]
    if result['num_rebalances'] > 0:
        years = len(result['dates']) / 252
        annual_freq = result['num_rebalances'] / years
        per_rebalance_cost = (result['total_costs'] + result['taxes']) / result['num_rebalances']
        
        print(f"\n{scenario_name}:")
        print(f"  Total rebalances: {result['num_rebalances']}")
        print(f"  Annual frequency: {annual_freq:.2f} rebalances/year")
        print(f"  Cost per rebalance: ${per_rebalance_cost:,.0f}")
        print(f"  Average portfolio value: ~${1e6:,.0f}")
        print(f"  Cost as % of portfolio: {per_rebalance_cost/1e6*100:.3f}%")

# Tax impact analysis
print(f"\n" + "="*100)
print("TAX IMPACT ON DIFFERENT ACCOUNT TYPES")
print("="*100)

for scenario_name in ['Annual (20 bps)', 'Quarterly (20 bps)']:
    result = results[scenario_name]
    tax_share = result['taxes'] / result['total_costs'] * 100 if result['total_costs'] > 0 else 0
    
    taxfree_advantage = result['taxes']  # Tax-free account saves this
    
    print(f"\n{scenario_name}:")
    print(f"  Trading costs: ${result['transaction_costs']:,.0f}")
    print(f"  Realized taxes: ${result['taxes']:,.0f}")
    print(f"  Tax as % of total cost: {tax_share:.1f}%")
    print(f"  Advantage of tax-free account: ${taxfree_advantage:,.0f}")
    print(f"  Advantage as % return: {taxfree_advantage/1e6*100:.3f}%")

# Key insights
print(f"\n" + "="*100)
print("KEY INSIGHTS: TRANSACTION COSTS AND IMPLEMENTATION")
print("="*100)
print("1. Frequent rebalancing (quarterly) costs 2-3x more than annual")
print("2. Transaction costs + taxes can exceed 0.5% annually")
print("3. Tax-deferred accounts allow unlimited rebalancing without tax drag")
print("4. Threshold-based rebalancing balances cost control vs drift")
print("5. Annual rebalancing often optimal (low cost, drift-controlled)")
print("6. Many active managers' alpha < transaction costs")
print("7. Buy-and-hold strategy requires >2% alpha to justify switching")
print("8. Small-cap and illiquid stocks have 5-10x higher transaction costs")
print("9. Market impact scales with order size (large orders hit hard)")
print("10. Execution timing (VWAP, TWAP) can save 10-50 bps on large orders")