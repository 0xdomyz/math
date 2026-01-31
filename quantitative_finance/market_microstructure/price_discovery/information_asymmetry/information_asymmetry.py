"""
Information Asymmetry & Adverse Selection: Modeling Trading Losses
Analyzes how information asymmetry widens spreads and costs market makers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
import seaborn as sns

# ============================================================================
# 1. GLOSTEN-MILGROM MODEL: Sequential Trade with Information Asymmetry
# ============================================================================

def glosten_milgrom_equilibrium(fundamental_value=100, prob_informed=0.20, 
                               prob_good_news=0.60, persistence=0.85,
                               n_trades=100):
    """
    Simulate sequential trading with Glosten-Milgrom dynamics.
    Information asymmetry causes spreads to widen as MM learns.
    
    Parameters:
    - fundamental_value: Initial true value
    - prob_informed: Fraction of informed traders (constant)
    - prob_good_news: P(good news | informed buys)
    - persistence: How long information advantage persists
    - n_trades: Number of sequential trades
    """
    
    prices = np.zeros(n_trades + 1)
    prices[0] = fundamental_value
    
    spreads = np.zeros(n_trades)
    bid_prices = np.zeros(n_trades)
    ask_prices = np.zeros(n_trades)
    
    true_value = fundamental_value
    lambda_param = prob_informed  # Current belief about prob informed
    
    trade_flow = np.zeros(n_trades)  # 1 = buy, -1 = sell
    information_revealed = np.zeros(n_trades)
    
    for t in range(n_trades):
        # Generate trade: informed or uninformed
        is_informed = np.random.rand() < prob_informed
        
        # Informed trader trades in direction of private information
        if is_informed:
            if true_value > prices[t]:
                trade_flow[t] = 1  # Informed buys if undervalued
            else:
                trade_flow[t] = -1  # Informed sells if overvalued
        else:
            trade_flow[t] = np.random.choice([-1, 1])  # Random walk for uninformed
        
        # Current midpoint
        midpoint = prices[t]
        
        # Adverse selection cost: expected loss if facing informed
        adverse_selection_cost = lambda_param * np.abs(true_value - midpoint) * 0.5
        
        # Market maker sets bid-ask to break even
        ask = midpoint + adverse_selection_cost  # Selling to potentially informed
        bid = midpoint - adverse_selection_cost  # Buying from potentially informed
        
        bid_prices[t] = bid
        ask_prices[t] = ask
        spreads[t] = ask - bid
        
        # Execute trade
        if trade_flow[t] > 0:  # Buy order
            execution_price = ask
        else:  # Sell order
            execution_price = bid
        
        prices[t + 1] = execution_price
        
        # Bayesian update: learn about probability trader was informed
        # P(Informed | Buy) = P(Buy | Informed) * P(Informed) / P(Buy)
        
        if trade_flow[t] > 0:  # Buy observed
            # If informed, likely buying (true value high)
            prob_buy_if_informed = prob_good_news if true_value > midpoint else (1 - prob_good_news)
        else:  # Sell observed
            # If informed, likely selling (true value low)
            prob_sell_if_informed = (1 - prob_good_news) if true_value > midpoint else prob_good_news
            prob_buy_if_informed = 1 - prob_sell_if_informed
        
        prob_buy_if_uninformed = 0.5
        prob_buy = (prob_buy_if_informed * prob_informed + 
                   prob_buy_if_uninformed * (1 - prob_informed))
        
        # Update belief about probability next trader is informed
        # (simplified: don't update lambda itself, focus on price impact)
        
        # Information revelation: each trade reveals private information slightly
        information_revealed[t] = 1 if is_informed else 0
        
        # Update true value with persistence (informed info leaks gradually)
        if is_informed:
            signal = 1 if trade_flow[t] > 0 else -1
            true_value = true_value + persistence * signal * np.abs(true_value - midpoint) * 0.05
    
    return {
        'prices': prices,
        'bid_prices': bid_prices,
        'ask_prices': ask_prices,
        'spreads': spreads,
        'trade_flow': trade_flow,
        'information_revealed': information_revealed
    }

# ============================================================================
# 2. ADVERSE SELECTION COST CALCULATION
# ============================================================================

def calculate_adverse_selection_cost(bid_ask_spread, midpoint, execution_price, 
                                    is_informed_counterparty):
    """
    Decompose bid-ask spread into components and measure realized adverse selection loss.
    """
    
    effective_spread = 2 * np.abs(execution_price - midpoint)
    half_spread = effective_spread / 2
    
    # Adverse selection cost (permanent impact)
    if is_informed_counterparty:
        adverse_selection = half_spread  # Full half-spread loss
    else:
        adverse_selection = 0  # No loss to uninformed
    
    return {
        'half_spread': half_spread,
        'adverse_selection': adverse_selection,
        'effective_spread_bps': (effective_spread / midpoint) * 10000
    }

# ============================================================================
# 3. INFORMATION DETECTION: Unusual Order Flow Analysis
# ============================================================================

def detect_informed_order_flow(buy_volume, sell_volume, n_lookback=20):
    """
    Detect unusual order flow imbalance that signals informed trading.
    Uses rolling statistics to identify anomalies.
    """
    
    n_trades = len(buy_volume)
    
    # Order imbalance: (Buy - Sell) / (Buy + Sell)
    order_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)
    
    # Detect anomalies using rolling z-score
    imbalance_anomaly_score = np.zeros(n_trades)
    buy_ratio_anomaly = np.zeros(n_trades)
    
    for t in range(n_lookback, n_trades):
        window = order_imbalance[t - n_lookback:t]
        window_buy = buy_volume[t - n_lookback:t]
        window_total = window_buy.sum() + (sell_volume[t - n_lookback:t].sum())
        
        # Z-score of current imbalance
        if window.std() > 0:
            z_score = (order_imbalance[t] - window.mean()) / window.std()
        else:
            z_score = 0
        
        imbalance_anomaly_score[t] = z_score
        
        # Buy ratio vs expected
        current_buy_ratio = buy_volume[t] / (buy_volume[t] + sell_volume[t] + 1e-10)
        expected_buy_ratio = 0.5
        if window_buy.std() > 0:
            buy_ratio_anomaly[t] = (current_buy_ratio - expected_buy_ratio) / (window_buy.std() / window_total)
        else:
            buy_ratio_anomaly[t] = 0
    
    return order_imbalance, imbalance_anomaly_score, buy_ratio_anomaly

# ============================================================================
# 4. INFORMATION ASYMMETRY SIMULATION: Multi-Scenario
# ============================================================================

def simulate_market_structure(n_days=100, n_trades_per_day=200, 
                             prob_informed_scenarios=[0.05, 0.20, 0.40]):
    """
    Compare market outcomes under different degrees of information asymmetry.
    """
    
    results_by_scenario = {}
    
    for prob_inf in prob_informed_scenarios:
        scenario_label = f"Informed Prob = {prob_inf:.0%}"
        
        daily_spreads = []
        daily_prices = []
        daily_losses = []
        
        for day in range(n_days):
            # Generate order flow
            buy_volume = np.random.poisson(lam=100, size=n_trades_per_day)
            sell_volume = np.random.poisson(lam=100, size=n_trades_per_day)
            
            # Informed traders create directional flow
            informed_trades = np.random.choice(n_trades_per_day, 
                                            size=int(n_trades_per_day * prob_inf),
                                            replace=False)
            buy_volume[informed_trades] += np.random.poisson(lam=20, size=len(informed_trades))
            
            # Calculate spreads based on order flow
            order_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)
            
            # Spread increases with order flow imbalance (adverse selection)
            base_spread = 0.05  # 5 bps baseline
            adverse_selection_component = prob_inf * np.abs(order_imbalance) * 50
            spreads = base_spread + adverse_selection_component
            
            daily_spreads.append(spreads)
            
            # Track cumulative losses to market maker from informed traders
            losses_per_trade = spreads * prob_inf  # Loss per spread
            daily_losses.append(losses_per_trade.sum())
        
        results_by_scenario[scenario_label] = {
            'spreads': np.concatenate(daily_spreads),
            'losses': np.array(daily_losses)
        }
    
    return results_by_scenario

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*70)
print("INFORMATION ASYMMETRY & ADVERSE SELECTION ANALYSIS")
print("="*70)

# Scenario 1: Glosten-Milgrom Sequential Trade Model
print("\n--- SCENARIO 1: Glosten-Milgrom Model ---")
print("Sequential trading with information asymmetry increases spreads\n")

gm_result = glosten_milgrom_equilibrium(n_trades=150)

print("Trade Sequence Statistics:")
print(f"Initial Price: ${gm_result['prices'][0]:.2f}")
print(f"Final Price: ${gm_result['prices'][-1]:.2f}")
print(f"Average Bid-Ask Spread: {gm_result['spreads'].mean():.4f} ({gm_result['spreads'].mean()/gm_result['prices'][0]*10000:.2f} bps)")
print(f"Max Spread: {gm_result['spreads'].max():.4f} ({gm_result['spreads'].max()/gm_result['prices'][0]*10000:.2f} bps)")
print(f"Informed Trades Detected: {gm_result['information_revealed'].sum():.0f} / {len(gm_result['information_revealed'])}")

# Calculate average loss to market maker
mm_loss_per_trade = gm_result['spreads'].mean() / 2
print(f"Avg Loss per Trade (half-spread): ${mm_loss_per_trade:.4f}")
print(f"Cumulative Loss over {len(gm_result['spreads'])} trades: ${mm_loss_per_trade * len(gm_result['spreads']) * gm_result['prices'][0] / 100:.2f}")

# Scenario 2: Order Flow Imbalance Detection
print("\n--- SCENARIO 2: Informed Order Flow Detection ---")
print("Detecting unusual trading patterns that signal informed trading\n")

np.random.seed(42)
n_days = 50
buy_orders = np.random.poisson(lam=75, size=n_days)
sell_orders = np.random.poisson(lam=75, size=n_days)

# Introduce informed trading cluster
informed_period = np.zeros(n_days)
informed_period[15:25] = 1  # Days 15-25 have informed traders
buy_orders[15:25] += np.random.poisson(lam=30, size=10)

order_imb, imb_anomaly, buy_anomaly = detect_informed_order_flow(buy_orders, sell_orders)

detected_informed = np.where(np.abs(imb_anomaly) > 1.5)[0]
print(f"Days with Unusual Order Flow (|z-score| > 1.5): {len(detected_informed)}")
print(f"True Informed Period: Days 15-25")
print(f"Detection Accuracy: {np.sum(informed_period[detected_informed]) / (detected_informed.shape[0] + 1e-10):.1%}")
print(f"Missed Days: {np.sum(informed_period) - np.sum(informed_period[detected_informed]):.0f}")

# Scenario 3: Multi-Scenario Comparison
print("\n--- SCENARIO 3: Information Asymmetry Scenarios ---")
print("How spreads and market maker losses scale with informed trader presence\n")

results = simulate_market_structure(n_days=50)

for scenario_label, scenario_data in results.items():
    avg_spread = scenario_data['spreads'].mean()
    total_loss = scenario_data['losses'].sum()
    
    print(f"{scenario_label}:")
    print(f"  Average Daily Spread: {avg_spread:.4f} ({avg_spread*10000:.2f} bps)")
    print(f"  Total Losses (50 days): ${total_loss*100:.2f}")
    print()

# Scenario 4: Stoll Decomposition - Spread Components
print("\n--- SCENARIO 4: Bid-Ask Spread Decomposition (Stoll Model) ---")
print("Spread = Order Processing + Inventory Risk + Adverse Selection\n")

# Typical values
order_processing_component = 0.0010  # 10 bps
inventory_risk_volatility = 0.15
inventory_risk_component = 0.0005 * inventory_risk_volatility  # 5 bps * vol
adverse_selection_low = 0.0005  # 5 bps (5% informed traders)
adverse_selection_high = 0.0020  # 20 bps (25% informed traders)

spread_low_asymmetry = order_processing_component + inventory_risk_component + adverse_selection_low
spread_high_asymmetry = order_processing_component + inventory_risk_component + adverse_selection_high

print("SPREAD DECOMPOSITION (Stock at $100, 15% Volatility):\n")
print("Low Information Asymmetry (5% informed traders):")
print(f"  Order Processing:        {order_processing_component*10000:6.2f} bps")
print(f"  Inventory Risk:          {inventory_risk_component*10000:6.2f} bps")
print(f"  Adverse Selection:       {adverse_selection_low*10000:6.2f} bps")
print(f"  Total Spread:            {spread_low_asymmetry*10000:6.2f} bps")

print("\nHigh Information Asymmetry (25% informed traders):")
print(f"  Order Processing:        {order_processing_component*10000:6.2f} bps")
print(f"  Inventory Risk:          {inventory_risk_component*10000:6.2f} bps")
print(f"  Adverse Selection:       {adverse_selection_high*10000:6.2f} bps")
print(f"  Total Spread:            {spread_high_asymmetry*10000:6.2f} bps")

print(f"\nAdverse Selection Premium: {(adverse_selection_high - adverse_selection_low)*10000:.2f} bps")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Glosten-Milgrom bid-ask evolution
ax1 = plt.subplot(2, 3, 1)
periods = np.arange(len(gm_result['ask_prices']))
ax1.fill_between(periods, gm_result['bid_prices'], gm_result['ask_prices'], 
                  alpha=0.3, label='Bid-Ask Spread')
ax1.plot(periods, gm_result['prices'], label='Execution Prices', linewidth=1.5, color='black')
ax1.set_xlabel('Trade Number')
ax1.set_ylabel('Price ($)')
ax1.set_title('Glosten-Milgrom: Spread Evolution with Adverse Selection')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Spread over time
ax2 = plt.subplot(2, 3, 2)
spreads_bps = gm_result['spreads'] / gm_result['prices'][0] * 10000
ax2.plot(spreads_bps, linewidth=1.5, color='red', label='Bid-Ask Spread')
ax2.fill_between(range(len(spreads_bps)), spreads_bps, alpha=0.3, color='red')
ax2.set_xlabel('Trade Number')
ax2.set_ylabel('Spread (bps)')
ax2.set_title('Adverse Selection: Spreads Widen Over Time')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Order flow imbalance with anomalies
ax3 = plt.subplot(2, 3, 3)
ax3.plot(order_imb, label='Order Imbalance', linewidth=1, alpha=0.7)
ax3.bar(range(len(imb_anomaly)), imb_anomaly, alpha=0.5, color=['red' if x > 1.5 else 'blue' for x in imb_anomaly])
ax3.axhline(y=1.5, color='red', linestyle='--', linewidth=1, label='Alert Threshold')
ax3.axhline(y=-1.5, color='red', linestyle='--', linewidth=1)
ax3.axvspan(15, 25, alpha=0.1, color='orange', label='Informed Period')
ax3.set_xlabel('Day')
ax3.set_ylabel('Anomaly Z-Score')
ax3.set_title('Order Flow Anomaly Detection (Informed Trading Signal)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Scenario comparison - spreads
ax4 = plt.subplot(2, 3, 4)
for scenario_label, scenario_data in results.items():
    spreads_values = scenario_data['spreads']
    ax4.hist(spreads_values, bins=30, alpha=0.6, label=scenario_label)
ax4.set_xlabel('Bid-Ask Spread (dollars)')
ax4.set_ylabel('Frequency')
ax4.set_title('Spread Distribution: Impact of Information Asymmetry')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Daily losses by scenario
ax5 = plt.subplot(2, 3, 5)
scenario_labels = list(results.keys())
daily_loss_means = [results[s]['losses'].mean() for s in scenario_labels]
daily_loss_stds = [results[s]['losses'].std() for s in scenario_labels]

x_pos = np.arange(len(scenario_labels))
ax5.bar(x_pos, daily_loss_means, yerr=daily_loss_stds, capsize=5, alpha=0.7, 
        color=['green', 'orange', 'red'])
ax5.set_ylabel('Daily Loss to Market Maker ($)')
ax5.set_title('Market Maker Losses: Scale with Informed Trader Density')
ax5.set_xticks(x_pos)
ax5.set_xticklabels([s.replace('Informed Prob = ', '') for s in scenario_labels])
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Spread decomposition stacked bar
ax6 = plt.subplot(2, 3, 6)
components = ['Order\nProcessing', 'Inventory\nRisk', 'Adverse\nSelection']
low_values = [order_processing_component*10000, inventory_risk_component*10000, adverse_selection_low*10000]
high_values = [order_processing_component*10000, inventory_risk_component*10000, adverse_selection_high*10000]

x = np.arange(len(components))
width = 0.35

bars1 = ax6.bar(x - width/2, low_values, width, label='Low Asymmetry (5% informed)', alpha=0.8)
bars2 = ax6.bar(x + width/2, high_values, width, label='High Asymmetry (25% informed)', alpha=0.8)

ax6.set_ylabel('Spread Component (bps)')
ax6.set_title('Stoll Spread Decomposition')
ax6.set_xticks(x)
ax6.set_xticklabels(components)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('information_asymmetry_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved: information_asymmetry_analysis.png")

plt.show()

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. ADVERSE SELECTION MECHANISM:
   - Each informed trade costs market maker half-spread on average
   - Uninformed trades compensate market maker for informed losses
   - Equilibrium: 4-5 uninformed trades per informed trade needed

2. SPREAD WIDENING:
   - Low asymmetry environment: 5-10 bps spreads
   - High asymmetry environment: 20-50 bps spreads
   - Critical threshold: ~20% informed traders → spreads double

3. DETECTION CHALLENGES:
   - Order flow imbalance detectable but noisy
   - False positives from correlated liquidity shocks
   - Informed traders may disguise flow (splitting orders, timing)

4. MARKET MAKER EQUILIBRIUM:
   - Spreads adjust to make market-making break-even
   - High adverse selection → lower trading volume (liquidity evaporates)
   - Feedback loop: Widening spreads → More uninformed traders exit → Spreads widen more

5. PRICE IMPACT:
   - Informed traders move prices permanently
   - Uninformed traders cause only temporary spread changes
   - Information eventually reveals but asymmetry persists temporarily
""")
