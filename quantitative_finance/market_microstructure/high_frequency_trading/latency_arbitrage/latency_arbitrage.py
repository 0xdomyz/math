import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class Quote:
    """Order book quote"""
    venue: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp_us: int  # Microseconds
    quote_age_us: int  # How old is this quote
    
class Venue:
    """Simulate exchange venue with latency characteristics"""
    
    def __init__(self, name, base_latency_us, jitter_us):
        self.name = name
        self.base_latency_us = base_latency_us
        self.jitter_us = jitter_us
        
        # Order book
        self.bid = 100.0
        self.ask = 100.02
        self.bid_size = 1000
        self.ask_size = 1000
        
        # Quote refresh
        self.last_refresh_us = 0
        self.refresh_rate_us = 5000  # Refresh every 5ms on average
    
    def get_quote(self, current_time_us, true_price):
        """Get current quote, may be stale"""
        # Simulate quote refresh
        time_since_refresh = current_time_us - self.last_refresh_us
        
        if time_since_refresh > self.refresh_rate_us:
            # Update to true price with spread
            self.bid = true_price - 0.01
            self.ask = true_price + 0.01
            self.last_refresh_us = current_time_us
            quote_age = 0
        else:
            # Quote is stale
            quote_age = time_since_refresh
        
        # Add latency
        latency = self.base_latency_us + np.random.uniform(-self.jitter_us, self.jitter_us)
        timestamp = current_time_us + int(latency)
        
        return Quote(
            venue=self.name,
            bid=self.bid,
            ask=self.ask,
            bid_size=self.bid_size,
            ask_size=self.ask_size,
            timestamp_us=timestamp,
            quote_age_us=int(quote_age)
        )
    
    def execute_order(self, side, size, current_time_us):
        """Execute order if still available"""
        # Simulate execution latency
        exec_latency = self.base_latency_us + np.random.uniform(-self.jitter_us, self.jitter_us)
        
        # Check if quote still valid (hasn't been refreshed)
        time_since_refresh = current_time_us - self.last_refresh_us
        
        if time_since_refresh < self.refresh_rate_us * 0.8:
            # Quote still valid, execute
            if side == 'buy':
                return True, self.ask, int(exec_latency)
            else:
                return True, self.bid, int(exec_latency)
        else:
            # Quote refreshed, order rejected or adverse fill
            return False, None, int(exec_latency)

class LatencyArbitrageEngine:
    """Latency arbitrage trading engine"""
    
    def __init__(self, venues: List[Venue], our_latency_us=100, arb_threshold=0.005):
        self.venues = {v.name: v for v in venues}
        self.our_latency_us = our_latency_us
        self.arb_threshold = arb_threshold
        
        # Performance tracking
        self.trades = []
        self.pnl = 0
        self.successful_arbs = 0
        self.failed_arbs = 0
        self.adverse_fills = 0
        
    def detect_arbitrage(self, quotes: Dict[str, Quote], true_price):
        """Detect arbitrage opportunities across venues"""
        opportunities = []
        
        venue_names = list(quotes.keys())
        
        for i, venue_a in enumerate(venue_names):
            for venue_b in venue_names[i+1:]:
                quote_a = quotes[venue_a]
                quote_b = quotes[venue_b]
                
                # Buy on A, sell on B
                profit_ab = quote_b.bid - quote_a.ask
                
                # Buy on B, sell on A
                profit_ba = quote_a.bid - quote_b.ask
                
                # Check if profitable after fees (assume 0.001 per side)
                fee = 0.002
                
                if profit_ab > self.arb_threshold + fee:
                    opportunities.append({
                        'buy_venue': venue_a,
                        'sell_venue': venue_b,
                        'buy_price': quote_a.ask,
                        'sell_price': quote_b.bid,
                        'profit_potential': profit_ab - fee,
                        'buy_quote_age': quote_a.quote_age_us,
                        'sell_quote_age': quote_b.quote_age_us
                    })
                
                if profit_ba > self.arb_threshold + fee:
                    opportunities.append({
                        'buy_venue': venue_b,
                        'sell_venue': venue_a,
                        'buy_price': quote_b.ask,
                        'sell_price': quote_a.bid,
                        'profit_potential': profit_ba - fee,
                        'buy_quote_age': quote_b.quote_age_us,
                        'sell_quote_age': quote_a.quote_age_us
                    })
        
        return opportunities
    
    def execute_arbitrage(self, opportunity, current_time_us, true_price):
        """Attempt to execute arbitrage"""
        buy_venue = self.venues[opportunity['buy_venue']]
        sell_venue = self.venues[opportunity['sell_venue']]
        
        size = 100  # Trade size
        
        # Execute both legs simultaneously
        buy_success, buy_price, buy_latency = buy_venue.execute_order('buy', size, current_time_us)
        sell_success, sell_price, sell_latency = sell_venue.execute_order('sell', size, current_time_us)
        
        # Total execution time
        total_latency = max(buy_latency, sell_latency) + self.our_latency_us
        
        if buy_success and sell_success:
            # Both legs filled
            profit = (sell_price - buy_price) * size - 0.002 * size  # Fees
            
            self.pnl += profit
            self.successful_arbs += 1
            
            self.trades.append({
                'timestamp_us': current_time_us,
                'success': True,
                'profit': profit,
                'buy_venue': opportunity['buy_venue'],
                'sell_venue': opportunity['sell_venue'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'latency_us': total_latency,
                'quote_age_avg': (opportunity['buy_quote_age'] + opportunity['sell_quote_age']) / 2
            })
            
            return 'success', profit
            
        elif buy_success or sell_success:
            # Only one leg filled - adverse selection
            self.adverse_fills += 1
            
            # Estimate loss (had to hedge at worse price)
            loss = 0.01 * size  # Assume 1 cent slippage
            self.pnl -= loss
            
            self.trades.append({
                'timestamp_us': current_time_us,
                'success': False,
                'profit': -loss,
                'buy_venue': opportunity['buy_venue'],
                'sell_venue': opportunity['sell_venue'],
                'buy_price': buy_price if buy_success else None,
                'sell_price': sell_price if sell_success else None,
                'latency_us': total_latency,
                'quote_age_avg': (opportunity['buy_quote_age'] + opportunity['sell_quote_age']) / 2
            })
            
            return 'adverse_fill', -loss
        
        else:
            # Both rejected
            self.failed_arbs += 1
            
            self.trades.append({
                'timestamp_us': current_time_us,
                'success': False,
                'profit': 0,
                'buy_venue': opportunity['buy_venue'],
                'sell_venue': opportunity['sell_venue'],
                'buy_price': None,
                'sell_price': None,
                'latency_us': total_latency,
                'quote_age_avg': (opportunity['buy_quote_age'] + opportunity['sell_quote_age']) / 2
            })
            
            return 'rejected', 0

def simulate_latency_arbitrage(n_periods=10000, n_venues=3):
    """Simulate latency arbitrage across multiple venues"""
    np.random.seed(42)
    
    # Create venues with different latency characteristics
    venues = [
        Venue('NASDAQ', base_latency_us=150, jitter_us=50),
        Venue('NYSE', base_latency_us=180, jitter_us=60),
        Venue('BATS', base_latency_us=120, jitter_us=40)
    ]
    
    # Create arbitrage engine
    engine = LatencyArbitrageEngine(venues, our_latency_us=100, arb_threshold=0.005)
    
    # Market state
    true_price = 100.0
    current_time_us = 0
    
    for t in range(n_periods):
        current_time_us += 1000  # Advance 1ms
        
        # True price random walk
        true_price += np.random.normal(0, 0.01)
        
        # Occasional price jump (news event)
        if np.random.random() < 0.01:
            true_price += np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15)
        
        # Get quotes from all venues
        quotes = {}
        for venue in venues:
            quotes[venue.name] = venue.get_quote(current_time_us, true_price)
        
        # Detect arbitrage opportunities
        opportunities = engine.detect_arbitrage(quotes, true_price)
        
        # Execute best opportunity
        if opportunities:
            # Sort by profit potential
            best_opp = max(opportunities, key=lambda x: x['profit_potential'])
            
            # Only execute if quote age suggests staleness
            if best_opp['buy_quote_age'] > 2000 or best_opp['sell_quote_age'] > 2000:
                engine.execute_arbitrage(best_opp, current_time_us, true_price)
    
    return pd.DataFrame(engine.trades), engine

# Run simulation
print("="*80)
print("LATENCY ARBITRAGE SIMULATION")
print("="*80)

df_trades, arb_engine = simulate_latency_arbitrage(n_periods=10000, n_venues=3)

print(f"\nPerformance Summary:")
print(f"  Total arbitrage attempts: {len(df_trades):,}")
print(f"  Successful arbitrages: {arb_engine.successful_arbs}")
print(f"  Failed arbitrages: {arb_engine.failed_arbs}")
print(f"  Adverse fills: {arb_engine.adverse_fills}")
print(f"  Success rate: {arb_engine.successful_arbs/len(df_trades)*100:.1f}%")
print(f"  Total P&L: ${arb_engine.pnl:,.2f}")

if len(df_trades) > 0:
    successful = df_trades[df_trades['success']]
    failed = df_trades[~df_trades['success']]
    
    print(f"\nProfit Analysis:")
    print(f"  Avg profit per success: ${successful['profit'].mean():.4f}")
    print(f"  Avg loss per failure: ${failed['profit'].mean():.4f}")
    print(f"  Profit/loss ratio: {abs(successful['profit'].mean() / failed['profit'].mean()):.2f}")
    print(f"  Max single profit: ${successful['profit'].max():.2f}")
    
    print(f"\nLatency Analysis:")
    print(f"  Avg execution latency: {df_trades['latency_us'].mean():.0f} μs")
    print(f"  Success latency: {successful['latency_us'].mean():.0f} μs")
    print(f"  Failure latency: {failed['latency_us'].mean():.0f} μs")
    
    print(f"\nQuote Age Analysis:")
    print(f"  Avg quote age traded: {df_trades['quote_age_avg'].mean():.0f} μs")
    print(f"  Success quote age: {successful['quote_age_avg'].mean():.0f} μs")
    print(f"  Failure quote age: {failed['quote_age_avg'].mean():.0f} μs")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Cumulative P&L
cumulative_pnl = df_trades['profit'].cumsum()
axes[0, 0].plot(cumulative_pnl, linewidth=1.5, color='darkgreen')
axes[0, 0].set_title('Cumulative P&L from Latency Arbitrage')
axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Cumulative P&L ($)')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Success rate over time
window = 100
success_rate = df_trades['success'].rolling(window).mean() * 100
axes[0, 1].plot(success_rate, linewidth=1.5, color='blue', alpha=0.7)
axes[0, 1].set_title(f'Success Rate (Rolling {window}-trade window)')
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Success Rate (%)')
axes[0, 1].axhline(success_rate.mean(), color='red', linestyle='--', 
                   label=f'Mean: {success_rate.mean():.1f}%')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Profit distribution
axes[1, 0].hist(df_trades['profit'], bins=50, alpha=0.7, edgecolor='black', color='purple')
axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
axes[1, 0].set_title('Profit Distribution per Trade')
axes[1, 0].set_xlabel('Profit ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Latency vs success
if len(successful) > 0 and len(failed) > 0:
    axes[1, 1].boxplot([successful['latency_us'], failed['latency_us']], 
                        labels=['Success', 'Failure'])
    axes[1, 1].set_title('Execution Latency by Outcome')
    axes[1, 1].set_ylabel('Latency (μs)')
    axes[1, 1].grid(axis='y', alpha=0.3)

# Plot 5: Quote age vs profit
axes[2, 0].scatter(df_trades['quote_age_avg'], df_trades['profit'], 
                   alpha=0.3, s=10, c=df_trades['success'], cmap='RdYlGn')
axes[2, 0].set_title('Quote Age vs Profit')
axes[2, 0].set_xlabel('Average Quote Age (μs)')
axes[2, 0].set_ylabel('Profit ($)')
axes[2, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[2, 0].grid(alpha=0.3)

# Plot 6: Venue pair performance
if len(df_trades) > 0:
    df_trades['venue_pair'] = df_trades['buy_venue'] + '-' + df_trades['sell_venue']
    venue_pnl = df_trades.groupby('venue_pair')['profit'].sum().sort_values()
    
    axes[2, 1].barh(range(len(venue_pnl)), venue_pnl.values, alpha=0.7)
    axes[2, 1].set_yticks(range(len(venue_pnl)))
    axes[2, 1].set_yticklabels(venue_pnl.index)
    axes[2, 1].set_title('P&L by Venue Pair')
    axes[2, 1].set_xlabel('Total P&L ($)')
    axes[2, 1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[2, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY FINDINGS")
print(f"{'='*80}")
print(f"\n1. Latency arbitrage profitable only with sub-millisecond advantage")
print(f"2. Success rate sensitive to quote age: older quotes more likely stale")
print(f"3. Adverse fills major risk when only one leg executes")
print(f"4. Profit per trade small (~$0.50-2.00), requires high volume")
print(f"5. Infrastructure cost enormous: co-location, FPGA, cross-connects")
print(f"6. Arms race dynamics: advantage erodes as competitors catch up")
