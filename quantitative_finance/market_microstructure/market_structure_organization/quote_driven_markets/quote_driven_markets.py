import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class QuoteDrivenMarket:
    def __init__(self, initial_price=100.0, fundamental_value=100.0):
        self.price = initial_price
        self.fundamental = fundamental_value
        self.dealer_inventory = 0
        self.dealer_inventory_target = 0
        self.dealer_pnl = 0
        self.quote_history = []
        self.trade_log = []
        self.spread_history = []
        self.inventory_history = []
        self.bid_history = []
        self.ask_history = []
        
    def get_spread(self, inventory, volatility=0.01):
        """Calculate spread based on Stoll's model"""
        # Spread components:
        # 1. Adverse selection (fear of informed traders)
        # 2. Inventory costs (compensation for holding risk)
        
        adverse_selection_cost = 0.005 * volatility  # 0.5% base
        inventory_cost = 0.0001 * abs(inventory)  # Cost per unit of inventory
        
        spread = adverse_selection_cost + inventory_cost
        return max(0.001, spread)  # Minimum spread
    
    def set_quotes(self, fundamental, inventory, volatility):
        """Dealer sets bid and ask based on fundamental and inventory"""
        mid = fundamental
        spread = self.get_spread(inventory, volatility)
        
        # Inventory adjustment (widen ask if short, widen bid if long)
        inventory_adjustment = 0.002 * inventory  # Adjust mid by inventory
        
        bid = mid - spread / 2 + inventory_adjustment
        ask = mid + spread / 2 + inventory_adjustment
        
        return bid, ask
    
    def execute_trade(self, side, quantity, informed_prob=0.1):
        """Execute trade from customer"""
        
        # Update fundamental with random news
        news = np.random.normal(0, 0.002)
        self.fundamental *= (1 + news)
        
        # Determine if order is informed
        is_informed = np.random.random() < informed_prob
        
        # Get current quotes
        volatility = abs(self.fundamental - self.price) / self.price + 0.01
        bid, ask = self.set_quotes(self.fundamental, self.dealer_inventory, volatility)
        
        if side == 'buy':
            # Customer buys at dealer's ask
            execution_price = ask
            
            # If informed, customer wants to buy near fundamental
            if is_informed and self.price < self.fundamental:
                execution_price = ask  # Pay dealer's ask anyway
            
            self.dealer_inventory -= quantity  # Dealer sells, inventory decreases
            self.dealer_pnl += quantity * (ask - self.fundamental)  # PnL from trade
            
        else:  # sell
            # Customer sells at dealer's bid
            execution_price = bid
            
            # If informed, customer wants to sell near fundamental
            if is_informed and self.price > self.fundamental:
                execution_price = bid  # Get dealer's bid anyway
            
            self.dealer_inventory += quantity  # Dealer buys, inventory increases
            self.dealer_pnl += quantity * (self.fundamental - bid)  # PnL from trade
        
        # Update market price (slowly move toward fundamental)
        self.price = 0.9 * self.price + 0.1 * self.fundamental
        
        # Record trade
        self.trade_log.append({
            'side': side,
            'price': execution_price,
            'quantity': quantity,
            'inventory': self.dealer_inventory,
            'informed': is_informed
        })
        
        self.spread_history.append(ask - bid)
        self.inventory_history.append(self.dealer_inventory)
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        
        return execution_price

# Scenario 1: Dealer managing inventory
print("Scenario 1: Dealer Inventory Management")
print("=" * 80)

market1 = QuoteDrivenMarket()

# Simulate customer orders (random arrival)
for t in range(200):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000, 5000])
    
    price = market1.execute_trade(side, quantity, informed_prob=0.05)

print(f"Initial Price: $100.00")
print(f"Final Price: ${market1.price:.2f}")
print(f"Fundamental Value: ${market1.fundamental:.2f}")
print(f"Total Trades: {len(market1.trade_log)}")
print(f"Dealer Inventory: {market1.dealer_inventory:,.0f} shares")
print(f"Dealer PnL: ${market1.dealer_pnl:,.2f}")
print(f"Average Spread: ${np.mean(market1.spread_history):.4f}")
print(f"Max Spread: ${np.max(market1.spread_history):.4f}")

# Scenario 2: High informed trader proportion
print(f"\n\nScenario 2: High Informed Trader Proportion")
print("=" * 80)

market2 = QuoteDrivenMarket()

# More informed traders increase spread
for t in range(150):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    
    price = market2.execute_trade(side, quantity, informed_prob=0.3)  # 30% informed

print(f"Initial Price: $100.00")
print(f"Final Price: ${market2.price:.2f}")
print(f"Dealer Inventory: {market2.dealer_inventory:,.0f} shares")
print(f"Dealer PnL: ${market2.dealer_pnl:,.2f}")
print(f"Average Spread: ${np.mean(market2.spread_history):.4f}")
print(f"Avg Spread vs Low Informed: {np.mean(market2.spread_history) / np.mean(market1.spread_history):.2f}x wider")

# Scenario 3: Crisis (fundamental drops sharply)
print(f"\n\nScenario 3: Crisis Scenario (Fundamental Crash)")
print("=" * 80)

market3 = QuoteDrivenMarket(fundamental_value=100.0)

# Normal period
for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    price = market3.execute_trade(side, quantity, informed_prob=0.1)

# Crisis: Fundamental drops 20%
market3.fundamental = 80.0
spreads_crisis = []

for t in range(100):
    side = 'buy' if np.random.random() < 0.5 else 'sell'
    quantity = np.random.choice([100, 500, 1000])
    price = market3.execute_trade(side, quantity, informed_prob=0.5)  # More informed in crisis
    spreads_crisis.append(market3.spread_history[-1])

print(f"Price Pre-Crisis: $100.00")
print(f"Fundamental Pre-Crisis: $100.00")
print(f"Price Post-Crisis: ${market3.price:.2f}")
print(f"Fundamental Post-Crisis: $80.00")
print(f"Dealer Inventory at Crisis: {market3.dealer_inventory:,.0f} shares")
print(f"Dealer PnL: ${market3.dealer_pnl:,.2f}")
print(f"Average Spread Pre-Crisis: ${np.mean(market1.spread_history):.4f}")
print(f"Average Spread Crisis: ${np.mean(spreads_crisis):.4f}")
print(f"Spread Widening: {np.mean(spreads_crisis) / np.mean(market1.spread_history):.1f}x")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Bid-Ask evolution
times = range(len(market1.bid_history))
axes[0, 0].fill_between(times, market1.bid_history, market1.ask_history, alpha=0.3, color='blue')
axes[0, 0].plot(times, market1.bid_history, linewidth=1, label='Bid', color='blue')
axes[0, 0].plot(times, market1.ask_history, linewidth=1, label='Ask', color='red')
axes[0, 0].axhline(y=market1.fundamental, color='green', linestyle='--', linewidth=2, label='Fundamental')
axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Scenario 1: Dealer Quotes (Bid-Ask Evolution)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Dealer inventory
times = range(len(market1.inventory_history))
axes[0, 1].plot(times, market1.inventory_history, linewidth=2, color='purple')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Inventory (shares)')
axes[0, 1].set_title('Scenario 1: Dealer Inventory Management')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].fill_between(times, 0, market1.inventory_history, alpha=0.3)

# Plot 3: Spread comparison
scenarios = ['Low\nInformed\n(5%)', 'High\nInformed\n(30%)']
avg_spreads = [np.mean(market1.spread_history), np.mean(market2.spread_history)]
colors = ['blue', 'orange']

bars = axes[1, 0].bar(scenarios, avg_spreads, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Average Bid-Ask Spread ($)')
axes[1, 0].set_title('Spread Widening with Informed Traders')
axes[1, 0].grid(alpha=0.3, axis='y')

for bar, spread in zip(bars, avg_spreads):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${spread:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Crisis spread spike
time_crisis = range(len(spreads_crisis))
axes[1, 1].plot(time_crisis, spreads_crisis, linewidth=2, color='red', label='Crisis Period')
axes[1, 1].axhline(y=np.mean(market1.spread_history), color='blue', linestyle='--', linewidth=2, label='Normal Average')
axes[1, 1].set_xlabel('Trade Number (Crisis Period)')
axes[1, 1].set_ylabel('Bid-Ask Spread ($)')
axes[1, 1].set_title('Scenario 3: Spread Widening During Crisis')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print(f"\n\nComparative Summary:")
print("=" * 80)
print(f"{'Metric':<30} {'Scenario 1':<20} {'Scenario 2':<20}")
print("-" * 70)
print(f"{'Average Spread':<30} {f'${np.mean(market1.spread_history):.4f}':<20} {f'${np.mean(market2.spread_history):.4f}':<20}")
print(f"{'Max Spread':<30} {f'${np.max(market1.spread_history):.4f}':<20} {f'${np.max(market2.spread_history):.4f}':<20}")
print(f"{'Final Inventory':<30} {f'{market1.dealer_inventory:,.0f}':<20} {f'{market2.dealer_inventory:,.0f}':<20}")
print(f"{'Dealer PnL':<30} {f'${market1.dealer_pnl:,.2f}':<20} {f'${market2.dealer_pnl:,.2f}':<20}")
print(f"{'Total Trades':<30} {f'{len(market1.trade_log)}':<20} {f'{len(market2.trade_log)}':<20}")
