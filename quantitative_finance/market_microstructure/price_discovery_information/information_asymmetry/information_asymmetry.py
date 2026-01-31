import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Glosten-Milgrom Sequential Trade Model
class GlostenMilgromMarket:
    def __init__(self, V_high=110, V_low=90, prior_prob_high=0.5, 
                 informed_fraction=0.3):
        """
        V_high, V_low: high and low asset values
        prior_prob_high: prior probability of high value
        informed_fraction: fraction of informed traders
        """
        self.V_high = V_high
        self.V_low = V_low
        self.prior_high = prior_prob_high
        self.alpha = informed_fraction
        
        # True value (unknown to market maker at start)
        self.true_value = V_high if np.random.random() < prior_prob_high else V_low
        
        # Market maker's belief
        self.belief_high = prior_prob_high
        
        # Price history
        self.bid_history = []
        self.ask_history = []
        self.trade_history = []
        self.belief_history = []
        
    def compute_quotes(self):
        """Market maker sets bid and ask based on beliefs"""
        expected_value = self.belief_high * self.V_high + \
                        (1 - self.belief_high) * self.V_low
        
        # Bid: expected value conditional on sell order
        # P(informed | sell) * V_low + P(uninformed | sell) * E[V]
        prob_informed_given_sell = self.alpha / (self.alpha + (1 - self.alpha) / 2)
        prob_uninformed_given_sell = 1 - prob_informed_given_sell
        
        bid = prob_informed_given_sell * self.V_low + \
              prob_uninformed_given_sell * expected_value
        
        # Ask: expected value conditional on buy order
        prob_informed_given_buy = self.alpha / (self.alpha + (1 - self.alpha) / 2)
        prob_uninformed_given_buy = 1 - prob_informed_given_buy
        
        ask = prob_informed_given_buy * self.V_high + \
              prob_uninformed_given_buy * expected_value
        
        return bid, ask
    
    def generate_order(self):
        """Generate buy or sell order from informed or uninformed trader"""
        is_informed = np.random.random() < self.alpha
        
        if is_informed:
            # Informed trader knows true value
            if self.true_value == self.V_high:
                order = 'buy'
            else:
                order = 'sell'
        else:
            # Uninformed trader trades randomly
            order = 'buy' if np.random.random() < 0.5 else 'sell'
        
        return order, is_informed
    
    def update_belief(self, order):
        """Update belief using Bayes' rule after observing order"""
        # P(V=VH | order) = P(order | V=VH) * P(V=VH) / P(order)
        
        if order == 'buy':
            # P(buy | V=VH) = alpha + (1-alpha)/2
            prob_buy_given_high = self.alpha + (1 - self.alpha) / 2
            # P(buy | V=VL) = 0 + (1-alpha)/2
            prob_buy_given_low = (1 - self.alpha) / 2
        else:  # sell
            # P(sell | V=VH) = 0 + (1-alpha)/2
            prob_buy_given_high = (1 - self.alpha) / 2
            # P(sell | V=VL) = alpha + (1-alpha)/2
            prob_buy_given_low = self.alpha + (1 - self.alpha) / 2
        
        # Bayes' rule
        numerator = prob_buy_given_high * self.belief_high
        denominator = prob_buy_given_high * self.belief_high + \
                     prob_buy_given_low * (1 - self.belief_high)
        
        if denominator > 0:
            self.belief_high = numerator / denominator
    
    def simulate_trade(self):
        """Simulate one trade"""
        # Set quotes
        bid, ask = self.compute_quotes()
        
        # Generate order
        order, is_informed = self.generate_order()
        
        # Execute trade
        if order == 'buy':
            price = ask
            pnl = self.true_value - price
        else:
            price = bid
            pnl = price - self.true_value
        
        # Record
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        self.belief_history.append(self.belief_high)
        self.trade_history.append({
            'order': order,
            'price': price,
            'is_informed': is_informed,
            'pnl': pnl,
            'belief': self.belief_high
        })
        
        # Update belief
        self.update_belief(order)
        
        return order, price, is_informed, pnl

# Run simulation
n_trades = 200
informed_fraction = 0.3

market = GlostenMilgromMarket(V_high=110, V_low=90, prior_prob_high=0.5,
                              informed_fraction=informed_fraction)

print("Glosten-Milgrom Sequential Trade Model")
print("=" * 70)
print(f"\nModel Parameters:")
print(f"High Value: ${market.V_high}")
print(f"Low Value: ${market.V_low}")
print(f"True Value: ${market.true_value}")
print(f"Informed Fraction (α): {informed_fraction*100:.0f}%")
print(f"Prior Belief P(V=High): {market.prior_high*100:.0f}%")

# Simulate trades
for t in range(n_trades):
    market.simulate_trade()

# Analysis
trades = market.trade_history
bids = np.array(market.bid_history)
asks = np.array(market.ask_history)
spreads = asks - bids
beliefs = np.array(market.belief_history)

# Informed vs uninformed performance
informed_pnl = [t['pnl'] for t in trades if t['is_informed']]
uninformed_pnl = [t['pnl'] for t in trades if not t['is_informed']]

print(f"\n\nTrading Results:")
print(f"Total Trades: {len(trades)}")
print(f"Informed Trades: {len(informed_pnl)} ({len(informed_pnl)/len(trades)*100:.1f}%)")
print(f"Uninformed Trades: {len(uninformed_pnl)} ({len(uninformed_pnl)/len(trades)*100:.1f}%)")

print(f"\nProfitability:")
print(f"Informed Trader Avg PnL: ${np.mean(informed_pnl):.2f} per trade")
print(f"Uninformed Trader Avg PnL: ${np.mean(uninformed_pnl):.2f} per trade")
print(f"Informed Total: ${np.sum(informed_pnl):.2f}")
print(f"Uninformed Total: ${np.sum(uninformed_pnl):.2f}")

# Market maker PnL (opposite of traders)
mm_pnl = -np.sum([t['pnl'] for t in trades])
print(f"Market Maker Total PnL: ${mm_pnl:.2f}")
print(f"Market Maker Avg per Trade: ${mm_pnl/len(trades):.2f}")

# Spread statistics
print(f"\nSpread Analysis:")
print(f"Initial Spread: ${spreads[0]:.2f}")
print(f"Final Spread: ${spreads[-1]:.2f}")
print(f"Mean Spread: ${spreads.mean():.2f}")
print(f"Max Spread: ${spreads.max():.2f}")
print(f"Min Spread: ${spreads.min():.2f}")

# Theoretical spread
theoretical_spread = informed_fraction * (market.V_high - market.V_low)
print(f"Theoretical Spread (α(VH-VL)): ${theoretical_spread:.2f}")

# Belief convergence
final_belief = beliefs[-1]
correct_belief = 1.0 if market.true_value == market.V_high else 0.0
print(f"\nBelief Evolution:")
print(f"Initial Belief P(V=High): {market.prior_high*100:.0f}%")
print(f"Final Belief P(V=High): {final_belief*100:.1f}%")
print(f"True State: {'High' if market.true_value == market.V_high else 'Low'}")
print(f"Belief Accuracy: {'Correct' if (final_belief > 0.5) == (market.true_value == market.V_high) else 'Incorrect'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Bid, Ask, True Value
time = range(len(bids))
axes[0, 0].plot(time, bids, label='Bid', linewidth=2, alpha=0.7)
axes[0, 0].plot(time, asks, label='Ask', linewidth=2, alpha=0.7)
axes[0, 0].axhline(market.true_value, color='black', linestyle='--', 
                  linewidth=2, label=f'True Value (${market.true_value})')
axes[0, 0].fill_between(time, bids, asks, alpha=0.2, color='gray', label='Spread')

# Mark informed trades
informed_trades = [i for i, t in enumerate(trades) if t['is_informed']]
for i in informed_trades:
    axes[0, 0].scatter(i, trades[i]['price'], c='red', marker='x', s=30, 
                      alpha=0.6, zorder=5)

axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].set_title('Quotes and Trades (Red X = Informed Trade)')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Spread evolution
axes[0, 1].plot(time, spreads, linewidth=2, alpha=0.7)
axes[0, 1].axhline(theoretical_spread, color='r', linestyle='--', 
                  linewidth=2, label=f'Theoretical: ${theoretical_spread:.2f}')
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Bid-Ask Spread ($)')
axes[0, 1].set_title('Spread Evolution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Plot 3: Belief updating
axes[1, 0].plot(time, beliefs * 100, linewidth=2)
axes[1, 0].axhline(correct_belief * 100, color='green', linestyle='--', 
                  linewidth=2, label='True State')
axes[1, 0].axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
axes[1, 0].set_xlabel('Trade Number')
axes[1, 0].set_ylabel('Belief P(V = High) (%)')
axes[1, 0].set_title('Bayesian Learning: Market Maker Belief Evolution')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim([-5, 105])

# Plot 4: Cumulative PnL
cumulative_informed = np.cumsum(informed_pnl) if len(informed_pnl) > 0 else []
cumulative_uninformed = np.cumsum(uninformed_pnl) if len(uninformed_pnl) > 0 else []

informed_indices = [i for i, t in enumerate(trades) if t['is_informed']]
uninformed_indices = [i for i, t in enumerate(trades) if not t['is_informed']]

if len(cumulative_informed) > 0:
    axes[1, 1].plot(informed_indices, cumulative_informed, label='Informed Traders',
                   linewidth=2, color='green')
if len(cumulative_uninformed) > 0:
    axes[1, 1].plot(uninformed_indices, cumulative_uninformed, label='Uninformed Traders',
                   linewidth=2, color='red')

axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_xlabel('Trade Number')
axes[1, 1].set_ylabel('Cumulative PnL ($)')
axes[1, 1].set_title('Profitability: Informed vs Uninformed')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical test: Do informed traders earn positive PnL?
if len(informed_pnl) > 0:
    t_stat, p_value = stats.ttest_1samp(informed_pnl, 0)
    print(f"\nStatistical Test: Informed Trader PnL")
    print(f"H0: Mean PnL = 0")
    print(f"t-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: REJECT H0 - Informed traders earn significant profits")
    else:
        print("Result: Cannot reject H0")

# Adverse selection component of spread
# After informed buy: price should move up (permanent impact)
# After uninformed buy: price should revert (temporary impact)

# Group trades by informed status and direction
informed_buys = [t for t in trades if t['is_informed'] and t['order'] == 'buy']
informed_sells = [t for t in trades if t['is_informed'] and t['order'] == 'sell']
uninformed_buys = [t for t in trades if not t['is_informed'] and t['order'] == 'buy']
uninformed_sells = [t for t in trades if not t['is_informed'] and t['order'] == 'sell']

print(f"\nAdverse Selection Analysis:")
print(f"Informed Buys: {len(informed_buys)} (expect price to stay high)")
print(f"Informed Sells: {len(informed_sells)} (expect price to stay low)")
print(f"Uninformed Buys: {len(uninformed_buys)}")
print(f"Uninformed Sells: {len(uninformed_sells)}")

# Win rate
if len(informed_pnl) > 0:
    win_rate_informed = sum(1 for pnl in informed_pnl if pnl > 0) / len(informed_pnl)
    print(f"Informed Win Rate: {win_rate_informed*100:.1f}%")

if len(uninformed_pnl) > 0:
    win_rate_uninformed = sum(1 for pnl in uninformed_pnl if pnl > 0) / len(uninformed_pnl)
    print(f"Uninformed Win Rate: {win_rate_uninformed*100:.1f}%")
