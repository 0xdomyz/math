import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Glosten-Milgrom Model Simulation
class GlostenMilgromSimulation:
    def __init__(self, V_L=90, V_H=110, mu_0=0.5, alpha=0.3, n_trades=200):
        """
        V_L, V_H: low and high values
        mu_0: prior probability of V_H
        alpha: fraction of informed traders
        n_trades: number of sequential trades
        """
        self.V_L = V_L
        self.V_H = V_H
        self.mu_0 = mu_0
        self.alpha = alpha
        self.n_trades = n_trades
        
        # Draw true value
        self.V_true = V_H if np.random.random() < mu_0 else V_L
        
        # History
        self.beliefs = [mu_0]
        self.bids = []
        self.asks = []
        self.spreads = []
        self.trades = []
        self.expected_values = []
        
    def compute_prices(self, mu_t):
        """Compute bid and ask given belief μ_t"""
        # Expected value
        E_V = mu_t * self.V_H + (1 - mu_t) * self.V_L
        
        # Ask price: E[V | buy]
        # P(buy | V_H) = alpha + (1-alpha)/2
        # P(buy | V_L) = (1-alpha)/2
        prob_buy_high = self.alpha + (1 - self.alpha) / 2
        prob_buy_low = (1 - self.alpha) / 2
        prob_buy = prob_buy_high * mu_t + prob_buy_low * (1 - mu_t)
        
        if prob_buy > 0:
            mu_given_buy = prob_buy_high * mu_t / prob_buy
        else:
            mu_given_buy = mu_t
        
        ask = mu_given_buy * self.V_H + (1 - mu_given_buy) * self.V_L
        
        # Bid price: E[V | sell]
        prob_sell_high = (1 - self.alpha) / 2
        prob_sell_low = self.alpha + (1 - self.alpha) / 2
        prob_sell = prob_sell_high * mu_t + prob_sell_low * (1 - mu_t)
        
        if prob_sell > 0:
            mu_given_sell = prob_sell_high * mu_t / prob_sell
        else:
            mu_given_sell = mu_t
        
        bid = mu_given_sell * self.V_H + (1 - mu_given_sell) * self.V_L
        
        return bid, ask, E_V
    
    def generate_order(self, mu_t):
        """Generate order from informed or uninformed trader"""
        is_informed = np.random.random() < self.alpha
        
        if is_informed:
            # Informed knows true value
            if self.V_true == self.V_H:
                order = 'buy'
            else:
                order = 'sell'
        else:
            # Uninformed trades randomly
            order = 'buy' if np.random.random() < 0.5 else 'sell'
        
        return order, is_informed
    
    def update_belief(self, mu_t, order):
        """Bayesian update after observing order"""
        prob_buy_high = self.alpha + (1 - self.alpha) / 2
        prob_buy_low = (1 - self.alpha) / 2
        prob_sell_high = (1 - self.alpha) / 2
        prob_sell_low = self.alpha + (1 - self.alpha) / 2
        
        if order == 'buy':
            prob_order = prob_buy_high * mu_t + prob_buy_low * (1 - mu_t)
            if prob_order > 0:
                mu_new = prob_buy_high * mu_t / prob_order
            else:
                mu_new = mu_t
        else:  # sell
            prob_order = prob_sell_high * mu_t + prob_sell_low * (1 - mu_t)
            if prob_order > 0:
                mu_new = prob_sell_high * mu_t / prob_order
            else:
                mu_new = mu_t
        
        return mu_new
    
    def run_simulation(self):
        """Run full simulation"""
        mu_t = self.mu_0
        
        for t in range(self.n_trades):
            # Compute prices
            bid, ask, E_V = self.compute_prices(mu_t)
            spread = ask - bid
            
            # Generate order
            order, is_informed = self.generate_order(mu_t)
            
            # Execute trade
            if order == 'buy':
                price = ask
            else:
                price = bid
            
            # Compute profit/loss
            if order == 'buy':
                pnl = self.V_true - price
            else:
                pnl = price - self.V_true
            
            # Record
            self.bids.append(bid)
            self.asks.append(ask)
            self.spreads.append(spread)
            self.expected_values.append(E_V)
            self.trades.append({
                'order': order,
                'price': price,
                'is_informed': is_informed,
                'pnl': pnl,
                'belief': mu_t
            })
            
            # Update belief
            mu_t = self.update_belief(mu_t, order)
            self.beliefs.append(mu_t)
        
        return self

# Run simulation
sim = GlostenMilgromSimulation(V_L=90, V_H=110, mu_0=0.5, alpha=0.3, n_trades=200)
sim.run_simulation()

print("Glosten-Milgrom Sequential Trade Model")
print("=" * 70)
print(f"\nParameters:")
print(f"V_L = ${sim.V_L}, V_H = ${sim.V_H}")
print(f"True Value: ${sim.V_true}")
print(f"Informed Fraction (α): {sim.alpha*100:.0f}%")
print(f"Prior Belief: {sim.mu_0*100:.0f}%")

# Analysis
beliefs = np.array(sim.beliefs)
bids = np.array(sim.bids)
asks = np.array(sim.asks)
spreads = np.array(sim.spreads)
expected_values = np.array(sim.expected_values)

print(f"\nBelief Convergence:")
print(f"Initial Belief P(V=V_H): {beliefs[0]*100:.1f}%")
print(f"Final Belief P(V=V_H): {beliefs[-1]*100:.1f}%")
print(f"True State: {'V_H' if sim.V_true == sim.V_H else 'V_L'}")
print(f"Correct Final Belief: {'Yes' if (beliefs[-1] > 0.5) == (sim.V_true == sim.V_H) else 'No'}")

print(f"\nSpread Evolution:")
print(f"Initial Spread: ${spreads[0]:.2f}")
print(f"Final Spread: ${spreads[-1]:.2f}")
print(f"Mean Spread: ${spreads.mean():.2f}")
print(f"Spread Reduction: {(1 - spreads[-1]/spreads[0])*100:.1f}%")

# Theoretical initial spread
theoretical_spread = sim.alpha * (sim.V_H - sim.V_L)
print(f"Theoretical Initial Spread: ${theoretical_spread:.2f}")

# Profitability
informed_pnl = [t['pnl'] for t in sim.trades if t['is_informed']]
uninformed_pnl = [t['pnl'] for t in sim.trades if not t['is_informed']]

print(f"\nProfitability:")
print(f"Informed Trades: {len(informed_pnl)}")
print(f"Informed Avg PnL: ${np.mean(informed_pnl):.2f}")
print(f"Uninformed Trades: {len(uninformed_pnl)}")
print(f"Uninformed Avg PnL: ${np.mean(uninformed_pnl):.2f}")
print(f"Uninformed Total Loss: ${np.sum(uninformed_pnl):.2f}")

# Theoretical uninformed loss = half-spread
avg_half_spread = spreads.mean() / 2
print(f"Theoretical Uninformed Loss per Trade: ${-avg_half_spread:.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Belief evolution
time = range(len(beliefs))
correct_value = 100 if sim.V_true == sim.V_H else 0

axes[0, 0].plot(time, beliefs * 100, linewidth=2, label='Belief P(V=V_H)')
axes[0, 0].axhline(correct_value, color='green', linestyle='--', linewidth=2,
                  label=f'True State: {"V_H" if sim.V_true == sim.V_H else "V_L"}')
axes[0, 0].axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
axes[0, 0].fill_between(time, 0, beliefs * 100, alpha=0.3)
axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Belief P(V = V_H) (%)')
axes[0, 0].set_title('Bayesian Learning and Belief Convergence')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_ylim([-5, 105])

# Plot 2: Bid, ask, and true value
time_prices = range(len(bids))

axes[0, 1].plot(time_prices, bids, label='Bid', linewidth=2, alpha=0.7, color='blue')
axes[0, 1].plot(time_prices, asks, label='Ask', linewidth=2, alpha=0.7, color='red')
axes[0, 1].axhline(sim.V_true, color='black', linestyle='--', linewidth=2,
                  label=f'True Value (${sim.V_true})')
axes[0, 1].fill_between(time_prices, bids, asks, alpha=0.2, color='gray', label='Spread')

# Mark trades
for i, trade in enumerate(sim.trades):
    color = 'green' if trade['is_informed'] else 'orange'
    marker = '^' if trade['order'] == 'buy' else 'v'
    axes[0, 1].scatter(i, trade['price'], c=color, marker=marker, s=15, alpha=0.4)

axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Price ($)')
axes[0, 1].set_title('Bid-Ask Quotes (Green=Informed, Orange=Uninformed)')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Spread convergence
axes[1, 0].plot(time_prices, spreads, linewidth=2, color='purple')
axes[1, 0].axhline(theoretical_spread, color='red', linestyle='--', linewidth=2,
                  label=f'Theoretical: α(V_H-V_L) = ${theoretical_spread:.2f}')
axes[1, 0].fill_between(time_prices, 0, spreads, alpha=0.3, color='purple')
axes[1, 0].set_xlabel('Trade Number')
axes[1, 0].set_ylabel('Spread ($)')
axes[1, 0].set_title('Spread Narrowing as Information Revealed')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Cumulative PnL
informed_cumsum = np.cumsum(informed_pnl) if len(informed_pnl) > 0 else []
uninformed_cumsum = np.cumsum(uninformed_pnl) if len(uninformed_pnl) > 0 else []

informed_indices = [i for i, t in enumerate(sim.trades) if t['is_informed']]
uninformed_indices = [i for i, t in enumerate(sim.trades) if not t['is_informed']]

if len(informed_cumsum) > 0:
    axes[1, 1].plot(informed_indices, informed_cumsum, label='Informed',
                   linewidth=2, color='green')
if len(uninformed_cumsum) > 0:
    axes[1, 1].plot(uninformed_indices, uninformed_cumsum, label='Uninformed',
                   linewidth=2, color='red')

axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_xlabel('Trade Number')
axes[1, 1].set_ylabel('Cumulative PnL ($)')
axes[1, 1].set_title('Cumulative Profits: Informed Gain, Uninformed Lose')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Convergence speed analysis
def find_convergence_time(beliefs, true_high, threshold=0.9):
    """Find time to reach threshold confidence"""
    target = threshold if true_high else 1 - threshold
    
    for i, b in enumerate(beliefs):
        if true_high and b >= target:
            return i
        elif not true_high and b <= (1 - target):
            return i
    return len(beliefs)

conv_time = find_convergence_time(beliefs, sim.V_true == sim.V_H, threshold=0.9)
print(f"\nConvergence Speed:")
print(f"Trades to 90% confidence: {conv_time}")
print(f"Rate: {conv_time/len(sim.trades)*100:.1f}% of total trades")

# Buy/sell imbalance
n_buys = sum(1 for t in sim.trades if t['order'] == 'buy')
n_sells = len(sim.trades) - n_buys
print(f"\nOrder Flow:")
print(f"Buy Orders: {n_buys} ({n_buys/len(sim.trades)*100:.1f}%)")
print(f"Sell Orders: {n_sells} ({n_sells/len(sim.trades)*100:.1f}%)")
print(f"Expected (if V=V_H): Buy > Sell")
print(f"Expected (if V=V_L): Sell > Buy")
print(f"Actual: {'Consistent' if (n_buys > n_sells) == (sim.V_true == sim.V_H) else 'Inconsistent'}")

# Zero-sum check
total_pnl = sum(t['pnl'] for t in sim.trades)
print(f"\nZero-Sum Verification:")
print(f"Total PnL (all traders): ${total_pnl:.2f}")
print(f"Market Maker PnL: ${-total_pnl:.2f}")
print(f"MM Breaks Even: {'Yes' if abs(total_pnl) < 1 else 'No'}")
