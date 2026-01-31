# Glosten-Milgrom Model

## 1. Concept Skeleton
**Definition:** Sequential trade model where market maker sets bid-ask quotes learning from informed traders via Bayesian updating, explaining spread as compensation for adverse selection from asymmetric information  
**Purpose:** Derive bid-ask spread endogenously from information asymmetry, explain why spreads widen with informed trading probability and narrow with liquidity  
**Prerequisites:** Bayesian updating, conditional probability, sequential rationality, martingale pricing, adverse selection concepts

## 2. Comparative Framing
| Model | Glosten-Milgrom | Kyle | Easley-O'Hara | Huang-Stoll |
|-------|-----------------|------|---------------|-------------|
| **Trading** | Sequential unit trades | Batch auction | PIN framework | Decomposition |
| **Learning** | Bayesian updating | Gaussian filtering | ML estimation | Spread components |
| **Spread Driver** | Informed probability | Market depth | Volume clustering | Inventory + selection |
| **Dynamics** | Discrete quote updates | Continuous price | Statistical inference | Dealer optimization |

## 3. Examples + Counterexamples

**Simple Example:**  
Asset worth $100 or $102 equally likely. 60% informed traders. Market maker sets bid $100.80, ask $101.20 reflecting adverse selection: buyers likely informed (high value), sellers likely informed (low value)

**Failure Case:**  
All traders informed (α=1): Market maker cannot break even at any spread, market breaks down, no trading occurs

**Edge Case:**  
Zero informed traders (α=0): Spread collapses to zero (transaction cost only), efficient market hypothesis holds

## 4. Layer Breakdown
```
Glosten-Milgrom Framework:
├─ Model Setup:
│   ├─ Asset Value:
│   │   ├─ Binary: v ∈ {v_L, v_H} (low/high)
│   │   ├─ Prior: P(v = v_H) = θ₀ (e.g., 0.5)
│   │   ├─ True value unknown to market maker
│   │   └─ Known to informed traders
│   ├─ Trader Types:
│   │   ├─ Informed: Observe true value v
│   │   │   ├─ Probability α (informed fraction)
│   │   │   ├─ Buy if v = v_H, sell if v = v_L
│   │   │   └─ Earn spread as profit
│   │   └─ Uninformed (Noise): Trade randomly
│   │       ├─ Probability 1 - α
│   │       ├─ Buy with prob 0.5, sell with prob 0.5
│   │       ├─ Liquidity traders (rebalancing, etc.)
│   │       └─ Lose to adverse selection on average
│   └─ Market Maker:
│       ├─ Risk-neutral, competitive
│       ├─ Sets bid (b) and ask (a) quotes
│       ├─ Zero expected profit per trade
│       ├─ Updates beliefs via Bayes' rule
│       └─ Quote adjustment after each trade
├─ Equilibrium Quotes:
│   ├─ Ask Price (a):
│   │   ├─ Formula: a = E[v | buy]
│   │   ├─ Conditional on observing buy order
│   │   ├─ a = [α·1 + (1-α)·θ]v_H + [0·1 + (1-α)(1-θ)]v_L / [α + (1-α)]
│   │   ├─ Simplifies: a = θ₁v_H + (1-θ₁)v_L
│   │   └─ where θ₁ = updated belief after buy
│   ├─ Bid Price (b):
│   │   ├─ Formula: b = E[v | sell]
│   │   ├─ Conditional on observing sell order
│   │   ├─ Symmetric derivation
│   │   └─ b < a (bid-ask spread exists)
│   ├─ Spread Decomposition:
│   │   ├─ Spread = a - b
│   │   ├─ Increasing in α (more adverse selection)
│   │   ├─ Increasing in value uncertainty (v_H - v_L)
│   │   ├─ Zero if α = 0 (no informed traders)
│   │   └─ Infinite if α = 1 (no liquidity traders)
│   └─ Zero-Profit Condition:
│       ├─ E[gain from uninformed] = E[loss to informed]
│       ├─ Market maker breaks even on average
│       ├─ Spread exactly compensates for adverse selection
│       └─ Competitive assumption crucial
├─ Bayesian Learning:
│   ├─ Belief Updating:
│   │   ├─ Start: P(v = v_H) = θ₀
│   │   ├─ Observe buy: Apply Bayes' rule
│   │   ├─ θ₁ = P(v = v_H | buy)
│   │   ├─ = P(buy | v_H)·θ₀ / P(buy)
│   │   └─ = [α + (1-α)·0.5]θ₀ / [α + (1-α)·0.5]
│   ├─ Sequential Updates:
│   │   ├─ After each trade, update θ
│   │   ├─ Sequence of buys → θ increases (learning high value)
│   │   ├─ Sequence of sells → θ decreases (learning low value)
│   │   └─ Prices converge to true value over time
│   ├─ Martingale Property:
│   │   ├─ E[p_{t+1} | p_t] = p_t
│   │   ├─ Prices follow random walk
│   │   ├─ No predictable patterns
│   │   └─ Semi-strong efficiency
│   └─ Convergence:
│       ├─ With enough trades, θ → 0 or 1
│       ├─ Market eventually learns true value
│       ├─ Spread narrows as uncertainty resolves
│       └─ Asymptotic efficiency
├─ Extensions & Variations:
│   ├─ Multi-Value Model:
│   │   ├─ v ~ continuous distribution (not binary)
│   │   ├─ Beliefs updated via Gaussian filtering
│   │   ├─ Quotes: bid/ask based on conditional means
│   │   └─ Closer to Kyle model structure
│   ├─ Variable Trade Sizes:
│   │   ├─ Large trades signal more information
│   │   ├─ Spread increasing in trade size
│   │   ├─ Stealth trading incentive (slice orders)
│   │   └─ Empirically validated
│   ├─ Dealer Inventory:
│   │   ├─ Add inventory holding cost
│   │   ├─ Skew quotes to manage position
│   │   ├─ Spread = Adverse selection + Inventory cost
│   │   └─ Stoll (1978) decomposition
│   ├─ Order Processing Costs:
│   │   ├─ Add fixed cost per trade
│   │   ├─ Spread minimum even if α = 0
│   │   ├─ Explains spreads in liquid stocks
│   │   └─ Empirical baseline
│   └─ Time-Varying Information:
│       ├─ New information arrives periodically
│       ├─ Resets uncertainty (θ back to prior)
│       ├─ Persistent spreads (not converging to zero)
│       └─ Matches empirical data
├─ Economic Implications:
│   ├─ Market Breakdown:
│   │   ├─ If α too high, spread very wide
│   │   ├─ Uninformed exit (losing too much)
│   │   ├─ Market unravels (adverse selection spiral)
│   │   └─ Akerlof "lemons" problem in trading
│   ├─ Insider Trading Regulation:
│   │   ├─ Banning informed trading (α → 0)
│   │   ├─ Narrows spreads, benefits liquidity traders
│   │   ├─ But reduces price discovery efficiency
│   │   └─ Trade-off: Fairness vs information
│   ├─ Market Transparency:
│   │   ├─ Pre-trade transparency helps uninformed
│   │   ├─ Post-trade transparency aids learning
│   │   ├─ Dark pools hide information flow
│   │   └─ Optimal transparency level debated
│   └─ High-Frequency Trading:
│       ├─ Faster learning → faster convergence
│       ├─ But also faster adverse selection
│       ├─ HFTs as informed or uninformed?
│       └─ Empirical evidence mixed
├─ Empirical Implementation:
│   ├─ Spread Estimation:
│   │   ├─ Measure bid-ask spread directly
│   │   ├─ Decompose into components (Roll estimator)
│   │   ├─ Adverse selection via regression
│   │   └─ Time-series of spread evolution
│   ├─ Informed Probability (α):
│   │   ├─ PIN model (Easley et al.)
│   │   ├─ Maximum likelihood estimation
│   │   ├─ Trade direction clustering
│   │   └─ Correlates with spreads
│   ├─ Learning Tests:
│   │   ├─ Do prices converge after information events?
│   │   ├─ Quote updates frequency
│   │   ├─ Order flow informativeness
│   │   └─ Event study methodology
│   └─ Cross-Sectional Evidence:
│       ├─ Spreads wider for small-cap stocks (higher α)
│       ├─ Narrower for index stocks (lower α)
│       ├─ Wider around earnings (temporary α spike)
│       └─ Positive correlation: spread vs PIN
└─ Comparison with Kyle Model:
    ├─ Similarities:
    │   ├─ Both model adverse selection
    │   ├─ Informed traders profit at expense of uninformed
    │   ├─ Prices aggregate information
    │   └─ Equilibrium zero-profit for market maker
    ├─ Differences:
    │   ├─ Sequential (GM) vs batch (Kyle) trading
    │   ├─ Discrete (GM) vs continuous (Kyle) prices
    │   ├─ Bayesian (GM) vs Gaussian (Kyle) updating
    │   └─ Spread (GM) vs lambda (Kyle) focus
    ├─ Complementarity:
    │   ├─ GM: Explains bid-ask spread origin
    │   ├─ Kyle: Explains market depth and liquidity
    │   ├─ Both crucial for market microstructure
    │   └─ Unified in Back-Baruch (2004) model
    └─ Empirical Relevance:
        ├─ GM better for high-frequency data
        ├─ Kyle better for institutional orders
        ├─ Both validate adverse selection importance
        └─ Spread/lambda estimates correlated
```

**Interaction:** Trade arrives → Market maker updates beliefs (Bayes) → Sets new quotes → Next trade → Repeat → Price converges to true value

## 5. Mini-Project
Simulate Glosten-Milgrom model with learning dynamics:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from dataclasses import dataclass

@dataclass
class GlostenMilgromParams:
    """Glosten-Milgrom model parameters"""
    v_L: float         # Low asset value
    v_H: float         # High asset value
    alpha: float       # Probability of informed trader
    theta_0: float     # Prior belief P(v = v_H)
    
class GlostenMilgromModel:
    """Glosten-Milgrom sequential trade model"""
    
    def __init__(self, params: GlostenMilgromParams):
        self.params = params
        
        # Current belief
        self.theta = params.theta_0
        
        # Trade history
        self.trades = []
        self.beliefs = [params.theta_0]
        self.spreads = []
        
    def expected_value(self, theta):
        """Expected value given belief theta"""
        return theta * self.params.v_H + (1 - theta) * self.params.v_L
    
    def compute_quotes(self):
        """
        Compute bid and ask quotes
        
        Ask = E[v | buy]
        Bid = E[v | sell]
        """
        alpha = self.params.alpha
        theta = self.theta
        v_H = self.params.v_H
        v_L = self.params.v_L
        
        # Probability of buy
        # P(buy) = P(buy | v_H)·P(v_H) + P(buy | v_L)·P(v_L)
        # P(buy | v_H) = α (informed buy) + (1-α)·0.5 (uninformed)
        # P(buy | v_L) = 0 (informed don't buy) + (1-α)·0.5 (uninformed)
        
        P_buy_given_H = alpha + (1 - alpha) * 0.5
        P_buy_given_L = (1 - alpha) * 0.5
        P_buy = P_buy_given_H * theta + P_buy_given_L * (1 - theta)
        
        # Updated belief after buy: P(v_H | buy)
        if P_buy > 0:
            theta_buy = (P_buy_given_H * theta) / P_buy
        else:
            theta_buy = theta
        
        # Ask quote
        ask = theta_buy * v_H + (1 - theta_buy) * v_L
        
        # Probability of sell
        P_sell_given_H = (1 - alpha) * 0.5
        P_sell_given_L = alpha + (1 - alpha) * 0.5
        P_sell = P_sell_given_H * theta + P_sell_given_L * (1 - theta)
        
        # Updated belief after sell: P(v_H | sell)
        if P_sell > 0:
            theta_sell = (P_sell_given_H * theta) / P_sell
        else:
            theta_sell = theta
        
        # Bid quote
        bid = theta_sell * v_H + (1 - theta_sell) * v_L
        
        # Mid-quote
        mid = 0.5 * (bid + ask)
        
        # Spread
        spread = ask - bid
        
        return {
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'spread': spread,
            'theta_buy': theta_buy,
            'theta_sell': theta_sell
        }
    
    def update_belief(self, trade_direction):
        """
        Update belief after observing trade
        
        trade_direction: 'buy' or 'sell'
        """
        quotes = self.compute_quotes()
        
        if trade_direction == 'buy':
            self.theta = quotes['theta_buy']
        else:
            self.theta = quotes['theta_sell']
        
        self.beliefs.append(self.theta)
    
    def simulate_trade(self, true_value):
        """
        Simulate one trade
        
        true_value: Either v_L or v_H
        """
        alpha = self.params.alpha
        
        # Determine trader type
        is_informed = np.random.random() < alpha
        
        if is_informed:
            # Informed trader knows true value
            if true_value == self.params.v_H:
                direction = 'buy'
            else:
                direction = 'sell'
        else:
            # Uninformed trades randomly
            direction = 'buy' if np.random.random() < 0.5 else 'sell'
        
        # Get quotes before trade
        quotes = self.compute_quotes()
        
        # Execute trade
        if direction == 'buy':
            price = quotes['ask']
        else:
            price = quotes['bid']
        
        # Trader profit/loss
        trader_pnl = (true_value - price) if direction == 'buy' else (price - true_value)
        
        # Market maker profit/loss
        mm_pnl = -trader_pnl
        
        # Record trade
        self.trades.append({
            'direction': direction,
            'price': price,
            'true_value': true_value,
            'is_informed': is_informed,
            'trader_pnl': trader_pnl,
            'mm_pnl': mm_pnl,
            'bid': quotes['bid'],
            'ask': quotes['ask'],
            'spread': quotes['spread'],
            'theta_before': self.beliefs[-1]
        })
        
        self.spreads.append(quotes['spread'])
        
        # Update belief
        self.update_belief(direction)
        
        return direction, price
    
    def run_simulation(self, n_trades, true_value):
        """Run full simulation with n_trades"""
        for i in range(n_trades):
            self.simulate_trade(true_value)
        
        return pd.DataFrame(self.trades)

def comparative_analysis():
    """Compare different levels of informed trading"""
    
    base_params = {
        'v_L': 98,
        'v_H': 102,
        'theta_0': 0.5
    }
    
    alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for alpha in alpha_values:
        params = GlostenMilgromParams(**base_params, alpha=alpha)
        model = GlostenMilgromModel(params)
        
        # True value is high
        true_value = params.v_H
        
        # Run simulation
        df = model.run_simulation(n_trades=100, true_value=true_value)
        
        # Calculate metrics
        avg_spread = df['spread'].mean()
        final_belief = model.beliefs[-1]
        convergence_rate = abs(final_belief - params.theta_0)
        
        # Profitability
        informed_trades = df[df['is_informed']]
        uninformed_trades = df[~df['is_informed']]
        
        avg_informed_pnl = informed_trades['trader_pnl'].mean() if len(informed_trades) > 0 else 0
        avg_uninformed_pnl = uninformed_trades['trader_pnl'].mean() if len(uninformed_trades) > 0 else 0
        
        results.append({
            'alpha': alpha,
            'avg_spread': avg_spread,
            'final_belief': final_belief,
            'convergence': convergence_rate,
            'informed_pnl': avg_informed_pnl,
            'uninformed_pnl': avg_uninformed_pnl,
            'mm_pnl': df['mm_pnl'].sum()
        })
    
    return pd.DataFrame(results)

# Run simulations
print("="*80)
print("GLOSTEN-MILGROM MODEL SIMULATION")
print("="*80)

# Baseline simulation
params_base = GlostenMilgromParams(v_L=98, v_H=102, alpha=0.3, theta_0=0.5)
model_base = GlostenMilgromModel(params_base)

print(f"\nParameters:")
print(f"  Low value (v_L): ${params_base.v_L:.2f}")
print(f"  High value (v_H): ${params_base.v_H:.2f}")
print(f"  Value range: ${params_base.v_H - params_base.v_L:.2f}")
print(f"  Informed probability (α): {params_base.alpha:.1%}")
print(f"  Prior belief (θ₀): {params_base.theta_0:.1%}")

# Initial quotes
initial_quotes = model_base.compute_quotes()
print(f"\nInitial Quotes:")
print(f"  Bid: ${initial_quotes['bid']:.4f}")
print(f"  Ask: ${initial_quotes['ask']:.4f}")
print(f"  Mid: ${initial_quotes['mid']:.4f}")
print(f"  Spread: ${initial_quotes['spread']:.4f} ({initial_quotes['spread']/initial_quotes['mid']*10000:.1f} bps)")

# Simulate with true value = v_H
true_value = params_base.v_H
df_trades = model_base.run_simulation(n_trades=200, true_value=true_value)

print(f"\nSimulation Results (N=200 trades, true value=${true_value}):")
print(f"  Final belief P(v_H): {model_base.beliefs[-1]:.4f}")
print(f"  Initial spread: ${df_trades['spread'].iloc[0]:.4f}")
print(f"  Final spread: ${df_trades['spread'].iloc[-1]:.4f}")
print(f"  Average spread: ${df_trades['spread'].mean():.4f}")

# Trader profitability
informed = df_trades[df_trades['is_informed']]
uninformed = df_trades[~df_trades['is_informed']]

print(f"\nTrader Profitability:")
print(f"  Informed traders: {len(informed)} trades, avg P&L ${informed['trader_pnl'].mean():.4f}")
print(f"  Uninformed traders: {len(uninformed)} trades, avg P&L ${uninformed['trader_pnl'].mean():.4f}")
print(f"  Market maker: total P&L ${df_trades['mm_pnl'].sum():.4f} (should be ~0)")

# Learning dynamics
buy_trades = df_trades[df_trades['direction'] == 'buy']
sell_trades = df_trades[df_trades['direction'] == 'sell']

print(f"\nTrade Direction (true value is HIGH):")
print(f"  Buy trades: {len(buy_trades)} ({len(buy_trades)/len(df_trades)*100:.1f}%)")
print(f"  Sell trades: {len(sell_trades)} ({len(sell_trades)/len(df_trades)*100:.1f}%)")
print(f"  Informed buy: {len(buy_trades[buy_trades['is_informed']])/len(buy_trades)*100:.1f}% of buys")
print(f"  Informed sell: {len(sell_trades[sell_trades['is_informed']])/len(sell_trades)*100:.1f}% of sells")

# Comparative statics
print("\n" + "="*80)
print("COMPARATIVE STATICS: INFORMED TRADING PROBABILITY")
print("="*80)

df_comp = comparative_analysis()

print("\nEffect of α on Market Quality:")
print(df_comp.to_string(index=False))

print(f"\nKey Observations:")
print(f"  1. Spread increases with α (more adverse selection)")
print(f"  2. Convergence rate increases with α (faster learning)")
print(f"  3. Informed traders always profit on average")
print(f"  4. Uninformed traders lose more as α increases")
print(f"  5. Market maker breaks even (competitive assumption)")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot 1: Belief evolution
axes[0, 0].plot(model_base.beliefs, linewidth=2)
axes[0, 0].axhline(params_base.theta_0, color='blue', linestyle='--', label='Prior')
axes[0, 0].axhline(1.0, color='green', linestyle='--', label='True value (v_H)')
axes[0, 0].set_title('Belief Evolution: P(v = v_H)')
axes[0, 0].set_xlabel('Trade Number')
axes[0, 0].set_ylabel('Belief (θ)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Spread evolution
axes[0, 1].plot(df_trades['spread'], linewidth=1.5, alpha=0.7)
axes[0, 1].set_title('Bid-Ask Spread Over Time')
axes[0, 1].set_xlabel('Trade Number')
axes[0, 1].set_ylabel('Spread ($)')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Cumulative P&L
df_trades['cum_informed_pnl'] = informed['trader_pnl'].cumsum() if len(informed) > 0 else []
df_trades['cum_uninformed_pnl'] = uninformed['trader_pnl'].cumsum() if len(uninformed) > 0 else []

if len(informed) > 0:
    axes[1, 0].plot(informed.index, informed['trader_pnl'].cumsum(), 
                    label='Informed', linewidth=2, color='green')
if len(uninformed) > 0:
    axes[1, 0].plot(uninformed.index, uninformed['trader_pnl'].cumsum(), 
                    label='Uninformed', linewidth=2, color='red')

axes[1, 0].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].set_title('Cumulative Trader P&L')
axes[1, 0].set_xlabel('Trade Number')
axes[1, 0].set_ylabel('Cumulative P&L ($)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Trade prices
axes[1, 1].scatter(df_trades.index[df_trades['direction']=='buy'], 
                   df_trades['price'][df_trades['direction']=='buy'], 
                   c='green', marker='^', alpha=0.5, label='Buy', s=30)
axes[1, 1].scatter(df_trades.index[df_trades['direction']=='sell'], 
                   df_trades['price'][df_trades['direction']=='sell'], 
                   c='red', marker='v', alpha=0.5, label='Sell', s=30)
axes[1, 1].axhline(true_value, color='blue', linestyle='--', linewidth=2, label='True value')
axes[1, 1].set_title('Transaction Prices')
axes[1, 1].set_xlabel('Trade Number')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# Plot 5: Spread vs alpha
axes[2, 0].plot(df_comp['alpha'], df_comp['avg_spread'], marker='o', linewidth=2, markersize=8)
axes[2, 0].set_title('Spread vs Informed Probability')
axes[2, 0].set_xlabel('α (Informed Probability)')
axes[2, 0].set_ylabel('Average Spread ($)')
axes[2, 0].grid(alpha=0.3)

# Plot 6: Profitability vs alpha
axes[2, 1].plot(df_comp['alpha'], df_comp['informed_pnl'], marker='o', linewidth=2, 
                label='Informed', color='green', markersize=8)
axes[2, 1].plot(df_comp['alpha'], df_comp['uninformed_pnl'], marker='o', linewidth=2,
                label='Uninformed', color='red', markersize=8)
axes[2, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
axes[2, 1].set_title('Average P&L vs Informed Probability')
axes[2, 1].set_xlabel('α (Informed Probability)')
axes[2, 1].set_ylabel('Average P&L per Trade ($)')
axes[2, 1].legend()
axes[2, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n{'='*80}")
print(f"KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. Bid-ask spread compensates market maker for adverse selection")
print(f"2. Bayesian learning: beliefs converge to truth via order flow")
print(f"3. Informed traders consistently profit, uninformed lose on average")
print(f"4. Spread increases with informed probability (α) and value uncertainty")
print(f"5. Market breaks down if α too high (uninformed exit)")
print(f"6. Sequential price discovery slower than Kyle batch model")
```

## 6. Challenge Round
When does Glosten-Milgrom model break down?
- **Market unraveling**: If α → 1, spreads widen infinitely, uninformed traders exit, market collapses
- **Perfect correlation**: If all informed trade same side, learning too fast, no profit opportunity
- **Continuous values**: Binary assumption restrictive, extension to continuous distributions complex
- **Strategic uninformed**: Model assumes uninformed don't respond to spreads, but they do in practice
- **Dealer competition**: Competitive assumption violated if few dealers with market power

How does model inform regulation?
- **Insider trading bans**: Reduce α, narrow spreads, but reduce informational efficiency
- **Disclosure requirements**: Force information into public domain, reduce asymmetry
- **Trading halts**: Prevent adverse selection during extreme information events
- **Tick size regulation**: Minimum spread below adverse selection component unviable
- **Market maker obligations**: Compensate for adverse selection via rebates/priority

## 7. Key References
- [Glosten, Milgrom (1985): Bid, Ask, and Transaction Prices](https://www.sciencedirect.com/science/article/abs/pii/0304405X85900444)
- [Easley, O'Hara (1987): Price, Trade Size, and Information in Securities Markets](https://www.sciencedirect.com/science/article/abs/pii/0304405X87900298)
- [Copeland, Galai (1983): Information Effects on the Bid-Ask Spread](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1983.tb03834.x)

---
**Status:** Foundational adverse selection model | **Complements:** Kyle Model, PIN Estimation, Spread Decomposition
