import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Build a behavioral portfolio model incorporating loss aversion

def prospect_theory_value(outcome, ref_point=0, alpha=0.88, beta=0.88, lambda_=2.25):
    """
    Compute prospect theory value for outcome relative to reference point.
    
    Parameters:
    - outcome: realized outcome (e.g., return percentage)
    - ref_point: mental reference point (e.g., cost basis, expected return)
    - alpha, beta: diminishing sensitivity parameters
    - lambda_: loss aversion coefficient
    """
    x = outcome - ref_point  # Deviation from reference point
    
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * ((-x) ** beta)


def simulate_investor_decisions(annual_returns, initial_price, reference_price, lambda_coeff=2.25):
    """
    Simulate investor selling decision based on prospect theory.
    Investors more likely to sell winners (lock in gain) than losers (avoid loss realization).
    """
    sell_probabilities = []
    
    for price in annual_returns:
        # Unrealized return relative to purchase price
        unrealized_return = (price - reference_price) / reference_price
        
        # Prospect theory value of holding vs selling
        value_hold = prospect_theory_value(unrealized_return, ref_point=0, lambda_=lambda_coeff)
        value_sell = 0  # Baseline (no emotional attachment after selling)
        
        # Probability of selling increases if holding causes loss sensation
        # Use logistic function to map value difference to probability
        value_diff = value_hold - value_sell
        sell_prob = 1 / (1 + np.exp(value_diff * 0.01))  # Logistic sigmoid
        
        sell_probabilities.append(sell_prob)
    
    return np.array(sell_probabilities)


def portfolio_performance_comparison(initial_wealth, years=10, risk_free_rate=0.02):
    """
    Compare rational vs behavioral investor portfolios over time.
    
    Rational: Rebalance consistently, sell based on valuation
    Behavioral: Sell winners, hold losers, overweight familiar assets
    """
    
    np.random.seed(42)
    
    # Generate annual returns (stock market simulation)
    annual_returns_market = np.random.normal(0.08, 0.16, years)
    annual_returns_bond = np.random.normal(0.03, 0.05, years)
    
    # Initial positions
    rational_wealth = initial_wealth
    behavioral_wealth = initial_wealth
    
    rational_stock_weight = 0.60
    behavioral_stock_weight = 0.60
    
    rational_history = [rational_wealth]
    behavioral_history = [behavioral_wealth]
    
    stock_purchase_price = 100  # Reference price for loss aversion
    current_stock_price = 100
    
    for year, (market_ret, bond_ret) in enumerate(zip(annual_returns_market, annual_returns_bond)):
        
        # RATIONAL INVESTOR: Rebalance to target 60/40
        stock_value = rational_wealth * rational_stock_weight
        bond_value = rational_wealth * (1 - rational_stock_weight)
        
        stock_value *= (1 + market_ret)
        bond_value *= (1 + bond_ret)
        
        rational_wealth = stock_value + bond_value
        
        # Always rebalance to 60/40
        rational_stock_weight = stock_value / rational_wealth
        rational_history.append(rational_wealth)
        
        # BEHAVIORAL INVESTOR: Disposition effect
        current_stock_price *= (1 + market_ret)
        unrealized_return = (current_stock_price - stock_purchase_price) / stock_purchase_price
        
        stock_value = behavioral_wealth * behavioral_stock_weight
        bond_value = behavioral_wealth * (1 - behavioral_stock_weight)
        
        stock_value *= (1 + market_ret)
        bond_value *= (1 + bond_ret)
        
        # Compute selling probability based on prospect theory
        sell_prob = simulate_investor_decisions(
            np.array([current_stock_price]), 
            stock_purchase_price,
            stock_purchase_price,
            lambda_coeff=2.25
        )[0]
        
        # If unrealized gain large, behavioral investor more likely to sell (lock in gain)
        if unrealized_return > 0.15 and np.random.random() < sell_prob:
            # Sell some winners
            behavioral_stock_weight = max(0.40, behavioral_stock_weight - 0.10)
            stock_purchase_price = current_stock_price  # Reset reference point
        
        # If unrealized loss large, behavioral investor reluctant to sell (avoid loss realization)
        elif unrealized_return < -0.15 and np.random.random() > sell_prob:
            # Hold losers (don't rebalance down)
            behavioral_stock_weight = min(0.70, behavioral_stock_weight + 0.05)
        
        behavioral_wealth = stock_value + bond_value
        behavioral_history.append(behavioral_wealth)
    
    return np.array(rational_history), np.array(behavioral_history), annual_returns_market


# Main Analysis
print("=" * 90)
print("LOSS AVERSION & PROSPECT THEORY IN PORTFOLIO DECISIONS")
print("=" * 90)

# 1. Prospect theory value function visualization
print("\n1. PROSPECT THEORY VALUE FUNCTION")
print("-" * 90)

outcomes = np.linspace(-50, 50, 1000)
values = [prospect_theory_value(x, ref_point=0, lambda_=2.25) for x in outcomes]

print(f"Value of +$100 gain: {prospect_theory_value(100):.2f}")
print(f"Value of -$100 loss: {prospect_theory_value(-100):.2f}")
print(f"Loss aversion ratio: {abs(prospect_theory_value(-100)) / prospect_theory_value(100):.2f}x")
print(f"  → Losses valued ~2.25× more than equivalent gains")

# 2. Selling behavior simulation
print("\n2. DISPOSITION EFFECT: Probability of Selling by Price Movement")
print("-" * 90)

price_movements = np.array([0.1, 0.05, 0, -0.05, -0.10])  # +10%, +5%, flat, -5%, -10%
sell_probs = simulate_investor_decisions(
    np.array([100 * (1 + pm) for pm in price_movements]),
    initial_price=100,
    reference_price=100
)

print(f"Price Movement | Sell Probability")
for move, prob in zip(price_movements, sell_probs):
    print(f"  {move:+6.1%}       | {prob:6.1%} {'(SELL - lock in gain)' if move > 0 else '(HOLD - avoid loss)'}")

# 3. Long-term portfolio comparison
print("\n3. LONG-TERM PERFORMANCE: Rational vs Behavioral Investor")
print("-" * 90)

rational_wealth, behavioral_wealth, market_returns = portfolio_performance_comparison(
    initial_wealth=100000, years=20
)

final_rational = rational_wealth[-1]
final_behavioral = behavioral_wealth[-1]
underperformance = (1 - final_behavioral / final_rational) * 100

print(f"Initial Wealth: $100,000")
print(f"Rational Investor (consistent 60/40): ${final_rational:,.0f}")
print(f"Behavioral Investor (disposition effect): ${final_behavioral:,.0f}")
print(f"Behavioral Underperformance: {underperformance:.1f}%")
print(f"Annual drag from behavioral bias: {underperformance/20:.1f}% per year")

# 4. Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Prospect theory value function
ax = axes[0, 0]
ax.plot(outcomes, values, linewidth=3, color='darkblue')
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.fill_between(outcomes[outcomes >= 0], 0, np.array(values)[outcomes >= 0], 
                alpha=0.3, color='green', label='Gains (less steep)')
ax.fill_between(outcomes[outcomes < 0], 0, np.array(values)[outcomes < 0], 
                alpha=0.3, color='red', label='Losses (steeper = more painful)')
ax.set_xlabel('Outcome ($ relative to reference)')
ax.set_ylabel('Prospect Theory Value')
ax.set_title('S-Shaped Value Function: Loss Aversion', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Selling probability by price movement
ax = axes[0, 1]
price_moves = np.linspace(-0.30, 0.30, 100)
sell_probs_range = simulate_investor_decisions(
    np.array([100 * (1 + pm) for pm in price_moves]),
    initial_price=100,
    reference_price=100
)
ax.plot(price_moves * 100, sell_probs_range * 100, linewidth=2, color='darkblue')
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(price_moves[price_moves > 0] * 100, 0, sell_probs_range[price_moves > 0] * 100,
                alpha=0.3, color='green', label='Gains (sell probability high)')
ax.fill_between(price_moves[price_moves < 0] * 100, 0, sell_probs_range[price_moves < 0] * 100,
                alpha=0.3, color='red', label='Losses (sell probability low)')
ax.set_xlabel('Price Movement (%)')
ax.set_ylabel('Probability of Selling (%)')
ax.set_title('Disposition Effect: Behavioral Selling Pattern', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Loss aversion coefficient impact
ax = axes[0, 2]
lambda_values = np.linspace(1.0, 3.0, 50)
value_ratios = []
for lam in lambda_values:
    val_gain = 100 ** 0.88
    val_loss = lam * (100 ** 0.88)
    value_ratios.append(val_loss / val_gain)

ax.plot(lambda_values, value_ratios, linewidth=2, color='darkblue')
ax.axvline(2.25, color='red', linestyle='--', linewidth=2, label='Typical λ = 2.25')
ax.fill_between(lambda_values, 1, np.array(value_ratios), alpha=0.2)
ax.set_xlabel('Loss Aversion Coefficient (λ)')
ax.set_ylabel('Loss/Gain Value Ratio')
ax.set_title('Loss Aversion Strength Impact', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Wealth accumulation comparison
ax = axes[1, 0]
years = np.arange(len(rational_wealth))
ax.plot(years, rational_wealth / 1000, linewidth=2.5, label='Rational (60/40 rebalance)', color='green')
ax.plot(years, behavioral_wealth / 1000, linewidth=2.5, label='Behavioral (disposition effect)', color='red')
ax.fill_between(years, rational_wealth / 1000, behavioral_wealth / 1000, alpha=0.2, color='gray')
ax.set_xlabel('Years')
ax.set_ylabel('Wealth ($1000s)')
ax.set_title('Long-Term Performance: Rational vs Behavioral', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Annual returns distribution
ax = axes[1, 1]
ax.hist(market_returns * 100, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(market_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean = {market_returns.mean()*100:.1f}%')
ax.set_xlabel('Annual Return (%)')
ax.set_ylabel('Frequency')
ax.set_title('Market Return Distribution (Simulation)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 6: Cumulative wealth gap
ax = axes[1, 2]
wealth_gap = (rational_wealth - behavioral_wealth) / 1000
ax.bar(years, wealth_gap, color=['green' if gap > 0 else 'red' for gap in wealth_gap], alpha=0.7)
ax.set_xlabel('Years')
ax.set_ylabel('Wealth Gap ($1000s)')
ax.set_title('Rational vs Behavioral Wealth Difference', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('prospect_theory_loss_aversion.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved: prospect_theory_loss_aversion.png")
plt.show()

# 5. Key findings
print("\n4. KEY INSIGHTS & DECISION RULES")
print("-" * 90)
print("""
LOSS AVERSION IN ACTION:
├─ Reference Point: Investor's mental anchor (e.g., purchase price)
├─ Disposition Effect: Sell gains quickly (lock in pleasure), hold losses (avoid pain realization)
└─ Cost: Suboptimal portfolio; realized losses higher, potential gains missed

PROBABILITY DISTORTION:
├─ Overweight rare events: Assign 10% probability to 0.1% tail risk
├─ Consequence: Over-allocate to tail hedges (expensive insurance)
└─ Result: Lower expected returns from overpriced protection

BEHAVIORAL PORTFOLIO CONSTRUCTION:
├─ Safe layer: Bonds/CDs (minimize loss aversion activation)
├─ Aspiration layer: Growth stocks (for gains satisfaction)
└─ Speculation layer: Lottery-like positions (probability overweighting)

EMPIRICAL ANOMALIES EXPLAINED:
├─ Momentum (price trends persist): Recency bias + narrow framing
├─ Mean reversion: Overreaction + regret theory
├─ Calendar anomalies (January effect): Mental accounting resets
└─ Active underperformance: Overconfidence + overtrading

COMBATING BEHAVIORAL BIASES:
├─ Broad framing: Evaluate portfolio as whole, not individual positions
├─ Rebalancing rules: Mechanical (annual/quarterly) vs emotional
├─ Tax-loss harvesting: Systematize selling losses (don't fight loss aversion instinct)
├─ Diversification discipline: Avoid concentration from endowment effect
└─ Reference point management: Track benchmark, not purchase price
""")

print("=" * 90)