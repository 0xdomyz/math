# Prospect Theory & Loss Aversion in Portfolio Decisions

## 1. Concept Skeleton
**Definition:** Behavioral theory of decision-making under risk where investors exhibit non-rational preferences: overweight losses relative to gains (loss aversion), evaluate outcomes relative to reference points (mental accounting), and distort probabilities (overweight low probabilities)  
**Purpose:** Explain why investors deviate from rational MPT predictions (under-diversify, hold losers too long, chase winners), quantify behavioral biases in portfolio construction  
**Prerequisites:** Rational choice theory, utility theory, expected value framework, behavioral economics

---

## 2. Comparative Framing

| Aspect | Rational Expected Utility Theory | Prospect Theory (Behavioral) |
|--------|----------------------------------|------------------------------|
| **Reference Point** | Absolute wealth level | Relative change from current position (mental reference) |
| **Gain/Loss Evaluation** | Linear value function; gains and losses symmetric | S-shaped value function; losses valued 2-3x more painful than equivalent gains |
| **Loss Aversion Coefficient** | λ = 1 (neutral) | λ ≈ 2.25 (loss is 2.25× worse than equivalent gain) |
| **Probability Weighting** | Uses true probabilities | Overweights low probabilities, underweights high probabilities |
| **Example: Gamble Preference** | Coin flip: win $100 or lose $100 → indifferent (EV = 0) | Coin flip: win $100 or lose $100 → REJECT (loss pain > gain pleasure) |
| **Portfolio Implication** | Optimal diversification (rational 60/40) | Home bias, concentrated holdings (irrational emotional attachment) |
| **Risk Tolerance** | Stable, time-independent | Depends on recent performance (win → risk-seeking; loss → risk-averse) |
| **Stop-Loss Behavior** | Sell when fair value changes; emotional neutral | Sell winners (lock in gains), hold losers (avoid crystallizing loss) |
| **Market Anomaly Prediction** | No anomalies; markets efficient | Predicts momentum, mean reversion, seasonal effects from behavioral distortions |

**Key Insight:** Prospect theory + loss aversion explains (1) equity risk premium puzzle (why stocks outperform when rational models say less), (2) home bias (familiarity reduces perceived risk), (3) performance chasing (recency bias), (4) active trading underperformance (behavioral overconfidence).

---

## 3. Examples + Counterexamples

**Example 1: Loss Aversion in Portfolio Rebalancing**
- Portfolio: 60% stocks (currently worth $600k), 40% bonds ($400k) = $1M
- Market rally: Stocks → $650k, bonds → $400k = $1.05M
- Rational action: Rebalance to 60/40 → Sell $30k stocks, buy $30k bonds (trim winners)
- Prospect theory prediction: Investor REFUSES to sell winners (gains feel good; selling feels like losing gain)
- Result: Portfolio drifts to 62% stocks (risk creep); investor becomes increasingly exposed to crash
- When crash comes: Stocks → $400k, bonds → $390k = $790k
- Loss aversion kicks in: "I can't sell at 62% loss! I'll wait for recovery."
- Outcome: Held through drawdown; rational 60/40 investor had exited at $700k

**Example 2: Disposition Effect (Selling Winners, Holding Losers)**
- Investor buys 100 shares of Stock A at $100 = $10k
- Stock rallies to $150 (unrealized gain +$5k) → Investor sells immediately (lock in pleasure)
- Investor buys 100 shares of Stock B at $100 = $10k
- Stock falls to $50 (unrealized loss -$5k) → Investor holds, hoping for recovery
- Rational behavior: Sell B if fundamentals don't support it; hold A if superior (symmetric)
- Empirical evidence: Investors realize gains 50% more frequently than losses (Odean 1998)
- Tax consequence: Suboptimal; missing tax-loss harvesting benefits; selling winners = higher taxes
- Implication: Behavioral bias costs ~1-2% annually in taxes alone

**Example 3: Overweighting Small Probability Events (Lottery Ticket Bias)**
- Investor obsesses over 1% probability EM sovereign default (despite 99% success)
- Reduces EM allocation from 20% to 5% "just to be safe"
- Meanwhile ignores 50% probability of mediocre stock picking (behavioral overconfidence)
- Result: Under-diversified into home country; takes massive unrecognized idiosyncratic risk
- Actual outcome: Home country hits recession; regrets not holding 20% EM buffer

**Example 4: Reference Point Dependence (Anchoring)**
- Investor buys stock at $100 (mental anchor)
- Stock falls to $60 (feels like 40% loss relative to anchor)
- Rationally: Only $60 matters; previous price irrelevant (sunk cost)
- Behaviorally: Investor experiences $40 "loss" relative to mental anchor
- When stock recovers to $90: "Almost broke even!" (feels like gain relative to $60, not a loss relative to $100)
- Consequence: Sells at $90 due to false sense of recovery (sold too early vs fundamental value)

**Counterexample: Overconfidence & Excessive Trading**
- Rational prediction: Investors understand they can't beat market; minimize trading (pass-through costs)
- Prospect theory + overconfidence: Investors feel losses acutely → overestimate ability to recover losses via stock picking
- Result: Actively trade to "make back losses" → incur transaction costs → create further losses
- Empirical: Frequent traders underperform by 6-8% annually (Barber & Odean 2000)
- Paradox: Loss aversion (avoid realizing losses) + overconfidence (believe can recover) create harmful combination

**Edge Case: Narrow Framing (Mental Accounting)**
- Investor has stock portfolio ($500k) + bond portfolio ($500k)
- Mentally accounts separately: "Stocks for growth, bonds for stability"
- Stock portfolio: -10% ($50k loss); bonds: +3% ($15k gain); net: $965k
- Behavioral response: Anguish over $50k stock loss; joy at $15k bond gain
- True perspective: Aggregate -$35k loss (2.5% on total wealth)
- Consequence: Sells stocks to "stop the bleeding" (narrow frame); realizes 2.5% loss at worst moment (regret)

---

## 4. Layer Breakdown

```
Prospect Theory & Loss Aversion Architecture:

├─ S-Shaped Value Function (Core of Prospect Theory):
│   ├─ Concave region (gains, x > 0):
│   │   └─ v(x) = x^α, where α ≈ 0.88 (diminishing sensitivity)
│   │   └─ Implication: $1 gain worth less than $1 loss
│   │       Example: +$100 feels good, but -$100 feels terrible (not symmetric)
│   │
│   ├─ Convex region (losses, x < 0):
│   │   └─ v(x) = -λ|x|^β, where β ≈ 0.88, λ ≈ 2.25
│   │   └─ Loss aversion: λ = 2.25 means losses valued 2.25× more than gains
│   │
│   ├─ Kink at reference point (x = 0):
│   │   └─ Steeper slope for losses than gains (discontinuity in derivative)
│   │   └─ Explains why investors reluctant to realize losses (steep pain curve)
│   │
│   └─ Mathematical formulation:
│       v(x) = x^α if x ≥ 0 (gains)
│       v(x) = -λ(−x)^β if x < 0 (losses, λ ≈ 2.25)
│
├─ Reference Points & Mental Accounting:
│   ├─ Mental reference points (not rational wealth):
│   │   ├─ Purchase price (anchoring bias): "Bought at $100" → reference
│   │   ├─ Previous portfolio value: "Was worth $1M" → reference
│   │   ├─ Expected portfolio value: "Should be at $1.1M" → reference
│   │   └─ Market benchmark: "S&P 500 up 10%, I'm only up 5%" → reference
│   │
│   ├─ Narrow vs Broad Framing:
│   │   ├─ Narrow frame: Evaluate each position independently
│   │   │   └─ Stock A: -$20k (loss) → feel pain
│   │   │   └─ Stock B: +$15k (gain) → feel pleasure
│   │   │   └─ Aggregate: Dismiss gain when evaluating loss (narrow attention)
│   │   │
│   │   ├─ Broad frame: Evaluate portfolio as whole
│   │   │   └─ Portfolio: -$5k (small loss) → manageable pain
│   │   │   └─ More rational; reduces emotional volatility
│   │   │
│   │   └─ Consequence: Narrow framers sell winners (A) to offset loss (avoid broad loss sensation)
│   │       Broad framers hold both and rebalance rationally
│   │
│   └─ Regret Theory Extension:
│       ├─ Regret: Pain of action vs counterfactual inaction
│       ├─ "If only I sold when price was $100..." → Regret amplified
│       ├─ Leads to inaction (regret aversion): Hold losing positions (avoid regret of realized loss)
│       └─ Consequence: Procrastination on portfolio rebalancing; home bias persists
│
├─ Probability Weighting in Portfolio Decisions:
│   ├─ Rational weighting: Use true probabilities
│   │   └─ 90% chance +5% return: E[R] = 0.9×5% + 0.1×(-10%) = 3.5%
│   │
│   ├─ Behavioral weighting: π(p) function (not = p)
│   │   ├─ Overweight low probabilities: π(0.01) ≈ 0.06 (6× overweight)
│   │   ├─ Underweight high probabilities: π(0.99) ≈ 0.95 (underweight)
│   │   ├─ Peak around p = 0.33 (overweighting moderate probabilities)
│   │   └─ Result: Overestimate tail risks (rare events), underestimate normal risks
│   │
│   ├─ Portfolio implications:
│   │   ├─ Overweight lottery tickets (1% chance 1000% gain) → underdiversify
│   │   ├─ Overweight catastrophe insurance → overpay for tail hedges
│   │   ├─ Underestimate normal volatility → insufficient risk management for 50/50 scenarios
│   │   └─ Example: Buy gold (tail hedge) at high cost while under-allocating to stocks (true expected return)
│   │
│   └─ Calibration: Better at extreme probabilities than moderate (U-shaped error)
│
├─ Implications for Portfolio Construction:
│   ├─ Home Bias Puzzle:
│   │   ├─ Rational: Global diversification reduces variance
│   │   ├─ Behavioral: Familiarity → lower perceived risk (overconfidence in home market knowledge)
│   │   ├─ Result: US investors hold 90% domestic stocks (should be 35-40% per CAPM)
│   │   └─ Cost: ~1% annually from suboptimal diversification
│   │
│   ├─ Endowment Effect (ownership bias):
│   │   ├─ Investors overvalue existing holdings (just because they own them)
│   │   ├─ Reluctant to sell; prefer status quo
│   │   └─ Consequence: Portfolio becomes stale; unable to rebalance when warranted
│   │
│   ├─ Disposition Effect Chain:
│   │   ├─ Buy position at anchor price
│   │   ├─ Gain phase: Sell too early (lock in pleasure)
│   │   ├─ Loss phase: Hold too long (avoid pain of realization)
│   │   ├─ Result: Selling winners too soon, holding losers too long
│   │   └─ Performance: Suboptimal; misses upside in held losers, holds drawdown risk
│   │
│   └─ Performance Chasing (Recency Bias):
│       ├─ Recent winner bias: Best recent performance → allocate to it
│       ├─ Ignores mean reversion: Past winner likely to regress
│       └─ Result: Buy high, sell low → contra-performance
│
└─ Behavioral Portfolio Theory (Shefrin-Statman):
    ├─ Mental wealth layers:
    │   ├─ Layer 1 (Safety): Current liabilities + emergency fund
    │   ├─ Layer 2 (Aspirations): Growth targets
    │   └─ Layer 3 (Retirement): Long-term wealth
    │
    ├─ Portfolio constructed by adding:
    │   ├─ Safe layer: Bonds, CDs, insurance (minimize loss aversion pain)
    │   ├─ Aspiration layer: Stocks, alternatives (utility from gains)
    │   └─ Lottery layer: Small high-risk positions (prospect theory overweighting of tails)
    │
    ├─ Result: Explains why investors hold diversified portfolio (rational core)
    │   + concentrated bets (behavioral fringes) + insurance tails
    │
    └─ Implication: Can't be dismissed as pure irrational; layers serve psychological needs
```

**Mathematical Formulas:**

Standard prospect theory value function:
$$v(x) = \begin{cases} x^\alpha & \text{if } x \geq 0 \\ -\lambda(-x)^\beta & \text{if } x < 0 \end{cases}$$

With typical parameters α ≈ 0.88, β ≈ 0.88, λ ≈ 2.25

Loss aversion coefficient:
$$\lambda = \frac{|v(-\$100)|}{v(+\$100)} = \frac{100^{0.88} \times 2.25}{100^{0.88}} = 2.25$$

Probability weighting function:
$$\pi(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}$$

With typical γ ≈ 0.61 for gains, 0.69 for losses

---

## 5. Mini-Project: Simulating Loss Aversion in Portfolio Decisions

```python
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
```

---

## 6. Challenge Round

1. **Loss Aversion Quantification:** Given λ = 2.25 (loss aversion coefficient), how should an investor adjust portfolio risk compared to rational theory? If rational allocation suggests 70% stocks, should behavioral investor reduce to 60%? 50%? How much does loss aversion justify underallocation?

2. **Narrow vs Broad Framing Trade-off:** When does narrow framing become beneficial (helps avoid overtrading) vs harmful (promotes disposition effect)? Design a rebalancing framework that uses narrow framing strategically.

3. **Reference Point Manipulation:** If an investor mentally anchors to purchase price ($100 = $50 loss, feels bad) vs market price ($75 = reference, feels neutral), how would you re-anchor them? Is it unethical to manipulate reference points for better decisions?

4. **Probability Weighting Mismatch:** Suppose rational allocation suggests 5% to tail hedge (legitimate 0.5% annual loss scenario). Behavioral investor overweights tail (assigns 5% probability to 0.1% event). What should they allocate? Is overhedging always wrong?

5. **Behavioral Portfolio Theory Design:** Using Shefrin-Statman layers, design a portfolio for a retiree (needs safety) vs young professional (can afford risk). How would you allocate between safe/aspiration/speculation layers?

---

## 7. Key References

- **Kahneman, D. & Tversky, A. (1979).** "Prospect Theory: An Analysis of Decision under Risk" – Foundational work establishing S-shaped value function and probability weighting; Nobel Prize paper.

- **Odean, T. (1998).** "Are Investors Reluctant to Realize Their Losses?" – Empirical evidence that investors realize gains 50% more than losses (disposition effect); documents tax consequences.

- **Barber, B.M. & Odean, T. (2000).** "Trading is Hazardous to Your Wealth" – Shows active traders underperform by 6.5% annually due to overconfidence and overtrading from loss aversion.

- **Shefrin, H. & Statman, M. (2000).** "Behavioral Portfolio Theory" – Multi-layer mental accounting model explaining why investors hold both safe and risky assets.

- **Thaler, R.H. (1985).** "Mental Accounting and Consumer Choice" – Narrow vs broad framing implications for portfolio rebalancing decisions.

- **Barberis, N., Huang, M., & Santos, T. (2001).** "Prospect Theory and Asset Prices" – Theoretical model linking prospect theory directly to market anomalies and return predictability.

- **Academicpedia: Behavioral Finance** – https://www.investopedia.com/behavioral-finance-4689749 – Accessible behavioral economics overview.

