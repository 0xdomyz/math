# Risk of Ruin: Probability of Losing All Capital

## 1. Concept Skeleton
**Definition:** Probability that a gambler's bankroll reaches zero before achieving target profit or quitting voluntarily  
**Purpose:** Quantify catastrophic risk; determine minimum bankroll for given edge, variance, and session length  
**Prerequisites:** Expected value, variance, probability theory, random walks

## 2. Comparative Framing
| Concept | Risk of Ruin | Probability of Profit | Breakeven Point | Volatility |
|---------|-------------|----------------------|-----------------|-----------|
| **Definition** | P(bankroll = 0) | P(final wealth > initial) | Sessions to EV materialize | σ of outcomes |
| **Formula** | Complex; depends on EV, σ, B | Depends on n, EV, σ | n ≈ (σ/EV)² | σ = √(n × variance) |
| **Bad Outcome** | Ruin (total loss) | Loss (less than EV predicts) | Convergence slow | High swings possible |
| **Triggers** | Too small bankroll; negative EV | Bad luck early; inadequate n | Wrong n selected; variance dominates | Undercapitalized game |
| **Application** | Bankroll planning | Session outcome prediction | Time horizon planning | Risk management |
| **Example (Roulette)** | Certain (100% with infinite play) | 2-3% per session | After millions of bets | Very high |

## 3. Examples + Counterexamples

**Simple Example:**  
$100 bankroll, $1 bets on roulette (-2.7% edge), variance high. Risk of Ruin ≈ 99%+ (will go broke)

**Failure Case:**  
Believing "positive expectation always wins eventually": True long-term, but ruin before convergence possible (finite bankroll constraint)

**Edge Case:**  
Card counter with +1% edge, small bankroll ($500). Could be +EV but ROR high if plays aggressively. Reduce bet size to lower ROR.

## 4. Layer Breakdown
```
Risk of Ruin Framework:
├─ I. FOUNDATIONAL CONCEPTS:
│   ├─ Gambler's Ruin (Classic Problem):
│   │   ├─ Setup: Start with B dollars, aim for target W
│   │   ├─ Each bet: ±$1 with prob p (win) and 1-p (lose)
│   │   ├─ Question: What's probability reach W before reach 0?
│   │   ├─ Solution: Depends on p and starting position
│   │   └─ Example: Fair game (p=0.5) → 50/50 odds regardless of B or W
│   ├─ Ruin vs Non-Ruin States:
│   │   ├─ Ruin: Bankroll reaches $0 (game over, capital lost)
│   │   ├─ Survival: Bankroll > $0 (can continue playing)
│   │   ├─ Target Reached: Achieve profit goal and quit
│   │   ├─ Voluntarily Exit: Quit before ruin or target
│   │   └─ Strategy: Set stop-loss to prevent ruin
│   ├─ Random Walk Interpretation:
│   │   ├─ Each bet: Position changes by ±(outcome)
│   │   ├─ Path: Cumulative position forms random walk
│   │   ├─ Absorption: Hitting 0 or target (terminal state)
│   │   ├─ Drift: Negative EV → walk drifts downward
│   │   └─ Question: Probability walk hits bottom (0) before escaping upward
│   └─ Martingale Property:
│       ├─ Fair game (p=0.5): Expected value = 0 (drift = 0)
│       ├─ Positive expectation (p>0.5): Drift upward
│       ├─ Negative expectation (p<0.5): Drift downward
│       ├─ Implication: EV sign determines long-term ruin risk
│       └─ Formula: Drift = 2×p - 1 (for ±$1 bets)
├─ II. RISK OF RUIN FORMULAS:
│   ├─ Classic Gambler's Ruin (Equal probabilities, $1 bets):
│   │   ├─ For p ≠ 0.5 (favorable or unfavorable):
│   │   │   ├─ Let q = 1 - p, r = q/p
│   │   │   ├─ If p > 0.5: RoR = r^B (r < 1, so decays exponentially)
│   │   │   ├─ If p = 0.5: RoR = 1 - B/W (linear in starting position)
│   │   │   ├─ If p < 0.5: RoR = 1 (certain ruin if play forever)
│   │   │   └─ Interpretation: Negative EV guarantees ruin (p<0.5)
│   │   ├─ Example (Roulette, p = 18/37 ≈ 0.486):
│   │   │   ├─ r = 19/18 ≈ 1.056
│   │   │   ├─ B = $100 bankroll
│   │   │   ├─ RoR = 1.056^100 ≈ ∞ (approximation breaks down, but RoR ≈ 99%+)
│   │   │   └─ Interpretation: Certain to go broke unless quit early
│   │   └─ Exact for negative EV:
│   │       ├─ RoR ≈ 1 - (small positive term proportional to EV)
│   │       └─ For games with house edge, assume RoR ≈ 1 (near certain)
│   ├─ Continuous Approximation (Large n, small step):
│   │   ├─ Use exponential formula: RoR ≈ exp(-2 × EV × B / σ²)
│   │   ├─ Components:
│   │   │   ├─ EV: Expected value per bet (negative for casinos)
│   │   │   ├─ B: Bankroll in dollars
│   │   │   ├─ σ²: Variance per bet
│   │   │   └─ Factor of 2: Derived from diffusion theory
│   │   ├─ Interpretation: Larger bankroll → exponentially lower RoR
│   │   ├─ Example (Blackjack card counter):
│   │   │   ├─ EV = +0.01 (1% per hand)
│   │   │   ├─ σ = 1.2 (typical for blackjack)
│   │   │   ├─ B = $5,000
│   │   │   ├─ RoR = exp(-2 × 0.01 × 5000 / 1.2²)
│   │   │   ├─ RoR = exp(-100/1.44) ≈ exp(-69.4) ≈ 0 (negligible)
│   │   │   └─ Conclusion: Very safe bet for card counter
│   │   └─ Example (Roulette player):
│   │       ├─ EV = -0.027 per $1 bet
│   │       ├─ σ ≈ 1.0
│   │       ├─ B = $100
│   │       ├─ RoR = exp(-2 × (-0.027) × 100 / 1²)
│   │       ├─ RoR = exp(5.4) ≈ 0.996 (99.6% certain ruin)
│   │       └─ Conclusion: Bankrupt almost certainly within reasonable time
│   ├─ Kelly Criterion Connection:
│   │   ├─ Kelly bet: f* = EV / σ² (per dollar bet)
│   │   ├─ Half-Kelly (conservative): f = f*/2
│   │   ├─ Quarter-Kelly: f = f*/4 (safer still)
│   │   ├─ With optimal Kelly: RoR minimized (asymptotically 0)
│   │   ├─ With half-Kelly: RoR low, growth slower
│   │   └─ Overbetting Kelly: RoR increases dramatically
│   └─ Relationship to Ruin Probability:
│       ├─ RoR and Kelly criterion related by growth rate
│       ├─ Optimal f: Maximizes log wealth → minimizes RoR
│       └─ Practical: Use Kelly (or half-Kelly for safety) to minimize RoR
├─ III. ROR BY GAME & BANKROLL:
│   ├─ Negative EV Games (Casinos):
│   │   ├─ Roulette with $100 bankroll:
│   │   │   ├─ RoR ≈ 99%+ (almost certain ruin)
│   │   │   ├─ Expected session length: ~37 hands (before bust)
│   │   │   └─ Implication: Insufficient capital for long play
│   │   ├─ Roulette with $1,000 bankroll:
│   │   │   ├─ RoR ≈ 98% (slightly safer, still bad)
│   │   │   ├─ Expected session length: ~300-400 hands
│   │   │   └─ Implication: Better, but still likely to lose all
│   │   ├─ Blackjack (basic strategy) with $500 bankroll:
│   │   │   ├─ RoR ≈ 85-90% (worse than roulette due to strategy)
│   │   │   ├─ Lower variance helps slightly
│   │   │   └─ Implication: -0.5% HE too strong to overcome
│   │   └─ Slots with $100 bankroll:
│   │       ├─ RoR ≈ 99.9%+ (extreme variance speeds ruin)
│   │       ├─ Could bust in single big loss
│   │       └─ Implication: Worst case scenario for capital preservation
│   ├─ Zero EV Games (Fair Bets):
│   │   ├─ Coin flip, $1 bet, $100 bankroll
│   │   ├─ Expected time to ruin: ~100 bets (gambler's ruin)
│   │   ├─ RoR = 50% chance before reaching $200 target
│   │   └─ Implication: Even fair game risky with finite capital
│   ├─ Positive EV Games (Advantage Play):
│   │   ├─ Card counter (+1% EV) with $500 bankroll:
│   │   │   ├─ RoR ≈ 0.4% (very safe)
│   │   │   ├─ Expected long-term profit: +5% of capital per session
│   │   │   └─ Implication: Sustainable play if avoid detection
│   │   ├─ Poker (+2 BB/100 for skilled player) with $2,000 bankroll:
│   │   │   ├─ BB = $10, RoR ≈ 0.1% (extremely safe)
│   │   │   ├─ Bankroll = 20 BB (conservative; pros use 30-50 BB)
│   │   │   └─ Implication: Skill generates safety through positive EV
│   │   └─ Sports betting (+2% EV) with $5,000 bankroll:
│   │       ├─ RoR ≈ 1-2% (low risk)
│   │       ├─ Bet sizing: Max 2-5% of bankroll per bet
│   │       └─ Implication: Model-based edge allows sustainable betting
│   └─ Scaling Effects:
│       ├─ Doubling bankroll: RoR ≈ RoR^2 (exponential drop)
│       ├─ Halving bets: RoR reduced dramatically
│       ├─ Example: $100 → $1000 (10×) reduces RoR by orders of magnitude
│       └─ Strategy: When undercapitalized, reduce bets aggressively
├─ IV. TIME HORIZON & ROR:
│   ├─ Session RoR vs Career RoR:
│   │   ├─ Single session (100 bets): RoR moderate for negative EV
│   │   ├─ Career (10,000 bets): RoR near certain for negative EV
│   │   ├─ Implication: Negative EV games ruin all players given enough play
│   │   └─ Strategy: Negative EV players must quit before convergence
│   ├─ Time Scaling:
│   │   ├─ More hands → RoR increases (for negative EV)
│   │   ├─ Formula: RoR(n sessions) ≈ RoR(1 session)^n
│   │   ├─ Example: Single session RoR = 20%, 10 sessions RoR = 20%^10 ≈ 0 (negligible)
│   │   └─ Inverse: Fewer hands → RoR decreases (quit early to survive)
│   └─ Long-term Sustainability:
│       ├─ +EV games: RoR decreases with more hands (asymptotes to 0)
│       ├─ -EV games: RoR increases with more hands (asymptotes to 1)
│       ├─ Implication: +EV games scale well; -EV games don't
│       └─ Strategy: +EV play as long as desired; -EV avoid extended play
├─ V. RISK MANAGEMENT STRATEGIES:
│   ├─ Bankroll Sizing:
│   │   ├─ Rule of thumb: Bankroll ≥ 50 × avg bet for low RoR
│   │   ├─ Example: $1 bets → $50-100 minimum bankroll
│   │   ├─ For poker: 20-50 buy-ins (standardized)
│   │   ├─ For sports: 2-5% per bet of total bankroll
│   │   └─ Principle: Larger bankroll → exponentially lower RoR
│   ├─ Bet Sizing:
│   │   ├─ Kelly Criterion: f* = EV / σ² (optimal, risky)
│   │   ├─ Half-Kelly: f = f*/2 (safer, standard)
│   │   ├─ Quarter-Kelly: f = f*/4 (very conservative)
│   │   ├─ Fixed unit: Bet same amount always (simple, suboptimal)
│   │   └─ Principle: Smaller bets → lower RoR (trade-off with growth)
│   ├─ Stop-Loss Limits:
│   │   ├─ Preset loss: Quit if down X% (usually 30-50%)
│   │   ├─ Example: 50% stop-loss → quit if lose $50 from $100 bankroll
│   │   ├─ Effect: Truncates downside, avoids ruin
│   │   ├─ Trade-off: Reduces losses but also prevents recovery
│   │   └─ Principle: Pre-commitment prevents emotional decisions
│   ├─ Profit Targets:
│   │   ├─ Set exit point when up X% (usually 30-50% gain)
│   │   ├─ Example: 50% target → quit if win $50
│   │   ├─ Effect: Lock in gains, protect from regression
│   │   ├─ Rationale: Variance will swing back eventually
│   │   └─ Principle: "Quit while you're ahead"
│   ├─ Diversification:
│   │   ├─ Mix +EV and 0 EV games
│   │   ├─ Combine high-variance and low-variance
│   │   ├─ Spread across multiple venues (reduce correlation)
│   │   └─ Principle: Reduces exposure to single-game ruin
│   └─ Bankroll Replenishment:
│       ├─ For professionals: Income from other sources replenishes bankroll
│       ├─ For casual players: Budget fixed amount (non-essential income)
│       ├─ Principle: Don't risk grocery money or rent
│       └─ Strategy: Only gambling capital that can afford to lose
├─ VI. CALCULATING ROR (PRACTICAL):
│   ├─ Using Exponential Approximation:
│   │   ├─ RoR ≈ exp(-2 × |EV| × B / σ²) for negative EV
│   │   ├─ Requires: EV, σ², B inputs
│   │   ├─ Steps:
│   │   │   ├─ 1. Compute EV per bet
│   │   │   ├─ 2. Compute σ per bet
│   │   │   ├─ 3. Plug into formula
│   │   │   ├─ 4. Take exponential
│   │   │   └─ 5. Convert to percentage
│   │   └─ Accuracy: Good for moderate values; breaks down for extreme
│   ├─ Simulation Method:
│   │   ├─ Monte Carlo: Simulate thousands of independent betting sequences
│   │   ├─ Track: Which paths hit zero (ruin)
│   │   ├─ Calculate: Fraction of paths reaching ruin
│   │   ├─ Advantage: Works for complex distributions
│   │   └─ Disadvantage: Computationally intensive
│   ├─ Iterative Calculation:
│   │   ├─ Markov chain: State = current bankroll
│   │   ├─ Transition: Probability of moving to next state
│   │   ├─ Absorption: Probability of reaching ruin state
│   │   ├─ Solve: System of linear equations
│   │   └─ Accuracy: Exact if states discretized finely
│   └─ Online Calculators:
│       ├─ Many freely available online (search "risk of ruin calculator")
│       ├─ Input: EV, bankroll, variance
│       ├─ Output: RoR percentage
│       └─ Use: Quick estimates before committing capital
├─ VII. ROR MISCONCEPTIONS:
│   ├─ "Low RoR means guaranteed profit":
│   │   ├─ False: Even 1% RoR means 1 in 100 ruin path
│   │   ├─ Truth: Need discipline; RoR just one risk metric
│   │   ├─ Example: Card counter with 0.5% RoR still needs capital discipline
│   │   └─ Lesson: Plan for possibility of ruin despite low RoR
│   ├─ "Increasing bets after losses reduces RoR":
│   │   ├─ False: Martingale systems don't change odds
│   │   ├─ Truth: Increasing bets increases RoR (larger swings)
│   │   ├─ Example: Doubling after loss → bankrupt faster on bad streaks
│   │   └─ Lesson: Fixed or Kelly-based sizing only
│   ├─ "Fair games have zero RoR":
│   │   ├─ False: Zero RoR only with infinite capital
│   │   ├─ Truth: Fair game + finite capital → finite ruin probability
│   │   ├─ Example: Coin flip, $100 bankroll → 50% chance loss $50 eventually
│   │   └─ Lesson: Finite capital always at risk even in fair games
│   └─ "RoR irrelevant for short sessions":
│       ├─ False: Short sessions can still cause ruin
│       ├─ Truth: Variance in short-term can be large enough to bankrupt
│       ├─ Example: Slots with high variance: $100 bankroll → ruin in few spins possible
│       └─ Lesson: Always consider RoR regardless of session length planned
└─ VIII. PRACTICAL DECISION-MAKING:
    ├─ Computing Minimum Bankroll:
    │   ├─ Given EV, σ, target RoR (e.g., 1%)
    │   ├─ Solve: B = -ln(RoR target) × σ² / (2 × EV)
    │   ├─ Example: EV=0.01, σ²=1, RoR_target=0.01
    │   ├─ B = -ln(0.01) × 1 / (2 × 0.01) ≈ 230
    │   └─ Interpretation: Need ~$230 for 1% RoR
    ├─ Evaluating Game Opportunities:
    │   ├─ If RoR > 50%: Avoid (high bankruptcy risk)
    │   ├─ If 10% < RoR < 50%: Risky; only if can afford loss
    │   ├─ If 1% < RoR < 10%: Moderate risk; acceptable for advantage players
    │   ├─ If RoR < 1%: Low risk; safe to play if no bigger catastrophes
    │   └─ Context: Depends on personal risk tolerance
    └─ When to Increase Bankroll:
        ├─ If running well: Grow bankroll → reduce RoR → safer future play
        ├─ If running badly: Reduce bet size (don't increase bankroll)
        ├─ Principle: Counter-intuitive but optimal strategy
        ├─ Example: Downswing → reduce bets 50%, wait for upswing
        └─ Goal: Survive to profitability
```

**Core Insight:** RoR quantifies bankruptcy probability; exponential with bankroll size. For -EV games, RoR ≈ 100% over time (accept this).

## 5. Mini-Project
Compute risk of ruin for different bankroll/bet combinations and simulate paths:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

np.random.seed(42)

print("="*70)
print("RISK OF RUIN: BANKRUPTCY PROBABILITY ANALYSIS")
print("="*70)

# ============================================================================
# 1. ROR FORMULA & CALCULATION
# ============================================================================

print("\n" + "="*70)
print("1. COMPUTING RISK OF RUIN")
print("="*70)

def compute_ror_exponential(ev, sigma2, bankroll):
    """
    Compute risk of ruin using exponential approximation.
    RoR ≈ exp(-2 × |EV| × B / σ²)
    """
    if ev >= 0:
        return 0.0  # +EV → no ruin risk (asymptotically)
    
    exponent = -2 * abs(ev) * bankroll / sigma2
    ror = np.exp(exponent)
    
    return min(ror, 1.0)  # Cap at 100%

# Games and scenarios
scenarios = [
    {'name': 'Roulette, $100 bankroll', 'ev': -1/37, 'sigma2': 1.0, 'b': 100},
    {'name': 'Roulette, $500 bankroll', 'ev': -1/37, 'sigma2': 1.0, 'b': 500},
    {'name': 'Blackjack, $200 bankroll', 'ev': -0.005, 'sigma2': 0.8, 'b': 200},
    {'name': 'Card Counter, $1000 bankroll', 'ev': 0.01, 'sigma2': 1.2, 'b': 1000},
    {'name': 'Poker, $2000 bankroll', 'ev': 0.015, 'sigma2': 2.0, 'b': 2000},
]

print("\nRisk of Ruin for Different Scenarios:")
print(f"{'Scenario':<40} {'EV':>8} {'B':>7} {'σ²':>6} {'RoR':>7}")
print("=" * 70)

ror_results = []
for scenario in scenarios:
    ror = compute_ror_exponential(scenario['ev'], scenario['sigma2'], scenario['b'])
    ror_results.append(ror)
    print(f"{scenario['name']:<40} {scenario['ev']:>8.4f} {scenario['b']:>7} {scenario['sigma2']:>6.2f} {ror*100:>6.2f}%")

# ============================================================================
# 2. BANKROLL SENSITIVITY
# ============================================================================

print("\n" + "="*70)
print("2. BANKROLL IMPACT ON ROR")
print("="*70)

# Roulette scenario: vary bankroll
bankrolls = np.array([50, 100, 200, 500, 1000, 2000, 5000])
ev_roulette = -1/37
sigma2_roulette = 1.0

rors_roulette = [compute_ror_exponential(ev_roulette, sigma2_roulette, b) for b in bankrolls]

print("\nRoulette (p=18/37): RoR vs Bankroll")
for b, ror in zip(bankrolls, rors_roulette):
    print(f"  Bankroll ${b:5d}: RoR = {ror*100:6.2f}%")

# ============================================================================
# 3. BET SIZING ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("3. BET SIZING & RISK OF RUIN")
print("="*70)

# Card counter: how bet size affects RoR
ev_counter = 0.01  # +1% per hand
sigma2_counter = 1.2
bankroll_counter = 500
bet_sizes = np.array([1, 2, 5, 10, 20])

print("\nCard Counter ($500 bankroll, +1% EV):")
print(f"{'Bet Size':>10} {'Effective EV':>15} {'Effective σ²':>15} {'RoR':>8}")
print("-" * 50)

for bet in bet_sizes:
    # Scaling: larger bets = larger effective EV and variance per bet
    eff_ev = ev_counter * bet
    eff_sigma2 = sigma2_counter * bet**2
    ror = compute_ror_exponential(eff_ev, eff_sigma2, bankroll_counter)
    print(f"${bet:>9} {eff_ev:>15.4f} {eff_sigma2:>15.2f} {ror*100:>7.2f}%")

# ============================================================================
# 4. SIMULATING ROR (MONTE CARLO)
# ============================================================================

print("\n" + "="*70)
print("4. MONTE CARLO SIMULATION: ROR VERIFICATION")
print("="*70)

def simulate_ruin(ev, sigma, initial_bankroll, n_simulations=10000, max_bets=10000):
    """
    Simulate betting sequences and track ruin events.
    """
    ruin_count = 0
    
    for _ in range(n_simulations):
        bankroll = initial_bankroll
        
        for bet_num in range(max_bets):
            # Outcome is normally distributed approximation
            outcome = np.random.normal(ev, sigma)
            bankroll += outcome
            
            if bankroll <= 0:
                ruin_count += 1
                break  # Bankrupted
    
    return ruin_count / n_simulations

# Test for a few scenarios
print("\nVerifying formula vs simulation (10,000 Monte Carlo paths):")
print(f"{'Scenario':<40} {'Formula':>10} {'Simulation':>12} {'Diff':>8}")
print("=" * 70)

for scenario in scenarios[:3]:  # Test first 3
    formula_ror = compute_ror_exponential(scenario['ev'], scenario['sigma2'], scenario['b'])
    simulated_ror = simulate_ruin(scenario['ev'], np.sqrt(scenario['sigma2']), scenario['b'], n_simulations=1000)
    diff = abs(formula_ror - simulated_ror)
    print(f"{scenario['name']:<40} {formula_ror*100:>9.2f}% {simulated_ror*100:>11.2f}% {diff*100:>7.2f}%")

# ============================================================================
# 5. MINIMUM BANKROLL CALCULATION
# ============================================================================

print("\n" + "="*70)
print("5. MINIMUM BANKROLL FOR TARGET ROR")
print("="*70)

def compute_min_bankroll(ev, sigma2, target_ror):
    """Solve for B given target RoR."""
    if ev >= 0:
        return 0  # +EV doesn't need bankroll for RoR
    
    b = -np.log(target_ror) * sigma2 / (2 * abs(ev))
    return b

# For different games, compute bankroll needed for 1% RoR
target_ror = 0.01  # 1%

print(f"\nBankroll Required for {target_ror*100:.1f}% Risk of Ruin:")
print(f"{'Game':<30} {'EV per hand':>15} {'σ² per hand':>15} {'Min Bankroll':>15}")
print("=" * 75)

games_minb = [
    ('Roulette', -1/37, 1.0),
    ('Blackjack (basic)', -0.005, 0.8),
    ('Card Counter', 0.01, 1.2),
    ('Poker Pro', 0.02, 2.0),
]

for game_name, ev, sigma2 in games_minb:
    if ev < 0:
        min_b = compute_min_bankroll(ev, sigma2, target_ror)
        print(f"{game_name:<30} {ev:>15.5f} {sigma2:>15.2f} ${min_b:>14.0f}")
    else:
        # +EV: compute for illustrative RoR (say, 10% over different horizon)
        min_b = compute_min_bankroll(ev, sigma2, 0.01)
        print(f"{game_name:<30} {ev:>15.5f} {sigma2:>15.2f} ${min_b:>14.0f} (for 1% RoR)")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bankroll vs RoR (roulette)
ax1 = axes[0, 0]
bankrolls_plot = np.logspace(1.5, 4, 50)  # 30 to 10,000
rors_plot = [compute_ror_exponential(ev_roulette, sigma2_roulette, b) for b in bankrolls_plot]

ax1.semilogy(bankrolls_plot, np.array(rors_plot)*100, linewidth=2, color='red')
ax1.fill_between(bankrolls_plot, np.array(rors_plot)*100, 100, alpha=0.3, color='red')
ax1.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='1% RoR threshold')
ax1.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='10% RoR threshold')
ax1.set_xlabel('Bankroll ($, log scale)')
ax1.set_ylabel('Risk of Ruin (%)')
ax1.set_title('Roulette: Bankroll Impact on RoR')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend()
ax1.set_ylim(0.01, 100)

# Plot 2: Scenario comparison
ax2 = axes[0, 1]
scenario_names = [s['name'].split(',')[0] for s in scenarios]
colors = ['red' if r > 0.5 else 'orange' if r > 0.1 else 'yellow' if r > 0.01 else 'green' for r in ror_results]
bars = ax2.bar(range(len(scenario_names)), np.array(ror_results)*100, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Risk of Ruin (%)')
ax2.set_title('RoR Across Different Games')
ax2.set_xticks(range(len(scenario_names)))
ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y', which='both')

# Add value labels
for bar, ror in zip(bars, ror_results):
    height = bar.get_height()
    if ror > 0.001:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ror*100:.2f}%', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., 0.1,
                f'{ror*100:.4f}%', ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

# Plot 3: Bet sizing impact (card counter)
ax3 = axes[1, 0]
rors_bet = []
for bet in bet_sizes:
    eff_ev = ev_counter * bet
    eff_sigma2 = sigma2_counter * bet**2
    ror = compute_ror_exponential(eff_ev, eff_sigma2, bankroll_counter)
    rors_bet.append(ror)

ax3.plot(bet_sizes, np.array(rors_bet)*100, marker='o', linewidth=2, markersize=10, color='blue')
ax3.fill_between(bet_sizes, 0, np.array(rors_bet)*100, alpha=0.3, color='blue')
ax3.set_xlabel('Bet Size ($)')
ax3.set_ylabel('Risk of Ruin (%)')
ax3.set_title('Card Counter: Bet Size Impact on RoR')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Simulated ruin paths
ax4 = axes[1, 1]
# Roulette: simulate a few paths
n_paths = 20
n_bets = 500
for path_num in range(n_paths):
    bankroll = 100
    path = [bankroll]
    
    for _ in range(n_bets):
        outcome = np.random.choice([1, -1], p=[18/37, 19/37])
        bankroll += outcome
        path.append(bankroll)
        
        if bankroll <= 0:
            break
    
    color = 'red' if bankroll <= 0 else 'green'
    ax4.plot(path, alpha=0.6, color=color)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax4.set_xlabel('Bet Number')
ax4.set_ylabel('Bankroll ($)')
ax4.set_title('Roulette: Sample Bankruptcy Paths (red=ruined, green=survived)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('risk_of_ruin_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: risk_of_ruin_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Risk of Ruin exponentially related to bankroll size")
print("✓ Larger bankroll → dramatically lower RoR")
print("✓ Negative EV games → RoR approaches 100% with extended play")
print("✓ +EV games → RoR decreases with more capital and time")
```

## 6. Challenge Round
**When do RoR calculations mislead?**
- Assumption violations: Non-independent bets or changing distributions → RoR formula breaks
- Extreme events: Heavy-tailed distributions (jackpots) → actual ruin faster than predicted
- Adaptive play: Changing bet size based on results → invalidates constant EV/σ assumptions
- Quitting decision: Pre-set stop-loss truncates distribution → formula underestimates true RoR
- Sequence dependency: Correlated outcomes → RoR higher than independent assumption predicts

## 7. Key References
- [Wikipedia - Gambler's Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin) - Mathematical foundation
- [Blackwell & Girshick (1954), "Theory of Games and Statistical Decisions"](https://www.amazon.com/Theory-Games-Statistical-Decisions-Blackwell/dp/0486639170) - Theoretical treatment
- [Thorp (1962), "Beat the Dealer"](https://en.wikipedia.org/wiki/Edward_Thorp) - Applied ROR to advantage play

---
**Status:** Critical for bankroll management; determines minimum capital needed | **Complements:** Expected value, Variance, Kelly Criterion | **Enables:** Safe play decisions