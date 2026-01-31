# Variance and Volatility: Measuring Outcome Spread

## 1. Concept Skeleton
**Definition:** Variance = E[(X - μ)²]; squared deviation from mean; measure of outcome uncertainty  
**Purpose:** Quantify risk/swing potential; distinguish between stable games (low variance) vs volatile (high variance)  
**Prerequisites:** Expected value, standard deviation, probability distributions

## 2. Comparative Framing
| Concept | Variance | Standard Deviation | Range | Coefficient of Variation |
|---------|----------|------------------|-------|-------------------------|
| **Definition** | Average squared deviation from mean | √Variance; same units as X | Max - Min | σ/μ; normalized risk |
| **Formula** | Var(X) = E[(X-μ)²] | σ = √Var(X) | Max outcome - Min outcome | CV = σ / \|μ\| |
| **Interpretation** | Larger = wilder swings | Risk metric (same units as outcomes) | Extreme range | Risk per unit return |
| **Example (Roulette)** | Large variance; many outcomes | σ ≈ 1 | ±$1 | Very high (negative μ) |
| **Example (Fair Coin)** | Medium variance; two outcomes | σ = 1 | ±$1 | N/A (μ=0) |
| **Application** | Bankroll sizing, probability bounds | Confidence intervals, risk assessment | Worst-case planning | Compare risky bets |

## 3. Examples + Counterexamples

**Simple Example:**  
Bet $1 on roulette (win $1 or lose $1, roughly equal). Variance high despite EV negative (large swings around -$0.027 mean)

**Failure Case:**  
Assuming variance irrelevant because EV negative: Actually, high variance increases ruin risk before LLN (bankrupt before convergence)

**Edge Case:**  
Two bets, same EV (-2%). Game A: σ=$0.50; Game B: σ=$2.00. Game B riskier (wilder swings) despite same EV. Choose based on bankroll.

## 4. Layer Breakdown
```
Variance and Volatility Framework:
├─ I. DEFINITION & CALCULATION:
│   ├─ Variance Formula:
│   │   ├─ Var(X) = E[(X - μ)²] = Σ(x_i - μ)² × P(x_i)
│   │   ├─ Alternative: Var(X) = E[X²] - (E[X])²
│   │   ├─ Properties: Always ≥ 0; units are squared
│   │   ├─ Interpretation: Average squared deviation
│   │   └─ Example: Var({1,1,-1,-1}) = (1-0)² + (1-0)² + (-1-0)² + (-1-0)² = 4
│   ├─ Standard Deviation:
│   │   ├─ σ = √Var(X)
│   │   ├─ Properties: Same units as X; easier to interpret
│   │   ├─ Interpretation: Typical deviation from mean
│   │   ├─ Example: σ({1,1,-1,-1}) = √4 = 2
│   │   └─ Rule of thumb: 68% outcomes within ±1σ (normal approx)
│   ├─ Coefficient of Variation:
│   │   ├─ CV = σ / |μ|
│   │   ├─ Purpose: Normalize risk by return (scale-free)
│   │   ├─ Application: Compare risky bets with different scales
│   │   └─ Example: σ=$10, μ=$100 (CV=0.1) vs σ=$10, μ=$1 (CV=10)
│   └─ Calculation Steps:
│       ├─ Step 1: Compute mean μ = E[X]
│       ├─ Step 2: For each outcome, compute (x_i - μ)²
│       ├─ Step 3: Weight by probability and sum
│       ├─ Step 4: Take square root for σ
│       └─ Result: Variance and std dev in comparable units
├─ II. VARIANCE IN GAMBLING:
│   ├─ Interpretation for Gamblers:
│   │   ├─ Low variance: Stable outcomes, predictable sessions
│   │   ├─ High variance: Wild swings, unpredictable results
│   │   ├─ Example: Roulette (high) vs blackjack basic strategy (med-low)
│   │   ├─ Player preference: Depends on risk tolerance
│   │   └─ Implication: Bankroll must cover variance, not just EV
│   ├─ Variance and Time Horizon:
│   │   ├─ Scaling: Var(S_n) = n × Var(X) for n independent bets
│   │   ├─ Std Dev: σ(S_n) = √n × σ(X) (grows with √n)
│   │   ├─ Example: σ(single) = $1, σ(100 bets) = $10
│   │   ├─ Implication: Longer play → larger absolute swings
│   │   └─ Interpretation: Must prepare capital for wider variance range
│   ├─ Variance and Ruin Risk:
│   │   ├─ High variance games:
│   │   │   ├─ Benefit: Higher upside swings possible
│   │   │   ├─ Risk: Bankruptcy swings also likely
│   │   │   ├─ Bankroll: Must be larger to sustain
│   │   │   └─ Example: Slots with σ=$5 require bigger cushion than blackjack
│   │   ├─ Low variance games:
│   │   │   ├─ Benefit: More predictable, less ruin risk
│   │   │   ├─ Risk: Slower capital erosion (if negative EV)
│   │   │   ├─ Strategy: Can bet higher % safely
│   │   │   └─ Example: Blackjack basic strategy smoother than roulette
│   │   └─ Risk of Ruin Formula:
│   │       ├─ R ≈ exp(-2 × |EV| × B / σ²)
│   │       ├─ Components: B = bankroll, EV = edge, σ² = variance
│   │       ├─ Implication: Higher variance → higher ruin risk
│   │       └─ Strategy: Reduce variance through lower bet size or game selection
│   └─ Variance vs Edge Trade-off:
│       ├─ Game A: EV = -2%, σ = $0.50 (low risk)
│       ├─ Game B: EV = +1%, σ = $3.00 (high risk)
│       ├─ Choice: Depends on bankroll and risk tolerance
│       ├─ Rule: Prefer +EV even if high variance (long-term wins)
│       └─ Alternative: Mix games (some stable, some volatile)
├─ III. VARIANCE IN SPECIFIC GAMES:
│   ├─ Roulette:
│   │   ├─ Each bet: Variance ≈ 1 (binary outcome)
│   │   ├─ Volatility: High (each spin ±$1)
│   │   ├─ Clustering: Streaks look non-random but are natural
│   │   └─ Implication: Large bankroll needed for long play
│   ├─ Blackjack:
│   │   ├─ Per hand: Variance ≈ 0.5-0.8 (multiple outcomes: win/push/lose)
│   │   ├─ Volatility: Medium-low (more outcomes reduce variance)
│   │   ├─ Streaks: Less dramatic than roulette
│   │   └─ Implication: Smoother results, easier bankroll planning
│   ├─ Poker:
│   │   ├─ Variance source: Multiple outcomes, all-in situations
│   │   ├─ High variance: Able to get all-in with aces, runner-runner loses
│   │   ├─ Volatility: Very high (session swings common)
│   │   ├─ Implication: Long-term play needed; short-term luck dominates
│   │   └─ Solution: Large bankroll (recommended: 30-50 buy-ins)
│   ├─ Slots:
│   │   ├─ Per spin: Very high variance (huge payouts, low frequency)
│   │   ├─ Volatility: Extreme (jackpot hits rare)
│   │   ├─ Swings: Possible to go from -$100 to +$5,000 in one spin
│   │   └─ Implication: Highly unpredictable; worst game for bankroll preservation
│   └─ Sports Betting:
│       ├─ Per bet: Variance depends on odds and outcome probability
│       ├─ Low-odds favorites: Low variance
│       ├─ High-odds underdogs: High variance
│       ├─ Parlay bets: Extreme variance (compounded)
│       └─ Implication: Diversify bets to smooth variance
├─ IV. USING VARIANCE FOR RISK MANAGEMENT:
│   ├─ Bankroll Sizing:
│   │   ├─ Rule of thumb: Bankroll ≥ k × σ × √n
│   │   ├─ k = safety factor (e.g., 3-5 for 99.7% safety)
│   │   ├─ n = number of bets in session
│   │   ├─ Example: σ=$1, n=1000 bets, k=3
│   │   ├─ Bankroll ≥ 3 × $1 × √1000 ≈ $95
│   │   └─ Reality: Multiply by bet size too (above is per $1 bet)
│   ├─ Bet Sizing:
│   │   ├─ Fixed bet: Bet same amount always (simplest)
│   │   ├─ Proportional: Bet % of bankroll (Kelly Criterion)
│   │   ├─ Conservative: Bet lower % when variance high
│   │   └─ Formula: Max bet ≤ Bankroll / (k × √(n sessions))
│   ├─ Confidence Intervals:
│   │   ├─ 68% CI: μ ± 1σ (1 standard deviation)
│   │   ├─ 95% CI: μ ± 1.96σ (2 standard deviations)
│   │   ├─ 99% CI: μ ± 3σ (3 standard deviations)
│   │   ├─ Example: EV = -$10/session, σ = $50
│   │   │   ├─ 68% CI: -$10 ± $50 = [-$60, $40]
│   │   │   ├─ 95% CI: -$10 ± $98 = [-$108, $88]
│   │   │   └─ Interpretation: 95% chance session between -$108 and +$88
│   │   └─ Application: Plan bankroll for worst-case scenario
│   └─ Hedging High Variance:
│       ├─ Strategy 1: Play multiple lower-variance games
│       ├─ Strategy 2: Make smaller bets on high-variance games
│       ├─ Strategy 3: Mix +EV and 0 EV (diversify)
│       └─ Goal: Smooth outcomes while maintaining +EV
├─ V. VOLATILITY INDEX & CLASSIFICATION:
│   ├─ Low Volatility (σ < 0.5):
│   │   ├─ Example: Blackjack basic strategy
│   │   ├─ Characteristics: Predictable, smooth results
│   │   ├─ Bankroll: Can sustain 50+ sessions
│   │   └─ Advantage: Easy planning, low ruin risk
│   ├─ Medium Volatility (0.5 ≤ σ < 1.5):
│   │   ├─ Example: Roulette, most table games
│   │   ├─ Characteristics: Some swings, reasonable predictability
│   │   ├─ Bankroll: Need 30-40 sessions funding
│   │   └─ Challenge: Runs create variance
│   ├─ High Volatility (σ ≥ 1.5):
│   │   ├─ Example: Poker, video poker, slots
│   │   ├─ Characteristics: Wild swings, long-term only
│   │   ├─ Bankroll: Need 50+ sessions minimum
│   │   └─ Risk: Bankrupt before convergence
│   └─ Extreme Volatility (σ > 3):
│       ├─ Example: Keno, progressive slots
│       ├─ Characteristics: Jackpot-based, highly unpredictable
│       ├─ Bankroll: Essentially unlimited for ruin safety
│       └─ Verdict: Avoid unless entertainment value offset
├─ VI. VARIANCE LIMITATIONS & MISCONCEPTIONS:
│   ├─ Volatility Clustering:
│   │   ├─ Observation: High variance today → high variance tomorrow
│   │   ├─ False implication: Trend will continue
│   │   ├─ Truth: Each bet independent (in fair games)
│   │   └─ Lesson: Variance is property of game, not sequence
│   ├─ Confusing Variance with Bias:
│   │   ├─ High variance ≠ biased game
│   │   ├─ Example: Fair roulette has high variance (random ±$1)
│   │   ├─ Implication: Can't detect bias from variance alone
│   │   └─ Solution: Track long-term frequency vs theory
│   ├─ Assuming Normality:
│   │   ├─ Most analysis assumes normal distribution
│   │   ├─ Reality: Some games (jackpot) are heavy-tailed
│   │   ├─ Implication: Extreme events more likely than normal predicts
│   │   └─ Solution: Use non-parametric bounds (Chebyshev)
│   └─ Ignoring Skewness:
│       ├─ Variance only captures spread, not asymmetry
│       ├─ Example: Game A: -$100 or +$50 vs Game B: -$50 or +$100
│       ├─ Both have same variance, but different distributions
│       ├─ Implication: Risk-averse prefer Game B (upside skew)
│       └─ Solution: Also examine skewness and kurtosis
├─ VII. FORMULAS & CALCULATIONS:
│   ├─ Variance:
│   │   ├─ Var(X) = E[X²] - (E[X])²
│   │   └─ Var(X) = Σ(x_i - μ)² × P(x_i)
│   ├─ Standard Deviation:
│   │   ├─ σ = √Var(X)
│   │   └─ Units same as X
│   ├─ Variance Scaling (n independent bets):
│   │   ├─ Var(S_n) = n × Var(X)
│   │   ├─ σ(S_n) = √n × σ(X)
│   │   └─ SE(mean) = σ / √n (shrinks with n)
│   ├─ Coefficient of Variation:
│   │   ├─ CV = σ / |μ|
│   │   └─ Unitless ratio for comparison
│   └─ Confidence Interval (normal approximation):
│       ├─ CI = μ ± z × σ / √n
│       ├─ z = 1.96 for 95% CI
│       └─ Example: EV=-$0.027, σ=$1, n=100: CI = -$0.027 ± 0.196
└─ VIII. PRACTICAL GUIDANCE:
    ├─ Game Selection Strategy:
    │   ├─ Prefer +EV over low variance (growth matters most)
    │   ├─ Among -EV games, choose low variance (minimize loss rate)
    │   ├─ Combine low & high variance for diversification
    │   └─ Example: 70% blackjack (stable) + 30% poker (volatile)
    ├─ Bankroll Allocation:
    │   ├─ Higher variance → smaller % of bankroll per bet
    │   ├─ Lower variance → can increase bet size
    │   ├─ Rebalance as variance estimates improve
    │   └─ Rule: Max bet = Bankroll / (Volatility Index × 2)
    ├─ Session Planning:
    │   ├─ Estimate session variance (hours × hands/hour × σ_per_hand)
    │   ├─ Set stop-loss at 2-3σ below target
    │   ├─ Set profit target at 1σ above expected (reasonable)
    │   └─ Avoid tilt by accepting variance as normal
    └─ Monitoring & Adjustment:
        ├─ Track actual variance vs theoretical
        ├─ If actual < theoretical: Game smoother than expected (good)
        ├─ If actual > theoretical: Game riskier than expected (reduce bet)
        └─ Recalibrate estimates quarterly
```

**Core Insight:** Variance measures risk; high variance requires larger bankroll but doesn't prevent profit (+EV games). Always balance EV and variance.

## 5. Mini-Project
Analyze variance across games and compute safe bet sizing:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)

print("="*70)
print("VARIANCE AND VOLATILITY: RISK ANALYSIS")
print("="*70)

# ============================================================================
# 1. COMPUTE VARIANCE FOR DIFFERENT GAMES
# ============================================================================

print("\n" + "="*70)
print("1. VARIANCE CALCULATION ACROSS GAMES")
print("="*70)

class Game:
    """Represents a gambling game with variance properties."""
    
    def __init__(self, name, outcomes, probabilities, bet_amount=1.0):
        self.name = name
        self.outcomes = np.array(outcomes)
        self.probabilities = np.array(probabilities)
        self.bet_amount = bet_amount
        
        # EV
        self.ev = np.sum(self.outcomes * self.probabilities)
        
        # Variance & Std Dev
        self.variance = np.sum((self.outcomes - self.ev)**2 * self.probabilities)
        self.std_dev = np.sqrt(self.variance)
        
        # CV
        self.cv = self.std_dev / abs(self.ev) if self.ev != 0 else np.inf
    
    def display(self):
        print(f"\n{self.name}:")
        print(f"   EV: ${self.ev:.4f}")
        print(f"   Variance: {self.variance:.4f}")
        print(f"   Std Dev (σ): ${self.std_dev:.4f}")
        print(f"   CV (Risk per $ return): {self.cv:.2f}")

# Define games
games = [
    Game("Roulette (EU, red)", [1, -1], [18/37, 19/37]),
    Game("Blackjack (basic)", [1, -1, 0.5], [0.48, 0.49, 0.03]),
    Game("Craps (pass)", [1, -1], [0.49, 0.51]),
    Game("Slots", [9, 99, 999, -1], [0.04, 0.005, 0.0001, 0.9549]),
    Game("Poker", [100, -100], [0.55, 0.45]),  # Hypothetical +EV
]

print("\nVariance Comparison:")
for game in games:
    game.display()

# ============================================================================
# 2. BANKROLL SIZING BASED ON VARIANCE
# ============================================================================

print("\n" + "="*70)
print("2. SAFE BET SIZING & BANKROLL REQUIREMENTS")
print("="*70)

def compute_bet_sizing(game, bankroll, sessions=1, hands_per_session=100, safety_factor=3):
    """
    Compute safe bet size and ruin risk.
    
    safety_factor: k where bankroll ≥ k × σ × √(n hands)
                   k=3 → 99.7% safety (3 sigma)
    """
    total_hands = sessions * hands_per_session
    
    # Variance scales as n
    total_std_dev = game.std_dev * np.sqrt(total_hands)
    
    # Safe bet = bankroll / (safety_factor × total_std_dev)
    safe_bet = bankroll / (safety_factor * total_std_dev)
    
    # Expected outcome
    expected_outcome = game.ev * total_hands * safe_bet
    
    # Simple ruin probability (approximation)
    z_score = (expected_outcome) / total_std_dev
    ruin_prob = stats.norm.sf(-z_score)  # P(X < -bankroll)
    
    return {
        'safe_bet': safe_bet,
        'expected_outcome': expected_outcome,
        'total_std_dev': total_std_dev,
        'ruin_prob': ruin_prob
    }

bankroll = 500
results = []

print(f"\nBankroll: ${bankroll}, 5 sessions × 100 hands each:")
for game in games:
    sizing = compute_bet_sizing(game, bankroll, sessions=5, hands_per_session=100)
    results.append({
        'Game': game.name,
        'Safe Bet ($)': sizing['safe_bet'],
        'Expected ($)': sizing['expected_outcome'],
        'Total σ': sizing['total_std_dev'],
        'Ruin %': sizing['ruin_prob'] * 100
    })

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# ============================================================================
# 3. CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*70)
print("3. CONFIDENCE INTERVALS FOR OUTCOMES")
print("="*70)

def compute_ci(game, n_bets, bet_amount=1.0, confidence=0.95):
    """Compute confidence interval for outcome distribution."""
    total_ev = game.ev * n_bets * bet_amount
    total_std = game.std_dev * np.sqrt(n_bets) * bet_amount
    
    z = stats.norm.ppf((1 + confidence) / 2)
    lower = total_ev - z * total_std
    upper = total_ev + z * total_std
    
    return lower, upper

n_test = 1000
bet = 1.0

print(f"\n{n_test} bets of ${bet} each, 95% confidence interval:")
for game in games:
    lower, upper = compute_ci(game, n_test, bet)
    midpoint = game.ev * n_test * bet
    print(f"\n{game.name}:")
    print(f"   Expected: ${midpoint:.2f}")
    print(f"   95% CI: [${lower:.2f}, ${upper:.2f}]")
    print(f"   Range: ${upper - lower:.2f}")

# ============================================================================
# 4. VARIANCE IMPACT ON OUTCOMES (SIMULATION)
# ============================================================================

print("\n" + "="*70)
print("4. SIMULATING VARIANCE IMPACT")
print("="*70)

def simulate_outcomes(game, n_bets, bet_amount=1.0, n_sims=1000):
    """Simulate many independent plays."""
    outcomes = []
    
    for _ in range(n_sims):
        bets = np.random.choice(game.outcomes, size=n_bets, p=game.probabilities)
        total = np.sum(bets) * bet_amount
        outcomes.append(total)
    
    return np.array(outcomes)

n_bets_test = 100
bet_amount_test = 5

print(f"\n{n_bets_test} bets of ${bet_amount_test} ({1000} simulations):")
for game in games:
    outcomes = simulate_outcomes(game, n_bets_test, bet_amount_test)
    
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    min_outcome = np.min(outcomes)
    max_outcome = np.max(outcomes)
    prob_profit = np.mean(outcomes > 0)
    
    print(f"\n{game.name}:")
    print(f"   Mean: ${mean_outcome:.2f}")
    print(f"   Std: ${std_outcome:.2f}")
    print(f"   Range: [${min_outcome:.2f}, ${max_outcome:.2f}]")
    print(f"   P(profit): {prob_profit*100:.1f}%")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Variance comparison
ax1 = axes[0, 0]
names = [g.name.split('(')[0].strip() for g in games]
stds = [g.std_dev for g in games]
evs = [g.ev for g in games]
colors = ['green' if ev > 0 else 'red' for ev in evs]

ax1.bar(names, stds, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Standard Deviation ($)')
ax1.set_title('Volatility Comparison (σ)')
ax1.grid(True, alpha=0.3, axis='y')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Safe bet sizing
ax2 = axes[0, 1]
ax2.barh(df_results['Game'], df_results['Safe Bet ($)'], color='blue', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Safe Bet Size ($)')
ax2.set_title(f'Safe Bet Sizing (Bankroll: ${bankroll})')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Distribution of outcomes (roulette vs poker)
ax3 = axes[1, 0]
roulette_outcomes = simulate_outcomes(games[0], 100, 5)
poker_outcomes = simulate_outcomes(games[4], 100, 5)

ax3.hist(roulette_outcomes, bins=30, alpha=0.6, label='Roulette', color='red', edgecolor='black')
ax3.hist(poker_outcomes, bins=30, alpha=0.6, label='Poker (+EV)', color='green', edgecolor='black')
ax3.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('Final Outcome ($)')
ax3.set_ylabel('Frequency')
ax3.set_title('Outcome Distributions (100 bets)')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: EV vs Variance (risk-return plot)
ax4 = axes[1, 1]
evs_plot = [g.ev for g in games]
stds_plot = [g.std_dev for g in games]
colors_plot = ['green' if e > 0 else 'red' for e in evs_plot]

ax4.scatter(stds_plot, evs_plot, s=300, c=colors_plot, alpha=0.7, edgecolor='black')
for name, std, ev in zip(names, stds_plot, evs_plot):
    ax4.annotate(name, (std, ev), xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Risk (Std Dev, σ)')
ax4.set_ylabel('Expected Value ($)')
ax4.set_title('Risk-Return Profile')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variance_volatility_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: variance_volatility_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Variance measures risk (spread of outcomes)")
print("✓ High variance → larger bankroll needed for same bet size")
print("✓ Safe bet size inversely proportional to √(variance)")
print("✓ Combine EV and variance for complete risk assessment")
```

## 6. Challenge Round
**When does variance analysis fail?**
- Distribution non-normal: Skewed or heavy-tailed → percentiles different from normal prediction
- Adaptive betting: Bet size changes with bankroll → variance scaling assumptions break
- Dependent outcomes: Correlated bets → variance calculations underestimate true risk
- Rare events: Extreme outcomes outside typical variance range (jackpot slots)
- Time-varying variance: Game conditions change → constant σ assumption violated

## 7. Key References
- [Khan Academy - Variance & Standard Deviation](https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data) - Foundational
- [Wikipedia - Volatility](https://en.wikipedia.org/wiki/Volatility_(finance)) - Statistical perspective
- [Wizard of Odds - Game Mathematics](https://www.wizardofodds.com/gambling/) - Specific game variances

---
**Status:** Risk quantification essential for bankroll management | **Complements:** Expected value, Risk of Ruin | **Enables:** Safe bet sizing, confidence intervals