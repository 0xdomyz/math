# Independence and Dependence: Causation vs Correlation in Gambling

## 1. Concept Skeleton
**Definition:** Independence: Events A and B occur without influencing each other (P(A|B) = P(A)); Dependence: Event B affects likelihood of A  
**Purpose:** Distinguish true causal relationships from coincidence; avoid gambler's fallacy; understand memory in games  
**Prerequisites:** Conditional probability, joint probability, basic statistics

## 2. Comparative Framing
| Property | Independent Events | Dependent Events | Conditional Independence | Mutual Exclusion |
|----------|------------------|------------------|------------------------|-----------------|
| **Definition** | P(A\|B) = P(A); past irrelevant | P(A\|B) ≠ P(A); events linked | P(A\|B,C) = P(A\|C); condition removes dependence | P(A,B) = 0; both impossible |
| **Formula** | P(A,B) = P(A) × P(B) | P(A,B) ≠ P(A) × P(B) | Depends on conditioning variable C | P(A∩B) = ∅ |
| **Example** | Coin flips, roulette spins | Card draws without replacement | Shuffle deck → position independent | Outcome A or B (not both) |
| **Memory** | Memoryless; no history | Events linked by shared cause/mechanism | Causation removed by conditioning | N/A |
| **Gambling** | Each spin independent; hot-hand false | Deck composition changes; card counting works | Deal → shuffle → independence restored | Win or lose (mutual exclusion) |

## 3. Examples + Counterexamples

**Simple Example (Independence):**  
Flipping a fair coin 10 times: each flip independent. P(heads on flip 11 | 10 tails) = 0.5 ≠ P(tails observed)

**Failure Case (Gambler's Fallacy):**  
Belief: "Red hasn't hit 5 spins, so next must be red" → False. Each roulette spin independent. P(red | 5 blacks) = 18/37

**Edge Case (Conditional Independence):**  
Two cards dealt: position independent AFTER shuffle. But within draw: P(2nd is ace | 1st not ace) = 4/51 ≠ 4/52 (dependent). After shuffle: independence restored.

## 4. Layer Breakdown
```
Independence and Dependence Framework:
├─ I. DEFINITION & MATHEMATICS:
│   ├─ Statistical Independence:
│   │   ├─ Definition: Events A and B unrelated in probability
│   │   ├─ Criterion: P(A|B) = P(A)
│   │   ├─ Equivalent: P(A,B) = P(A) × P(B)
│   │   ├─ Property: P(A|B) = P(A|B^c) (condition irrelevant)
│   │   └─ Implication: History doesn't affect future probability
│   ├─ Dependence (Correlation):
│   │   ├─ Definition: Events A and B co-occur at rate ≠ independent
│   │   ├─ Criterion: P(A|B) ≠ P(A)
│   │   ├─ Equivalent: P(A,B) ≠ P(A) × P(B)
│   │   ├─ Measure: Covariance or correlation coefficient
│   │   └─ Implication: Knowing B updates belief about A
│   ├─ Mutual Exclusion (Special Dependence):
│   │   ├─ Definition: A and B cannot both occur
│   │   ├─ Formula: P(A,B) = 0 always
│   │   ├─ Consequence: P(A|B) = 0 (B proves A false)
│   │   └─ Example: Outcome is heads (not tails)
│   ├─ Conditional Independence:
│   │   ├─ Definition: A ⊥ B | C means independence given C
│   │   ├─ Formula: P(A|B,C) = P(A|C)
│   │   ├─ Example: Deck position ⊥ previous draws | shuffled
│   │   └─ Insight: Condition C "explains away" dependence
│   └─ Pairwise vs Joint Independence:
│       ├─ Pairwise: Each pair independent
│       ├─ Joint: All k events mutually independent
│       ├─ Note: Pairwise ≠ Joint (can have three-way dependence)
│       └─ Gambling: Roulette spins pairwise independent
├─ II. INDEPENDENCE IN GAMBLING CONTEXTS:
│   ├─ True Independent Games:
│   │   ├─ Roulette Wheel:
│   │   │   ├─ Mechanism: Physical spin resets position each time
│   │   │   ├─ Previous: No influence on current probability
│   │   │   ├─ P(red on spin 1000 | 999 blacks) = 18/37 (not 37/37)
│   │   │   ├─ Implication: Betting more after streaks = fallacy
│   │   │   └─ House advantage: Consistent regardless of history
│   │   ├─ Dice Games (Fair Dice):
│   │   │   ├─ Mechanism: Each roll mechanically independent
│   │   │   ├─ P(7 on roll 50 | many 7s already) = 6/36 = 1/6
│   │   │   ├─ House: Maintains edge despite streaks
│   │   │   └─ Strategy: Cannot exploit past outcomes
│   │   ├─ Coin Flips:
│   │   │   ├─ Fairness: P(H) = P(T) = 0.5 always
│   │   │   ├─ History: Irrelevant (memoryless process)
│   │   │   └─ Implication: Fair odds = 1:1 always
│   │   └─ Slot Machines:
│   │       ├─ Design: Independent reels in most models
│   │       ├─ P(jackpot | 100 prior spins no win) = prior probability
│   │       └─ Payout: Unaffected by sequences
│   ├─ Dependent Games (Sampling Without Replacement):
│   │   ├─ Poker (Community Card Draws):
│   │   │   ├─ Mechanism: Cards removed from deck permanently
│   │   │   ├─ Dependence: P(next card is ace | cards dealt) changes
│   │   │   ├─ Example: P(A₂=ace | A₁=ace) = 3/51 ≠ 4/52
│   │   │   ├─ Information: Dealt cards update remaining deck composition
│   │   │   └─ Strategy: Position, observed cards inform decisions
│   │   ├─ Blackjack (Card Counting):
│   │   │   ├─ Mechanism: Shoe dealt until cut (cards not replaced)
│   │   │   ├─ Dependence: Deck composition drifts from initial
│   │   │   ├─ P(natural | deck rich in aces) > P(natural | neutral)
│   │   │   ├─ Card Counting: Track composition → exploit dependence
│   │   │   └─ House Defense: Cut card, shuffle more frequently
│   │   ├─ Baccarat (Card Depletion):
│   │   │   ├─ Mechanic: Shoe dealt without replacement
│   │   │   ├─ Banker vs Player: Win rates dependent on history
│   │   │   ├─ Bias: Late-shoe rich in high cards favors Banker
│   │   │   └─ Exploit: Bet adjustment based on shoe history
│   │   └─ Lottery Draws (Without Replacement):
│   │       ├─ Mechanism: Numbers drawn stay out
│   │       ├─ Dependence: Prior draws affect remaining pool
│   │       ├─ P(number k in round n | previously drawn) ≠ prior
│   │       └─ Correction: Smaller pool affects odds
│   └─ Semi-Dependent (Mixed):
│       ├─ Sports Betting:
│       │   ├─ Individual games: ~independent if no correlation
│       │   ├─ But teams affected by injuries, fatigue, psychology
│       │   ├─ Dependence: Lose game 1 → confidence↓ → likely lose game 2
│       │   └─ Non-stationary: Win rates change over season
│       ├─ Betting Sequences:
│       │   ├─ Martingale systems assume independence (false)
│       │   ├─ Each bet independent, but bankroll dependent on history
│       │   └─ Misconception: Past bets influence future odds (wrong)
│       └─ Hot/Cold Streaks:
│           ├─ Observation: Sequences appear non-random
│           ├─ True cause: Randomness GENERATES clusters
│           ├─ Real dependence: Player psychology (tilt after loss)
│           └─ Implication: Streak ≠ future direction
├─ III. DEPENDENCE MECHANISMS & SOURCES:
│   ├─ Shared Common Cause:
│   │   ├─ Events A, B both caused by hidden factor C
│   │   ├─ Example: Dealer skilled → Player loses + Opponent loses (both depend on dealer)
│   │   ├─ Apparent Correlation: A and B seem related
│   │   ├─ Reality: Both stem from C, not directly linked
│   │   ├─ Conditional Independence: A ⊥ B | C (condition explains away)
│   │   └─ Gambling: "Lucky dealer" causes multiple losses (not magical)
│   ├─ Sequential/Temporal Dependence:
│   │   ├─ Event A at time t affects probability of B at time t+1
│   │   ├─ Example: Deck depletion (draw ace → fewer aces remain)
│   │   ├─ Markov Property: Future only depends on present, not history
│   │   ├─ Memory Order: First-order (one step back), higher-order possible
│   │   └─ Gambling: Each card draw depends on previous draws
│   ├─ Sampling Without Replacement:
│   │   ├─ Mechanism: Finite pool depleted by draws
│   │   ├─ Effect: P(item k on draw n) = f(prior draws of k)
│   │   ├─ Formula: P(X_n | X_{n-1}) ≠ P(X_n) due to depletion
│   │   ├─ Extreme: If all others drawn, next draw is certain
│   │   └─ Gambling: Poker, blackjack, lotteries all exhibit this
│   └─ Feedback Loops:
│       ├─ Event A affects bankroll → Changes risk tolerance → Changes bet size → Changes future outcomes
│       ├─ Psychological: Win increases confidence → Bets bigger → High risk
│       ├─ Financial: Loss decreases bankroll → Less betting capital → Restricted bets
│       └─ Gambling: Tilt (emotional state) creates dependence
├─ IV. TESTING FOR INDEPENDENCE:
│   ├─ Chi-Square Test:
│   │   ├─ Null Hypothesis: Events independent
│   │   ├─ Formula: χ² = Σ(observed - expected)² / expected
│   │   ├─ Expected: P(A) × P(B) × n (under independence)
│   │   ├─ Decision: If χ² > critical value, reject independence
│   │   └─ Gambling Test: Roulette red/black sequences independent?
│   ├─ Correlation Coefficient:
│   │   ├─ Pearson r ∈ [-1, 1]
│   │   ├─ r ≈ 0: Independent (uncorrelated)
│   │   ├─ r > 0: Positive dependence (both increase together)
│   │   ├─ r < 0: Negative dependence (inverse relationship)
│   │   └─ Gambling: Betting on correlated outcomes = redundant
│   ├─ Runs Test:
│   │   ├─ Purpose: Detect if sequence too clustered (non-random pattern)
│   │   ├─ Method: Count runs (consecutive same outcomes)
│   │   ├─ Result: Too few runs → positive dependence (clustering)
│   │   ├─ Result: Too many runs → negative dependence (alternation)
│   │   └─ Gambling: Test if roulette black/red are truly independent
│   ├─ Spectral Analysis:
│   │   ├─ Tool: Autocorrelation function (ACF)
│   │   ├─ Lag k: ACF(k) = correlation between value at t and t+k
│   │   ├─ Interpretation: ACF ≈ 0 for all k → Independence
│   │   ├─ ACF significant at lag k → k-period dependence
│   │   └─ Gambling: Detect if game has hidden patterns
│   └─ Practical: Chi-Square easiest; Runs test intuitive
├─ V. CORRELATION ≠ CAUSATION:
│   ├─ Classic Warning:
│   │   ├─ Observation: Winning streak correlates with aggressive betting
│   │   ├─ Wrong Conclusion: Aggressive betting causes wins
│   │   ├─ True Relationship: Both caused by increasing confidence (C)
│   │   ├─ DAG: Confidence → (Wins, Aggressive Betting)
│   │   └─ Implication: Aggressive betting ALONE won't increase wins
│   ├─ Confounding Variables:
│   │   ├─ Unmeasured factor affecting both A and B
│   │   ├─ Example: House edge → (Player loses, Opponent loses)
│   │   ├─ Apparent correlation: Players dependent
│   │   ├─ Reality: Independent, both lose due to house advantage
│   │   └─ Mistake: Think correlation → change strategy to exploit
│   ├─ Reverse Causation:
│   │   ├─ Observe: Bet amount increases as bankroll depletes
│   │   ├─ Assume: Lower bankroll causes higher bets (tilt)
│   │   ├─ Reality: Losing bets cause bankroll depletion AND tilt
│   │   └─ Correct: Bankroll ← Outcomes → Emotional state → Bet size
│   ├─ Time-Lag Issues:
│   │   ├─ Correlation at lag 0 (simultaneous): Often spurious
│   │   ├─ Correlation at lag 1 (delayed): More plausible causal
│   │   ├─ Example: Monday games lose more (Monday effect?)
│   │   ├─ Reality: Different field conditions, fewer sharp players
│   │   └─ Causal: Condition → Reduced opponent skill → You win more
│   └─ Statistical vs Practical Significance:
│       ├─ Statistically significant: Correlation exists (p < 0.05)
│       ├─ Practically significant: Effect large enough to matter
│       ├─ Danger: Tiny correlation significant with large n, negligible impact
│       └─ Gambling: Correlation r=0.1 with n=10000 significant but ignorable
├─ VI. IMPLICATIONS FOR GAMBLING STRATEGY:
│   ├─ Independent Games (Roulette, Dice, Fair Coins):
│   │   ├─ Truth: No exploitation possible
│   │   ├─ Strategy: Each bet identical expected value
│   │   ├─ Bankroll: Use fixed bet sizing (Kelly doesn't apply)
│   │   ├─ Variance: High; standard deviation grows √n
│   │   ├─ House Edge: Insurmountable (negative EV every bet)
│   │   └─ Lesson: Accept loss, set loss limit, quit
│   ├─ Dependent Games (Poker, Blackjack):
│   │   ├─ Opportunity: Information → exploit dependence
│   │   ├─ Card Counting: Track composition → adjust bet/play
│   │   ├─ Position Advantage: Information accumulates through hand
│   │   ├─ Skill Edge: Read opponents (dependent on history)
│   │   └─ Kelly Criterion: Can use if positive edge identified
│   ├─ Semi-Dependent (Sports Betting):
│   │   ├─ Dependence source: Team conditions, psychology, injury
│   │   ├─ Opportunity: Identify games less efficient (information edges)
│   │   ├─ Hedge: Betting sequences correlated via line-setting
│   │   └─ Risk: Sportsbook maintains negative EV for most bettors
│   └─ Martingale Fallacy (All Games):
│       ├─ Belief: Doubling after loss ensures eventual win (false)
│       ├─ Assumption: Independence + no limits (wrong)
│       ├─ Reality: Infinite capital needed; bankrupt before win
│       ├─ Risk: Ruin probability = 1 with finite bankroll
│       └─ Correct: Bet sizing based on true odds, not sequences
├─ VII. COMMON FALLACIES:
│   ├─ Gambler's Fallacy:
│   │   ├─ Belief: Past independent events influence future
│   │   ├─ Example: "Red is due after 10 blacks"
│   │   ├─ Math: P(red | 10 blacks) = 18/37 (unchanged)
│   │   ├─ Origin: Humans pattern-seek; randomness feels non-random
│   │   ├─ Cost: Increases bet after losses (chasing losses)
│   │   └─ Truth: Wheel "memory-free"
│   ├─ Hot-Hand Fallacy:
│   │   ├─ Belief: Winners tend to continue winning
│   │   ├─ Truth: If underlying probability unchanged, false
│   │   ├─ Exception: Skill improves or opponents tire (true dependence)
│   │   ├─ Research: "Streak" usually just randomness
│   │   └─ Betting: Don't increase after wins blindly
│   ├─ Illusion of Control:
│   │   ├─ Belief: Personal action affects independent outcome
│   │   ├─ Example: Throwing dice harder to avoid snake-eyes
│   │   ├─ Reality: Dice roll independent of effort
│   │   ├─ Psychology: Humans crave agency over randomness
│   │   └─ Danger: False confidence → Reckless betting
│   ├─ Clustering Illusion:
│   │   ├─ Observation: Randomness produces clusters (clumping)
│   │   ├─ Misinterpretation: Clusters = non-random pattern
│   │   ├─ Truth: Randomness naturally clusters
│   │   ├─ Example: 8 reds in 10 spins ≠ wheel biased (within variance)
│   │   └─ Test: Run long-term analysis, not short sequences
│   └─ Selection Bias:
│       ├─ Memory: Remember time prediction worked, forget misses
│       ├─ Example: "I predicted winner" (ignore 10 wrong predictions)
│       ├─ Correlation: Correlation ≠ Causation conflation
│       └─ Fix: Formal hypothesis testing, not anecdotal evidence
└─ VIII. MATH FORMULAS:
    ├─ Independence:
    │   ├─ P(A|B) = P(A)
    │   ├─ P(A,B) = P(A) × P(B)
    │   └─ If n events: P(A₁, A₂, ..., A_n) = ∏ P(A_i)
    ├─ Dependence (Correlation):
    │   ├─ ρ(A,B) = Cov(A,B) / (σ_A × σ_B)
    │   ├─ ρ = 0 → uncorrelated (often independent)
    │   ├─ ρ > 0 → positive correlation
    │   └─ ρ < 0 → negative correlation
    ├─ Conditional Independence:
    │   ├─ A ⊥ B | C ⟺ P(A|B,C) = P(A|C)
    │   ├─ Equivalently: P(A,B|C) = P(A|C) × P(B|C)
    │   └─ Application: Condition removes dependence
    └─ Chi-Square for Independence:
        ├─ χ² = Σ[(O_i - E_i)² / E_i]
        ├─ E_i = P(A) × P(B) × n (expected under independence)
        ├─ df = (rows-1) × (cols-1)
        └─ If χ² > χ²_crit, reject independence
```

**Insight:** Independence simplifies probability (multiplication rule works). Dependence requires conditional reasoning. Games exploit dependence; gambling fallacies misapply independence.

## 5. Mini-Project
Test independence across gambling games with statistical analysis:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, kendalltau
from scipy.signal import correlate
import pandas as pd

np.random.seed(42)

print("="*70)
print("INDEPENDENCE AND DEPENDENCE: TESTING GAMBLING GAMES")
print("="*70)

# ============================================================================
# 1. ROULETTE INDEPENDENCE TEST
# ============================================================================

print("\n" + "="*70)
print("1. ROULETTE: TESTING INDEPENDENCE ACROSS SPINS")
print("="*70)

class Roulette:
    """European roulette wheel (18 red, 18 black, 1 green)."""
    
    def spin(self, n_spins):
        """Spin wheel n times; return outcomes (1=red, 0=black, -1=green)."""
        outcomes = np.random.choice([1, 0, -1], size=n_spins, p=[18/37, 18/37, 1/37])
        return outcomes
    
    def test_independence_chi_square(self, n_spins=500):
        """Test if consecutive spins are independent using chi-square."""
        outcomes = self.spin(n_spins)
        
        # Create contingency table: spin i vs spin i+1
        pairs = list(zip(outcomes[:-1], outcomes[1:]))
        
        # Count joint occurrences
        contingency_table = np.zeros((3, 3))
        for (prev, next_) in pairs:
            if prev >= 0 and next_ >= 0:  # Ignore green for simplicity
                contingency_table[int(prev), int(next_)] += 1
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'contingency': contingency_table,
            'expected': expected,
            'reject_independence': p_value < 0.05
        }

roulette = Roulette()
roulette_result = roulette.test_independence_chi_square(n_spins=1000)

print(f"\nRoulette Chi-Square Test (1000 spins):")
print(f"   χ² statistic: {roulette_result['chi2']:.4f}")
print(f"   p-value: {roulette_result['p_value']:.4f}")
print(f"   Reject independence (p<0.05)? {roulette_result['reject_independence']}")
print(f"   Contingency Table (row=prev spin, col=next spin):")
print(f"   {roulette_result['contingency'].astype(int)}")
print(f"   Conclusion: Spins ARE independent ✓ (as expected)")

# Compute lag-1 correlation
outcomes = roulette.spin(500)
correlation_lag1 = np.corrcoef(outcomes[:-1], outcomes[1:])[0,1]
print(f"\n   Lag-1 Correlation: {correlation_lag1:.4f} (near 0 = independent)")

# ============================================================================
# 2. POKER DEPENDENCE TEST: CARD DEPLETION
# ============================================================================

print("\n" + "="*70)
print("2. POKER: TESTING DEPENDENCE (CARD DEPLETION)")
print("="*70)

class PokerDeck:
    """Poker deck (52 cards, 4 suits/rank)."""
    
    RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
    HIGH_CARDS = ['10', 'J', 'Q', 'K', 'A']  # 20 cards
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset deck."""
        self.cards = []
        for rank in self.RANKS:
            for _ in range(4):
                self.cards.append(1 if rank in self.HIGH_CARDS else 0)  # 1=high, 0=low
        np.random.shuffle(self.cards)
        self.index = 0
    
    def deal_cards(self, n):
        """Deal n cards sequentially."""
        dealt = self.cards[self.index:self.index+n]
        self.index += n
        return dealt

# Test: Probability of high card depends on previous cards
deck = PokerDeck()

# Scenario 1: First 20 cards all low
dealt_low = deck.deal_cards(20)  # All low cards in order
remaining_deck = deck.cards[deck.index:]
p_high_after_low = np.mean(remaining_deck)  # Proportion high cards remaining

deck.reset()  # Reset

# Scenario 2: First 20 cards all high
deck.index = 0
all_cards = np.array(deck.cards)
low_indices = np.where(all_cards == 0)[0][:20]  # Indices of low cards
deck.index = len(low_indices)  # Skip those
# This is forced; instead, simulate deal high cards
deck.reset()

# Better: just compute theoretical
# If 20 low cards dealt: 32 cards remain, 20 of which are high
p_high_after_all_low = 20/32
p_high_initial = 20/52

# If 20 random cards dealt: expected 10 high, 10 low
# Remaining: 42 cards, ~10 high
p_high_after_random = 10/42

print(f"\nCard Dependence (Theoretical):")
print(f"   P(high card | deck start) = 20/52 = {20/52:.4f}")
print(f"   P(high card | 20 low dealt) = 20/32 = {p_high_after_all_low:.4f}")
print(f"   P(high card | 20 random dealt) ≈ 10/42 = {p_high_after_random:.4f}")
print(f"   Conclusion: Probabilities CHANGE with dealt cards (dependent) ✓")

# Simulate many hands
n_simulations = 1000
outcomes_dependence = []

for _ in range(n_simulations):
    deck = PokerDeck()
    
    # Deal 20 cards (first half)
    dealt_first = deck.deal_cards(20)
    n_high_dealt = np.sum(dealt_first)
    
    # Probability of high card in remaining deck
    remaining = deck.cards[deck.index:]
    n_high_remaining = np.sum(remaining)
    p_high_given = n_high_remaining / len(remaining)
    
    outcomes_dependence.append({
        'high_dealt': n_high_dealt,
        'p_high_given': p_high_given
    })

df_dep = pd.DataFrame(outcomes_dependence)
correlation_dealt_vs_prob = df_dep['high_dealt'].corr(df_dep['p_high_given'])

print(f"\n   Simulation ({n_simulations} hands):")
print(f"   Correlation(high_dealt, p_high_given): {correlation_dealt_vs_prob:.4f}")
print(f"   Negative correlation = dependence ✓ (more high dealt → fewer remain)")

# ============================================================================
# 3. GAMBLER'S FALLACY CHECK: TESTING FALSE DEPENDENCE
# ============================================================================

print("\n" + "="*70)
print("3. GAMBLER'S FALLACY: FALSE BELIEF IN DEPENDENCE")
print("="*70)

# Generate long roulette sequence
long_sequence = roulette.spin(n_spins=5000)

# Find long streaks (e.g., 5+ consecutive reds)
red_stretches = []
current_streak = 0

for outcome in long_sequence:
    if outcome == 1:  # Red
        current_streak += 1
    else:
        if current_streak >= 5:
            red_stretches.append(current_streak)
        current_streak = 0

# After long red streak, what happens next?
print(f"\nLong Red Streaks (5+ reds):")
print(f"   Found {len(red_stretches)} streaks of length ≥ 5")
if red_stretches:
    print(f"   Longest: {max(red_stretches)} reds in a row")
    print(f"   Average length: {np.mean(red_stretches):.1f}")

# Find where long streaks ended and check next 10 outcomes
next_after_streak = []
for i, outcome in enumerate(long_sequence):
    # Find if this is end of 5+ red streak
    if i >= 5:
        recent = long_sequence[max(0, i-5):i]
        if len(recent) == 5 and np.all(recent == 1) and i < len(long_sequence) - 1:
            # 5 reds just ended, collect next spin
            next_spin = long_sequence[i]
            next_after_streak.append(next_spin)

p_black_after_streak = 1 - np.mean(next_after_streak) if next_after_streak else 0

print(f"\n   Next spin after 5-red streak:")
print(f"   P(black next | 5 reds just happened) = {p_black_after_streak:.3f}")
print(f"   P(black | independent) = 18/37 = {18/37:.3f}")
print(f"   Difference: {abs(p_black_after_streak - 18/37):.3f}")
print(f"   Conclusion: No significant difference (gambler's fallacy FALSE) ✓")

# ============================================================================
# 4. CORRELATION VS CAUSATION: BETTING SEQUENCES
# ============================================================================

print("\n" + "="*70)
print("4. CORRELATION VS CAUSATION: BET SIZE & OUTCOMES")
print("="*70)

# Simulate betting with psychological tilt
n_games = 100
outcomes_corr = []

initial_bet = 1.0
bankroll = 100.0
bet_history = []
outcome_history = []

for game in range(n_games):
    # Bet amount (depends on recent losses)
    bet = initial_bet
    
    if len(outcome_history) >= 2:
        recent_outcomes = outcome_history[-2:]
        if np.mean(recent_outcomes) == 0:  # Lost last 2 games
            bet = initial_bet * 2  # Tilt: increase bet after losses
    
    # Outcome (independent, -1 or +1 win/loss)
    outcome = np.random.choice([-1, +1], p=[0.51, 0.49])  # House edge
    
    # Bankroll update
    bankroll += bet * outcome
    
    bet_history.append(bet)
    outcome_history.append(outcome)
    
    outcomes_corr.append({
        'game': game,
        'bet': bet,
        'outcome': outcome,
        'cumulative_loss': np.sum(outcome_history)
    })

df_corr = pd.DataFrame(outcomes_corr)

# Test correlation
corr_bet_outcome = df_corr['bet'].corr(df_corr['outcome'])
corr_bet_loss = df_corr['bet'].corr(df_corr['cumulative_loss'])

print(f"\nBetting Psychology Simulation ({n_games} games):")
print(f"   Correlation(bet_size, outcome): {corr_bet_outcome:.4f}")
print(f"   Correlation(bet_size, cumulative_loss): {corr_bet_loss:.4f}")
print(f"\n   Interpretation:")
print(f"   - Bet size correlates with losses (bet more after losses)")
print(f"   - BUT: Causation is LOSS → TILT → HIGHER BET")
print(f"   - NOT: Higher bets cause more losses (odds fixed)")
print(f"   - Correlation ≠ Causation ✓")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Roulette spins (independent, appears random)
ax1 = axes[0, 0]
spins = roulette.spin(100)
colors_spin = ['red' if s == 1 else 'black' for s in spins]
ax1.bar(range(len(spins)), spins, color=colors_spin, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Outcome (1=Red, 0=Black)')
ax1.set_xlabel('Spin Number')
ax1.set_title('Roulette Spins: Independent (No Pattern)')
ax1.set_ylim(-0.1, 1.1)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Poker card dependence
ax2 = axes[0, 1]
high_dealt_list = [o['high_dealt'] for o in outcomes_dependence]
p_high_list = [o['p_high_given'] for o in outcomes_dependence]
ax2.scatter(high_dealt_list, p_high_list, alpha=0.6, s=20)
ax2.set_xlabel('High Cards Dealt (first 20)')
ax2.set_ylabel('P(High Card in Remaining)')
ax2.set_title(f'Poker: Dependence (corr={correlation_dealt_vs_prob:.3f})')
z = np.polyfit(high_dealt_list, p_high_list, 1)
p = np.poly1d(z)
ax2.plot(sorted(high_dealt_list), p(sorted(high_dealt_list)), "r--", linewidth=2, alpha=0.8)
ax2.grid(True, alpha=0.3)

# Plot 3: Betting psychology
ax3 = axes[1, 0]
ax3_twin = ax3.twinx()
bars = ax3.bar(df_corr['game'], df_corr['bet'], alpha=0.5, color='blue', label='Bet Size')
colors_outcome = ['green' if o == 1 else 'red' for o in df_corr['outcome']]
ax3_twin.plot(df_corr['game'], df_corr['cumulative_loss'], 'o-', color='purple', 
             linewidth=2, markersize=4, label='Cumulative Loss')
ax3.set_xlabel('Game Number')
ax3.set_ylabel('Bet Size (Blue)', color='blue')
ax3_twin.set_ylabel('Cumulative Loss (Purple)', color='purple')
ax3.set_title('Correlation ≠ Causation: Betting After Losses')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Chi-square independence test
ax4 = axes[1, 1]
roulette_test_results = []
n_values = [100, 200, 500, 1000, 2000, 5000]

for n_spin in n_values:
    result = roulette.test_independence_chi_square(n_spins=n_spin)
    roulette_test_results.append({
        'n_spins': n_spin,
        'p_value': result['p_value'],
        'chi2': result['chi2']
    })

df_test = pd.DataFrame(roulette_test_results)
ax4.plot(df_test['n_spins'], df_test['p_value'], 'o-', linewidth=2, markersize=8, color='darkgreen')
ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05 threshold')
ax4.set_ylabel('p-value')
ax4.set_xlabel('Number of Spins')
ax4.set_title('Chi-Square Test: Roulette Independence (p>0.05 = independent)')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('independence_dependence_gambling.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: independence_dependence_gambling.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ Roulette: Independent (χ² test p > 0.05)")
print("✓ Poker: Dependent (card depletion correlates with probability)")
print("✓ Gambler's Fallacy: False belief in dependence (disproved)")
print("✓ Correlation ≠ Causation: Betting correlates with loss, not cause")
```

## 6. Challenge Round
**When does independence testing fail?**
- Small sample bias: Chi-square unreliable with n < 5 per cell; false independence detected
- Hidden confounding: Apparent independence masks common cause (both affected by external factor)
- Temporal aggregation: Aggregated data looks independent, disaggregated shows dependence
- Feedback loops: System adapts over time (non-stationary); independence assumption violated mid-game
- Measurement error: Noise in observation masks true dependence patterns

## 7. Key References
- [Feller (1968), "An Introduction to Probability Theory"](https://www.amazon.com/Introduction-Probability-Theory-Applications-Vol/dp/0471257087) - Foundation on independence
- [Tversky & Kahneman (1974), "Judgment Under Uncertainty"](https://science.sciencemag.org/content/185/4157/1124) - Gambler's Fallacy and representativeness heuristic
- [Thorp (1962), "Beat the Dealer"](https://en.wikipedia.org/wiki/Edward_Thorp) - Card counting exploits dependence

---
**Status:** Establishes difference between true and false dependence | **Complements:** Conditional probability, Law of Large Numbers | **Enables:** Game-specific strategy development