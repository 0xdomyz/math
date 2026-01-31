# Expected Value (EV): Core Metric for Gambling Decisions

## 1. Concept Skeleton
**Definition:** E[X] = Σ(outcome_i × probability_i); long-term average outcome per bet  
**Purpose:** Single metric to compare games, determine +EV vs -EV opportunities, guide bet sizing  
**Prerequisites:** Basic probability, outcomes and payoffs, probability distributions

## 2. Comparative Framing
| Concept | Expected Value | Most Likely Outcome | Best Case | Worst Case |
|---------|----------------|-------------------|-----------|-----------|
| **Definition** | Weighted average of all outcomes | Mode (most frequent result) | Maximum possible gain | Maximum possible loss |
| **Formula** | E[X] = Σ x_i × P(x_i) | argmax P(X=x) | max(outcomes) | min(outcomes) |
| **Example (Bet $1 Red)** | -1/37 ≈ -$0.027 | Lose $1 (more likely) | Win $1 | Lose $1 |
| **Reflects** | Long-term average | Single most probable event | Optimistic scenario | Pessimistic scenario |
| **Relevance** | Highest for strategy (repeated decisions) | Lowest (misleading for one-time) | Useful for upside | Useful for downside |
| **Gambling** | Determines long-term profitability | Not directly useful alone | Rarely happens in short-term | Rare but possible |

## 3. Examples + Counterexamples

**Simple Example:**  
Flip coin, win $1 heads, lose $1 tails (fair coin). E[X] = 1×(0.5) + (-1)×(0.5) = 0 (fair game, break-even)

**Failure Case:**  
Roulette on American wheel (double 0): E[X] = -$0.053 per $1 bet, but one session might show +$100 gain (short-term variance dominates)

**Edge Case:**  
Sports bet: 60% win prob, +100 payout (bet $100 to win $100). E[X] = 100×(0.6) + (-100)×(0.4) = $20 (positive EV, bet it!)

## 4. Layer Breakdown
```
Expected Value Framework:
├─ I. DEFINITION & CALCULATION:
│   ├─ Basic Formula:
│   │   ├─ E[X] = Σ x_i × P(X = x_i) (discrete outcomes)
│   │   ├─ E[X] = ∫ x × f(x) dx (continuous)
│   │   ├─ Components:
│   │   │   ├─ x_i: Outcome value (win/loss amount)
│   │   │   ├─ P(x_i): Probability of outcome
│   │   │   └─ Summation: Average across all possibilities
│   │   └─ Interpretation: What you expect to win/lose on average per bet
│   ├─ Linearity of Expectation:
│   │   ├─ E[X + Y] = E[X] + E[Y] (works for any X, Y)
│   │   ├─ E[c × X] = c × E[X] (for constant c)
│   │   └─ Application: Multiple independent bets
│   ├─ Calculation Steps:
│   │   ├─ Step 1: List all possible outcomes
│   │   ├─ Step 2: Assign probability to each outcome
│   │   ├─ Step 3: Multiply outcome × probability
│   │   ├─ Step 4: Sum all products
│   │   └─ Result: Expected value (in $ or units)
│   └─ Example Walkthrough (Roulette):
│       ├─ Bet $1 on red
│       ├─ Outcome 1: Win $1 if red (prob 18/37)
│       ├─ Outcome 2: Lose $1 if not red (prob 19/37)
│       ├─ E[X] = (+1)×(18/37) + (-1)×(19/37)
│       ├─ E[X] = 18/37 - 19/37 = -1/37 ≈ -$0.027
│       └─ Interpretation: Average loss = 2.7¢ per $1 bet
├─ II. EXPECTED VALUE IN GAMBLING CONTEXTS:
│   ├─ Casino Perspective (House EV):
│   │   ├─ Definition: Average profit per bet (for casino)
│   │   ├─ Formula: House EV = 1 - Player EV / (bet amount)
│   │   ├─ Example: Roulette House EV = 2.7% (European)
│   │   ├─ Significance: Guarantees profit over time
│   │   ├─ Scale: Billions of bets → certain profitability
│   │   └─ Implication: House always wins long-term (unless biased wheel)
│   ├─ Player Perspective (Negative EV):
│   │   ├─ Definition: Average loss per bet
│   │   ├─ Formula: Player EV = (Prob win × Win amount) - (Prob lose × Lose amount)
│   │   ├─ Example: -1/37 per $1 on roulette
│   │   ├─ Significance: Determines rate of capital depletion
│   │   ├─ Scale: More bets → larger cumulative loss
│   │   └─ Implication: Negative EV games are unbeatable
│   ├─ Advantage Player (Positive EV):
│   │   ├─ Definition: Average profit per bet (rare in casino games)
│   │   ├─ Example: Card counting in blackjack (+1% to +2% possible)
│   │   ├─ Example: Sports betting with informational edge (+3% to +5%)
│   │   ├─ Strategy: Maximize bets when EV positive
│   │   ├─ Bankroll: Kelly Criterion sizing
│   │   └─ Implication: Sustained profitability possible
│   └─ Zero EV (Fair Game):
│       ├─ Definition: E[X] = 0 (neither player advantage)
│       ├─ Example: Fair coin flip ($1 vs $1)
│       ├─ Reality: Rare in commercial gambling (always HE)
│       ├─ Implication: Break-even long-term (but variance possible)
│       └─ Academic: Only for theory; casinos never offer fair games
├─ III. SPECIFIC GAME EXPECTED VALUES:
│   ├─ Roulette:
│   │   ├─ European (single 0): EV = -1/37 ≈ -2.70%
│   │   ├─ American (0,00): EV = -2/38 ≈ -5.26%
│   │   └─ All bets same EV: Straight, red/black, evens, odds all -2.7% (EU)
│   ├─ Blackjack:
│   │   ├─ Basic Strategy (perfect play): EV ≈ -0.5% to -0.6%
│   │   ├─ Average player (poor strategy): EV ≈ -2% to -4%
│   │   ├─ Card Counter (tracking high cards): EV ≈ +0.5% to +2%
│   │   └─ Implication: Skill greatly affects expected value
│   ├─ Poker:
│   │   ├─ House rake (5-10%): Reduces player EV by rake amount
│   │   ├─ Skilled player (+1.5 BB/100): Positive EV through skill
│   │   ├─ Average player: Often negative after rake
│   │   └─ Advantage: Skill dominates (not just luck)
│   ├─ Craps:
│   │   ├─ Pass Line: EV ≈ -1.41%
│   │   ├─ Don't Pass: EV ≈ -1.36%
│   │   ├─ Field Bet: EV ≈ -5.56%
│   │   └─ Odds Bets: EV = 0 (fair after point established)
│   ├─ Slot Machines:
│   │   ├─ Typical RTP: 92-96% (EV = -(4% to 8%))
│   │   ├─ Progressive slots: Often lower RTP (higher HE)
│   │   ├─ Variance: High (spiky payouts)
│   │   └─ Implication: Worst EV among popular games
│   ├─ Keno:
│   │   ├─ EV typically -25% to -40% (worst casino game)
│   │   ├─ High house edge reflects high variance/low win freq
│   │   └─ Implication: Should be avoided if -EV available
│   └─ Sports Betting:
│       ├─ Sportsbook cut (juice/vig): Typically -4% to -5% baseline
│       ├─ Sharp bettor (model-based): Can achieve +1% to +3%
│       ├─ Average bettor: Usually -5% to -10%
│       └─ Implication: Informational edge critical
├─ IV. EXPECTED VALUE PER UNIT TIME:
│   ├─ Definition: EV per bet × bets per hour = expected hourly win/loss
│   ├─ Formula: EV_hourly = EV_per_bet × hands_per_hour
│   ├─ Example (Poker):
│   │   ├─ Game: $1/$2 cash game
│   │   ├─ Expected win rate: +2 BB/hand (skilled player)
│   │   ├─ Hands/hour: ~30 hands
│   │   ├─ EV_hourly = 2 BB × 30 hands × $2 = $120/hour
│   │   └─ Implication: Income potential from skill edge
│   ├─ Example (Blackjack):
│   │   ├─ Game: $10/hand
│   │   ├─ Basic strategy EV: -0.5%
│   │   ├─ Hands/hour: ~60 hands
│   │   ├─ EV_hourly = -0.005 × $10 × 60 = -$3/hour
│   │   └─ Implication: Slow capital drain
│   └─ Application: Decision-making (pursue +EV activity if hourly return adequate)
├─ V. EDGE DETECTION & COMPARISON:
│   ├─ Identifying +EV Opportunities:
│   │   ├─ Method: Calculate E[X] for each option
│   │   ├─ Compare: E[X] values across games/bets
│   │   ├─ Rule: Choose highest E[X]
│   │   ├─ Threshold: Only play if E[X] > 0 (or minimum acceptable)
│   │   └─ Challenge: Requires accurate probability estimates
│   ├─ Implied Odds (Sports Betting):
│   │   ├─ Definition: Convert betting odds to implied probability
│   │   ├─ Example: -110 odds (bet $110 to win $100)
│   │   ├─ Implied prob: 110 / (110+100) ≈ 0.524
│   │   ├─ Bookmaker markup: Implied > true probability
│   │   ├─ Edge detection: If true prob > implied, positive EV
│   │   └─ Method: Build models → compare true vs implied
│   ├─ Game Selection Discipline:
│   │   ├─ Rule: Only play +EV games (E[X] > 0)
│   │   ├─ Rationale: Guaranteed long-term profit
│   │   ├─ Challenge: Casinos offer -EV only (except poker rake structure)
│   │   ├─ Alternative: Play -EV games with lowest edge
│   │   └─ Example: Blackjack (-0.5%) > Slots (-5%)
│   └─ Personal Skill Edge:
│       ├─ In poker: Winning players have +EV vs rake
│       ├─ In blackjack: Card counters achieve +EV vs HE
│       ├─ In sports: Quantitative models find +EV lines
│       ├─ Requirement: Skill edge > game's built-in edge
│       └─ Measurement: Track long-term win rate vs theoretical
├─ VI. EXPECTED VALUE LIMITATIONS:
│   ├─ Assumes Many Repetitions:
│   │   ├─ Theory: E[X] accurate as n→∞
│   │   ├─ Reality: Single session high variance possible
│   │   ├─ Example: One hand blackjack can win/lose massively
│   │   ├─ Implication: EV not useful for single bets
│   │   └─ Lesson: EV is long-term metric
│   ├─ Ignores Utility (Risk Aversion):
│   │   ├─ Theory: E[X] maximization optimal
│   │   ├─ Reality: Humans dislike risk (concave utility)
│   │   ├─ Example: Risk-averse player avoids all-in (even +EV)
│   │   ├─ Implication: Kelly Criterion more realistic
│   │   └─ Lesson: Bankroll size affects decision-making
│   ├─ Assumes Probability Estimates Correct:
│   │   ├─ Challenge: Accurately computing P(outcome) hard
│   │   ├─ Error: Misestimate probabilities → wrong EV
│   │   ├─ Example: Sports bettor overestimates team strength
│   │   ├─ Implication: Biased models → -EV perceived as +EV
│   │   └─ Lesson: Validation critical (compare predictions to outcomes)
│   ├─ Non-Stationary Distributions:
│   │   ├─ Issue: Probabilities change over time
│   │   ├─ Example: Card composition changes in blackjack
│   │   ├─ Example: Team skill evolves over season
│   │   ├─ Implication: Static EV calculation misleading
│   │   └─ Lesson: Adapt estimates; re-calculate as conditions change
│   └─ Ruin Before Convergence:
│       ├─ Issue: Negative EV players bust before LLN kicks in
│       ├─ Formula: Risk of Ruin depends on bankroll
│       ├─ Implication: Even if long-term break-even, ruin possible
│       └─ Lesson: Bankroll size must sustain variance
├─ VII. FORMULAS & CALCULATIONS:
│   ├─ Discrete Expected Value:
│   │   ├─ E[X] = Σ x_i × P(x_i)
│   │   └─ Example: Dice (E[X] = 1×(1/6) + ... + 6×(1/6) = 3.5)
│   ├─ Continuous Expected Value:
│   │   ├─ E[X] = ∫ x × f(x) dx
│   │   └─ Example: Uniform distribution E[X] = (a+b)/2
│   ├─ Expected Value of Bet:
│   │   ├─ E[profit] = (Win amount)×P(win) + (Loss amount)×P(lose)
│   │   └─ Simplified: E[profit] = (payout - 1) × P(win) - P(lose)
│   ├─ House Edge Calculation:
│   │   ├─ House Edge % = |Player EV| / (typical bet size)
│   │   └─ Example: Roulette = (1/37) / 1 = 2.7%
│   ├─ Combined EV (Multiple Independent Bets):
│   │   ├─ E[X+Y] = E[X] + E[Y]
│   │   └─ Total EV = (EV per bet) × (number of bets)
│   └─ Hourly EV:
│       ├─ EV_hourly = EV_per_bet × bet_size × bets_per_hour
│       └─ Example: $1 EV per hand × $10/hand × 30 hands = $300/hr
└─ VIII. STRATEGIC APPLICATIONS:
    ├─ Bet Sizing:
    │   ├─ Kelly Criterion: f* = (EV per dollar) / variance
    │   ├─ Conservative: Bet only when EV clear and positive
    │   └─ Risk control: Never overbid despite high EV
    ├─ Game Selection:
    │   ├─ Rank games by EV: Pick highest available
    │   ├─ +EV games: Only play if found
    │   ├─ -EV games: Minimize (lowest edge)
    │   └─ Zero EV: Educational only
    ├─ Session Management:
    │   ├─ Stop loss: Quit when down X% (prevent ruin)
    │   ├─ Profit target: Quit when up Y (lock in gains)
    │   ├─ Time limit: Play max H hours (control exposure)
    │   └─ Rationale: Variance dominates short-term
    └─ Evaluation & Tracking:
        ├─ Track actual vs expected: Compare results to EV predictions
        ├─ Calibration: Are estimates accurate?
        ├─ Adjustment: Refine estimates based on outcomes
        └─ Accountability: Verify skill edge exists
```

**Core Insight:** EV is the fundamental metric determining long-term profitability. E[X] > 0 → profit, E[X] < 0 → loss, E[X] = 0 → break-even.

## 5. Mini-Project
Calculate EV across multiple games and simulate outcomes:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

np.random.seed(42)

print("="*70)
print("EXPECTED VALUE: GAME COMPARISON & SIMULATION")
print("="*70)

# ============================================================================
# 1. CALCULATE EV FOR DIFFERENT GAMES
# ============================================================================

print("\n" + "="*70)
print("1. EXPECTED VALUE CALCULATION ACROSS GAMES")
print("="*70)

class CasinoGame:
    """Represents a casino game with outcomes and probabilities."""
    
    def __init__(self, name, outcomes, probabilities, bet_amount=1.0):
        """
        Args:
            name: Game name
            outcomes: List of outcome values (profit/loss)
            probabilities: List of outcome probabilities (must sum to 1)
            bet_amount: Default bet size ($)
        """
        self.name = name
        self.outcomes = np.array(outcomes)
        self.probabilities = np.array(probabilities)
        self.bet_amount = bet_amount
        
        assert np.isclose(np.sum(probabilities), 1.0), "Probabilities must sum to 1"
        
        # Calculate EV
        self.ev = np.sum(self.outcomes * self.probabilities)
        self.ev_percentage = self.ev / bet_amount * 100
        
        # Calculate variance and std dev
        self.variance = np.sum((self.outcomes - self.ev)**2 * self.probabilities)
        self.std_dev = np.sqrt(self.variance)
    
    def display_summary(self):
        """Print game summary."""
        print(f"\n{self.name}:")
        print(f"   Outcomes: {self.outcomes}")
        print(f"   Probabilities: {self.probabilities}")
        print(f"   Expected Value: ${self.ev:.4f} per ${self.bet_amount:.2f} bet")
        print(f"   EV %: {self.ev_percentage:.2f}%")
        print(f"   Std Dev: ${self.std_dev:.4f}")
        print(f"   Risk-Adjusted: EV/σ = {self.ev/self.std_dev:.4f}")

# Define games
games = []

# Roulette (European)
roulette_ev = CasinoGame(
    "Roulette (European, bet red)",
    outcomes=[1, -1],  # Win $1 or lose $1
    probabilities=[18/37, 19/37],  # Red prob, not red prob
    bet_amount=1.0
)
games.append(roulette_ev)

# Blackjack (basic strategy)
blackjack_ev = CasinoGame(
    "Blackjack (basic strategy)",
    outcomes=[1, -1.05, 0.5],  # Win $1, lose, or push (tie)
    probabilities=[0.48, 0.49, 0.03],
    bet_amount=1.0
)
games.append(blackjack_ev)

# Craps (Pass line)
craps_ev = CasinoGame(
    "Craps (Pass line)",
    outcomes=[1, -1],
    probabilities=[49/100, 51/100],  # Simplified
    bet_amount=1.0
)
games.append(craps_ev)

# Slots (typical)
slots_ev = CasinoGame(
    "Slot Machine (typical)",
    outcomes=[9, 99, 999, -1],  # Various payouts
    probabilities=[0.04, 0.005, 0.0001, 0.9549],
    bet_amount=1.0
)
games.append(slots_ev)

# Fair coin (for reference)
fair_coin = CasinoGame(
    "Fair Coin Flip",
    outcomes=[1, -1],
    probabilities=[0.5, 0.5],
    bet_amount=1.0
)
games.append(fair_coin)

print("\nGame Comparison:")
for game in games:
    game.display_summary()

# ============================================================================
# 2. RANK GAMES BY EV & RISK-ADJUSTED RETURNS
# ============================================================================

print("\n" + "="*70)
print("2. RANKING GAMES BY EXPECTED VALUE")
print("="*70)

df_games = pd.DataFrame({
    'Game': [g.name for g in games],
    'EV ($)': [g.ev for g in games],
    'EV (%)': [g.ev_percentage for g in games],
    'Std Dev ($)': [g.std_dev for g in games],
    'Sharpe Ratio (EV/σ)': [g.ev / g.std_dev for g in games],
    'Sortino Ratio': [g.ev / g.std_dev for g in games]  # Simplified
})

df_games_sorted = df_games.sort_values('EV ($)', ascending=False)
print("\nRanked by Expected Value (best to worst):")
print(df_games_sorted.to_string(index=False))

# ============================================================================
# 3. SIMULATE BETTING OUTCOMES
# ============================================================================

print("\n" + "="*70)
print("3. SIMULATING BETTING OVER TIME")
print("="*70)

def simulate_game(game, n_bets, bet_size=1.0, n_simulations=100):
    """
    Simulate n independent bets; return cumulative outcomes.
    
    Returns:
        paths: (n_simulations, n_bets) array of cumulative profits
        expected_path: Average path across simulations
    """
    paths = []
    
    for sim in range(n_simulations):
        # Generate outcomes based on probabilities
        bets = np.random.choice(
            game.outcomes,
            size=n_bets,
            p=game.probabilities
        )
        
        # Multiply by bet size and compute cumulative sum
        profits = bets * bet_size
        cumsum = np.cumsum(profits)
        paths.append(cumsum)
    
    paths = np.array(paths)
    expected_path = np.mean(paths, axis=0)
    
    return paths, expected_path

# Simulate each game
n_bets = 1000
simulations_per_game = {}

print(f"\nSimulating {n_bets} bets per game ({100} simulations each):")
for game in games:
    paths, expected = simulate_game(game, n_bets, bet_size=game.bet_amount, n_simulations=100)
    simulations_per_game[game.name] = {'paths': paths, 'expected': expected}
    
    final_avg = np.mean(paths[:, -1])
    final_std = np.std(paths[:, -1])
    theoretical_loss = game.ev * n_bets
    
    print(f"\n{game.name}:")
    print(f"   Theoretical total EV: ${theoretical_loss:.2f}")
    print(f"   Simulated avg outcome: ${final_avg:.2f} ± ${final_std:.2f}")
    print(f"   % of bets profitable: {100 * np.mean(paths[:, -1] > 0):.1f}%")

# ============================================================================
# 4. CALCULATING EV FOR SPECIFIC BETS
# ============================================================================

print("\n" + "="*70)
print("4. EV CALCULATION FOR CUSTOM BETS")
print("="*70)

# Sports betting scenario
print("\nSports Betting Scenario:")
print("   Bet: $100 on Team A to win")
print("   Your estimated P(Team A wins): 0.55")
print("   Sportsbook odds: -110 (American)")
print("   ")
print("   Implied probability: 110 / (110 + 100) = 0.524")
print("   ")

your_prob = 0.55
implied_prob = 110 / (110 + 100)
bet_amount = 100
win_payout = 100  # Net gain if win

ev_sports = (your_prob * win_payout) - ((1 - your_prob) * bet_amount)
print(f"   Your EV: (0.55 × $100) - (0.45 × $100) = ${ev_sports:.2f}")
print(f"   Your EV %: {(ev_sports / bet_amount) * 100:.2f}%")
print(f"   Decision: {'✓ BET' if ev_sports > 0 else '✗ PASS'}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: EV comparison (bar chart)
ax1 = axes[0, 0]
colors_ev = ['green' if ev > 0 else 'red' for ev in df_games['EV (%)']]
bars = ax1.barh(df_games['Game'], df_games['EV (%)'], color=colors_ev, alpha=0.7, edgecolor='black')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Expected Value (%)')
ax1.set_title('Expected Value Comparison Across Games')
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (game, ev) in enumerate(zip(df_games['Game'], df_games['EV (%)'])):
    ax1.text(ev + 0.1 if ev > 0 else ev - 0.1, i, f'{ev:.2f}%', 
            va='center', ha='left' if ev > 0 else 'right', fontsize=9)

# Plot 2: Roulette simulation (cumulative profit)
ax2 = axes[0, 1]
roulette_paths = simulations_per_game['Roulette (European, bet red)']['paths']
roulette_expected = simulations_per_game['Roulette (European, bet red)']['expected']

# Plot sample paths
for path in roulette_paths[:20]:  # Plot first 20 simulations
    ax2.plot(path, alpha=0.1, color='red')

ax2.plot(roulette_expected, color='red', linewidth=3, label='Expected Path')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Bet Number')
ax2.set_ylabel('Cumulative Profit ($)')
ax2.set_title('Roulette: Cumulative Outcomes (20 simulations)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribution of outcomes by game
ax3 = axes[1, 0]
final_outcomes_all = []
game_labels_all = []

for game_name, sim_data in simulations_per_game.items():
    final_outcomes = sim_data['paths'][:, -1]
    final_outcomes_all.append(final_outcomes)
    game_labels_all.append(game_name.split(' (')[0])  # Shorten label

bp = ax3.boxplot(final_outcomes_all, labels=game_labels_all, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_ev):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_ylabel('Final Outcome ($) after 1000 bets')
ax3.set_title('Distribution of Final Outcomes by Game')
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 4: Risk vs Reward (scatter)
ax4 = axes[1, 1]
ax4.scatter(df_games['Std Dev ($)'], df_games['EV ($)'], s=200, alpha=0.7, c=colors_ev)

for idx, row in df_games.iterrows():
    ax4.annotate(row['Game'].split(' (')[0], 
                (row['Std Dev ($)'], row['EV ($)']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')
ax4.set_xlabel('Risk (Std Dev, $)')
ax4.set_ylabel('Expected Value ($)')
ax4.set_title('Risk-Return Profile')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('expected_value_analysis.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: expected_value_analysis.png")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ EV metric ranks games by long-term profitability")
print("✓ Negative EV games → guaranteed loss over time")
print("✓ Only play +EV games or minimize -EV exposure")
print("✓ Risk-return trade-off: High std dev with negative EV is worst")
```

## 6. Challenge Round
**When does EV analysis fail or mislead?**
- Insufficient sample size: One session high variance; EV prediction off
- Probability estimation error: Wrong P(outcome) → incorrect EV → bad decisions
- Non-stationary distributions: Game changes mid-sequence; static EV obsolete
- Ruin before convergence: Bankrupt before LLN takes effect despite +EV
- Misuse of hourly EV: Using EV per bet when time/intensity varies

## 7. Key References
- [Khan Academy - Expected Value](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/expected-value-library) - Foundational concept
- [Wizard of Odds - Game Mathematics](https://www.wizardofodds.com/gambling/) - Specific game EV calculations
- [Thorp (1962), "Beat the Dealer"](https://en.wikipedia.org/wiki/Edward_Thorp) - Applied EV to blackjack advantage play

---
**Status:** Fundamental metric for all gambling decisions | **Complements:** Probability, Variance, Risk of Ruin | **Enables:** Game selection, bet sizing, profitability assessment