# Game Theory in Gambling

## 1. Concept Skeleton
**Definition:** Mathematical framework analyzing strategic interactions where outcomes depend on all participants' decisions; applies to competitive games and optimal play  
**Purpose:** Find optimal strategies resistant to opponent adaptation, identify mixed strategies, determine Nash equilibrium  
**Prerequisites:** Probability, decision theory, linear algebra, game theory basics

## 2. Comparative Framing
| Concept | Zero-Sum | Cooperative | Symmetric | Asymmetric |
|---------|----------|------------|-----------|-----------|
| **Total Payoff** | Always = 0 | Sum > 0 (shared) | All players equal | Players differ |
| **Conflict** | Maximum | Minimum | Pure | Mixed |
| **Example** | Poker | Partnership | Matching pennies | Chess |
| **Solution** | Nash Eq | Coalition | Mixed Strat | Minimax |

## 3. Examples + Counterexamples

**Simple Example:**  
Matching pennies: Each player hides coin (heads/tails). If match, player 1 wins; if different, player 2 wins. Optimal: 50-50 randomization.

**Failure Case:**  
Pure strategy in dynamic game. Opponent detects pattern, exploits it. Mixed (randomized) strategy forces opponent's indifference.

**Edge Case:**  
Bluffing in poker: If always bluff or never bluff, exploitable. Equilibrium: Bluff proportional to pot odds.

## 4. Layer Breakdown
```
Game Theory Framework:
├─ Zero-Sum Games:
│   ├─ One player's gain = opponent's loss
│   ├─ Payoff matrix: All outcomes sum to zero
│   ├─ Nash Equilibrium: Each player plays best response to opponent
│   ├─ Minimax Theorem: Max of min payoff = Min of max payoff
│   └─ Examples: Poker, roulette (player vs house), sports betting
├─ Nash Equilibrium:
│   ├─ No player can unilaterally improve by deviating
│   ├─ Pure strategy: Always play same action
│   ├─ Mixed strategy: Randomize across actions with specific probabilities
│   ├─ Calculation: Make opponent indifferent between their actions
│   └─ Result: Often unique for 2-player zero-sum games
├─ Bluffing & Mixed Strategies:
│   ├─ Pure bluffing: Always bluff → opponent folds always
│   │   Then never bluff → opponent calls always (exploited)
│   ├─ Mixed strategy: Bluff frequency = pot_odds / (pot_odds + bet)
│   ├─ Indifference: Opponent indifferent between fold/call
│   ├─ Example: 3:1 pot odds → bluff 25% of value hands
│   └─ Deviation: Exploitable; opponent can counter-exploit
├─ Information Games:
│   ├─ Perfect information: All prior moves known (chess, checkers)
│   ├─ Imperfect information: Hidden cards/information (poker)
│   ├─ Asymmetric information: Players know different amounts
│   ├─ Signaling: Actions reveal hidden information (hand strength in poker)
│   └─ Bluffing: Lie or misrepresent for advantage
├─ Solving Games:
│   ├─ Backward induction: Perfect info, work backward from end
│   ├─ Linear programming: Optimize mixed strategy probabilities
│   ├─ Simplex method: Solve for equilibrium payoffs
│   ├─ Lemke-Howson: Find Nash equilibrium algorithmically
│   └─ Approximation: Iterative methods for complex games
├─ Strategic Concepts:
│   ├─ Dominated strategy: Always worse, never use
│   ├─ Best response: Given opponent's strategy, your optimal action
│   ├─ Indifference: Opponent doesn't care which action you take
│   ├─ Exploitation: Punish opponent's deviation from equilibrium
│   └─ Counter-exploitation: Detect opponent's strategy, adapt
└─ Practical Application:
    ├─ Poker: Mix value bets + bluffs to exploit pot odds
    ├─ Roulette: No strategic element (random house rules)
    ├─ Sports betting: Line movement reflects aggregate player strategies
    ├─ Blackjack: House fixed strategy, optimal play is deterministic
    └─ Sports match: Team strategies interact (symmetric in setup)
```

## 5. Mini-Project
Solve simple games with game theory:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Example 1: Matching Pennies (classic zero-sum)
print("=== Game Theory: Matching Pennies ===\n")

# Payoff matrix (player 1's perspective)
# Player 2 plays Heads or Tails
# Player 1 plays Heads or Tails
# If match (both H or both T): Player 1 wins $1
# If don't match: Player 1 loses $1

print("Payoff Matrix (Player 1's winnings):")
print("              P2 Heads  P2 Tails")
print("P1 Heads        +1        -1")
print("P1 Tails        -1        +1)
print()

# Find mixed strategy Nash equilibrium
# Player 1: p(Heads) = p, p(Tails) = 1-p
# Player 2: q(Heads) = q, q(Tails) = 1-q

# For P1 to mix, P2 must make H and T equally attractive to P1
# EV(P1 plays H) = p*1 + (1-p)*(-1) = 2p - 1
# EV(P1 plays T) = p*(-1) + (1-p)*1 = 1 - 2p
# Indifference: 2p - 1 = 1 - 2p → p = 0.5

# Similarly, q = 0.5

print("Nash Equilibrium Mixed Strategy:")
print("Player 1: Play Heads 50%, Tails 50%")
print("Player 2: Play Heads 50%, Tails 50%")
print("Expected payoff for Player 1: $0 (fair game)")

# Example 2: Poker bluffing equilibrium
print("\n\n=== Poker Bluffing Strategy ===\n")

# Simplified: Player 1 (button) can bet $10 or check
# Player 2 (big blind) can call or fold
# If both check: showdown, P1 wins 60% (strong hand)
# If P1 bets, P2 calls: P1 wins 60%
# If P1 bets, P2 folds: P1 wins pot ($5)

pot = 20  # Current pot
bet_amount = 10
win_prob_showdown = 0.60

print("Game setup:")
print(f"  Pot: ${pot}")
print(f"  Bet: ${bet_amount}")
print(f"  P1 win % at showdown: {win_prob_showdown*100:.0f}%\n")

# P1 strategy: bet with value, check with air (bluff frequency)
# For P2 to call optimally, P2 needs positive EV from calling
# P2's call EV = -bet_amount * (prob P1 has strong hand) + bet_amount * (1 - prob strong)
#              = bet_amount * (1 - 2*prob_strong)
# If prob_strong = 0.5, EV(call) = 0 (indifferent)

# So P1 should bet value+bluff such that P2 sees 50% strong hands
# If P1 value bets 100% when strong, bluff frequency must maintain 50% overall

# Using pot odds: P2 folds if bluff ratio too low, calls if too high
# Optimal bluff frequency (simplified) = 1 / (1 + pot/bet) ≈ bet / (pot + bet)

optimal_bluff_freq = bet_amount / (pot + bet_amount)
value_bet_freq = 1 - optimal_bluff_freq

print(f"Optimal bluff frequency: {optimal_bluff_freq:.1%}")
print(f"Value bet frequency: {value_bet_freq:.1%}")
print(f"Ratio bluff:value = {optimal_bluff_freq:value_bet_freq:.1f}:1")
print()
print("Interpretation: When P2 calls, {p_strong:.0%} are value bets, {p_bluff:.0%} are bluffs")
print(f"P2 indifferent between calling and folding")

# Example 3: Rock-Paper-Scissors with payoffs
print("\n\n=== Rock-Paper-Scissors Equilibrium ===\n")

# Standard RPS: each beats one, loses to one
# Payoff matrix
payoff_matrix = np.array([
    [0, -1, 1],   # Rock: beats Scissors, loses to Paper
    [1, 0, -1],   # Paper: beats Rock, loses to Scissors
    [-1, 1, 0]    # Scissors: beats Paper, loses to Rock
])

print("Payoff Matrix:")
print("          Rock  Paper  Scissors")
print(f"Rock      {payoff_matrix[0,0]:3}  {payoff_matrix[0,1]:3}    {payoff_matrix[0,2]:3}")
print(f"Paper     {payoff_matrix[1,0]:3}  {payoff_matrix[1,1]:3}    {payoff_matrix[1,2]:3}")
print(f"Scissors  {payoff_matrix[2,0]:3}  {payoff_matrix[2,1]:3}    {payoff_matrix[2,2]:3}\n")

# By symmetry, Nash equilibrium is 1/3 each
eq_strategy = np.array([1/3, 1/3, 1/3])
expected_payoff = eq_strategy @ payoff_matrix @ eq_strategy

print(f"Nash Equilibrium: Rock {1/3:.1%}, Paper {1/3:.1%}, Scissors {1/3:.1%}")
print(f"Expected payoff at equilibrium: {expected_payoff:.3f} (fair game)")

# Example 4: Exploiting non-equilibrium play
print("\n\n=== Exploiting Non-Equilibrium Strategy ===\n")

# Suppose opponent plays: Rock 50%, Paper 30%, Scissors 20%
opponent_strategy = np.array([0.5, 0.3, 0.2])

# Calculate payoff for each of our actions against this strategy
my_payoffs = payoff_matrix @ opponent_strategy

print("Opponent's strategy: Rock 50%, Paper 30%, Scissors 20%")
print(f"Payoff if we play Rock:     {my_payoffs[0]:+.2f}")
print(f"Payoff if we play Paper:    {my_payoffs[1]:+.2f}")
print(f"Payoff if we play Scissors: {my_payoffs[2]:+.2f}")
print()
print(f"Best response: Play {['Rock', 'Paper', 'Scissors'][np.argmax(my_payoffs)]}")
print(f"Expected payoff from best response: {np.max(my_payoffs):+.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Bluffing frequency equilibrium
bet_sizes = np.array([5, 10, 20, 50, 100])
pot_sizes = 100
bluff_freqs = bet_sizes / (pot_sizes + bet_sizes)

axes[0, 0].plot(bet_sizes, bluff_freqs * 100, 'o-', linewidth=2, markersize=8, color='darkblue')
axes[0, 0].fill_between(bet_sizes, bluff_freqs * 100, alpha=0.3)
axes[0, 0].set_xlabel('Bet Size ($)')
axes[0, 0].set_ylabel('Optimal Bluff Frequency (%)')
axes[0, 0].set_title(f'Bluff Frequency vs Bet Size\n(Pot = ${pot_sizes})')
axes[0, 0].grid(alpha=0.3)

# Plot 2: Payoff matrix heatmap (RPS)
im = axes[0, 1].imshow(payoff_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
axes[0, 1].set_xticks([0, 1, 2])
axes[0, 1].set_yticks([0, 1, 2])
axes[0, 1].set_xticklabels(['Rock', 'Paper', 'Scissors'])
axes[0, 1].set_yticklabels(['Rock', 'Paper', 'Scissors'])
axes[0, 1].set_title('Rock-Paper-Scissors Payoff Matrix')
for i in range(3):
    for j in range(3):
        axes[0, 1].text(j, i, f'{payoff_matrix[i, j]:+d}', ha='center', va='center', fontweight='bold')

# Plot 3: Mixing strategy - multiple plays
num_plays = 100
np.random.seed(42)
equilibrium_plays = np.random.choice([0, 1, 2], size=num_plays, p=eq_strategy)
plays_count = np.bincount(equilibrium_plays, minlength=3)

axes[1, 0].bar(['Rock', 'Paper', 'Scissors'], plays_count, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Nash Equilibrium Plays\n({num_plays} games)')
axes[1, 0].grid(alpha=0.3, axis='y')

# Plot 4: Exploitation payoff surface
opponent_rock_probs = np.linspace(0, 1, 20)
opponent_paper_probs = np.linspace(0, 1, 20)
payoff_exploits = np.zeros((20, 20, 3))

for i, r_prob in enumerate(opponent_rock_probs):
    for j, p_prob in enumerate(opponent_paper_probs):
        s_prob = 1 - r_prob - p_prob
        if s_prob >= 0:
            opp_strat = np.array([r_prob, p_prob, s_prob])
            payoff_exploits[i, j] = payoff_matrix @ opp_strat

best_payoffs = np.max(payoff_exploits, axis=2)
im2 = axes[1, 1].contourf(opponent_rock_probs, opponent_paper_probs, best_payoffs.T, levels=15, cmap='RdYlGn')
axes[1, 1].set_xlabel("Opponent's Rock %")
axes[1, 1].set_ylabel("Opponent's Paper %")
axes[1, 1].set_title('Max Payoff vs Opponent Strategy')
plt.colorbar(im2, ax=axes[1, 1], label='Expected Payoff')

plt.tight_layout()
plt.show()
```

## 6. Challenge Round
When does game theory analysis fail?
- Imperfect information with hidden tells (poker psychology)
- Dynamic games where timing matters (strategic multi-round interactions)
- More than 2 players (coalition formation, voting)
- Learning and adaptation (opponent improves over time)
- Behavioral deviations (humans don't play rationally)

## 7. Key References
- [Wikipedia: Game Theory](https://en.wikipedia.org/wiki/Game_theory)
- [Nash Equilibrium Explained](https://en.wikipedia.org/wiki/Nash_equilibrium)
- [Poker Game Theory (GTO)](https://www.pokernews.com/strategy/game-theory-optimal-gto-poker.html)

---
**Status:** Strategic decision framework | **Complements:** Expected Value, Probability, Bluffing Strategy
