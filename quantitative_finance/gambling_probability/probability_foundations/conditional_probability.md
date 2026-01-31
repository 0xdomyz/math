# Conditional Probability: Sequential Outcomes & Dependence

## 1. Concept Skeleton
**Definition:** Probability of event A occurring given that event B has already occurred; denoted P(A|B)  
**Purpose:** Analyze dependent events in gambling (card draws, sequential bets); essential for blackjack card counting, poker position adjustments  
**Prerequisites:** Basic probability, Bayes' theorem, joint and marginal probabilities

## 2. Comparative Framing
| Concept | Conditional P(A\|B) | Joint P(A,B) | Marginal P(A) | Independence P(A\|B)=P(A) |
|---------|-------------------|-----------|-----------|---------------------------|
| **Definition** | Probability of A given B occurred | Probability of both A and B | Probability of A alone | No relationship between events |
| **Formula** | P(A\|B) = P(A,B) / P(B) | P(A,B) = P(A\|B)×P(B) | Σ_B P(A,B) | P(A\|B) = P(A) |
| **Example** | P(deal ace \| deck missing 2 aces) | P(ace & king drawn) | P(any ace dealt) | P(heads \| fair coin) = 0.5 |
| **Dependency** | Depends on B | Shows relationship | No condition | No dependency |
| **Gambling** | Card counting, position adjustments | Pot odds vs draw odds | General game odds | Fair coin flips, dice |

## 3. Examples + Counterexamples

**Simple Example:**  
Deck: 52 cards, 4 aces. First card drawn is not an ace. P(second card is ace | first wasn't) = 4/51 ≠ 4/52

**Failure Case:**  
Roulette: Gambler's fallacy: Black hasn't hit 10 spins → P(black | 10 reds) ≠ P(black). True: P(black | history) = 18/37 (independent)

**Edge Case:**  
Two cards dealt without replacement: P(A1=ace | A2=ace) ≠ P(A1=ace). Actually symmetric for exchangeable draws.

## 4. Layer Breakdown
```
Conditional Probability Framework:
├─ I. DEFINITION & FORMULA:
│   ├─ Conditional Probability:
│   │   ├─ Definition: P(A|B) = probability of A given B occurred
│   │   ├─ Formula: P(A|B) = P(A ∩ B) / P(B)
│   │   ├─ Requirement: P(B) > 0 (can't condition on impossible event)
│   │   └─ Interpretation: Rescale probability space to "universe where B happened"
│   ├─ Joint Probability:
│   │   ├─ P(A,B) = P(A|B) × P(B) = P(B|A) × P(A)
│   │   ├─ Multiplication Rule: Chain multiple events
│   │   └─ Example: P(first 2 cards both aces) = (4/52) × (3/51)
│   ├─ Marginal Probability (Law of Total Probability):
│   │   ├─ P(A) = Σ_B P(A|B) × P(B)
│   │   ├─ Decompose over all possible conditions
│   │   └─ Example: P(blackjack) = P(blackjack | deck balanced) + ...
│   └─ Conditional Independence:
│       ├─ A and C independent given B: P(A|B,C) = P(A|B)
│       └─ Conditioning can make independent events dependent
├─ II. BAYES' THEOREM (Posterior Probability):
│   ├─ Forward Formula:
│   │   ├─ P(A|B) = P(B|A) × P(A) / P(B)
│   │   ├─ P(B) = Σ_A P(B|A) × P(A) (law of total probability)
│   │   └─ Use: Update beliefs with new evidence
│   ├─ Poker Application:
│   │   ├─ Prior: Opponent's likely hand before community cards
│   │   ├─ Likelihood: P(community | opponent's hand)
│   │   ├─ Posterior: Updated hand probability
│   │   └─ Decision: Fold/call based on posterior
│   └─ Odds Form (more intuitive):
│       ├─ Posterior Odds = Likelihood Ratio × Prior Odds
│       ├─ Posterior Odds = P(A|B) / P(A^c|B)
│       └─ LR = P(B|A) / P(B|A^c)
├─ III. DEPENDENCE & INDEPENDENCE:
│   ├─ Independent Events (Memoryless):
│   │   ├─ Definition: P(A|B) = P(A) (prior knowledge of B irrelevant)
│   │   ├─ Equivalent: P(A,B) = P(A) × P(B)
│   │   ├─ Examples: Coin flips, independent bets, roulette spins
│   │   └─ Gambler's Fallacy: Mistaking independent for dependent
│   ├─ Dependent Events (With Condition):
│   │   ├─ Definition: P(A|B) ≠ P(A)
│   │   ├─ Causes:
│   │   │   ├─ Sampling Without Replacement: Cards, lottery draws
│   │   │   ├─ Common Cause: Both events share parent factor
│   │   │   └─ Sequential: Events in time series
│   │   └─ Examples: Card draws, multiround games
│   ├─ Correlation vs Causation:
│   │   ├─ Dependence: Events co-occur in probability
│   │   ├─ Causation: One event causes the other
│   │   └─ Warning: Dependence ≠ Causation (confounders)
│   └─ Conditional Independence:
│       ├─ A ⊥ C | B means: knowing B makes A,C independent
│       ├─ Example: Shuffled deck position independent of previous positions
│       └─ Application: Simplify complex probability models
├─ IV. APPLICATIONS IN GAMBLING:
│   ├─ Card Counting (Blackjack):
│   │   ├─ Objective: Estimate P(high card | cards dealt)
│   │   ├─ Method: Track dealt cards, adjust deck composition
│   │   ├─ Strategy: Increase bet when P(blackjack) high
│   │   └─ Risk: Casino detection, counting cards illegal in many places
│   ├─ Poker Position & Information:
│   │   ├─ Early Position (few cards seen):
│   │   │   ├─ Limited information on other hands
│   │   │   ├─ Rely on P(hand | my cards), less adjustment needed
│   │   │   └─ Strategy: Tighter range
│   │   ├─ Late Position (many cards seen):
│   │   │   ├─ More information: opponents' actions, community cards
│   │   │   ├─ Update P(opponent hand | observed actions)
│   │   │   └─ Strategy: Wider range, exploitative
│   │   └─ Equity Calculation:
│   │       ├─ P(win | my hand, visible cards) given range of opponents
│   │       ├─ Pot odds: Compare to P(improved hand | draw odds)
│   │       └─ Decision: Call if EV positive
│   ├─ Roulette & True Independence:
│   │   ├─ Key: Each spin independent
│   │   ├─ P(red next | 10 reds in a row) = 18/37 (European)
│   │   ├─ Gambler's Fallacy: False belief "black due" (wrong)
│   │   └─ Correct: Red and black equally likely every spin
│   ├─ Craps & Sequential Outcomes:
│   │   ├─ Pass Line Bet:
│   │   │   ├─ Come-Out Roll: P(7 or 11 | first roll) = 8/36 (win)
│   │   │   ├─ Point Established: P(point | 4,5,6,8,9,10)
│   │   │   ├─ Final: P(make point | point established)
│   │   │   └─ Calculation: Product of sequential probabilities
│   │   └─ Strategy: Odds bets reduce house edge
│   └─ Lottery & Large Numbers:
│       ├─ Prior: P(win) = 1 in millions
│       ├─ Observing: No win after 100 tickets
│       ├─ Posterior: P(win | no prior wins) ≈ same (independent draws)
│       └─ Insight: Past non-wins don't increase future win probability
├─ V. BAYES FOR UPDATING BELIEFS:
│   ├─ Card Counting Example:
│   │   ├─ Prior: Deck composition (52 cards, 4 aces)
│   │   ├─ Observe: 3 cards dealt (no aces)
│   │   ├─ Posterior: P(next card is ace | 3 dealt, none aces) = 4/49
│   │   └─ Update: P(ace) changes continuously as cards dealt
│   ├─ Poker Hand Estimation:
│   │   ├─ Prior P(opponent has flush): 5% (before seeing action)
│   │   ├─ Observe: Opponent raises pre-flop on button
│   │   ├─ Update P(flush | aggressive pre-flop action): 7% (slightly higher)
│   │   └─ Posterior: Adjust by all observed information
│   └─ Hit Rate Calculation:
│       ├─ Prior: Typical hit rate 45%
│       ├─ Observe: First 100 trials, 52% hit
│       ├─ Posterior: Bayesian credible interval on true rate
│       └─ Question: Luck or skill difference?
├─ VI. MISCONCEPTIONS & BIASES:
│   ├─ Gambler's Fallacy:
│   │   ├─ False Belief: Past failures increase future success chance
│   │   ├─ Wrong: P(heads | 10 tails) ≠ 0.6 (still 0.5)
│   │   ├─ Reality: Independent events unaffected by history
│   │   └─ Cost: Leads to chasing losses, bad betting
│   ├─ Correlation Confusion:
│   │   ├─ Observing: Red hit 5 times consecutively
│   │   ├─ False Conclusion: "Red is running hot" (trend will continue)
│   │   ├─ Truth: P(red | sequence of reds) = same as always
│   │   └─ Conditional: Only true if roulette wheel is biased
│   ├─ Hot-Hand Fallacy:
│   │   ├─ Belief: Winning streak will continue
│   │   ├─ Reality: If underlying win probability unchanged, no edge
│   │   └─ Exception: If skill/conditions improving → might be true
│   └─ Confirmation Bias:
│       ├─ Remember: Times we got "due" and won (lucky coincidence)
│       ├─ Forget: Times we got "due" and lost
│       └─ Distortion: Memory bias makes independence feel like dependence
└─ VII. CALCULATIONS & FORMULAS:
    ├─ Chain Rule (Multiplication for Multiple Events):
    │   ├─ P(A,B,C) = P(A) × P(B|A) × P(C|A,B)
    │   ├─ Poker: P(dealt AA) = P(A1=A) × P(A2=A|A1=A)
    │   │        = (4/52) × (3/51)
    │   └─ Generalizes to n events
    ├─ Odds Ratios:
    │   ├─ Odds of A = P(A) / P(A^c)
    │   ├─ Posterior Odds = Likelihood Ratio × Prior Odds
    │   ├─ LR = P(evidence | A) / P(evidence | not A)
    │   └─ Intuitive for gambling: Compare alternative hands
    ├─ Sensitivity & Specificity (Medical Test Analogy):
    │   ├─ Sensitivity: P(test+ | disease) = true positive rate
    │   ├─ Specificity: P(test- | no disease) = true negative rate
    │   ├─ Bayes: Convert to P(disease | test+) = posterior
    │   └─ Gambling: Condition "strong hand signal" → true hand strength
    └─ Contingency Table (Organize Information):
        ├─ Rows: Event A (yes/no)
        ├─ Columns: Event B (yes/no)
        ├─ Cells: Joint counts
        └─ Margins: Marginal totals for P(A), P(B), P(A|B)
```

**Interaction:** Observe event B → Update P(A) using Bayes → Revise beliefs → Adjust betting/strategy

## 5. Mini-Project
Implement blackjack card counting simulation with conditional probability updates:
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# ============================================================================
# DECK REPRESENTATION & TRACKING
# ============================================================================

print("="*70)
print("CONDITIONAL PROBABILITY: CARD COUNTING IN BLACKJACK")
print("="*70)

class Deck:
    """Standard 52-card deck with composition tracking."""
    
    RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
    CARDS_PER_RANK = 4
    
    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.reset()
    
    def reset(self):
        """Reset deck to full composition."""
        self.cards = {}
        for rank in self.RANKS:
            self.cards[rank] = self.CARDS_PER_RANK * self.num_decks
        self.total_remaining = 52 * self.num_decks
    
    def draw_card(self):
        """Remove and return random card; update composition."""
        # Probability of each card type
        probabilities = np.array([self.cards[rank] for rank in self.RANKS])
        probabilities = probabilities / probabilities.sum()
        
        # Draw card
        card_idx = np.random.choice(len(self.RANKS), p=probabilities)
        card = self.RANKS[card_idx]
        
        # Update deck
        self.cards[card] -= 1
        self.total_remaining -= 1
        
        return card
    
    def peek_composition(self):
        """Show current deck composition (normalized probabilities)."""
        if self.total_remaining == 0:
            return {rank: 0 for rank in self.RANKS}
        
        return {rank: self.cards[rank] / self.total_remaining 
               for rank in self.RANKS}

# ============================================================================
# CONDITIONAL PROBABILITY CALCULATIONS
# ============================================================================

print("\n" + "="*70)
print("1. CONDITIONAL PROBABILITY OF CARD TYPES")
print("="*70)

deck = Deck(num_decks=1)
print(f"\nInitial Deck (52 cards):")
print(f"   P(ace) = {deck.cards['A']} / {deck.total_remaining} = {deck.cards['A']/deck.total_remaining:.4f}")
print(f"   P(10-value) = {deck.cards['10'] + deck.cards['J'] + deck.cards['Q'] + deck.cards['K']} / {deck.total_remaining} = {(deck.cards['10'] + deck.cards['J'] + deck.cards['Q'] + deck.cards['K'])/deck.total_remaining:.4f}")
print(f"   P(low card 2-6) = {sum([deck.cards[r] for r in ['2','3','4','5','6']])} / {deck.total_remaining} = {sum([deck.cards[r] for r in ['2','3','4','5','6']])/deck.total_remaining:.4f}")

# Simulate drawing cards and updating conditional probabilities
print("\n" + "-"*70)
print("Drawing cards sequentially (showing P(next card | cards drawn)):")
print("-"*70)

drawn_cards = []
composition_history = []

for draw_num in range(15):
    card = deck.draw_card()
    drawn_cards.append(card)
    
    # Current conditional probabilities
    current_comp = deck.peek_composition()
    composition_history.append(current_comp)
    
    p_ace_given = deck.cards['A'] / deck.total_remaining if deck.total_remaining > 0 else 0
    p_ten_given = (deck.cards['10'] + deck.cards['J'] + deck.cards['Q'] + deck.cards['K']) / deck.total_remaining if deck.total_remaining > 0 else 0
    p_low_given = sum([deck.cards[r] for r in ['2','3','4','5','6']]) / deck.total_remaining if deck.total_remaining > 0 else 0
    
    print(f"\nDraw {draw_num + 1}: Card = {card}")
    print(f"   P(next is ace | cards drawn) = {p_ace_given:.4f}")
    print(f"   P(next is 10-value | cards drawn) = {p_ten_given:.4f}")
    print(f"   P(next is low 2-6 | cards drawn) = {p_low_given:.4f}")
    print(f"   Cards remaining: {deck.total_remaining}")

# ============================================================================
# BAYES' THEOREM APPLICATION: UPDATE BELIEFS ABOUT DECK "RICHNESS"
# ============================================================================

print("\n" + "="*70)
print("2. BAYES THEOREM: UPDATING BELIEF ABOUT DECK COMPOSITION")
print("="*70)

def calculate_likelihood_of_observation(drawn_cards, true_deck_state):
    """
    P(observed cards | deck state)
    Simplified: assume drawn cards are from known subset
    """
    likelihood = 1.0
    temp_deck = true_deck_state.copy()
    
    for card in drawn_cards:
        if temp_deck[card] > 0:
            likelihood *= (temp_deck[card] / sum(temp_deck.values()))
            temp_deck[card] -= 1
        else:
            likelihood = 0  # Impossible observation
            break
    
    return likelihood

# Prior: Assume deck could be "rich in 10s" or "rich in low cards" or "balanced"
# Hypothesis 1: Deck rich in 10s (more 10s removed from average)
# Hypothesis 2: Deck rich in low cards (fewer 10s removed)
# Hypothesis 3: Deck balanced (average removal rate)

print(f"\nObserved Cards: {drawn_cards}")
print(f"   Count: {Counter(drawn_cards)}")

# Calculate posterior probability for "10-rich" vs "low-rich" deck
# Simplified: count actual high/low cards

high_count = sum([1 for c in drawn_cards if c in ['10','J','Q','K']])
low_count = sum([1 for c in drawn_cards if c in ['2','3','4','5','6']])

print(f"\nCard Summary:")
print(f"   High (10-K): {high_count} cards")
print(f"   Low (2-6): {low_count} cards")
print(f"   Ratio (high/low): {high_count/max(low_count,1):.2f}")

# Bayes update: if we see more lows, remaining deck is richer in highs
posterior_rich_in_highs = (low_count - high_count) / 15  # Shift based on observation

print(f"\nPosterior Belief (updated by observation):")
print(f"   P(deck rich in 10s | observed removal) = {posterior_rich_in_highs:.3f}")
print(f"   Interpretation: Remove more lows → deck becomes richer in highs")

# ============================================================================
# CARD COUNTING STRATEGY SIMULATION
# ============================================================================

print("\n" + "="*70)
print("3. CARD COUNTING STRATEGY: HI-LO SYSTEM")
print("="*70)

def hi_lo_count(cards):
    """
    Hi-Lo card counting system (simplistic).
    Count: +1 for 2-6, 0 for 7-9, -1 for 10-A
    Positive count → more low cards removed → deck rich in highs → bet high
    """
    running_count = 0
    for card in cards:
        if card in ['2','3','4','5','6']:
            running_count += 1
        elif card in ['7','8','9']:
            running_count += 0
        else:  # 10, J, Q, K, A
            running_count -= 1
    return running_count

# Simulate multiple shoe-games
n_shoes = 100
shoes_results = []

for shoe_num in range(n_shoes):
    deck = Deck(num_decks=1)
    shoe_cards = []
    counting_progression = []
    
    # Play out most of deck
    for _ in range(40):  # Draw 40 cards per shoe
        card = deck.draw_card()
        shoe_cards.append(card)
        running_count = hi_lo_count(shoe_cards)
        counting_progression.append(running_count)
    
    final_count = counting_progression[-1]
    p_high_given_count = (final_count > 0)  # Simplified: if count positive, deck rich in highs
    
    shoes_results.append({
        'running_count': final_count,
        'p_high_given': p_high_given_count,
        'high_cards_actual': sum([1 for c in shoe_cards if c in ['10','J','Q','K']])
    })

avg_count = np.mean([s['running_count'] for s in shoes_results])
high_cards_when_positive = np.mean([s['high_cards_actual'] for s in shoes_results 
                                   if s['running_count'] > 0])
high_cards_when_negative = np.mean([s['high_cards_actual'] for s in shoes_results 
                                   if s['running_count'] <= 0])

print(f"\nHi-Lo Card Counting Over {n_shoes} Shoes (40 cards each):")
print(f"   Average Running Count: {avg_count:.2f}")
print(f"   When Count Positive: Avg High Cards = {high_cards_when_positive:.1f}")
print(f"   When Count Negative: Avg High Cards = {high_cards_when_negative:.1f}")
print(f"   Implication: Positive count → more high cards in deck (P(high|+count) high)")

# ============================================================================
# CONDITIONAL PROBABILITY FOR HAND COMPARISONS
# ============================================================================

print("\n" + "="*70)
print("4. CONDITIONAL PROBABILITY: DEALER BUST GIVEN UP CARD")
print("="*70)

# Simplified: P(dealer busts | dealer up card)
# Dealer rules: hit on 16 or less, stand on 17+

def estimate_dealer_bust_prob(dealer_up_card, deck_composition):
    """
    Estimate P(dealer busts | dealer up card, deck composition)
    Simplified calculation.
    """
    bust_prob = 0.0
    
    # If dealer has 17+, won't bust
    if dealer_up_card in ['7','8','9','10','J','Q','K','A']:
        if dealer_up_card == '7':
            # 7+9 = 16, need to hit. May bust on 6+ (K through A)
            bust_prob = 0.15  # Rough estimate
        elif dealer_up_card in ['8','9']:
            bust_prob = 0.30
        else:  # 10-A
            bust_prob = 0.40  # Higher bust chance with strong up card
    else:  # 2-6 (weak up cards, more likely to bust)
        bust_prob = 0.35 + (6 - int(dealer_up_card)) * 0.08
    
    return bust_prob

# Example
print(f"\nDealer Up Card Analysis:")
for up_card in ['2','5','7','10','K']:
    bust_prob = estimate_dealer_bust_prob(up_card, {})
    print(f"   P(dealer busts | up card = {up_card}) = {bust_prob:.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Card removal and conditional probability change
ax1 = axes[0, 0]
deck_test = Deck(num_decks=1)
draws = []
p_ace_progression = []
p_ten_progression = []

for i in range(30):
    draws.append(deck_test.draw_card())
    p_ace = deck_test.cards['A'] / deck_test.total_remaining if deck_test.total_remaining > 0 else 0
    p_ten = (deck_test.cards['10'] + deck_test.cards['J'] + deck_test.cards['Q'] + deck_test.cards['K']) / deck_test.total_remaining if deck_test.total_remaining > 0 else 0
    p_ace_progression.append(p_ace)
    p_ten_progression.append(p_ten)

cards_drawn = range(len(draws))
ax1.plot(cards_drawn, p_ace_progression, 'o-', linewidth=2, label='P(Ace | drawn)', color='red')
ax1.plot(cards_drawn, p_ten_progression, 's-', linewidth=2, label='P(10-value | drawn)', color='blue')
ax1.axhline(y=4/52, color='red', linestyle='--', alpha=0.5, label='Initial P(Ace)')
ax1.axhline(y=16/52, color='blue', linestyle='--', alpha=0.5, label='Initial P(10-value)')
ax1.set_xlabel('Cards Drawn')
ax1.set_ylabel('Probability')
ax1.set_title('Conditional Probability Changes with Card Removal')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Hi-Lo count distribution
ax2 = axes[0, 1]
counts = [s['running_count'] for s in shoes_results]
ax2.hist(counts, bins=15, alpha=0.7, color='purple', edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral Count')
ax2.set_xlabel('Running Count (after 40 cards)')
ax2.set_ylabel('Frequency')
ax2.set_title('Hi-Lo Card Counting Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Bayes Update - High Cards vs Count Sign
ax3 = axes[1, 0]
counts_positive = [s['running_count'] for s in shoes_results if s['running_count'] > 0]
counts_negative = [s['running_count'] for s in shoes_results if s['running_count'] <= 0]
high_pos = [s['high_cards_actual'] for s in shoes_results if s['running_count'] > 0]
high_neg = [s['high_cards_actual'] for s in shoes_results if s['running_count'] <= 0]

bp = ax3.boxplot([high_neg, high_pos], labels=['Count ≤ 0\n(Deck Poor)', 'Count > 0\n(Deck Rich)'],
                 patch_artist=True)
for patch, color in zip(bp['boxes'], ['salmon', 'lightgreen']):
    patch.set_facecolor(color)
ax3.set_ylabel('High Cards Drawn (observed)')
ax3.set_title('Bayes: High Card Count vs Card Composition')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Dealer Bust Probability by Up Card
ax4 = axes[1, 1]
up_cards_list = ['2','3','4','5','6','7','8','9','10','K']
bust_probs = [estimate_dealer_bust_prob(card, {}) for card in up_cards_list]
colors_bust = ['green' if p > 0.35 else 'orange' for p in bust_probs]
bars = ax4.bar(up_cards_list, bust_probs, color=colors_bust, alpha=0.7, edgecolor='black')
ax4.set_ylabel('P(Dealer Busts | Up Card)')
ax4.set_xlabel('Dealer Up Card')
ax4.set_title('Conditional Probability: Dealer Bust Risk')
ax4.grid(True, alpha=0.3, axis='y')

# Add threshold line
ax4.axhline(y=0.35, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Pivot ~0.35')
ax4.legend()

plt.tight_layout()
plt.savefig('conditional_probability_gambling.png', dpi=100, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved: conditional_probability_gambling.png")
plt.show()
```

## 6. Challenge Round
When does conditional probability analysis fail in gambling?
- Non-stationarity: Assumed probabilities change over time (skill develops, psychological state shifts)
- Correlation confusion: Observing correlation between events → falsely assume dependence
- Information asymmetry: Dealer knows hidden information → conditional probs update when revealed
- Model error: Estimate P(A|B) incorrectly → Wrong betting decisions despite correct framework
- Cognitive bias: Gambler's fallacy overrides conditional logic even when understood

## 7. Key References
- [Feller (1968), "An Introduction to Probability Theory"](https://www.amazon.com/Introduction-Probability-Theory-Applications-Vol/dp/0471257087) - Foundational text on conditional probability
- [Thorp (1962), "Beat the Dealer"](https://en.wikipedia.org/wiki/Edward_Thorp) - Card counting applications
- [Bayes' Theorem Explanation](https://en.wikipedia.org/wiki/Bayes%27_theorem) - Posterior probability updating

---
**Status:** Foundation for sequential gambling analysis | **Complements:** Independence, Law of Large Numbers | **Enables:** Card counting, Bayes strategy