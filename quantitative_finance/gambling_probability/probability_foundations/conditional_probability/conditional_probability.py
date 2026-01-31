"""
Extracted from: conditional_probability.md
"""

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
