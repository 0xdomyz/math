"""
Extracted from: conditional_probability_detailed.md
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb

# Conditional probability: Drawing cards without replacement
def card_prob_conditional(cards_drawn, target_card, deck_size=52, target_count=4):
    """
    P(next draw is target_card | drawn cards_drawn cards)
    """
    remaining = deck_size - cards_drawn
    remaining_targets = target_count - (1 if target_card in range(cards_drawn) else 0)
    
    if remaining <= 0:
        return 0
    return remaining_targets / remaining

# Scenario 1: Conditional probability of drawing ace vs standard draw
deck_size = 52
ace_count = 4

draw_order = range(0, 11)  # Draw 1-10 cards
unconditional_ace = [ace_count / deck_size] * 11  # Always 4/52 if independent
conditional_ace = [ace_count / (deck_size - i) for i in draw_order]

# Scenario 2: Poker equity calculation (simplified)
# P(win | community cards observed)
def monte_carlo_equity(hole_cards, community_cards, n_sim=10000):
    """Simplified: estimate hand win probability"""
    wins = 0
    deck = list(range(52))
    
    # Remove known cards
    for card in hole_cards + community_cards:
        deck.remove(card)
    
    for _ in range(n_sim):
        opponent_cards = np.random.choice(deck, 2, replace=False)
        # Simplified: just compare card values
        if max(hole_cards) > max(opponent_cards):
            wins += 1
    
    return wins / n_sim

# Scenario 3: Bayesian update (card counting)
prior_dealer_bust = 0.28  # Dealer busts ~28% with dealer-up cards
prob_up_card_high = 0.31  # ~16 high cards / 52 deck

posterior_bust_given_high = (prob_up_card_high * prior_dealer_bust) / (
    prob_up_card_high * prior_dealer_bust + 
    (1 - prob_up_card_high) * (1 - prior_dealer_bust)
)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Conditional probability of drawing ace
axes[0, 0].plot(draw_order, unconditional_ace, 'o-', label='Unconditional (if independent)')
axes[0, 0].plot(draw_order, conditional_ace, 's-', label='Conditional (without replacement)')
axes[0, 0].set_title('P(Next Draw is Ace)')
axes[0, 0].set_xlabel('Cards Already Drawn')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Conditional probability after drawing cards
remaining_after_draws = [deck_size - i for i in draw_order]
axes[0, 1].plot(draw_order, remaining_after_draws, 'o-', color='green')
axes[0, 1].set_title('Remaining Cards in Deck')
axes[0, 1].set_xlabel('Cards Drawn')
axes[0, 1].set_ylabel('Cards Remaining')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Bayesian belief update
priors = [0.28]
likelihoods = [prob_up_card_high, 1 - prob_up_card_high]
posteriors = []

for likelihood in likelihoods:
    posterior = (likelihood * priors[0]) / (
        likelihood * priors[0] + (1 - likelihood) * (1 - priors[0])
    )
    posteriors.append(posterior)

axes[1, 0].bar(['Prior (Dealer Bust)', 'Posterior (High Card)'], priors + posteriors[:1], color=['blue','red'])
axes[1, 0].set_title('Bayesian Update: P(Dealer Bust | Up Card)')
axes[1, 0].set_ylabel('Probability')

# Plot 4: Conditional equity by board cards
board_cards = range(0, 6)  # 0=preflop, 1=flop, 5=river
equity_estimates = [0.50, 0.45, 0.48, 0.52, 0.65, 0.95]  # Simplified hand-dependent

axes[1, 1].plot(board_cards, equity_estimates, 'o-', color='purple', linewidth=2)
axes[1, 1].set_title('Conditional Equity (Poker Hand)')
axes[1, 1].set_xlabel('Community Cards Revealed')
axes[1, 1].set_ylabel('Win Probability')
axes[1, 1].set_xticks(board_cards)
axes[1, 1].set_xticklabels(['Preflop', 'Flop', '4th', '5th', 'Turn', 'River'])
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Conditional P(ace | 1 card drawn): {conditional_ace[1]:.4f}")
print(f"Unconditional P(ace): {unconditional_ace[0]:.4f}")
print(f"Posterior P(dealer bust | high card): {posteriors[0]:.4f}")
