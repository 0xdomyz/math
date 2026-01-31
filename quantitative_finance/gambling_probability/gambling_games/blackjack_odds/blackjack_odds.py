"""
Extracted from: blackjack_odds.md
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class BlackjackSimulator:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.deck = self.create_deck()
        self.reset_deck()
    
    def create_deck(self):
        """Create a shoe of cards"""
        deck = []
        for _ in range(self.num_decks):
            for rank in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
                deck.append(rank)
        np.random.shuffle(deck)
        return deck
    
    def reset_deck(self):
        self.deck = self.create_deck()
    
    def card_value(self, card):
        """Return value of card"""
        if card in ['J', 'Q', 'K']:
            return 10
        elif card == 'A':
            return 11
        else:
            return int(card)
    
    def hand_value(self, hand):
        """Calculate best value (accounting for Ace)"""
        total = sum(self.card_value(c) for c in hand)
        num_aces = hand.count('A')
        
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1
        
        return total
    
    def is_soft_hand(self, hand):
        """True if hand has usable Ace"""
        total = sum(self.card_value(c) for c in hand)
        num_aces = hand.count('A')
        
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1
        
        return 'A' in hand and total <= 21
    
    def basic_strategy_action(self, player_hand, dealer_upcard):
        """Return 'hit', 'stand', 'double', or 'split'"""
        p_value = self.hand_value(player_hand)
        
        if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
            # Pair splitting
            if player_hand[0] in ['8', 'A']:
                return 'split'
            if player_hand[0] in ['2', '3', '4', '5', '6', '7']:
                return 'split' if dealer_upcard in ['2', '3', '4', '5', '6', '7'] else 'hit'
        
        if self.is_soft_hand(player_hand):
            # Soft hand strategy
            if p_value >= 19:
                return 'stand'
            elif p_value == 18:
                return 'stand' if dealer_upcard in ['2', '3', '4', '5', '6', '8'] else 'hit'
            else:
                return 'hit'
        else:
            # Hard hand strategy
            if p_value >= 17:
                return 'stand'
            elif p_value == 16:
                return 'stand' if dealer_upcard in ['2', '3', '4', '5', '6'] else 'hit'
            elif p_value == 15:
                return 'stand' if dealer_upcard in ['2', '3', '4', '5', '6'] else 'hit'
            elif p_value == 12:
                return 'stand' if dealer_upcard in ['4', '5', '6'] else 'hit'
            else:
                return 'hit'
    
    def dealer_play(self, hand):
        """Dealer plays until 17+"""
        while self.hand_value(hand) < 17:
            if len(self.deck) < 10:
                self.reset_deck()
            hand.append(self.deck.pop())
        return hand
    
    def play_hand(self, use_basic_strategy=True):
        """Simulate single blackjack hand"""
        if len(self.deck) < 10:
            self.reset_deck()
        
        player_hand = [self.deck.pop(), self.deck.pop()]
        dealer_hand = [self.deck.pop(), self.deck.pop()]
        
        # Player plays
        while True:
            p_value = self.hand_value(player_hand)
            if p_value > 21:
                return 'BUST', p_value, self.hand_value(dealer_hand)
            
            if use_basic_strategy:
                action = self.basic_strategy_action(player_hand, dealer_hand[0])
            else:
                # Random action
                action = np.random.choice(['hit', 'stand'])
            
            if action == 'stand':
                break
            else:
                player_hand.append(self.deck.pop())
        
        # Dealer plays
        dealer_hand = self.dealer_play(dealer_hand)
        
        p_value = self.hand_value(player_hand)
        d_value = self.hand_value(dealer_hand)
        
        if d_value > 21:
            return 'DEALER_BUST', p_value, d_value
        elif p_value > d_value:
            return 'WIN', p_value, d_value
        elif p_value == d_value:
            return 'PUSH', p_value, d_value
        else:
            return 'LOSS', p_value, d_value

# Example 1: Basic strategy vs random play
print("=== Basic Strategy vs Random Play ===\n")

np.random.seed(42)

bj_sim = BlackjackSimulator(num_decks=6)

# Simulate with basic strategy
basic_strategy_results = Counter()
for _ in range(5000):
    result, p_val, d_val = bj_sim.play_hand(use_basic_strategy=True)
    basic_strategy_results[result] += 1

# Simulate with random play
bj_sim = BlackjackSimulator(num_decks=6)
random_play_results = Counter()
for _ in range(5000):
    result, p_val, d_val = bj_sim.play_hand(use_basic_strategy=False)
    random_play_results[result] += 1

print(f"{'Strategy':<25} {'Wins':<10} {'Losses':<10} {'Pushes':<10} {'Win Rate':<10}")
print("-" * 55)

for name, results in [("Basic Strategy", basic_strategy_results), ("Random Play", random_play_results)]:
    wins = results['WIN']
    losses = results['LOSS'] + results['BUST']
    pushes = results['PUSH']
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    print(f"{name:<25} {wins:<10} {losses:<10} {pushes:<10} {win_rate:<10.1%}")

# Example 2: Dealer bust probability by upcard
print("\n\n=== Dealer Bust Probability by Upcard ===\n")

bj_sim = BlackjackSimulator(num_decks=6)
upcard_results = {str(i): Counter() for i in range(2, 11)} | {'A': Counter(), 'T': Counter(), 'J': Counter(), 'Q': Counter(), 'K': Counter()}

for _ in range(5000):
    if len(bj_sim.deck) < 10:
        bj_sim.reset_deck()
    
    dealer_upcard = bj_sim.deck.pop()
    dealer_hand = [dealer_upcard, bj_sim.deck.pop()]
    dealer_hand = bj_sim.dealer_play(dealer_hand)
    
    d_value = bj_sim.hand_value(dealer_hand)
    if d_value > 21:
        upcard_results[dealer_upcard]['BUST'] += 1
    else:
        upcard_results[dealer_upcard]['STAND'] += 1

print(f"{'Dealer Upcard':<18} {'Bust Rate':<15} {'Stand Rate':<15}")
print("-" * 50)

for upcard in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']:
    total = sum(upcard_results[upcard].values())
    bust_rate = upcard_results[upcard]['BUST'] / total if total > 0 else 0
    stand_rate = upcard_results[upcard]['STAND'] / total if total > 0 else 0
    print(f"{upcard:<18} {bust_rate:<15.1%} {stand_rate:<15.1%}")

# Example 3: Player hand outcomes
print("\n\n=== Player Hand Outcome Distribution ===\n")

bj_sim = BlackjackSimulator(num_decks=6)
hand_outcomes = {}

for _ in range(10000):
    result, p_val, d_val = bj_sim.play_hand(use_basic_strategy=True)
    hand_key = f"{p_val}"
    if hand_key not in hand_outcomes:
        hand_outcomes[hand_key] = Counter()
    hand_outcomes[hand_key][result] += 1

print(f"{'Hand Value':<15} {'Wins':<12} {'Losses':<12} {'Pushes':<12}")
print("-" * 51)

for val in sorted([int(v) for v in hand_outcomes.keys()]):
    val_str = str(val)
    outcomes = hand_outcomes[val_str]
    total = sum(outcomes.values())
    wins = outcomes.get('WIN', 0) + outcomes.get('DEALER_BUST', 0)
    losses = outcomes.get('LOSS', 0) + outcomes.get('BUST', 0)
    pushes = outcomes.get('PUSH', 0)
    win_pct = wins / total if total > 0 else 0
    print(f"{val:<15} {win_pct:<12.1%} {1-win_pct-pushes/total:<12.1%} {pushes/total:<12.1%}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Basic strategy vs random win rates
strategies = ['Basic Strategy', 'Random Play']
win_data = [
    basic_strategy_results['WIN'] / sum(basic_strategy_results.values()),
    random_play_results['WIN'] / sum(random_play_results.values())
]
loss_data = [
    (basic_strategy_results['LOSS'] + basic_strategy_results['BUST']) / sum(basic_strategy_results.values()),
    (random_play_results['LOSS'] + random_play_results['BUST']) / sum(random_play_results.values())
]

x_pos = np.arange(len(strategies))
axes[0, 0].bar(x_pos - 0.2, win_data, 0.4, label='Wins', color='green', alpha=0.7)
axes[0, 0].bar(x_pos + 0.2, loss_data, 0.4, label='Losses', color='red', alpha=0.7)
axes[0, 0].set_ylabel('Win/Loss Rate')
axes[0, 0].set_title('Basic Strategy vs Random Play (5000 hands)')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(strategies)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')

# Plot 2: Dealer bust by upcard
upcards = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'A']
bust_rates = []
for upcard in upcards:
    total = sum(upcard_results[upcard].values())
    bust_rate = upcard_results[upcard]['BUST'] / total if total > 0 else 0
    bust_rates.append(bust_rate * 100)

colors_bust = ['green' if rate > 30 else 'red' for rate in bust_rates]
axes[0, 1].bar(upcards, bust_rates, color=colors_bust, alpha=0.7)
axes[0, 1].axhline(30, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0, 1].set_ylabel('Bust Rate (%)')
axes[0, 1].set_xlabel('Dealer Upcard')
axes[0, 1].set_title('Dealer Bust Probability by Upcard')
axes[0, 1].grid(alpha=0.3, axis='y')

# Plot 3: Hand value distribution
hand_values = sorted([int(v) for v in hand_outcomes.keys()])
win_rates = []
for val in hand_values:
    outcomes = hand_outcomes[str(val)]
    total = sum(outcomes.values())
    wins = outcomes.get('WIN', 0) + outcomes.get('DEALER_BUST', 0)
    win_rates.append(wins / total if total > 0 else 0)

axes[1, 0].plot(hand_values, np.array(win_rates)*100, 'o-', linewidth=2, markersize=6)
axes[1, 0].axhline(50, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Player Hand Value')
axes[1, 0].set_ylabel('Win Rate (%)')
axes[1, 0].set_title('Win Rate by Final Hand Value')
axes[1, 0].grid(alpha=0.3)

# Plot 4: House edge over time
np.random.seed(42)
bj_sim = BlackjackSimulator(num_decks=6)
cumulative_win = 0
cumulative_loss = 0
edges = []

for hand_num in range(1, 1001):
    result, p_val, d_val = bj_sim.play_hand(use_basic_strategy=True)
    if result == 'WIN' or result == 'DEALER_BUST':
        cumulative_win += 1
    elif result == 'LOSS' or result == 'BUST':
        cumulative_loss += 1
    
    if hand_num % 10 == 0:
        if cumulative_win + cumulative_loss > 0:
            win_pct = cumulative_win / (cumulative_win + cumulative_loss)
            house_edge = (1 - win_pct) * 100
            edges.append(house_edge)

hand_nums = np.arange(10, 1001, 10)
axes[1, 1].plot(hand_nums, edges, linewidth=2, color='darkblue')
axes[1, 1].axhline(0.5, color='green', linestyle='--', linewidth=2, label='Theoretical 0.5%')
axes[1, 1].fill_between(hand_nums, edges, 0.5, alpha=0.2)
axes[1, 1].set_xlabel('Number of Hands')
axes[1, 1].set_ylabel('House Edge (%)')
axes[1, 1].set_title('House Edge Convergence (Basic Strategy)')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
