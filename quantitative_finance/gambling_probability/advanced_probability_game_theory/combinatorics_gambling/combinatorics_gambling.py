"""
Extracted from: combinatorics_gambling.md
"""

import numpy as np
from math import comb, factorial
import itertools
from collections import Counter

# Poker hand classification
def classify_hand(cards):
    """Classify 5-card poker hand"""
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]
    
    rank_order = {'2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':11, 'Q':12, 'K':13, 'A':14}
    rank_nums = sorted([rank_order[r] for r in ranks])
    
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    is_flush = len(suit_counts) == 1
    is_straight = (rank_nums[-1] - rank_nums[0] == 4 and len(set(rank_nums)) == 5) or set(rank_nums) == {2, 3, 4, 5, 14}
    
    counts = sorted(rank_counts.values(), reverse=True)
    
    if is_straight and is_flush and rank_nums[-1] == 14 and rank_nums[0] == 10:
        return "Royal Flush", 10
    elif is_straight and is_flush:
        return "Straight Flush", 9
    elif counts == [4, 1]:
        return "Four of a Kind", 8
    elif counts == [3, 2]:
        return "Full House", 7
    elif is_flush:
        return "Flush", 6
    elif is_straight:
        return "Straight", 5
    elif counts == [3, 1, 1]:
        return "Three of a Kind", 4
    elif counts == [2, 2, 1]:
        return "Two Pair", 3
    elif counts == [2, 1, 1, 1]:
        return "One Pair", 2
    else:
        return "High Card", 1

# Example 1: Calculate exact poker hand probabilities
print("=== Exact Poker Hand Probabilities ===\n")

# Theoretical combinations (using combinatorics)
hand_counts = {
    "Royal Flush": 4,
    "Straight Flush": 36,
    "Four of a Kind": 624,
    "Full House": 3744,
    "Flush": 5108,
    "Straight": 10200,
    "Three of a Kind": 54912,
    "Two Pair": 123552,
    "One Pair": 1098240,
    "High Card": 1302540,
}

total_hands = comb(52, 5)

print(f"Total 5-card hands: {total_hands:,}\n")
print(f"{'Hand Type':<20} {'Count':<15} {'Probability':<15} {'Odds':<15}")
print("-" * 65)

cumulative_better = 0
for hand_type in ["Royal Flush", "Straight Flush", "Four of a Kind", "Full House", "Flush", 
                  "Straight", "Three of a Kind", "Two Pair", "One Pair", "High Card"]:
    count = hand_counts[hand_type]
    prob = count / total_hands
    odds_against = (total_hands - count) / count
    print(f"{hand_type:<20} {count:<15,} {prob:<15.6f} 1 in {odds_against:<14.0f}")
    cumulative_better += count

# Example 2: Texas Hold'em outs calculation
print("\n\n=== Texas Hold'em Outs & Pot Odds ===\n")

scenarios = [
    {"name": "Flush draw", "outs": 9, "cards_seen": 5},
    {"name": "Straight draw", "outs": 8, "cards_seen": 5},
    {"name": "Pair to trips", "outs": 2, "cards_seen": 5},
    {"name": "Open-ended straight draw", "outs": 8, "cards_seen": 5},
    {"name": "Flush + pair", "outs": 12, "cards_seen": 5},  # Outs w/redraw value
]

print(f"{'Scenario':<25} {'Outs':<8} {'Remaining':<12} {'Prob Hit':<15} {'Odds Against':<15}")
print("-" * 75)

for scenario in scenarios:
    remaining = 52 - scenario["cards_seen"] - 2  # Remove hole cards
    remaining_turn = remaining  # Cards remaining for next card
    prob_hit = scenario["outs"] / remaining_turn
    odds_against = (remaining_turn - scenario["outs"]) / scenario["outs"]
    
    print(f"{scenario['name']:<25} {scenario['outs']:<8} {remaining_turn:<12} {prob_hit:<15.1%} 1:{odds_against:<14.1f}")

# Example 3: Pot odds vs odds required
print("\n\n=== Pot Odds Comparison ===\n")

pot_sizes = [10, 20, 50, 100]
bet_amounts = [10, 10, 10, 10]
out_counts = [9, 9, 9, 9]  # All flush draws

print(f"{'Pot':<10} {'Bet':<10} {'Pot Odds':<15} {'Outs':<8} {'Hit Prob':<12} {'EV':<12}")
print("-" * 67)

for pot, bet, outs in zip(pot_sizes, bet_amounts, out_counts):
    pot_odds = (pot + bet) / bet
    remaining = 47  # Example: flop to river
    hit_prob = outs / remaining
    required_prob = 1 / pot_odds
    ev = (hit_prob * (pot + bet + bet)) - (1 - hit_prob) * bet - bet
    
    is_call = "CALL" if hit_prob > required_prob else "FOLD"
    print(f"${pot:<9} ${bet:<9} {pot_odds:<15.2f}:1 {outs:<8} {hit_prob:<12.1%} {ev:+7.1f} {is_call}")

# Example 4: Hand equity calculation
print("\n\n=== Two-Hand Equity (Heads-Up) ===\n")

# Example: Pair vs Overcards
pair_hand = ['7s', '7h']  # Pocket sevens
opp_hand = ['Ks', 'Qh']   # King-Queen

# Simulate outcomes
pair_wins = 0
opp_wins = 0
ties = 0

deck = [f"{r}{s}" for r in '23456789TJQKA' for s in 'shdc']
for card1 in deck:
    if card1 not in pair_hand + opp_hand:
        for card2 in deck:
            if card2 not in pair_hand + opp_hand + [card1]:
                for card3 in deck:
                    if card3 not in pair_hand + opp_hand + [card1, card2]:
                        # Complete hands
                        pair_complete = pair_hand + [card1, card2, card3]
                        opp_complete = opp_hand + [card1, card2, card3]
                        
                        pair_class, pair_rank = classify_hand(pair_complete)
                        opp_class, opp_rank = classify_hand(opp_complete)
                        
                        if pair_rank > opp_rank:
                            pair_wins += 1
                        elif opp_rank > pair_rank:
                            opp_wins += 1
                        else:
                            ties += 1

total_outcomes = pair_wins + opp_wins + ties
pair_equity = pair_wins / total_outcomes
opp_equity = opp_wins / total_outcomes
tie_equity = ties / total_outcomes

print(f"Hand 1 (pair): 7s 7h")
print(f"Hand 2 (over-cards): Ks Qh\n")
print(f"Total outcomes: {total_outcomes:,}\n")
print(f"Pair wins: {pair_wins:,} ({pair_equity:.1%})")
print(f"Opponent wins: {opp_wins:,} ({opp_equity:.1%})")
print(f"Ties: {ties:,} ({tie_equity:.1%})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Hand frequency
hand_names = list(hand_counts.keys())
hand_counts_list = [hand_counts[h] for h in hand_names]
colors = ['gold', 'silver', '#CD7F32', 'purple', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red']

axes[0, 0].barh(hand_names, hand_counts_list, color=colors, alpha=0.7)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('Frequency (log scale)')
axes[0, 0].set_title('Poker Hand Frequencies (5-card)')
axes[0, 0].grid(alpha=0.3, axis='x')

# Plot 2: Probability by hand rank
hand_probs = [hand_counts[h] / total_hands * 100 for h in hand_names]
cumulative_probs = np.cumsum(hand_probs[::-1])[::-1]

axes[0, 1].plot(range(len(hand_names)), cumulative_probs, 'o-', linewidth=2, markersize=8)
axes[0, 1].set_xticks(range(len(hand_names)))
axes[0, 1].set_xticklabels(hand_names, rotation=45, ha='right')
axes[0, 1].set_ylabel('Cumulative Probability (%)')
axes[0, 1].set_title('Cumulative Hand Probability')
axes[0, 1].grid(alpha=0.3)

# Plot 3: Outs to hit by turn/river
outs_range = np.arange(1, 16)
prob_turn = outs_range / 47  # 47 unknown cards on turn
prob_river = outs_range / 46  # 46 unknown cards on river (if miss turn)
prob_either = 1 - ((47-outs_range)/47) * ((46-outs_range)/46)

axes[1, 0].plot(outs_range, prob_turn * 100, 'o-', label='Turn only', linewidth=2)
axes[1, 0].plot(outs_range, prob_river * 100, 's-', label='River only', linewidth=2)
axes[1, 0].plot(outs_range, prob_either * 100, '^-', label='Turn or River', linewidth=2)
axes[1, 0].set_xlabel('Number of Outs')
axes[1, 0].set_ylabel('Probability of Hitting (%)')
axes[1, 0].set_title('Outs Probability')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Pot odds vs break-even
pot_odds_range = np.logspace(0, 2, 50)
break_even_prob = 1 / pot_odds_range

# Example hands
colors_hands = ['green', 'orange', 'red']
hand_outs = [9, 6, 4]
hand_names_outs = ['Flush draw (9 outs)', 'Straight draw (6)', 'Pair to trips (4)']

for outs, color, name in zip(hand_outs, colors_hands, hand_names_outs):
    hit_prob = outs / 47
    axes[1, 1].axhline(hit_prob, color=color, linestyle='--', alpha=0.5, label=f'{name}: {hit_prob:.1%}')

axes[1, 1].plot(pot_odds_range, break_even_prob, 'k-', linewidth=2, label='Break-even')
axes[1, 1].fill_between(pot_odds_range, break_even_prob, 1, alpha=0.2, color='green', label='Profitable calls')
axes[1, 1].set_xscale('log')
axes[1, 1].set_xlabel('Pot Odds (e.g., 3:1)')
axes[1, 1].set_ylabel('Win Probability Required')
axes[1, 1].set_title('Pot Odds vs Hit Probability')
axes[1, 1].set_ylim([0, 0.3])
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
