"""
Extracted from: texas_holdem_equity_calculation.md
"""

import random
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "shdc"

def deck():
    return [r+s for r in RANKS for s in SUITS]

def hand_score(cards):
    """Toy evaluator: counts pairs/trips/quads, then high-card fallback."""
    ranks = [c[0] for c in cards]
    counts = Counter(ranks)
    counts_sorted = sorted(counts.values(), reverse=True)
    high = max([RANKS.index(r) for r in ranks])
    return (counts_sorted, high)

def equity(hand_a, hand_b, trials=5000):
    wins = ties = 0
    for _ in range(trials):
        d = deck()
        for c in hand_a + hand_b:
            d.remove(c)
        board = random.sample(d, 5)
        score_a = hand_score(hand_a + board)
        score_b = hand_score(hand_b + board)
        if score_a > score_b:
            wins += 1
        elif score_a == score_b:
            ties += 1
    return (wins + 0.5 * ties) / trials

print("AA vs KK equity (toy):", equity(["As","Ah"],["Ks","Kh"]))
