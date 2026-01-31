"""
Independence and Dependence: Core definitions and helpers.
Extracted from: independence_dependence.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.stats import chi2_contingency, kendalltau, spearmanr


class Roulette:
    """European roulette wheel (18 red, 18 black, 1 green)."""

    def spin(self, n_spins):
        """Spin wheel n times; return outcomes (1=red, 0=black, -1=green)."""
        outcomes = np.random.choice(
            [1, 0, -1], size=n_spins, p=[18 / 37, 18 / 37, 1 / 37]
        )
        return outcomes

    def test_independence_chi_square(self, n_spins=500):
        """Test if consecutive spins are independent using chi-square."""
        outcomes = self.spin(n_spins)

        # Create contingency table: spin i vs spin i+1
        pairs = list(zip(outcomes[:-1], outcomes[1:]))

        # Count joint occurrences
        contingency_table = np.zeros((3, 3))
        for prev, next_ in pairs:
            if prev >= 0 and next_ >= 0:  # Ignore green for simplicity
                contingency_table[int(prev), int(next_)] += 1

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        return {
            "chi2": chi2,
            "p_value": p_value,
            "contingency": contingency_table,
            "expected": expected,
            "reject_independence": p_value < 0.05,
        }


class PokerDeck:
    """Poker deck (52 cards, 4 suits/rank)."""

    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    HIGH_CARDS = ["10", "J", "Q", "K", "A"]  # 20 cards

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
        dealt = self.cards[self.index : self.index + n]
        self.index += n
        return dealt
