"""
Poker Probabilities Analysis: core simulations.
Extracted from: poker_probabilities.py
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def calculate_hand_equity_simulation(
    hero_cards, villain_cards, board, num_simulations=100000
):
    """
    Simulate poker equity between two hands
    Cards format: ['As', 'Kd', '2h'] etc
    """
    deck = [f"{r}{s}" for r in "23456789TJQKA" for s in "hcds"]

    # Remove known cards
    for card in hero_cards + villain_cards + board:
        deck.remove(card)

    hero_wins = 0
    villain_wins = 0
    ties = 0

    for _ in range(num_simulations):
        sample_deck = deck.copy()
        np.random.shuffle(sample_deck)

        # Complete boards (5 community cards)
        cards_needed = 5 - len(board)
        remaining_board = board + sample_deck[:cards_needed]
        remaining_deck = sample_deck[cards_needed:]

        # Evaluate hands (simplified: value by combinations)
        hero_best = evaluate_hand(hero_cards + remaining_board)
        villain_best = evaluate_hand(villain_cards + remaining_board)

        if hero_best > villain_best:
            hero_wins += 1
        elif villain_best > hero_best:
            villain_wins += 1
        else:
            ties += 1

    total = hero_wins + villain_wins + ties
    hero_equity = hero_wins / total if total > 0 else 0
    villain_equity = villain_wins / total if total > 0 else 0
    tie_equity = ties / total if total > 0 else 0

    return hero_equity, villain_equity, tie_equity


def evaluate_hand(seven_cards):
    """
    Simple hand evaluator - counts hand strength
    Returns numeric value for 5-card best hand
    Simplified: just checking pairs, straights, flushes
    """
    ranks = [card[0] for card in seven_cards]
    suits = [card[1] for card in seven_cards]

    rank_values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }
    rank_nums = sorted([rank_values[r] for r in ranks])

    from collections import Counter

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Hand strength (simplified)
    is_flush = max(suit_counts.values()) >= 5
    is_straight = any(
        rank_nums[i : i + 5] == list(range(rank_nums[i], rank_nums[i] + 5))
        for i in range(len(rank_nums) - 4)
    ) or set(rank_nums[-5:]) == {10, 11, 12, 13, 14}

    counts_sorted = sorted(rank_counts.values(), reverse=True)

    if is_straight and is_flush:
        return 800  # Straight flush
    elif counts_sorted == [4, 1, 1, 1]:
        return 700  # Four of a kind
    elif counts_sorted == [3, 2, 1, 1]:
        return 600  # Full house
    elif is_flush:
        return 500  # Flush
    elif is_straight:
        return 400  # Straight
    elif counts_sorted == [3, 1, 1, 1, 1]:
        return 300  # Three of a kind
    elif counts_sorted == [2, 2, 1, 1, 1]:
        return 200  # Two pair
    elif counts_sorted == [2, 1, 1, 1, 1, 1]:
        return 100  # One pair
    else:
        return rank_nums[-1]  # High card
