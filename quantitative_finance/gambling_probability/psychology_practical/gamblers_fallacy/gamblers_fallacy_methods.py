"""
Gambler's Fallacy Analysis: core simulations.
Extracted from: gamblers_fallacy.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def simulate_roulette_fallacy(num_spins=10000):
    """
    Simulate roulette spins and track "due" betting
    Red = 1, Black = 0 (simplified, ignore green)
    """
    red_numbers = set(
        [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
    )

    spins = []
    streak_lengths = []
    current_color = None
    current_streak = 0

    for _ in range(num_spins):
        spin = np.random.randint(0, 37)  # European roulette
        if spin in red_numbers:
            color = "red"
        elif spin == 0:
            color = "green"
        else:
            color = "black"

        spins.append(color)

        if color == current_color and color != "green":
            current_streak += 1
        else:
            if current_streak > 0:
                streak_lengths.append(current_streak)
            current_color = color
            current_streak = 1

    return spins, streak_lengths


def test_independence(spins, color_to_test="red"):
    """
    Test if outcome is independent of previous outcome
    """
    # Count transitions
    prev_color = None
    after_red = {"red": 0, "black": 0, "green": 0}
    after_black = {"red": 0, "black": 0, "green": 0}

    for spin in spins:
        if prev_color == "red":
            after_red[spin] += 1
        elif prev_color == "black":
            after_black[spin] += 1
        prev_color = spin

    total_after_red = sum(after_red.values())
    total_after_black = sum(after_black.values())

    return after_red, after_black, total_after_red, total_after_black


def simulate_fallacy_betting(num_sessions=1000):
    """
    Simulate betting on 'due' outcomes vs random
    """
    fallacy_bankroll = 1000
    random_bankroll = 1000

    for _ in range(num_sessions):
        spins = [np.random.randint(0, 37) for _ in range(100)]

        # Gambler's fallacy: Bet on color after 3-streak
        current_streak = 0
        prev_color = None

        for spin in spins:
            red_set = set(
                [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
            )
            if spin in red_set:
                color = "red"
            elif spin == 0:
                color = "green"
            else:
                color = "black"

            # Fallacy bettor: Bet against streak of 3+
            if current_streak >= 3 and prev_color in ["red", "black"]:
                opposite_color = "black" if prev_color == "red" else "red"
                if color == opposite_color:
                    fallacy_bankroll += 10
                else:
                    fallacy_bankroll -= 10

            # Random bettor: Bet on random color each time
            if np.random.random() < 0.5:
                if color == "red":
                    random_bankroll += 10
                else:
                    random_bankroll -= 10
            else:
                if color == "black":
                    random_bankroll += 10
                else:
                    random_bankroll -= 10

            # Update streak
            if color == prev_color and color != "green":
                current_streak += 1
            else:
                current_streak = 1
                prev_color = color

    return fallacy_bankroll, random_bankroll
