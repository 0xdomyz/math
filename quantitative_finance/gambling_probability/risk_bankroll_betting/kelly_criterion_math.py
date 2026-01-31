"""
Kelly Criterion: core calculations.
Extracted from: kelly_criterion.py
"""

import matplotlib.pyplot as plt
import numpy as np


def kelly_fraction(prob_win, odds, b=1):
    """
    Calculate Kelly criterion
    prob_win: probability of winning
    odds: payout ratio (net odds received on win)
    b: odds parameter (1 for even money bets, 2 for 2:1 odds, etc.)
    """
    q = 1 - prob_win
    f_kelly = (b * prob_win - q) / b
    return max(f_kelly, 0)  # Don't bet if negative EV
