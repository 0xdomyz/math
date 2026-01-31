"""
Expected Value (EV) Analysis: core calculations.
Extracted from: expected_value.py
"""

import matplotlib.pyplot as plt
import numpy as np


def calculate_ev(prob_win, payout, bet_amount=1):
    """
    Calculate expected value
    prob_win: probability of winning
    payout: amount returned if win (including bet)
    bet_amount: amount risked
    """
    prob_loss = 1 - prob_win
    outcome_win = payout - bet_amount
    outcome_loss = -bet_amount

    ev = prob_win * outcome_win + prob_loss * outcome_loss
    ev_pct = (ev / bet_amount) * 100

    return ev, ev_pct
