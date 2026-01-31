"""
Extracted from: simulation_monte_carlo.md
"""

import numpy as np

spins = 100000
payout = 35
win_prob = 1/37  # single number European roulette
wins = np.random.binomial(spins, win_prob)

ev = (wins * payout - (spins - wins)) / spins
print("Estimated EV per $1 bet:", round(ev, 4))
