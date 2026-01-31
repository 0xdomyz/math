"""
Extracted from: probability_distributions_in_games.md
"""

import numpy as np

p = 0.52
n = 100
trials = 5000
wins = np.random.binomial(n, p, size=trials)

mean = wins.mean()
std = wins.std()
print("Mean wins:", round(mean, 2), "Std:", round(std, 2))
