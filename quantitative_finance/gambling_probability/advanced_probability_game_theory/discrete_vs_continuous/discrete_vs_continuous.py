"""
Extracted from: discrete_vs_continuous.md
"""

import numpy as np

wins = np.random.binomial(100, 1/6, size=1000)
waiting = np.random.exponential(scale=6, size=1000)

print("Avg dice wins:", wins.mean())
print("Avg waiting time:", round(waiting.mean(), 2))
