"""
Extracted from: bayesian_updating.md
"""

from math import comb

alpha, beta = 2, 2  # prior
wins, losses = 12, 8
posterior_alpha = alpha + wins
posterior_beta = beta + losses
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

print("Posterior win rate:", round(posterior_mean, 3))
