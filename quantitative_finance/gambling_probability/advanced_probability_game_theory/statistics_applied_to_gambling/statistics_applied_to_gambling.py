"""
Extracted from: statistics_applied_to_gambling.md
"""

import math

wins, n = 60, 100
z = 1.96
phat = wins / n
center = (phat + z*z/(2*n)) / (1 + z*z/n)
margin = z * math.sqrt((phat*(1-phat)+z*z/(4*n)) / n) / (1 + z*z/n)

print("95% CI:", round(center - margin, 3), "to", round(center + margin, 3))
