"""
Extracted from: over_under_betting.md
"""

import math

mean_total = 52
std_total = 10
line = 48.5
z = (line - mean_total) / std_total
p_over = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
print("Over probability:", round(p_over, 3))
