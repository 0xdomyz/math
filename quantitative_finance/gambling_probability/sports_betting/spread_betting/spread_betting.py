"""
Extracted from: spread_betting.md
"""

import math

mean_margin = 4
std_margin = 10
spread = 3.5

z = (spread - mean_margin) / std_margin
p_cover = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
print("Cover probability:", round(p_cover, 3))
