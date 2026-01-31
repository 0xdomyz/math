"""
Extracted from: implied_probability.md
"""

def implied_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

odds = +150
true_p = 0.45
imp_p = implied_prob(odds)
edge = true_p - imp_p
print("Edge:", round(edge, 4))
