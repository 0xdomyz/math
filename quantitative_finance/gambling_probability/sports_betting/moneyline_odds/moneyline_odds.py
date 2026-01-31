"""
Extracted from: moneyline_odds.md
"""

def implied_prob_american(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

for o in [-150, +200]:
    print(o, round(implied_prob_american(o), 4))
