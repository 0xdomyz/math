"""
Extracted from: ev_in_sports_betting.md
"""

def ev(prob, odds):
    if odds < 0:
        win = 100 / abs(odds)
    else:
        win = odds / 100
    return prob * win - (1 - prob) * 1

print("EV:", round(ev(0.55, -110), 4))
