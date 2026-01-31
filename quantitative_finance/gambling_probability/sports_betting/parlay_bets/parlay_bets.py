"""
Extracted from: parlay_bets.md
"""

p1, p2, payout = 0.52, 0.52, 2.7
p_win = p1 * p2
 ev = p_win * payout - (1 - p_win)
print("Parlay EV:", round(ev, 4))
