"""
Extracted from: commission_in_baccarat.md
"""

p_banker = 0.5068
p_player = 0.4932
p_tie = 0.0952

banker_ev = p_banker * 0.95 - (1 - p_banker - p_tie)
player_ev = p_player * 1.0 - (1 - p_player - p_tie)

print("Banker EV:", round(banker_ev, 4))
print("Player EV:", round(player_ev, 4))
