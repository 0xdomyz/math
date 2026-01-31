"""
Extracted from: game_selection_discipline.md
"""

rules = {"3:2_S17": 0.5, "6:5_H17": 1.9, "3:2_H17": 0.7}
for r, edge in rules.items():
    print(r, "house edge", edge, "%")
