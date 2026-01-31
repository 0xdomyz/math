"""
Extracted from: rake_in_poker.md
"""

pots_per_hour = 25
avg_pot = 60
rake_rate = 0.05
cap = 5

rake_per_pot = min(avg_pot * rake_rate, cap)
 hourly_cost = pots_per_hour * rake_per_pot
print("Rake per hour:", round(hourly_cost, 2))
