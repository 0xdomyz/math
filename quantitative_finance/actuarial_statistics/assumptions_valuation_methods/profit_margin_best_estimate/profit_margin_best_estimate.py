# Auto-extracted from markdown file
# Source: profit_margin_best_estimate.md

# --- Code Block 1 ---
best_estimate_cost = 100
best_estimate_return = 20
profit_margin_pct = 0.15

gross_premium = best_estimate_cost * (1 + profit_margin_pct)
profit = gross_premium - best_estimate_cost
print("Gross premium:", gross_premium, "Profit:", profit)

