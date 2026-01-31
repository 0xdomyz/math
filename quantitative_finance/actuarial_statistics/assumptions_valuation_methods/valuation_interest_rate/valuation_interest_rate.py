# Auto-extracted from markdown file
# Source: valuation_interest_rate.md

# --- Code Block 1 ---
reserve = 1000000
duration = 7.5
rate_change = 0.01

reserve_change = -reserve * duration * rate_change
print("Reserve impact:", reserve_change)

