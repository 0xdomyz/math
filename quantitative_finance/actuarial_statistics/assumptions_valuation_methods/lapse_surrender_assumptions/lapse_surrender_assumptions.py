# Auto-extracted from markdown file
# Source: lapse_surrender_assumptions.md

# --- Code Block 1 ---
base_lapse = 0.05
rate_shock = 0.02  # 200bps increase
elasticity = -0.10  # 10% lapse increase per 100bps
adjusted_lapse = base_lapse * (1 + elasticity * rate_shock / 0.01)
print("Adjusted lapse:", adjusted_lapse)

