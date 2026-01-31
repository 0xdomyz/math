# Auto-extracted from markdown file
# Source: critical_illness_insurance.md

# --- Code Block 1 ---
incidence = 0.003  # 0.3% annually
benefit = 100000
admin_rate = 0.15
profit_margin = 0.10
gross_premium = (incidence * benefit * (1 + admin_rate)) / (1 - profit_margin)
print("Premium:", gross_premium)

