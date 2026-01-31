# Auto-extracted from markdown file
# Source: defined_benefit_plans.md

# --- Code Block 1 ---
import numpy as np

salary = 50000
years_service = 10
benefit_rate = 0.015
final_benefit = salary * benefit_rate * years_service
pv_factor = 1 / 1.03**15  # 15 years to retirement, 3% discount
liability = final_benefit * pv_factor
print("Liability:", liability)

