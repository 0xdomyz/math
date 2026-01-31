# Auto-extracted from markdown file
# Source: retirement_plans.md

# --- Code Block 1 ---
import numpy as np

# Employee parameters
current_age = 45
retirement_age = 65
life_expectancy = 85  # assumed for simplicity (use mortality table in practice)
final_average_salary = 100_000
years_of_service = 20
accrual_rate = 0.015  # 1.5% per year of service

# Benefit calculation
annual_benefit = accrual_rate * final_average_salary * years_of_service
print(f"Annual Pension Benefit (starting at age 65): ${annual_benefit:,.2f}")

# Actuarial assumptions
discount_rate = 0.05  # 5% (corporate bond yield)
mortality_decrement = 0.02  # 2% annual probability of death (simplified; use life table in practice)

# Present value at retirement (annuity certain for life expectancy - retirement age)
years_in_retirement = life_expectancy - retirement_age
pv_at_retirement = 0

for year in range(1, years_in_retirement + 1):
    survival_prob = (1 - mortality_decrement) ** year
    pv_at_retirement += (annual_benefit * survival_prob) / ((1 + discount_rate) ** year)

print(f"PV of Benefits at Retirement (age {retirement_age}): ${pv_at_retirement:,.2f}")

# Discount back to current age (deferred annuity)
years_to_retirement = retirement_age - current_age
pv_current = pv_at_retirement / ((1 + discount_rate) ** years_to_retirement)

print(f"PV of Benefits at Current Age ({current_age}): ${pv_current:,.2f}")

# Sensitivity: increase discount rate to 6%
discount_rate_high = 0.06
pv_at_retirement_high = 0
for year in range(1, years_in_retirement + 1):
    survival_prob = (1 - mortality_decrement) ** year
    pv_at_retirement_high += (annual_benefit * survival_prob) / ((1 + discount_rate_high) ** year)

pv_current_high = pv_at_retirement_high / ((1 + discount_rate_high) ** years_to_retirement)
print(f"\nHigh Discount Rate (6%) PV at Current Age: ${pv_current_high:,.2f}")
print(f"PV Reduction: ${pv_current - pv_current_high:,.2f} ({(pv_current - pv_current_high)/pv_current:.1%})")

