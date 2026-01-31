# Auto-extracted from markdown file
# Source: long_term_disability.md

# --- Code Block 1 ---
import numpy as np

# Claim parameters
monthly_benefit = 5000  # Net after offsets
current_age = 45
benefit_to_age = 65
annual_discount_rate = 0.03
months_remaining = (benefit_to_age - current_age) * 12

# Simplified recovery/mortality termination rates per month (illustrative)
# Assume constant 1% per month probability of claim termination
monthly_termination_prob = 0.01
monthly_discount_factor = (1 + annual_discount_rate) ** (1/12)

# Calculate present value of future benefits
pv_benefits = 0
survival_prob = 1.0  # Probability claim is still active

for month in range(1, months_remaining + 1):
    # Benefit paid at end of month if claim still active
    pv_benefits += monthly_benefit * survival_prob / (monthly_discount_factor ** month)
    # Update survival probability (claim continues if not terminated)
    survival_prob *= (1 - monthly_termination_prob)

print(f"Current Age: {current_age}")
print(f"Benefit to Age: {benefit_to_age}")
print(f"Monthly Benefit: ${monthly_benefit:,}")
print(f"Months Remaining: {months_remaining}")
print(f"Annual Discount Rate: {annual_discount_rate:.1%}")
print(f"Monthly Termination Probability: {monthly_termination_prob:.1%}")
print(f"\nPresent Value of Claim Reserve: ${pv_benefits:,.2f}")

# Sensitivity: increase termination rate to 1.5% (better recovery outcomes)
survival_prob_adj = 1.0
pv_benefits_adj = 0
monthly_termination_prob_adj = 0.015

for month in range(1, months_remaining + 1):
    pv_benefits_adj += monthly_benefit * survival_prob_adj / (monthly_discount_factor ** month)
    survival_prob_adj *= (1 - monthly_termination_prob_adj)

print(f"\nAdjusted Reserve (1.5% termination): ${pv_benefits_adj:,.2f}")
print(f"Reserve Reduction: ${pv_benefits - pv_benefits_adj:,.2f}")

