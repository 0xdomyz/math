# Auto-extracted from markdown file
# Source: medical_health_insurance.md

# --- Code Block 1 ---
import numpy as np

# Prior year group experience
prior_year_claims = 1_800_000  # Total paid claims
prior_year_members = 200
prior_year_pmpm = prior_year_claims / (prior_year_members * 12)

# Medical trend and demographic adjustments
medical_trend = 0.07  # 7% annual increase
demographic_factor = 1.02  # Aging workforce, 2% increase in expected costs
admin_load = 0.15  # 15% for admin and profit

# Calculate expected PMPM for renewal year
expected_claims_pmpm = prior_year_pmpm * (1 + medical_trend) * demographic_factor
premium_pmpm = expected_claims_pmpm / (1 - admin_load)

print(f"Prior Year Claims: ${prior_year_claims:,}")
print(f"Prior Year Members: {prior_year_members}")
print(f"Prior Year PMPM (claims only): ${prior_year_pmpm:,.2f}")
print(f"Medical Trend: {medical_trend:.1%}")
print(f"Demographic Factor: {demographic_factor:.2f}")
print(f"Expected Claims PMPM (renewal): ${expected_claims_pmpm:,.2f}")
print(f"Admin Load: {admin_load:.1%}")
print(f"Renewal Premium PMPM: ${premium_pmpm:,.2f}")
print(f"Annual Premium per Member: ${premium_pmpm * 12:,.2f}")
print(f"Total Group Annual Premium (200 members): ${premium_pmpm * 12 * 200:,.2f}")

# Sensitivity: reduce trend to 5% (e.g., improved utilization management)
expected_claims_pmpm_low = prior_year_pmpm * 1.05 * demographic_factor
premium_pmpm_low = expected_claims_pmpm_low / (1 - admin_load)
print(f"\nLow-Trend Scenario (5%):")
print(f"Renewal Premium PMPM: ${premium_pmpm_low:,.2f}")
print(f"Savings per member per year: ${(premium_pmpm - premium_pmpm_low) * 12:,.2f}")

