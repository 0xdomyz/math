# Auto-extracted from markdown file
# Source: accrued_benefit_obligation.md

# --- Code Block 1 ---
salary = 60000
years_service = 8
vesting = 0.015
pv_factor = 1 / 1.04**10
abo = salary * years_service * vesting * pv_factor
print("ABO:", abo)

