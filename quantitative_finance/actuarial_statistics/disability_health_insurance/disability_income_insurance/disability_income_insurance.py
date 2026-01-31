# Auto-extracted from markdown file
# Source: disability_income_insurance.md

# --- Code Block 1 ---
salary = 60000
replacement = 0.60
disability_rate = 0.005
years = 2
pv_factor = 1 / 1.03
liability = salary * replacement * disability_rate * years * pv_factor
print("DI liability:", liability)

