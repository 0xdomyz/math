# Auto-extracted from markdown file
# Source: projected_benefit_obligation.md

# --- Code Block 1 ---
current_salary = 50000
salary_growth = 0.03
years_to_retirement = 15
final_salary = current_salary * (1 + salary_growth) ** years_to_retirement
benefit = final_salary * 0.015 * (8 + years_to_retirement)
pv = benefit / (1.04 ** years_to_retirement)
print("PBO:", pv)

