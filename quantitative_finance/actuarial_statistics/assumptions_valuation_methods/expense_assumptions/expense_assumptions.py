# Auto-extracted from markdown file
# Source: expense_assumptions.md

# --- Code Block 1 ---
premium = 1000
renewal_years = 9
acquisition = 200
renewal_exp = 50 * (1.03 ** np.arange(renewal_years))

total = acquisition + renewal_exp.sum()
print("Total expenses:", total)

